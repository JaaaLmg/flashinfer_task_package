#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>

namespace {

// MetaX C500 exposes a 64-lane warp.  One warp computes one
// (batch, query position, query head) output row.  With D=128 each lane owns
// two elements of Q and O.
constexpr int kWarpSize = 64;
constexpr uint64_t kFullWarpMask = ~uint64_t{0};

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset, kWarpSize);
    }
    return __shfl_sync(kFullWarpMask, value, 0, kWarpSize);
}

__global__ void ragged_prefill_baseline_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ kv_indptr,
    int64_t batch_size,
    int64_t max_seq_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    int64_t causal) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;

    int64_t work = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
    const int64_t total_work = batch_size * max_seq_len * num_qo_heads;
    if (work >= total_work) {
        return;
    }

    const int64_t qo_head = work % num_qo_heads;
    work /= num_qo_heads;
    const int64_t q_pos = work % max_seq_len;
    const int64_t batch = work / max_seq_len;

    const int64_t qo_begin = static_cast<int64_t>(qo_indptr[batch]);
    const int64_t qo_len = static_cast<int64_t>(qo_indptr[batch + 1]) - qo_begin;
    if (q_pos >= qo_len) {
        return;
    }

    const int64_t kv_begin = static_cast<int64_t>(kv_indptr[batch]);
    const int64_t kv_len = static_cast<int64_t>(kv_indptr[batch + 1]) - kv_begin;

    int64_t visible = kv_len;
    if (causal) {
        // FlashInfer uses bottom-right causal alignment when qo_len != kv_len.
        visible = kv_len - qo_len + q_pos + 1;
        visible = visible < 0 ? 0 : visible;
        visible = visible > kv_len ? kv_len : visible;
    }

    const int64_t group_size = num_qo_heads / num_kv_heads;
    const int64_t kv_head = qo_head / group_size;
    const int64_t q_row = qo_begin + q_pos;
    const float sm_scale = rsqrtf(static_cast<float>(head_dim_qk));

    const __nv_bfloat16* q_ptr =
        q + (q_row * num_qo_heads + qo_head) * head_dim_qk;

    // The contest shape is Dqk=Dvo=128.  Bounds keep the implementation safe
    // for dimensions smaller than 128; run_kernel only launches this kernel
    // for the required contest shape.
    float q_fragment[2];
    float out_acc[2] = {0.0f, 0.0f};
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int d = lane + i * kWarpSize;
        q_fragment[i] = d < head_dim_qk ? __bfloat162float(q_ptr[d]) : 0.0f;
    }

    float row_max = -1.0e20f;
    float row_sum = 0.0f;

    for (int64_t kv_pos = 0; kv_pos < visible; ++kv_pos) {
        const int64_t kv_row = kv_begin + kv_pos;
        const __nv_bfloat16* k_ptr =
            k + (kv_row * num_kv_heads + kv_head) * head_dim_qk;
        const __nv_bfloat16* v_ptr =
            v + (kv_row * num_kv_heads + kv_head) * head_dim_vo;

        float score = 0.0f;
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int d = lane + i * kWarpSize;
            if (d < head_dim_qk) {
                score += q_fragment[i] * __bfloat162float(k_ptr[d]);
            }
        }
        score = warp_sum(score) * sm_scale;

        // Numerically stable online softmax.  This intentionally mirrors the
        // simple starter implementation and keeps all state in FP32.
        const float new_max = fmaxf(row_max, score);
        const float alpha = row_max > -1.0e19f ? __expf(row_max - new_max) : 0.0f;
        const float beta = __expf(score - new_max);

#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const int d = lane + i * kWarpSize;
            if (d < head_dim_vo) {
                out_acc[i] = out_acc[i] * alpha + beta * __bfloat162float(v_ptr[d]);
            }
        }
        row_sum = row_sum * alpha + beta;
        row_max = new_max;
    }

    __nv_bfloat16* out_ptr =
        output + (q_row * num_qo_heads + qo_head) * head_dim_vo;
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int d = lane + i * kWarpSize;
        if (d < head_dim_vo) {
            out_ptr[d] = __float2bfloat16(out_acc[i] * inv_sum);
        }
    }
}

}  // namespace

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* k,
    const __nv_bfloat16* v,
    __nv_bfloat16* output,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim_qk,
    int64_t head_dim_vo,
    int64_t causal) {
    // Problem 20001 fixes these values.  Avoid launching an incorrectly sized
    // register fragment if this file is accidentally used for another task.
    if (num_qo_heads != 32 || num_kv_heads != 4 ||
        head_dim_qk != 128 || head_dim_vo != 128) {
        return;
    }

    constexpr int kThreads = 128;
    constexpr int kWarpsPerBlock = kThreads / kWarpSize;
    const int64_t total_work = batch_size * seq_len * num_qo_heads;
    const int blocks = static_cast<int>(
        (total_work + kWarpsPerBlock - 1) / kWarpsPerBlock);

    ragged_prefill_baseline_kernel<<<blocks, kThreads>>>(
        q, k, v, output, qo_indptr, kv_indptr,
        batch_size, seq_len, num_qo_heads, num_kv_heads,
        head_dim_qk, head_dim_vo, causal);
}
