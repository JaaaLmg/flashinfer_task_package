#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>

namespace {

constexpr int kWarpSize = 64;
constexpr int kNumQHeads = 32;
constexpr int kNumKVHeads = 4;
constexpr int kHeadDim = 128;
constexpr int kGroupSize = 8;
constexpr int kQTokenStride = kNumQHeads * kHeadDim;
constexpr int kKVTokenStride = kNumKVHeads * kHeadDim;
constexpr uint64_t kFullWarpMask = ~uint64_t{0};
constexpr float kSmScale = 0.08838834764831845f;  // 1 / sqrt(128)

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 32; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(kFullWarpMask, value, offset, kWarpSize);
    }
    return __shfl_sync(kFullWarpMask, value, 0, kWarpSize);
}

// For Lq=Lk=1 causal attention is exactly V.  One CTA copies all 32 output
// heads and performs the fixed GQA broadcast without reading Q or K.
__global__ void single_token_kernel(
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ kv_indptr,
    int batch_size) {
    const int b = blockIdx.x;
    if (b >= batch_size) return;
    const int qo_begin = qo_indptr[b];
    const int qo_len = qo_indptr[b + 1] - qo_begin;
    const int kv_begin = kv_indptr[b];
    const int kv_len = kv_indptr[b + 1] - kv_begin;
    if (qo_len != 1 || kv_len != 1) return;

    for (int i = threadIdx.x; i < kNumQHeads * kHeadDim; i += blockDim.x) {
        const int h = i / kHeadDim;
        const int d = i & (kHeadDim - 1);
        output[static_cast<int64_t>(qo_begin) * kQTokenStride + i] =
            v[static_cast<int64_t>(kv_begin) * kKVTokenStride +
              (h / kGroupSize) * kHeadDim + d];
    }
}

__global__ void ragged_prefill_stage_b_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ kv_indptr,
    int batch_size,
    int max_seq_len) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    constexpr int kWarpsPerBlock = 2;

    int work = (blockIdx.x * kWarpsPerBlock) + warp_in_block;
    const int total_work = batch_size * max_seq_len * kNumQHeads;
    if (work >= total_work) return;

    const int qo_head = work & (kNumQHeads - 1);
    work >>= 5;
    const int q_pos = work % max_seq_len;
    const int batch = work / max_seq_len;

    const int qo_begin = qo_indptr[batch];
    const int qo_len = qo_indptr[batch + 1] - qo_begin;
    if (q_pos >= qo_len) return;
    const int kv_begin = kv_indptr[batch];
    const int kv_len = kv_indptr[batch + 1] - kv_begin;
    int visible = kv_len - qo_len + q_pos + 1;
    visible = visible < 0 ? 0 : visible;
    visible = visible > kv_len ? kv_len : visible;

    const int kv_head = qo_head >> 3;
    const int q_row = qo_begin + q_pos;
    const __nv_bfloat16* q_ptr =
        q + static_cast<int64_t>(q_row) * kQTokenStride + qo_head * kHeadDim;
    const __nv_bfloat16* k_ptr =
        k + static_cast<int64_t>(kv_begin) * kKVTokenStride + kv_head * kHeadDim;
    const __nv_bfloat16* v_ptr =
        v + static_cast<int64_t>(kv_begin) * kKVTokenStride + kv_head * kHeadDim;

    const float q0 = __bfloat162float(q_ptr[lane]);
    const float q1 = __bfloat162float(q_ptr[lane + kWarpSize]);
    float out0 = 0.0f, out1 = 0.0f;
    float row_max = -INFINITY;
    float row_sum = 0.0f;

    for (int kv_pos = 0; kv_pos < visible; ++kv_pos) {
        float score = q0 * __bfloat162float(k_ptr[lane]) +
                      q1 * __bfloat162float(k_ptr[lane + kWarpSize]);
        score = warp_sum(score) * kSmScale;

        // The comparison is warp-uniform.  Exactly one exponential is needed:
        // either rescale the old state or weight the new value.
        if (score <= row_max) {
            const float beta = __expf(score - row_max);
            out0 += beta * __bfloat162float(v_ptr[lane]);
            out1 += beta * __bfloat162float(v_ptr[lane + kWarpSize]);
            row_sum += beta;
        } else {
            const float alpha = row_sum > 0.0f ? __expf(row_max - score) : 0.0f;
            out0 = out0 * alpha + __bfloat162float(v_ptr[lane]);
            out1 = out1 * alpha + __bfloat162float(v_ptr[lane + kWarpSize]);
            row_sum = row_sum * alpha + 1.0f;
            row_max = score;
        }
        k_ptr += kKVTokenStride;
        v_ptr += kKVTokenStride;
    }

    __nv_bfloat16* out_ptr =
        output + static_cast<int64_t>(q_row) * kQTokenStride + qo_head * kHeadDim;
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    out_ptr[lane] = __float2bfloat16(out0 * inv_sum);
    out_ptr[lane + kWarpSize] = __float2bfloat16(out1 * inv_sum);
}

}  // namespace

extern "C" void run_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* output, const int32_t* qo_indptr, const int32_t* kv_indptr,
    int64_t batch_size, int64_t seq_len, int64_t num_qo_heads,
    int64_t num_kv_heads, int64_t head_dim_qk, int64_t head_dim_vo,
    int64_t causal) {
    if (num_qo_heads != kNumQHeads || num_kv_heads != kNumKVHeads ||
        head_dim_qk != kHeadDim || head_dim_vo != kHeadDim || !causal) return;

    if (seq_len == 1) {
        single_token_kernel<<<static_cast<int>(batch_size), 256>>>(
            v, output, qo_indptr, kv_indptr, static_cast<int>(batch_size));
        return;
    }
    constexpr int kThreads = 128;
    constexpr int kWarpsPerBlock = kThreads / kWarpSize;
    const int64_t total_work = batch_size * seq_len * kNumQHeads;
    const int blocks = static_cast<int>((total_work + kWarpsPerBlock - 1) /
                                        kWarpsPerBlock);
    ragged_prefill_stage_b_kernel<<<blocks, kThreads>>>(
        q, k, v, output, qo_indptr, kv_indptr,
        static_cast<int>(batch_size), static_cast<int>(seq_len));
}
