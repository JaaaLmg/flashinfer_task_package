#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>

namespace {

constexpr int kWarpSize = 64;
constexpr uint64_t kFullWarpMask = ~uint64_t{0};
constexpr int kPageSize = 16;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(kFullWarpMask, value, offset, kWarpSize);
  }
  return __shfl_sync(kFullWarpMask, value, 0, kWarpSize);
}

__global__ void paged_prefill_baseline_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ kv_data,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ qo_indptr,
    const int32_t* __restrict__ kv_indptr,
    const int32_t* __restrict__ kv_indices,
    const int32_t* __restrict__ last_page_len,
    int batch_size,
    int max_seq_len,
    int num_qo_heads,
    int num_kv_heads,
    int head_dim,
    int causal) {
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp_in_block = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;

  int64_t work = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
  const int64_t total_work =
      static_cast<int64_t>(batch_size) * max_seq_len * num_qo_heads;
  if (work >= total_work) return;

  const int qo_head = work % num_qo_heads;
  work /= num_qo_heads;
  const int q_pos = work % max_seq_len;
  const int batch = work / max_seq_len;

  const int q_begin = qo_indptr[batch];
  const int q_len = qo_indptr[batch + 1] - q_begin;
  if (q_pos >= q_len) return;

  const int page_begin = kv_indptr[batch];
  const int num_pages = kv_indptr[batch + 1] - page_begin;
  const int kv_len = num_pages > 0
                         ? (num_pages - 1) * kPageSize + last_page_len[batch]
                         : 0;
  int visible = kv_len;
  if (causal) {
    visible = kv_len - q_len + q_pos + 1;
    visible = visible < 0 ? 0 : visible;
    visible = visible > kv_len ? kv_len : visible;
  }

  const int group_size = num_qo_heads / num_kv_heads;
  const int kv_head = qo_head / group_size;
  const int q_row = q_begin + q_pos;
  const float sm_scale = rsqrtf(static_cast<float>(head_dim));
  const __nv_bfloat16* q_ptr =
      q + (static_cast<int64_t>(q_row) * num_qo_heads + qo_head) * head_dim;

  float q_fragment[4];
  float out_acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    q_fragment[i] = d < head_dim ? __bfloat162float(q_ptr[d]) : 0.f;
  }

  float row_max = -1.0e20f;
  float row_sum = 0.f;
  const int64_t page_stride =
      static_cast<int64_t>(2) * kPageSize * num_kv_heads * head_dim;
  const int64_t kv_plane_stride =
      static_cast<int64_t>(kPageSize) * num_kv_heads * head_dim;

  for (int kv_pos = 0; kv_pos < visible; ++kv_pos) {
    const int logical_page = page_begin + (kv_pos >> 4);
    const int physical_page = __ldg(kv_indices + logical_page);
    const int entry = kv_pos & (kPageSize - 1);
    const int64_t token_offset =
        static_cast<int64_t>(entry) * num_kv_heads * head_dim +
        static_cast<int64_t>(kv_head) * head_dim;
    const __nv_bfloat16* k_ptr =
        kv_data + static_cast<int64_t>(physical_page) * page_stride + token_offset;
    const __nv_bfloat16* v_ptr = k_ptr + kv_plane_stride;

    float score = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      if (d < head_dim) {
        score += q_fragment[i] * __bfloat162float(k_ptr[d]);
      }
    }
    score = warp_sum(score) * sm_scale;

    const float new_max = fmaxf(row_max, score);
    const float alpha = row_max > -1.0e19f ? __expf(row_max - new_max) : 0.f;
    const float beta = __expf(score - new_max);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int d = lane + i * kWarpSize;
      if (d < head_dim) {
        out_acc[i] = out_acc[i] * alpha + beta * __bfloat162float(v_ptr[d]);
      }
    }
    row_sum = row_sum * alpha + beta;
    row_max = new_max;
  }

  __nv_bfloat16* out_ptr =
      output + (static_cast<int64_t>(q_row) * num_qo_heads + qo_head) * head_dim;
  const float inv_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int d = lane + i * kWarpSize;
    if (d < head_dim) out_ptr[d] = __float2bfloat16(out_acc[i] * inv_sum);
  }
}

}  // namespace

extern "C" void run_kernel(
    const __nv_bfloat16* q,
    const __nv_bfloat16* kv_data,
    __nv_bfloat16* output,
    const int32_t* qo_indptr,
    const int32_t* kv_indptr,
    const int32_t* kv_indices,
    const int32_t* last_page_len,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_qo_heads,
    int64_t num_kv_heads,
    int64_t head_dim,
    int64_t page_block_size,
    int64_t causal) {
  if (page_block_size != kPageSize ||
      (head_dim != 128 && head_dim != 256) ||
      num_qo_heads <= 0 || num_kv_heads <= 0 ||
      num_qo_heads % num_kv_heads != 0) {
    return;
  }

  constexpr int kThreads = 128;
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  const int64_t total_work = batch_size * seq_len * num_qo_heads;
  const int blocks = static_cast<int>(
      (total_work + kWarpsPerBlock - 1) / kWarpsPerBlock);
  paged_prefill_baseline_kernel<<<blocks, kThreads>>>(
      q, kv_data, output, qo_indptr, kv_indptr, kv_indices, last_page_len,
      static_cast<int>(batch_size), static_cast<int>(seq_len),
      static_cast<int>(num_qo_heads), static_cast<int>(num_kv_heads),
      static_cast<int>(head_dim), static_cast<int>(causal));
}
