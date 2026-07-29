// XPU-OJ 20003: FlashInfer MLA Paged Attention (DeepSeek MLA decode).
//
// Exact, unoptimised reference implementation used as the correctness anchor.
// One 64-lane warp produces one output row (batch, head):
//
//   score_j = (dot(q_nope[b,h,:], ckv[t_j,0,:]) +
//              dot(q_pe  [b,h,:], kpe[t_j,0,:])) * sm_scale
//   p       = softmax(score)
//   out     = sum_j p_j * ckv[t_j,0,:]
//
// with t_j = kv_indices[kv_indptr[b] + j] because page_size == 1.

#include <stdint.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <math.h>

namespace
{

  constexpr int kWarpSize = 64;
  constexpr uint64_t kFullWarpMask = ~uint64_t{0};
  // 512 / 64 lanes and 64 / 64 lanes for the DeepSeek MLA head dimensions.
  constexpr int kMaxCkvPerLane = 8;
  constexpr int kMaxPePerLane = 1;

  __device__ __forceinline__ float warp_sum(float value)
  {
#pragma unroll
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
    {
      value += __shfl_down_sync(kFullWarpMask, value, offset, kWarpSize);
    }
    return __shfl_sync(kFullWarpMask, value, 0, kWarpSize);
  }

  __global__ void mla_paged_baseline_kernel(
      const __nv_bfloat16 *__restrict__ q_nope,
      const __nv_bfloat16 *__restrict__ q_pe,
      const __nv_bfloat16 *__restrict__ ckv,
      const __nv_bfloat16 *__restrict__ kpe,
      __nv_bfloat16 *__restrict__ output,
      const int32_t *__restrict__ q_indptr,
      const int32_t *__restrict__ kv_indptr,
      const int32_t *__restrict__ kv_indices,
      const int32_t *__restrict__ kv_lens,
      int batch_size,
      int num_heads,
      int head_dim_ckv,
      int head_dim_kpe,
      float sm_scale)
  {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp_in_block = threadIdx.x / kWarpSize;
    const int warps_per_block = blockDim.x / kWarpSize;

    int64_t work = static_cast<int64_t>(blockIdx.x) * warps_per_block + warp_in_block;
    const int64_t total_work = static_cast<int64_t>(batch_size) * num_heads;
    if (work >= total_work)
      return;

    const int head = static_cast<int>(work % num_heads);
    const int batch = static_cast<int>(work / num_heads);

    const int q_row = q_indptr[batch];
    const int page_begin = kv_indptr[batch];
    const int page_count = kv_indptr[batch + 1] - page_begin;
    int kv_len = kv_lens[batch];
    if (kv_len > page_count)
      kv_len = page_count;
    if (kv_len < 0)
      kv_len = 0;

    const __nv_bfloat16 *q_nope_ptr =
        q_nope + (static_cast<int64_t>(q_row) * num_heads + head) * head_dim_ckv;
    const __nv_bfloat16 *q_pe_ptr =
        q_pe + (static_cast<int64_t>(q_row) * num_heads + head) * head_dim_kpe;

    float q_nope_frag[kMaxCkvPerLane];
    float q_pe_frag[kMaxPePerLane];
    float out_acc[kMaxCkvPerLane];
#pragma unroll
    for (int i = 0; i < kMaxCkvPerLane; ++i)
    {
      const int d = lane + i * kWarpSize;
      q_nope_frag[i] = d < head_dim_ckv ? __bfloat162float(q_nope_ptr[d]) : 0.f;
      out_acc[i] = 0.f;
    }
#pragma unroll
    for (int i = 0; i < kMaxPePerLane; ++i)
    {
      const int d = lane + i * kWarpSize;
      q_pe_frag[i] = d < head_dim_kpe ? __bfloat162float(q_pe_ptr[d]) : 0.f;
    }

    float row_max = -1.0e20f;
    float row_sum = 0.f;

    for (int kv_pos = 0; kv_pos < kv_len; ++kv_pos)
    {
      // page_size == 1, so one page entry is exactly one KV token.
      const int token = __ldg(kv_indices + page_begin + kv_pos);
      const __nv_bfloat16 *ckv_ptr =
          ckv + static_cast<int64_t>(token) * head_dim_ckv;
      const __nv_bfloat16 *kpe_ptr =
          kpe + static_cast<int64_t>(token) * head_dim_kpe;

      float score = 0.f;
#pragma unroll
      for (int i = 0; i < kMaxCkvPerLane; ++i)
      {
        const int d = lane + i * kWarpSize;
        if (d < head_dim_ckv)
          score += q_nope_frag[i] * __bfloat162float(ckv_ptr[d]);
      }
#pragma unroll
      for (int i = 0; i < kMaxPePerLane; ++i)
      {
        const int d = lane + i * kWarpSize;
        if (d < head_dim_kpe)
          score += q_pe_frag[i] * __bfloat162float(kpe_ptr[d]);
      }
      score = warp_sum(score) * sm_scale;

      const float new_max = fmaxf(row_max, score);
      const float alpha = row_max > -1.0e19f ? __expf(row_max - new_max) : 0.f;
      const float beta = __expf(score - new_max);
#pragma unroll
      for (int i = 0; i < kMaxCkvPerLane; ++i)
      {
        const int d = lane + i * kWarpSize;
        if (d < head_dim_ckv)
        {
          // MLA uses the compressed KV cache itself as V.
          out_acc[i] = out_acc[i] * alpha + beta * __bfloat162float(ckv_ptr[d]);
        }
      }
      row_sum = row_sum * alpha + beta;
      row_max = new_max;
    }

    __nv_bfloat16 *out_ptr =
        output + (static_cast<int64_t>(q_row) * num_heads + head) * head_dim_ckv;
    const float inv_sum = row_sum > 0.f ? 1.f / row_sum : 0.f;
#pragma unroll
    for (int i = 0; i < kMaxCkvPerLane; ++i)
    {
      const int d = lane + i * kWarpSize;
      if (d < head_dim_ckv)
        out_ptr[d] = __float2bfloat16(out_acc[i] * inv_sum);
    }
  }

} // namespace

extern "C" void run_kernel(
    const __nv_bfloat16 *q_nope,
    const __nv_bfloat16 *q_pe,
    const __nv_bfloat16 *ckv,
    const __nv_bfloat16 *kpe,
    __nv_bfloat16 *output,
    const int32_t *q_indptr,
    const int32_t *kv_indptr,
    const int32_t *kv_indices,
    const int32_t *kv_lens,
    int64_t batch_size,
    int64_t seq_len,
    int64_t num_heads,
    int64_t head_dim_ckv,
    int64_t head_dim_kpe,
    int64_t page_size,
    int64_t causal)
{
  (void)seq_len;
  (void)causal; // decode has a single query row per request, causal is a no-op.

  if (page_size != 1 || batch_size <= 0 || num_heads <= 0 ||
      head_dim_ckv <= 0 || head_dim_ckv > kMaxCkvPerLane * kWarpSize ||
      head_dim_kpe <= 0 || head_dim_kpe > kMaxPePerLane * kWarpSize)
  {
    return;
  }

  const float sm_scale =
      1.0f / sqrtf(static_cast<float>(head_dim_ckv + head_dim_kpe));

  constexpr int kThreads = 128;
  constexpr int kWarpsPerBlock = kThreads / kWarpSize;
  const int64_t total_work = batch_size * num_heads;
  const int blocks =
      static_cast<int>((total_work + kWarpsPerBlock - 1) / kWarpsPerBlock);
  mla_paged_baseline_kernel<<<blocks, kThreads>>>(
      q_nope, q_pe, ckv, kpe, output,
      q_indptr, kv_indptr, kv_indices, kv_lens,
      static_cast<int>(batch_size), static_cast<int>(num_heads),
      static_cast<int>(head_dim_ckv), static_cast<int>(head_dim_kpe),
      sm_scale);
}
