#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>

namespace {
constexpr int HQ = 32, HKV = 4, D = 128, G = 8;
constexpr int Q_STRIDE = HQ * D, KV_STRIDE = HKV * D;
constexpr int CTA_TILE_Q = 64;
constexpr int MAX_TASKS = 8192;

__global__ void build_regular_schedule(const int32_t* __restrict__ qi,
                                       int32_t* __restrict__ request,
                                       int32_t* __restrict__ qo_tile,
                                       int32_t* __restrict__ kv_tile,
                                       bool* __restrict__ valid,
                                       int tasks_per_batch, int total_tasks) {
  for (int bx = blockIdx.x * blockDim.x + threadIdx.x; bx < total_tasks;
       bx += blockDim.x * gridDim.x) {
    int b = bx / tasks_per_batch;
    int tile = bx - b * tasks_per_batch;
    int qlen = qi[b + 1] - qi[b];
    request[bx] = b;
    qo_tile[bx] = tile;
    kv_tile[bx] = 0;
    // A packed tile contains 64 (token, GQA-head) rows = 8 query tokens.
    valid[bx] = tile * (CTA_TILE_Q / G) < qlen;
  }
}

__global__ void set_chunk_size(int32_t* dst, int32_t value) { *dst = value; }

__global__ void single_token(const __nv_bfloat16* __restrict__ v,
                             __nv_bfloat16* __restrict__ o,
                             const int32_t* __restrict__ qi,
                             const int32_t* __restrict__ ki, int batch) {
  int b = blockIdx.x;
  if (b >= batch) return;
  int qb = qi[b], kb = ki[b];
  if (qi[b + 1] - qb != 1 || ki[b + 1] - kb != 1) return;
  for (int i = threadIdx.x; i < HQ * D; i += blockDim.x) {
    int h = i / D, d = i & 127;
    o[static_cast<int64_t>(qb) * Q_STRIDE + i] =
        v[static_cast<int64_t>(kb) * KV_STRIDE + (h >> 3) * D + d];
  }
}
}  // namespace

extern "C" void run_kernel(
    const __nv_bfloat16* q, const __nv_bfloat16* k, const __nv_bfloat16* v,
    __nv_bfloat16* o, const int32_t* qi, const int32_t* ki,
    int64_t batch, int64_t seq_len, int64_t hq, int64_t hkv,
    int64_t dq, int64_t dv, int64_t causal) {
  if (hq != HQ || hkv != HKV || dq != D || dv != D || !causal) return;
  if (seq_len == 1) {
    single_token<<<static_cast<int>(batch), 256>>>(v, o, qi, ki, static_cast<int>(batch));
    return;
  }

  int tasks_per_batch = (static_cast<int>(seq_len) + 7) / 8;
  int total_tasks = static_cast<int>(batch) * tasks_per_batch;
  if (total_tasks > MAX_TASKS) return;
  // The ABI has no workspace.  Allocate a small persistent schedule buffer on
  // first use; subsequent timed calls only rebuild its contents asynchronously.
  static int32_t* d_request = nullptr;
  static int32_t* d_qo_tile = nullptr;
  static int32_t* d_kv_tile = nullptr;
  static int32_t* d_chunk = nullptr;
  static bool* d_valid = nullptr;
  if (d_request == nullptr) {
    cudaMalloc(reinterpret_cast<void**>(&d_request), MAX_TASKS * sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_qo_tile), MAX_TASKS * sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_kv_tile), MAX_TASKS * sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_chunk), sizeof(int32_t));
    cudaMalloc(reinterpret_cast<void**>(&d_valid), MAX_TASKS * sizeof(bool));
  }
  build_regular_schedule<<<(total_tasks + 255) / 256, 256>>>(
      qi, d_request, d_qo_tile, d_kv_tile, d_valid,
      tasks_per_batch, total_tasks);
  set_chunk_size<<<1, 1>>>(d_chunk, static_cast<int32_t>(seq_len));

  using T = __nv_bfloat16;
  using Params = flashinfer::BatchPrefillRaggedParams<T, T, T, int32_t>;
  using Variant = flashinfer::DefaultAttention<false, false, false, false>;
  Params p;
  p.q = const_cast<T*>(q);
  p.k = const_cast<T*>(k);
  p.v = const_cast<T*>(v);
  p.o = o;
  p.lse = nullptr;
  p.q_indptr = const_cast<int32_t*>(qi);
  p.kv_indptr = const_cast<int32_t*>(ki);
  p.maybe_custom_mask = nullptr;
  p.maybe_mask_indptr = nullptr;
  p.maybe_q_rope_offset = nullptr;
  p.maybe_k_rope_offset = nullptr;
  p.maybe_alibi_slopes = nullptr;
  p.group_size = flashinfer::uint_fastdiv(G);
  p.num_qo_heads = HQ;
  p.num_kv_heads = HKV;
  p.q_stride_n = Q_STRIDE;
  p.q_stride_h = D;
  p.k_stride_n = KV_STRIDE;
  p.k_stride_h = D;
  p.v_stride_n = KV_STRIDE;
  p.v_stride_h = D;
  p.window_left = -1;
  p.logits_soft_cap = 0.f;
  p.sm_scale = 0.08838834764831845f;
  p.rope_rcp_scale = 1.f;
  p.rope_rcp_theta = 1.f / 10000.f;
  p.request_indices = d_request;
  p.qo_tile_indices = d_qo_tile;
  p.kv_tile_indices = d_kv_tile;
  p.merge_indptr = nullptr;
  p.o_indptr = const_cast<int32_t*>(qi);
  p.kv_chunk_size_ptr = d_chunk;
  p.block_valid_mask = d_valid;
  p.max_total_num_rows = 0;
  p.total_num_rows = nullptr;
  p.padded_batch_size = total_tasks;
  p.partition_kv = false;

  flashinfer::BatchPrefillWithRaggedKVCacheDispatched<
      CTA_TILE_Q, D, D, flashinfer::PosEncodingMode::kNone, false,
      flashinfer::MaskMode::kCausal, Variant, Params>(p, nullptr, nullptr, 0);
}
