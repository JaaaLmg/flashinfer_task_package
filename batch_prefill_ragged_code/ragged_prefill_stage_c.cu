#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>

namespace {
constexpr int WARP = 64, HQ = 32, HKV = 4, D = 128, G = 8;
constexpr int Q_STRIDE = HQ * D, KV_STRIDE = HKV * D;
constexpr int BC = 32;
constexpr uint64_t FULL_MASK = ~uint64_t{0};
constexpr float SCALE = 0.08838834764831845f;

__device__ __forceinline__ float warp_sum(float x) {
#pragma unroll
  for (int d = 32; d; d >>= 1) x += __shfl_down_sync(FULL_MASK, x, d, WARP);
  return __shfl_sync(FULL_MASK, x, 0, WARP);
}

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

// One 512-thread CTA computes all eight GQA heads for one (batch, q, kv-head).
// A K/V tile is fetched once into shared memory and consumed by all 8 warps.
__global__ void shared_gqa(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    __nv_bfloat16* __restrict__ o,
    const int32_t* __restrict__ qi,
    const int32_t* __restrict__ ki,
    int batch_size, int max_len) {
  __shared__ __nv_bfloat16 ks[BC][D];
  __shared__ __nv_bfloat16 vs[BC][D];

  int task = blockIdx.x;
  int hk = task & 3;
  task >>= 2;
  int qpos = task % max_len;
  int b = task / max_len;
  if (b >= batch_size) return;
  int qb = qi[b], qlen = qi[b + 1] - qb;
  if (qpos >= qlen) return;  // CTA-uniform, safe before barriers.
  int kb = ki[b], klen = ki[b + 1] - kb;
  int visible = klen - qlen + qpos + 1;
  visible = visible < 0 ? 0 : (visible > klen ? klen : visible);

  int warp = threadIdx.x >> 6, lane = threadIdx.x & 63;
  int qh = (hk << 3) + warp;
  int qrow = qb + qpos;
  const __nv_bfloat16* qp = q + static_cast<int64_t>(qrow) * Q_STRIDE + qh * D;
  float q0 = __bfloat162float(qp[lane]);
  float q1 = __bfloat162float(qp[lane + 64]);
  float out0 = 0.f, out1 = 0.f, m = -INFINITY, l = 0.f;

  for (int tile = 0; tile < visible; tile += BC) {
    int n = min(BC, visible - tile);
    for (int x = threadIdx.x; x < n * 2 * D; x += blockDim.x) {
      int t = x / (2 * D), rem = x - t * (2 * D);
      int which = rem >> 7, d = rem & 127;
      int64_t off = static_cast<int64_t>(kb + tile + t) * KV_STRIDE + hk * D + d;
      if (which == 0) ks[t][d] = k[off];
      else vs[t][d] = v[off];
    }
    __syncthreads();

    for (int j = 0; j < n; ++j) {
      float s = q0 * __bfloat162float(ks[j][lane]) +
                q1 * __bfloat162float(ks[j][lane + 64]);
      s = warp_sum(s) * SCALE;
      if (s <= m) {
        float beta = __expf(s - m);
        out0 += beta * __bfloat162float(vs[j][lane]);
        out1 += beta * __bfloat162float(vs[j][lane + 64]);
        l += beta;
      } else {
        float alpha = l > 0.f ? __expf(m - s) : 0.f;
        out0 = out0 * alpha + __bfloat162float(vs[j][lane]);
        out1 = out1 * alpha + __bfloat162float(vs[j][lane + 64]);
        l = l * alpha + 1.f;
        m = s;
      }
    }
    __syncthreads();
  }

  __nv_bfloat16* op = o + static_cast<int64_t>(qrow) * Q_STRIDE + qh * D;
  float inv = l > 0.f ? 1.f / l : 0.f;
  op[lane] = __float2bfloat16(out0 * inv);
  op[lane + 64] = __float2bfloat16(out1 * inv);
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
  } else {
    int64_t blocks = batch * seq_len * HKV;
    shared_gqa<<<static_cast<int>(blocks), 512>>>(
        q, k, v, o, qi, ki, static_cast<int>(batch), static_cast<int>(seq_len));
  }
}
