#include <stdint.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

template <int HEAD_DIM>
__global__ void unpack_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst_k,
                              __nv_bfloat16* dst_v, uint32_t total_vectors) {
  constexpr uint32_t kHeads = 4;
  constexpr uint32_t kPage = 16;
  constexpr uint32_t kVecsPerToken = kHeads * HEAD_DIM / 8;
  const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_vectors) return;
  const uint32_t token = idx / kVecsPerToken;
  const uint32_t vec = idx - token * kVecsPerToken;
  const uint32_t page = token >> 4;
  const uint32_t entry = token & 15;
  constexpr uint32_t kPageElems = 2 * kPage * kHeads * HEAD_DIM;
  constexpr uint32_t kPlaneElems = kPage * kHeads * HEAD_DIM;
  const uint32_t src_elem = page * kPageElems + entry * kHeads * HEAD_DIM + vec * 8;
  reinterpret_cast<uint4*>(dst_k)[idx] = reinterpret_cast<const uint4*>(src + src_elem)[0];
  reinterpret_cast<uint4*>(dst_v)[idx] =
      reinterpret_cast<const uint4*>(src + src_elem + kPlaneElems)[0];
}

extern "C" void debug_unpack(const __nv_bfloat16* src, __nv_bfloat16* k,
                             __nv_bfloat16* v, int64_t tokens, int64_t head_dim) {
  const uint32_t vectors = uint32_t(tokens) * 4 * uint32_t(head_dim) / 8;
  if (head_dim == 128) {
    unpack_kernel<128><<<(vectors + 255) / 256, 256>>>(src, k, v, vectors);
  }
}
