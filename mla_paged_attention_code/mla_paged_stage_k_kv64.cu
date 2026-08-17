// Stage K: 64-token MMA KV tiles halve loop/control overhead on the long,
// aligned path while retaining the exact official CKV+KPE softmax kernel.
#define MLA_PAGED_CTA_TILE_KV 64
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_CTA_TILE_KV
