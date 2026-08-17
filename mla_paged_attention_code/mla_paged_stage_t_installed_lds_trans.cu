// Stage T: test the installed official LDS-transpose trait for xcore1000.
// It preserves the same FP32 streaming softmax and BF16 MMA computation.
#define MLA_PAGED_USE_INSTALLED_HEADERS 1
#define MLA_PAGED_USE_LDS_TRANS 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_USE_LDS_TRANS
#undef MLA_PAGED_USE_INSTALLED_HEADERS
