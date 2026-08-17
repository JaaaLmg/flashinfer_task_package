// Stage AV: exact online-softmax rescale elision when max is unchanged.
#define MLA_PAGED_STAGE_AV_SKIP_NOOP_RESCALE 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_AV_SKIP_NOOP_RESCALE
