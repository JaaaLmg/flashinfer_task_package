// Stage AS: test the exact non-sharded QK/PV kernel mapping.  It removes the
// cross-warpgroup P all-gather at the cost of duplicated QK work.
#define MLA_PAGED_STAGE_AS_UNSHARDED_QK 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_AS_UNSHARDED_QK
