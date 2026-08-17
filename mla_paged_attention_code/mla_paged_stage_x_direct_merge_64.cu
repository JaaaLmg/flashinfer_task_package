// Stage X: vectorized one-wave direct merge after the exact upstream QK/PV path.
#define MLA_PAGED_SKIP_PERSISTENT_MERGE 1
#define MLA_PAGED_STAGE_W_DIRECT_MERGE 1
#define MLA_PAGED_STAGE_X_DIRECT_MERGE_64 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_X_DIRECT_MERGE_64
#undef MLA_PAGED_STAGE_W_DIRECT_MERGE
#undef MLA_PAGED_SKIP_PERSISTENT_MERGE
