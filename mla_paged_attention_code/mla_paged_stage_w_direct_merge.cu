// Stage W: full upstream QK/PV kernel without the unnecessary persistent
// grid barrier; merge all exact streaming-softmax partials in a second launch.
#define MLA_PAGED_SKIP_PERSISTENT_MERGE 1
#define MLA_PAGED_STAGE_W_DIRECT_MERGE 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_W_DIRECT_MERGE
#undef MLA_PAGED_SKIP_PERSISTENT_MERGE
