// Stage J: compare the official, shape-generic MLA scheduler policy against
// the inherited experimentally tuned partitioning.  The compute kernel and
// ABI are otherwise identical to the current legal candidate.
#define MLA_PAGED_USE_DEFAULT_SCHEDULE 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_USE_DEFAULT_SCHEDULE
