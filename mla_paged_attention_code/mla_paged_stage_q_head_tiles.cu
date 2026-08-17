// Stage Q: map B16/H128 long requests as two 64-head work tiles across all
// 104 C500 clusters.  This is an exact re-partitioning of the official
// split-KV states and uses normal input-derived scheduling only.
#define MLA_PAGED_STAGE_Q_HEAD_TILES 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_Q_HEAD_TILES
