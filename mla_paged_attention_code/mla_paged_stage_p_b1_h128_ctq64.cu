// Stage P: test the official CTQ64 kernel geometry for B=1/H=128.  It halves
// the block count relative to CTQ32 while using the same exact computation.
#define MLA_PAGED_STAGE_P_B1_H128_CTQ64 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_P_B1_H128_CTQ64
