// Stage M: two equal exact split-KV chunks for B16/H128 long contexts.
// This isolates merge overhead versus the inherited 4/5 chunk partition.
#define MLA_PAGED_STAGE_M_TWO_CHUNKS 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_M_TWO_CHUNKS
