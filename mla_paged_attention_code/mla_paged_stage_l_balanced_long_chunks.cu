// Stage L: use three balanced split-KV chunks for B16/H128 long contexts.
// Hypothesis: 48 one-work clusters reduce the critical path versus 64 chunks
// with 16 clusters carrying a second full/tail work item.
#define MLA_PAGED_STAGE_L_TUNED_CHUNKS 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_L_TUNED_CHUNKS
