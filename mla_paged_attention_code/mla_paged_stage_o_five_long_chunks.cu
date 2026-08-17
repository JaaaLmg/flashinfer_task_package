// Stage O: five exact KV partitions can shorten the per-cluster critical path
// for B16/H128 long contexts despite extra partial rows.
#define MLA_PAGED_STAGE_O_FIVE_CHUNKS 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_O_FIVE_CHUNKS
