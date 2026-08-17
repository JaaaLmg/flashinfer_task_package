// Stage S: compile the exact same full-KPE candidate against the headers used
// by the installed FlashInfer benchmark package.  This isolates SDK-version
// code generation from planner differences without changing the algorithm.
#define MLA_PAGED_USE_INSTALLED_HEADERS 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_USE_INSTALLED_HEADERS
