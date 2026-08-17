// Stage AH: allow mxcc to choose the register/occupancy trade-off for MLA CTAs.
#define MLA_PAGED_STAGE_AH_NO_LAUNCH_BOUNDS 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_AH_NO_LAUNCH_BOUNDS
