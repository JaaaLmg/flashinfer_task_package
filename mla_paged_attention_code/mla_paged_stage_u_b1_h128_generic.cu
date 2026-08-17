// Stage U: retain the measured schedule elsewhere but use the upstream generic
// work partition for the B=1/H=128 family.
#define MLA_PAGED_STAGE_U_B1_H128_GENERIC 1
#include "mla_paged_optimized.cu"
#undef MLA_PAGED_STAGE_U_B1_H128_GENERIC
