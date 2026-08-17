# GJ/GJ2 direct-BSM double-K architecture probe

- Baseline source SHA: `5057e880308dca92d8c1edc968a2f6e24143ac7330c506a79c553066242f3b99`.
- Candidate GJ source SHA: `6600cb7ac18d88d26baa7f2d60c9286955e1a418b58390c29581bb2b0e408aad`.
- Candidate GJ2 source SHA: `70e15ca2deb40654184636b36644ec3403fd055553f0b34012db42c95ff42214`.
- Build: `mxcc -O3 -std=c++17 --offload-arch=xcore1000 -I/opt/maca/tools/cu-bridge/include -shared -fPIC`.
- Resource output: Q128 used 222 MT registers / 48 ST registers / 2 resident warps per PEU;
  the extra K buffer did not increase static shared allocation.
- GJ used explicit `arrive_gvmcnt(0)` + `arrive_bsmcnt(0)` after `is_async=true` BSM loads;
  GJ2 used `is_async=false` plus `arrive(0)`.
- Both variants failed correctness on representative cases 3/4/6/8 (0/4 pass; match ratios
  0.16--0.41, severe errors present). The failure is therefore architectural/data-transport,
  not a score regression, and neither source is eligible for promotion.
- GJ3 additionally applied the guide's lane-dependent source permutation to each BSM segment;
  case 3 still failed at match ratio 0.212 (0/1), so source swizzling is not a repair.
- The existing `mcProfiler` baseline (`profiles/cq_case4_mcprofiler_summary.md`) shows the
  canonical Q128 launch is compute/softmax pressure dominated (high MMA/VALU, low BSM conflict),
  so a raw BSM transport rewrite has no evidence-backed promotion path after this correctness
  failure.
