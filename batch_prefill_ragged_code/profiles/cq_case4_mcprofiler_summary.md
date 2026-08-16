# CQ Q128 profiler summary

- Command: `cd /opt/mcProfiler-ubuntu18.04 && ./mcProfiler perf_exec --cmdline
  'python benchmark_stage_a.py --source ragged_prefill_optimized.cu --library
  ragged_prefill_optimized.so --cases 4 --max-repeats 1' --cwd
  /data/flashinfer_task_package/batch_prefill_ragged_code --casename cq_case4
  --kernelnames BatchPrefillWithRaggedKVCacheKernel --counts 2 --per-kernel`
- Source SHA-256: `3cee75cf0c376c81c235a0df25c9208220f36f28bc6de0d664091064e92192dc`
- Report database: `/opt/mcProfiler-ubuntu18.04/.20260816103623.db`
- Target kernel: Q128, `CTA_TILE_Q=128`, `NUM_MMA_KV=2`, four Q warps.
- Resource compile: 230 MT registers, 52 ST registers, 0 static shared bytes, two
  resident warps per PEU.
- Counters for the target launch: 4096 workgroups / 32768 waves; BF16 MMA cycles
  about 44.05M per DPC, VALU cycles about 75.7M, BSM read cycles about 10.55M,
  BSM conflict cycles about 71.7K. The low conflict count and high VALU/MMA counts
  point to compute/softmax instruction pressure rather than a bandwidth or bank-conflict
  bottleneck. This motivates reducing scalar softmax/state work without changing the
  exact attention math.
