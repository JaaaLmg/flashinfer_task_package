# Ragged Prefill 阶段 A Benchmark 报告

## 运行环境

- 时间：2026-07-15 17:41（Asia/Shanghai）
- GPU：MetaX C500（当前容器分配 25% Compute、约 16 GiB 显存配额）
- MXMACA：3.5.3.20
- PyTorch：2.8.0+metax3.5.3.9
- FlashInfer：0.2.6+metax3.5.3.9torch2.8
- 编译器：mxcc 1.0.0 (6477545d4d)
- 源码：`code/ragged_prefill_baseline.cu`
- 完整机器可读结果：`code/stage_a_benchmark_results.csv`

编译、运行命令：

```bash
python code/benchmark_stage_a.py \
  --cases all \
  --output code/stage_a_benchmark_results.csv \
  --max-repeats 3
```

## 正确性结论

- 15/15 类题目形状通过；
- 所有用例 `match_ratio = 1.0`；
- 所有用例 `severe_error_count = 0`；
- 最大绝对误差不超过 0.015625；
- 单 token 用例最大绝对误差为 0；
- 非 2 的幂尾段用例逐元素通过。

题目只公开了 ragged 用例的 batch、总长度和最大长度，没有公开每个隐藏段的精确 indptr。本地用例使用确定性的正整数段长，严格匹配题目公布的 batch、`total_q/total_kv` 和 `max_q/max_kv`。kernel 本身不依赖这些具体段长。

## 性能结果

`candidate_ms` 是阶段 A 精确 warp-per-query kernel 的设备时间；`flashinfer_ms` 是相同输入上 FlashInfer `wrapper.run` 的设备时间。有效 TFLOPS 按真实 bottom-right causal 可见 Query-Key 对数计算。

| ID | 类型 | B | max len | candidate ms | FlashInfer ms | 慢速倍数 | 有效 TFLOPS | 匹配率 | 最大误差 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ragged long | 33 | 987 | 89.731 | 1.208 | 74.28× | 0.759 | 1.0 | 0.015625 |
| 2 | equal | 1 | 1024 | 11.955 | 0.175 | 68.23× | 0.719 | 1.0 | 0.015625 |
| 3 | equal | 1 | 4096 | 189.910 | 1.856 | 102.30× | 0.724 | 1.0 | 0.015625 |
| 4 | equal | 1 | 16384 | 3120.756 | 26.594 | 117.35× | 0.705 | 1.0 | 0.015625 |
| 5 | equal | 4 | 1024 | 45.361 | 0.551 | 82.39× | 0.758 | 1.0 | 0.015625 |
| 6 | equal | 4 | 4096 | 753.119 | 6.932 | 108.65× | 0.730 | 1.0 | 0.015625 |
| 7 | equal | 16 | 1024 | 178.814 | 2.015 | 88.73× | 0.769 | 1.0 | 0.015625 |
| 8 | equal | 16 | 2048 | 712.826 | 7.219 | 98.75× | 0.772 | 1.0 | 0.015625 |
| 9 | q < kv uniform | 4 | 1024 | 34.323 | 0.396 | 86.66× | 0.751 | 1.0 | 0.001953 |
| 10 | q < kv mixed | 4 | 1280 | 26.245 | 0.310 | 84.68× | 0.755 | 1.0 | 0.001953 |
| 11 | q < kv two | 2 | 2048 | 29.042 | 0.340 | 85.35× | 0.740 | 1.0 | 0.001953 |
| 12 | ragged medium | 27 | 873 | 62.324 | 0.870 | 71.63× | 0.756 | 1.0 | 0.015625 |
| 13 | ragged short | 15 | 123 | 0.929 | 0.048 | 19.34× | 0.593 | 1.0 | 0.015625 |
| 14 | single token | 1 | 1 | 0.022 | 0.019 | 1.17× | 0.001 | 1.0 | 0 |
| 15 | non-power-of-two | 2 | 65 | 0.148 | 0.023 | 6.56× | 0.300 | 1.0 | 0.015625 |

## 阶段 A 结论

当前实现已经满足阶段 A 的目标：删除了 `prefix_mean_kernel` 和 `exact_len` 截断，所有输出均由精确 attention 计算；正确处理 ragged indptr、32/4 GQA、bottom-right causal、单 token 和非 2 的幂尾块；并建立了逐用例延迟基线。

性能结果也验证了优化方案中的判断：除极小用例外，warp-per-query 标量实现比 FlashInfer 慢约 68～117 倍，有效吞吐约 0.7 TFLOPS。阶段 B/C 应优先减少指数与地址指令，并按 KV head 共享 K/V；真正的数量级提升预计要依赖分块 FlashAttention 和 BF16 矩阵乘路径。
