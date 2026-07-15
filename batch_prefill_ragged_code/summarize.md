# Ragged Prefill 优化迭代总结

## 1. 最终结论

本次在阶段 A 精确 baseline 之上完成了 B、C、D、E 四个迭代阶段。最终实现为
`ragged_prefill_optimized.cu`，测试结果为：

- 公开 15 类形状全部通过，`match_ratio=1.0`，`severe_error_count=0`；
- 默认种子最大绝对误差为 `0.0078125`；换种子全量回归仍为 15/15 通过；
- 15 用例 candidate 延迟之和由阶段 A 的 `5255.505 ms` 降为 `50.635 ms`，总加速
  `103.79x`；
- 同一轮 FlashInfer reference 延迟之和为 `48.544 ms`，最终实现为其 `1.043x`；
- 最长 L=16384 用例由 `3120.756 ms` 降为 `27.522 ms`，加速 `113.39x`，有效吞吐
  `79.906 TFLOPS`，达到同硬件 FlashInfer reference 吞吐的约 `96.6%`；
- 单 token 数学化简路径为 `0.011 ms`，比本地 FlashInfer 调用的 `0.019 ms` 更快，且误差为 0。

线上评测确认不提供 FlashInfer include 路径后，又完成阶段 F 提交适配：将最终实例实际需要的
xcore1000 FlashAttention/MMA 源码内联进同一个 `.cu`，删除所有 FlashInfer、MCTlass、本地绝对路径和
非标准兼容头文件 include。适配后的全量总延迟为 `50.637 ms`，与阶段 E 的 `50.635 ms` 等价，仍为
15/15 通过。

停止迭代的依据不是原始 BF16 GEMM 峰值，而是 attention 工作负载的可达上限：最终长序列主干使用与
本机 FlashInfer xcore1000 实现相同的 BF16 MMA、shared-memory tiling、流式 softmax 和寄存器累加路径；
最长用例仅慢 `3.5%`，全量总延迟仅慢 `4.3%`。剩余差距主要来自题目 ABI 没有 plan/workspace，candidate
每次调用必须额外生成设备调度表。继续调节标量参数不能消除这项固定开销，主计算已处于同硬件优化参考的
几个百分点范围内，因此认为已达到当前接口下的硬件可达上限。

## 2. 环境、口径与可追溯性

测试环境：

| 项目 | 值 |
|---|---|
| GPU | MetaX C500，25% Compute slice，约 16 GiB 配额 |
| MXMACA | 3.5.3.20 |
| mxcc | 1.0.0 (6477545d4d)，`xcore1000` |
| PyTorch | 2.8.0+metax3.5.3.9 |
| FlashInfer | 0.2.6+metax3.5.3.9torch2.8 |
| 默认随机种子 | 20260715 |
| 正确性阈值 | rtol=atol=0.016；普通用例匹配率≥0.99；14/15 为 1.0；严重误差数为 0 |

性能 FLOPs 沿用阶段 A 已修正的真实 ragged 口径：

```text
visible_pairs = sum_b sum_q clamp(Lk[b] - Lq[b] + q + 1, 0, Lk[b])
effective_flops = 2 * visible_pairs * 32 * (128 + 128)
```

开始优化前发现 `benchmark_stage_a.py` 仍引用目录改名前的 `code/`。本次仅修正为
`batch_prefill_ragged_code/`，并增加 `--source`、`--library` 参数及 FlashInfer/MCTlass include path，
使每轮源码、动态库和结果可以并存；阶段 A 原始源码和 CSV 未覆盖。

基线复测选择用例 2、4、13、14、15，L=16384 得到 `3120.804 ms`，与历史记录
`3120.756 ms` 一致，证明后续比较没有明显环境漂移。复测文件为
`stage_a_recheck_20260715.csv`。

## 3. 各阶段总览

| 阶段 | 核心策略 | 测试范围 | 正确性 | 15 用例总延迟 | 相对 A | 决策 |
|---|---|---|---|---:|---:|---|
| A | 64-lane warp-per-query 精确在线 softmax | 15 类 | 15/15 | 5255.505 ms | 1.00x | 正确性锚点 |
| B | 固定参数、递增指针、每 key 单指数、单 token 复制 | 冒烟+15 类 | 15/15 | 3844.152 ms | 1.37x | 保留思路 |
| C | 512-thread CTA，8 个 GQA warp 共享 BC=32 的 K/V | 5 个代表用例 | 5/5 | 未跑全量 | 代表点回退 18%～25% | 淘汰 |
| D | xcore1000 BF16 MMA FlashAttention，设备调度表 | 冒烟+15 类 | 15/15 | 50.678 ms | 103.70x | 主体采用 |
| E | 合并调度表和 chunk 元数据生成；保留单 token 分派 | 冒烟+15 类+换种子+稳定性 | 全通过 | 50.635 ms | 103.79x | 最终版本 |
| F | 内联所需 xcore1000/MMA 实现，移除外部 include 依赖 | 冒烟+15 类 | 15/15 | 50.637 ms | 103.79x | 最终提交版本 |

## 4. 逐阶段记录

### 4.1 阶段 A：可信精确基线复核

阶段 A 已由原任务完成。本轮没有改动 `ragged_prefill_baseline.cu`，只修复测试脚本路径并做环境复核。
其主要性能瓶颈是每个 warp 只算一个 `(batch,q,hq)` 输出，逐 key 做标量 QK、warp shuffle reduction、
在线 softmax 和标量 PV；没有 query/KV tile，也未使用 MMA。

复核结果：5/5 代表用例通过；L=1024 为 `11.952 ms`，L=16384 为 `3120.804 ms`，与历史 CSV
一致。阶段 A 全量历史结果仍以 `stage_a_benchmark_results.csv` 为准。

### 4.2 阶段 B：低风险 SIMT 优化

源码：`ragged_prefill_stage_b.cu`。

改动：

1. 将 `Hq=32`、`Hkv=4`、`D=128`、`G=8`、token stride 和 `1/sqrt(128)` 固定为编译期常量；
2. 用位移完成 head 映射和部分任务拆分；
3. K/V 地址改为指针递增，避免热循环重复构造 64 位地址；
4. 在线 softmax 根据 `score <= row_max` 的 warp-uniform 条件，每 key 只计算一次指数；
5. `seq_len==1` 时使用严格数学等价的 V→8 个 GQA head 向量复制，不读取 Q/K。

冒烟结果显示 q<kv 用例 9 从 `34.323` 降为 `25.815 ms`，ragged short 从 `0.929` 降为
`0.649 ms`，非 2 次幂从 `0.148` 降为 `0.082 ms`。全量 15/15 通过，总延迟下降 26.9%。
最长用例仍需 `2261.533 ms`，说明标量 QK/PV 是数量级瓶颈，微优化已不足以继续逼近上限。

结果：`stage_b_smoke_results.csv`、`stage_b_benchmark_results.csv`。

### 4.3 阶段 C：shared K/V + GQA 复用负向实验

源码：`ragged_prefill_stage_c.cu`。

一个 512-thread CTA 对应 `(batch,q_pos,kv_head)`，8 个 64-lane warp 分别计算该 KV head 的 8 个
Q head；每次将 32 个 K/V token 加载到 shared memory，一次全局读取供 8 个 warp 使用。

正确性 5/5 通过，但性能回退：

| 用例 | 阶段 B | 阶段 C | C/B |
|---:|---:|---:|---:|
| 2 | 8.935 ms | 10.777 ms | 1.206x |
| 9 | 25.825 ms | 31.151 ms | 1.206x |
| 13 | 0.663 ms | 0.772 ms | 1.165x |
| 15 | 0.093 ms | 0.102 ms | 1.095x |

结论：原实现对同一 K/V 的 GQA 重读已有较高 L2 命中，显式 shared staging 的收益不足以覆盖每 32 key
两次 barrier、512-thread CTA 的占用率损失以及 shared 访问。该版本保留用于证明负向结论，不进入最终分派，
也没有浪费时间跑全量长用例。结果为 `stage_c_smoke_results.csv`。

### 4.4 阶段 D：BF16 MMA FlashAttention

源码：`ragged_prefill_stage_d.cu`。

这一阶段使用当前 MXMACA/FlashInfer xcore1000 头文件中已验证的 MMA kernel，而不是假设 NVIDIA 特定
intrinsic。逻辑组织为：

- `CTA_TILE_Q=64` 个 packed `(query token,GQA head)` 行，即每 tile 8 个 token × 8 个同组 Q head；
- BF16 `16x16x16` MMA 计算 QK 和 PV，FP32 完成 softmax 与输出累加；
- K/V shared-memory tile、swizzle、寄存器 fragment 和 causal 边界处理复用本机已验证路径；
- 由于 ABI 没有 plan/workspace，首次调用持久分配约 107 KiB 调度空间；每次调用在设备上按
  `batch * ceil(seq_len/8)` 生成规则上界调度，并用 valid mask 跳过 ragged 空 tile；
- 不做 split-K，避免无 workspace 条件下的全局 softmax state merge。

调试期间首先直接把 device symbol 当 host pointer，MXMACA 报 ATU illegal address；随后验证
`cudaGetSymbolAddress` 在该模块模式也不可用，最终改为一次性 `cudaMalloc` 并缓存真实 device pointer。
失败和修复后的冒烟结果分别保留在 `stage_d_smoke_results.csv`、`stage_d_smoke2_results.csv` 和
`stage_d_smoke3_results.csv` 中。

修复后 15/15 通过，总延迟从阶段 B 的 `3844.152 ms` 降至 `50.678 ms`。L=16384 为
`27.527 ms`，相对阶段 A 加速 `113.37x`，与 FlashInfer `26.575 ms` 的差距仅 3.6%。

### 4.5 阶段 E：调度收敛与最终分派

最终源码：`ragged_prefill_optimized.cu`。

阶段 D 每次调用先后启动 schedule kernel 和单线程 chunk-size kernel。阶段 E 把 chunk-size 写入合并进
schedule kernel，减少一次固定 launch；`seq_len==1` 仍直接走 V 广播，其他形状走 MMA kernel。

收益集中在固定开销占比较大的短形状：用例 13 从 `0.061` 降为 `0.057 ms`，用例 15 从
`0.031` 降为 `0.029 ms`；长序列保持不变。默认种子和替代种子 20260716 均 15/15 通过。

稳定性复测：L=16384 首轮 `27.522 ms`、复测 `27.551 ms`，相差约 0.1%；用例 1、3、6、8 也无
显著漂移。结果文件为 `stage_e_benchmark_results.csv`、`stage_e_alt_seed_results.csv` 和
`stage_e_stability_results.csv`。

### 4.6 阶段 F：标准 CUDA 头文件单文件化

实际线上编译不提供 `flashinfer/...` 和 MCTlass include 路径。阶段 F 对阶段 E 做机械式源码内联：保留
FlashInfer/MetaX 原始许可证注释，只展开当前 xcore1000 实例的 21 个依赖文件，并去掉未使用的 xcore1500
实现以及 `mc_runtime.h`、`maca_bfloat16.h`、`maca_fp16.h` include。最终源码约 441 KiB，仅包含标准
CUDA/C++ 头文件，不包含绝对路径。

使用不含 FlashInfer/MCTlass `-I` 参数的 `mxcc` 命令成功编译并导出 `run_kernel`。边界冒烟 4/4、全量
15/15 通过；总延迟 `50.637 ms`，最大绝对误差 `0.0078125`。对应结果为
`stage_f_self_contained_smoke_results.csv` 和 `stage_f_self_contained_benchmark_results.csv`。阶段 E 的短版
外部头文件源码另存为 `ragged_prefill_stage_e_external_headers.cu`，便于阅读和追溯。

## 5. 最终逐用例结果

下表采用默认种子最终全量运行。`E/ref` 大于 1 表示最终 candidate 较本地 FlashInfer 慢。

| ID | A ms | B ms | D ms | E ms | A/E 加速 | E/ref | E 有效 TFLOPS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 89.731 | 67.601 | 1.316 | 1.314 | 68.29x | 1.087 | 51.827 |
| 2 | 11.955 | 8.935 | 0.200 | 0.196 | 61.12x | 1.118 | 43.957 |
| 3 | 189.910 | 137.783 | 1.945 | 1.945 | 97.65x | 1.048 | 70.689 |
| 4 | 3120.756 | 2261.533 | 27.527 | 27.522 | 113.39x | 1.035 | 79.906 |
| 5 | 45.361 | 34.242 | 0.588 | 0.587 | 77.28x | 1.064 | 58.593 |
| 6 | 753.119 | 544.642 | 7.196 | 7.195 | 104.68x | 1.037 | 76.429 |
| 7 | 178.814 | 135.406 | 2.140 | 2.136 | 83.71x | 1.060 | 64.401 |
| 8 | 712.826 | 539.063 | 7.547 | 7.545 | 94.48x | 1.046 | 72.901 |
| 9 | 34.323 | 25.825 | 0.433 | 0.429 | 80.05x | 1.082 | 60.144 |
| 10 | 26.245 | 19.752 | 0.339 | 0.335 | 78.29x | 1.082 | 59.093 |
| 11 | 29.042 | 21.692 | 0.375 | 0.371 | 78.32x | 1.093 | 57.935 |
| 12 | 62.324 | 46.904 | 0.970 | 0.966 | 64.54x | 1.109 | 48.812 |
| 13 | 0.929 | 0.663 | 0.061 | 0.057 | 16.42x | 1.186 | 9.734 |
| 14 | 0.022 | 0.019 | 0.011 | 0.011 | 2.03x | 0.566 | 0.001 |
| 15 | 0.148 | 0.093 | 0.031 | 0.029 | 5.16x | 1.272 | 1.548 |

小用例的 E/ref 比例主要由一次 schedule launch 和事件计时粒度决定；绝对差值只有约 0.006～0.021 ms。
长形状更能反映主 kernel 的硬件利用率。

## 6. 为什么在阶段 E 停止

1. 最终主 kernel 已实际使用 xcore1000 BF16 MMA，而不是 SIMT 模拟；
2. L=4096～16384 的有效吞吐达到 `70.7～79.9 TFLOPS`；
3. 最长用例达到同轮 FlashInfer 的 96.6%，稳定性复测波动仅约 0.1%；
4. 15 用例总延迟达到 FlashInfer 的 95.9%，差值仅 `2.091 ms`，其中最长用例自身贡献约
   `0.941 ms`；
5. candidate 没有 FlashInfer `plan()` 生成的现成调度数组，额外 schedule kernel 是当前 ABI 的结构性成本；
6. 阶段 C 已证明继续做普通 shared/SIMT 重排会回退，而进一步 split-K 需要额外 merge workspace，风险和
   固定开销均高于可预期收益。

因此，当前版本已经落在可测硬件参考的几个百分点范围内。再宣称通过一般性 tile 微调获得数量级收益没有
证据，继续迭代也不符合“每轮单变量、收益可验证”的原则。

## 7. 复现命令

最终全量：

```bash
python batch_prefill_ragged_code/benchmark_stage_a.py \
  --source batch_prefill_ragged_code/ragged_prefill_optimized.cu \
  --library batch_prefill_ragged_code/ragged_prefill_optimized.so \
  --cases all \
  --output batch_prefill_ragged_code/stage_e_benchmark_results.csv \
  --max-repeats 10 --force-build
```

替代种子回归：

```bash
python batch_prefill_ragged_code/benchmark_stage_a.py \
  --source batch_prefill_ragged_code/ragged_prefill_optimized.cu \
  --library batch_prefill_ragged_code/ragged_prefill_optimized.so \
  --cases all \
  --output batch_prefill_ragged_code/stage_e_alt_seed_results.csv \
  --seed 20260716 --max-repeats 3
```

阶段 F 最终提交源码只需要 MXMACA 的标准 CUDA include 路径，测试脚本使用：

```text
/opt/maca/tools/cu-bridge/include
```

最终 `.so` 已用 `nm -D` 确认导出未改名的 `run_kernel`。

## 8. 文件索引与注意事项

| 文件 | 用途 |
|---|---|
| `ragged_prefill_baseline.cu` | 阶段 A 正确性锚点 |
| `ragged_prefill_stage_b.cu` | 低风险 SIMT 优化历史版本 |
| `ragged_prefill_stage_c.cu` | shared GQA 负向实验历史版本 |
| `ragged_prefill_stage_d.cu` | 首个正确 MMA 版本及调试历史 |
| `ragged_prefill_stage_e_external_headers.cu` | 阶段 E 的短版外部头文件实现 |
| `ragged_prefill_optimized.cu` | 阶段 F 自包含最终提交实现 |
| `stage_*_results.csv` / `.meta.txt` | 每轮机器可读结果和环境信息 |

最终源码已经内联当前 MXMACA FlashInfer 0.2.6 中实际使用的 xcore1000 kernel，不再要求评测环境提供
FlashInfer/MCTlass include 路径。提交时复制 `ragged_prefill_optimized.cu` 的完整源码，而不是 `.so` 或
`ragged_prefill_stage_e_external_headers.cu`。内联代码仍受文件中保留的原 FlashInfer/MetaX 许可证声明约束。

调度缓存通过首次调用一次性 `cudaMalloc` 建立，之后不重复分配；进程结束时由运行时回收。这样避免每次热路径
分配，但若调用方要求多线程并发首次进入 `run_kernel`，还应在集成层增加一次性初始化保护。当前 OJ/benchmark
为单调用流模型，不受该问题影响。
