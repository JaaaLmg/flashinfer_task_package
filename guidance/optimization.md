# FlashInfer Ragged Prefill 性能瓶颈与 CUDA MACA 优化方案

## 1. 结论先行

当前 `starter/示例冒烟代码.md` 适合验证接口，不适合作为比赛实现。主要原因不是某个 block size 没调好，而是计算组织方式存在数量级上的效率损失：

- 一个 warp 只处理一个 `(batch, q_pos, qo_head)`，QK 和 PV 都是标量运算；
- 同一个 KV head 被其 8 个 Q head 重复读取，且不同 Query 位置之间也完全不复用 K/V；
- 每处理一个 Key 就做 warp reduction 和两次指数运算；
- 没有 shared memory 分块、双缓冲或矩阵乘加路径；
- ragged grid 按最大 `seq_len` 展开，会启动无效 warp；
- 旧的 prefix-mean 尾部近似在当前题目上不正确，并有 ragged 越界风险；
- 仓库现有 benchmark 的 Q 长度为 1，却按方形 prefill 统计 FLOPs，不能作为优化依据。

推荐的正式方案是：以 FlashAttention 的 streaming softmax 为算法骨架，按 `(batch, kv_head, q_tile)` 分 CTA，把一个 KV head 对应的多个 Query 位置和多个 GQA head 合并成矩阵行；K/V 按块加载到 shared memory，使用 MXMACA 工具链实际支持的 BF16 矩阵乘加能力完成 `QK^T` 与 `PV`，全程 FP32 softmax/累加，并为单 token、短序列和长序列分别选择 kernel。

## 2. 优化前必须先修正的 P0 问题

### 2.1 删除 prefix-mean 近似

`prefix_mean_kernel` 不使用 Q/K，计算的不是 attention。当前题目输入是标准正态分布，并明确设置严格边界用例；通过统计近似“蒙过”旧数据既不可靠，也不符合正确性与反作弊要求。

更严重的是，它对每段直接循环到 `seq_len`，没有读取真实 `qo_len/kv_len`。如果在 ragged 用例触发，会跨段或越界。

正式版本应始终进行精确 attention。任何近似指数、低精度累加或特殊快速路径都必须以数学等价为前提，并逐个测试点验证。

### 2.2 以最新题目文档为唯一形状真相

不能采用旧教程中的以下假设：

```text
qo_len == kv_len == seq_len
batch 只取 1/4/16
输入为均匀分布
容差为 1e-2
```

当前题目是 15 个测试点，包含 ragged、`q_len < kv_len`、batch 2/15/27/33、正态输入以及更具体的误差规则。所有 kernel 都必须从 indptr 读取真实边界。

### 2.3 修正本地性能基线

现有 `bench_batch_prefill_ragged.py` 的每段 `q_len=1`，但 FLOPs 公式按 `q_len=seq_len` 计算。设 `L=seq_len`、`Dq=Dv=128`，实际主干 FLOPs 近似为：

```text
2 * B * Hq * L * (Dq + Dv)
```

脚本统计的是：

```text
B * Hq * L^2 * (Dq + Dv)
```

两者相差约 `L/2` 倍。以 `L=16384` 为例，CSV 中 37006 TFLOPS 除以 8192 后约为 4.5 TFLOPS，才接近该实际工作负载对应的量级。

应另建与 OJ 15 个形状一致的验证/计时入口，至少记录：每段 `Lq/Lk`、总可见 Query-Key 对数、延迟、有效 TFLOPS、正确率和最大误差。

## 3. starter 的计算与访存成本

对一个 Query head 与一个可见 Key 的配对，starter 大约完成：

- 128 维 QK 点积；
- 128 维 `weight * V` 累加；
- 读取 128 个 BF16 K 和 128 个 BF16 V，即约 512 字节；
- 一次 warp 点积归约；
- 在线 softmax 更新。

若把乘加记作 2 FLOPs，则 QK 与 PV 共约 512 FLOPs，未考虑缓存命中时算术强度只有约 1 FLOP/byte。更关键的是，一个 KV head 服务 8 个 Q head，但 starter 对这 8 个 head 分别读取相同 K/V，理论上存在至少 8 倍的组内复用空间；跨多个 Query 位置还存在更大的 tile 复用空间。

等长 causal 段的可见 Query-Key 对数是：

```text
L * (L + 1) / 2
```

主干 FLOPs 近似为：

```text
F = B * Hq * L * (L + 1) / 2 * 2 * (Dq + Dv)
```

当 `Dq=Dv=128` 时，每个可见配对约 512 FLOPs。长度 16384、batch 1 的用例约有 134M 个位置配对，乘 32 个 Q head 后超过 2 TFLOPs 工作量。标量 warp 循环无法充分利用高吞吐矩阵计算单元。

## 4. 逐项性能瓶颈

| 优先级 | 瓶颈 | 影响 | 根因 |
|---|---|---|---|
| P0 | 近似尾部不正确 | 可能 WA/越界 | 用均值代替 attention，忽略 ragged |
| P1 | 未使用矩阵乘加路径 | 长序列吞吐低 | 每 lane 做标量 BF16→FP32 乘加 |
| P1 | K/V 重复读取 | 带宽与缓存压力大 | warp-per-query-head，不共享 GQA/Query tile |
| P1 | 每 Key 做 softmax 更新 | 特殊函数和控制开销高 | 粒度过细，没有按 Key tile 处理 |
| P1 | 没有 shared memory tile | 数据复用差 | 全程 register + global memory |
| P2 | 动态 64 位除法、取模和寻址 | 指令开销 | 固定参数未特化，热循环地址重复计算 |
| P2 | ragged 空任务 | launch/调度浪费 | grid 按 `batch * seq_len * heads` 展开 |
| P2 | causal 工作量不均 | 尾部效应 | 不同 q_pos 循环次数差异大 |
| P2 | 两次指数/Key | SFU 压力 | 标量在线 softmax 写法未化简 |
| P3 | 标量 load/store | 指令数偏高 | 未利用对齐的向量化搬运 |
| P3 | 单一 kernel 配置 | 小/中/长形状不能兼顾 | 128 threads、单 warp 算法固定 |

### 4.1 标量计算路径

每个 score 由 32 个 lane 各算 4 个标量，再通过 shuffle 求和。这能正确覆盖 128 维，却无法像 BF16 矩阵乘加那样一次处理多个 Query 行和 Key 列。

### 4.2 GQA 复用完全丢失

`qo_head=0..7` 对应同一个 `kv_head=0`。starter 为 8 个 Q head 启动 8 个独立 warp，每个 warp 都从头读取相同 K/V。若一个 CTA 同时处理这 8 个 Q head，K/V tile 可以只从全局内存加载一次。

### 4.3 Query 位置之间也不复用

相邻 Query 位置会访问大部分相同的 K/V 前缀。starter 每个 Query warp 从 `kv_pos=0` 独立扫描。FlashAttention 分块可以让一个 K/V tile 同时服务多个 Query 位置。

### 4.4 softmax 粒度过细

每个 Key 都更新 `m/l/acc`。当前代码无论新 score 是否超过最大值，都计算 `alpha` 和 `beta` 两个指数。即使保留 warp 实现，也可利用条件：

```text
若 score <= m：alpha=1，beta=exp(score-m)
若 score >  m：alpha=exp(m-score)，beta=1
```

每次只需要一个指数，且分支对同一 warp 是一致的。不过这只是小修，不能替代 tile 化重构。

### 4.5 热循环中的通用整数运算

题目正式范围固定 `Hq=32, Hkv=4, D=128, G=8, causal=1`。starter 仍在运行时计算 group、动态 stride 和大量 64 位地址表达式。编译器可能做部分强度削弱，但显式特化更可靠：

- head 映射可使用常量除法/位移；
- 每次迭代递增 K/V 指针，而不是重算完整乘法；
- 段内长度和索引用 32 位即可覆盖当前范围；
- 仅全局偏移计算保留足够宽的类型。

### 4.6 ragged 的空 warp

精确 kernel 的 grid 使用 `batch_size * exact_len * 32`。某段短于 `seq_len` 时，多出的 warp 只读取 indptr 后返回。测试 1、12、13、15 都会出现不同程度的浪费。

## 5. 目标算法：分块 FlashAttention

### 5.1 CTA 的逻辑任务

推荐先尝试：

```text
一个 CTA = (batch b, KV head hk, Query token tile)
```

该 KV head 对应 8 个 Q head。若 Query tile 含 `Br` 个 token，则可把矩阵行组织为：

```text
M = Br * G
```

每行代表 `(q_pos, group_head)`，Q 子矩阵形状为 `M x 128`。每次读取 `Bc` 个 Key：

```text
Q_tile: M  x 128
K_tile: Bc x 128
S_tile = Q_tile @ K_tile^T       # M x Bc
O_tile += softmax(S_tile) @ V_tile
V_tile: Bc x 128
```

这样一个 K/V tile 同时服务 `Br * 8` 个 Query 行，实现 GQA 和 Query 位置两级复用。

如果 `Br * 8` 带来过高寄存器/共享内存压力，可退一步让每 CTA 处理 2 或 4 个同组 Q head，通过多个 CTA 覆盖 8 个 head。应以实际编译资源和计时决定，而不是预设“8 个一定最好”。

### 5.2 streaming softmax 状态

对 tile 内每个 Query 行维护 FP32：

```text
row_max[M]
row_sum[M]
out_acc[M, 128]
```

遍历 Key tile 时：

1. 算出 `S_tile` 并应用缩放和 mask；
2. 求每行当前 tile 最大值；
3. 合并旧 `row_max/row_sum`；
4. 计算 tile 内概率；
5. 用矩阵乘加更新 `out_acc`；
6. 最后除以 `row_sum` 并写 BF16。

不在全局内存保存完整注意力矩阵，显存复杂度从 `O(Lq*Lk)` 降为 tile 大小。

### 5.3 bottom-right mask 的 tile 化

对某 Query 位置：

```text
visible(q) = clamp(Lk - Lq + q + 1, 0, Lk)
```

对一个 Query tile，可先计算其中最大的 `visible`，只遍历可能被任何行看到的 Key tile。

Key tile 可分为三类：

- 完全可见：无需逐元素 mask；
- 完全不可见：整个 tile 跳过；
- 穿过边界：只在这个边界 tile 内做逐元素 mask。

大多数 tile 因而不承担 mask 分支成本。`q_len < kv_len` 只改变对角线偏移，不改变该分类方法。

## 6. CUDA MACA 上的数据通路设计

### 6.1 BF16 矩阵乘加

应先用 `mxcc` 做一个最小编译和性能探针，确认当前 MXMACA 环境对以下哪条路径支持最好：

1. CUDA 兼容的 `mma.h`/WMMA BF16 接口；
2. MXMACA 提供的等价矩阵乘加 intrinsic 或库内模板；
3. 若前两者不可用，再使用向量化 SIMT fallback。

不要在未验证的情况下把 NVIDIA 特定的架构代号、`cp.async` 或某种 MMA tile 当作 C500 必然支持。方案的目标是使用实际可编译、可反汇编确认、可测得收益的 MACA 路径。

### 6.2 shared memory 布局

每个 Key tile 至少需要保存 K 和 V；Q 可按 CTA 生命周期加载一次。以 BF16 粗略估算：

```text
Q shared bytes = M  * 128 * 2
K shared bytes = Bc * 128 * 2
V shared bytes = Bc * 128 * 2
```

例如 `Br=8, G=8, M=64, Bc=32` 时，三者合计约 32 KiB；若对 K/V 双缓冲，则再增加约 16 KiB。实际还要考虑 padding、softmax 临时量和编译器分配。

建议从以下候选开始测试：

| 场景 | `Br` | 同 CTA Q heads | `Bc` | 线程数 | 说明 |
|---|---:|---:|---:|---:|---|
| 短序列 | 4/8 | 4/8 | 16/32 | 128 | 控制启动和同步开销 |
| 中序列 | 8 | 4/8 | 32/64 | 128/256 | 平衡复用和占用率 |
| 长序列 | 8/16 | 2/4/8 | 64/128 | 256 | 提高 MMA 与流水化效率 |

这些只是搜索空间，不是预先认定的最优值。必须同时记录 shared memory、寄存器数、驻留 CTA 数和实际延迟。

### 6.3 向量化搬运

固定维度 128 BF16，即每个 head 行 256 字节。PyTorch 张量基地址和每行 stride 通常满足较强对齐，可尝试 16 字节向量 load/store，然后在 shared memory 中重排为 MMA 需要的布局。

尾块发生在 token 维，不发生在 head dimension，因此 D=128 的向量化无需为维度尾部付出分支。仍应保证指针类型转换满足实际对齐规则。

### 6.4 双缓冲和预取

当一个 K/V tile 正在参与 QK/PV 计算时，预取下一个 tile，以隐藏全局内存延迟。实现顺序建议是：

1. 先完成单缓冲、数值正确的 tile kernel；
2. 再加普通的两阶段 ping-pong shared memory；
3. 只有在工具链确认支持且 profile 表明有效时，才引入异步拷贝机制。

过早加入多阶段流水会增加同步和寄存器压力，使错误难以定位。

### 6.5 shared memory 冲突

Q/K/V 的 shared layout 应围绕实际 MMA fragment 访问方式设计。可通过 padding 或 swizzle 避免多个 lane 落在同一 bank。不能仅凭 CUDA 平台经验断言某个 swizzle 在 MACA 上最佳，应以 profiler 或微基准验证。

## 7. kernel 分派策略

单一 kernel 很难同时覆盖长度 1 和 16384。推荐按通用形状性质分派，而不是识别测试 ID。

### 7.1 单 token 路径

若某段 `Lq=Lk=1` 且 causal：softmax 只有一个元素，输出数学上恰好等于对应 KV head 的 V。可以直接向量化复制到 8 个 Q head，不读取 Q/K。

这是严格的数学化简，不是近似。当前用例 14 要求逐元素通过，BF16 到 BF16 的直接复制也最容易满足该要求。

由于 indptr 在设备端，主机不能无同步地读取每段长度。可在 `seq_len==1` 时启动专用 kernel，并在设备内读取 indptr 做边界保护。

### 7.2 短序列路径

对 `seq_len <= 128` 或类似阈值，MMA tile 的装载与同步开销可能大于收益。可保留一个经过修正的 warp/CTA SIMT kernel：

- 删除 prefix mean；
- 一个 CTA 共享同一 KV head 的 K/V；
- 向量化 BF16 load；
- 固定 128 维、32/4 heads；
- 一次指数更新；
- 正确处理 65 等非 2 的幂尾部。

阈值应通过用例 13、14、15附近的微基准确定。

### 7.3 中长序列路径

对 512～16384 采用 FlashAttention tile kernel。可以再按 `seq_len`、batch size 或 `q_len/kv_len` 上界选择 `Br/Bc`，但分派规则应是通用 shape-based 规则。

### 7.4 是否需要 split-K

长序列中一个 Query tile 要遍历很多 K tile。split-K 可增加并行度，但需要合并每个 split 的 `m/l/O` 状态。当前接口没有显式 workspace，直接使用原子操作也无法简单合并 softmax。

本题即使 batch=1，仍有 4 个 KV head 和多个 Query tile，通常已有足够 CTA。第一版不建议 split-K。只有 profile 明确显示长序列并行度不足，且能设计无额外分配的安全 workspace/合并方法时再考虑。

## 8. ragged 调度方案

### 8.1 第一版：规则上界 grid

使用三维或线性 grid 覆盖：

```text
batch_size * num_kv_heads * ceil(seq_len / Br)
```

CTA 内读取 `qo_indptr[b:b+2]` 与 `kv_indptr[b:b+2]`；若 q tile 超过真实 `Lq`，立即返回。

优点是单 kernel、无 workspace、无主机同步。虽然仍有空 CTA，但比 starter 的“每个 Q head 一个 warp”显著减少任务数量。

### 8.2 后续：减少空任务和尾部效应

若 ragged 用例 profile 显示空 CTA 比例或长短段不均衡显著，可尝试：

- grid-stride/persistent CTA，从全局计数器领取 q tile；
- 让 CTA 顺序交错长段与短段；
- 用一个轻量调度 kernel 生成 tile 描述，再由主 kernel 消费。

但第三种方案需要存储 tile 列表；接口无 workspace，不能在 `run_kernel` 中反复 `cudaMalloc`。除非评测规则允许并能安全管理持久缓存，否则不应采用。

### 8.3 indptr 读取优化

同一 CTA 只需由少数线程读取四个边界值，再广播到整个 block。不要让每个 warp、每次循环重复访问 indptr。indptr 很小，缓存命中通常好，但减少冗余指令仍有价值。

## 9. 数值精度方案

推荐的精度策略：

- Q/K/V 输入保持 BF16；
- QK 使用 BF16 乘法与 FP32 accumulator 的矩阵乘加模式；
- score 缩放、row max、指数、row sum 使用 FP32；
- `P @ V` 累加使用 FP32；
- 最终归一化后一次性转 BF16 写出。

优化指数函数时，可比较 `__expf` 与基于 `exp2` 的实现：

```text
exp(x) = exp2(x * log2(e))
```

若把 `log2(e)/sqrt(128)` 融入 score 缩放，可减少乘法。但任何近似都必须检查：

- 全部 15 个用例的匹配率；
- 最大绝对误差；
- 超差元素是否仍满足 8 倍误差上限；
- 用例 14、15 是否逐元素通过。

建议先使用较稳健的指数实现建立正确版，再单独评估 fast math。不要同时改变 tiling、MMA 和指数精度，否则出现误差时难以归因。

## 10. 从 starter 渐进优化的路线

### 阶段 A：建立可信正确版

1. 删除 `prefix_mean_kernel` 和 `exact_len` 截断；
2. 保留现有 warp-per-query 精确计算；
3. 对全部 15 个形状验证 ragged、GQA、mask 和尾块；
4. 建立每用例延迟表。

这一版会慢，但它是后续 CUDA 实现的可读对照，不应再依赖 Python baseline 才能定位所有索引错误。

### 阶段 B：低风险 SIMT 优化

1. 固定编译期参数 `Hq=32, Hkv=4, D=128, G=8`；
2. 指针递增代替热循环完整地址重算；
3. 每次在线更新只计算一次指数；
4. 尝试 shuffle-xor 全 warp reduction，减少最终广播；
5. 使用 BF16x2/更宽向量搬运；
6. 增加单 token 与短序列专用路径。

这一阶段可验证 MACA warp shuffle、向量 load 和 fast math 的真实性能与精度。

### 阶段 C：shared K/V + GQA 复用

先不引入矩阵乘加，让一个 CTA 内多个 warp 处理同一 KV head 的多个 Q head/位置，K/V tile 只加载一次。该版本用于验证 CTA 映射、shared layout 和 mask。

如果这一阶段没有明显降低全局读流量，应检查 L2 已经缓存了多少数据，以及同步成本是否过高。

### 阶段 D：BF16 MMA FlashAttention

把 QK 和 PV 改为实际支持的矩阵乘加路径，保持阶段 C 的调度和在线 softmax不变。优先打通一个保守 tile，例如较小 `Br/Bc`，再扩展搜索空间。

### 阶段 E：流水化与自动调参

依次加入：

- K/V 双缓冲；
- shared padding/swizzle；
- 不同长度的 `Br/Bc/threads/stages` 分派；
- ragged 调度改进；
- 精度通过后的指数优化。

每次只改变一个维度，并保存代码版本、编译资源、逐用例延迟和正确性报告。

## 11. 建议的自动调参空间

固定语义参数后，可搜索：

| 参数 | 候选 |
|---|---|
| Query token tile `Br` | 4、8、16 |
| Key tile `Bc` | 16、32、64、128 |
| 同 CTA GQA heads | 2、4、8 |
| threads/CTA | 128、256 |
| K/V pipeline stages | 1、2、3（资源允许时） |
| score/P 临时存储 | register、shared 混合 |
| shared layout | plain、padding、经验证的 swizzle |
| softmax | `expf`、`__expf`、经验证的 exp2 |

剪枝规则：

1. 编译失败或使用不支持的 intrinsic，淘汰；
2. 任一正确性用例失败，淘汰；
3. 寄存器溢出或 local memory spill 明显，淘汰；
4. shared memory 导致驻留 CTA 过少且延迟恶化，淘汰；
5. 分别保留短、中、长形状的 Pareto 最优配置，再设计 dispatch。

不要只按平均延迟选配置。题目排行榜的具体加权未在仓库中公开，应至少同时观察总延迟、最差回退和每类形状表现。

## 12. 正确性与性能验证矩阵

### 12.1 正确性

每次改动都应覆盖：

| 类别 | 重点检查 |
|---|---|
| 等长 causal | 对角线与最后一行 |
| `q_len < kv_len` | bottom-right 偏移 |
| 多段 ragged | 跨段读写、每段独立 mask |
| 非 2 的幂 | q tile、K tile 尾部 mask |
| 单 token | 直接复制路径与 head 广播 |
| GQA | `hq/8` 映射、8 个输出互不覆盖 |
| 数值 | FP32 softmax/accumulator、BF16 最终写回 |

除题目总体规则外，调试阶段建议计算逐元素误差分布，并随机抽取 `(batch, q_pos, hq)` 与 CPU/PyTorch 小矩阵逐行对照。

### 12.2 计时

- `run_kernel` 内不调用 `cudaDeviceSynchronize()`；
- 预热后用 GPU event 或评测器相同机制计时；
- 分开记录 kernel launch 数和各 kernel 时间；
- 至少报告中位数，并观察高分位抖动；
- 编译和 plan 不应混入 kernel 稳态时间，除非 OJ 明确如此计分；
- 对比时固定输入、流、时钟/功耗条件和重复次数。

### 12.3 推荐指标

对任意 ragged 输入，先计算总可见配对数：

```text
pairs = sum_b sum_t clamp(Lk[b] - Lq[b] + t + 1, 0, Lk[b])
```

有效主干 FLOPs 可估算为：

```text
2 * pairs * Hq * (Dq + Dv)
```

该指标忽略 softmax 标量操作，但比直接使用 `seq_len^2` 更能公平比较 ragged 用例。

## 13. profile 时要回答的问题

不要只看总耗时。每个候选 kernel 至少要回答：

1. 是否发生 register spill/local memory 访问？
2. 每 CTA 使用多少 shared memory，能同时驻留多少 CTA？
3. BF16 矩阵乘加单元是否真的被使用，利用率如何？
4. global load 是否合并，实际带宽和缓存命中如何？
5. shared memory 是否有明显 bank conflict？
6. 时间主要在 QK、softmax 还是 PV？
7. causal 边界 tile 比例和空 ragged CTA 比例是多少？
8. 长序列是否因单 CTA 循环过长出现尾部效应？
9. 短序列是否被 launch、同步和 indptr 元数据开销主导？

如果环境缺少完整 profiler，可用逐层 ablation 代替：固定同一形状，只改变是否共享 K/V、是否 MMA、是否双缓冲，并通过延迟和编译资源反推瓶颈。

## 14. 风险与对应措施

| 风险 | 表现 | 措施 |
|---|---|---|
| MMA API 在 MACA 上行为不同 | 编译失败或无加速 | 先做最小探针，保留 SIMT fallback |
| tile 太大 | spill、占用率低 | 减小同 CTA heads、`Br` 或 `Bc` |
| tile 太小 | MMA/带宽利用率低 | 增大 `Bc` 或流水阶段 |
| shared layout 不合适 | bank conflict | padding/swizzle 实测 |
| fast exp 误差 | 长序列少量超差 | 保留 FP32 稳健路径，逐项开关 |
| mask 写错 | q<kv 用例失败 | 用 `Lk-Lq` 显式构造边界 |
| ragged 越界 | 随机错误/崩溃 | 所有段长来自 indptr，尾 tile 双重保护 |
| 过度特判 | 隐藏用例退化 | 只做数学等价、shape-based dispatch |
| 动态显存分配 | 时间不稳/不允许 | 单 kernel 或静态片上状态，不在热路径 malloc |

## 15. 推荐实施优先级

按预期收益与风险综合排序：

1. **正确性清理**：删除近似、补齐 15 用例验证、修正 benchmark；
2. **按 KV head 合并 GQA**：让同组 Q head 共享 K/V；
3. **Query/Key 分块**：实现 streaming FlashAttention，复用相邻 Query 的 K/V；
4. **BF16 MMA + FP32 累加**：获得长序列的主要算力收益；
5. **短/中/长 kernel 分派**：避免小用例被复杂 tile 拖慢；
6. **向量化与固定参数特化**：减少搬运和整数指令；
7. **双缓冲、shared layout、softmax 微调**：榨取后续性能；
8. **ragged persistent 调度或 split-K**：仅在 profile 证明必要时投入。

最重要的里程碑不是“某个 kernel 能跑”，而是：所有 15 个用例稳定正确；每个优化步骤都有逐用例性能证据；在 MXMACA 上确实执行了预期的矩阵计算与数据复用路径。这样形成的实现和调优记录也最符合比赛对 Agent/Skill 可复现性的要求。
