# FlashInfer MLA Paged Attention 算子原理、Baseline 实现与优化方向

本文对应 XPU-OJ 20003，分析对象是 [`mla_paged_baseline.cu`](mla_paged_baseline.cu) 与
[`benchmark_mla.py`](benchmark_mla.py)。题目接口见
[`Agent 推理算子库优化 - FlashInfer MLA Paged Attention.md`](../xpuoj_problem/problem_20003/Agent%20推理算子库优化%20-%20FlashInfer%20MLA%20Paged%20Attention.md)，
组织方式对齐 [`batch_prefill_paged_code/explaination.md`](../batch_prefill_paged_code/explaination.md)。

**和前两题最重要的区别：MLA 是 decode，不是 prefill。** Ragged/Paged Prefill 的计算量是
`O(L²)`，属于 compute-bound，优化重点是让 MMA 吃满；MLA decode 每个 request 只有 1 个
query token，计算量是 `O(L)`，属于彻底的 **memory-bound**。因此本题的优化目标不是
TFLOPS，而是 **KV cache 的有效读取带宽**。把 Paged Prefill 那套 tile/MMA 思路原样搬过来
会得到一个正确但慢的实现。

---

## 1. 先建立整体认识

MLA（Multi-head Latent Attention）是 DeepSeek 提出的注意力变体。标准 MHA 的 KV cache 要为
每个 KV head 存一份 K 和一份 V；MLA 把 KV 压缩成一个 **所有 head 共享的低维 latent 向量**
`ckv`，再单独保留一小段带 RoPE 的 `kpe`。推理时：

```text
K_j = concat(ckv_j, kpe_j)        # 576 = 512 + 64
V_j = ckv_j                       # 512，直接复用 latent，不再单独存 V
```

这带来两个结构性后果，它们决定了整道题的优化空间：

1. **KV cache 体积极小。** 每个 token 只存 `512 + 64 = 576` 个 bf16，而不是
   `2 * num_heads * head_dim`。这正是 MLA 的卖点。
2. **KV 在所有 head 之间完全共享。** `num_heads` 个 query head 读的是同一份 `ckv/kpe`。
   等价于一个 `group_size = num_heads` 的极端 GQA。

第 2 点是本题最大的优化杠杆：**一次把 KV token 读进 shared memory / 寄存器，可以给
64 或 128 个 head 复用**。baseline 恰恰完全没有利用这一点，所以它把 KV cache 重复读了
`num_heads` 遍。

| 对比项 | Paged Prefill (20002) | MLA Paged Attention (20003) |
|---|---|---|
| Q 长度 | 每 request `L` 个 token | 每 request **1** 个 token |
| 计算复杂度 | `O(L²)` | `O(L)` |
| 瓶颈 | compute-bound，靠 MMA | **memory-bound，靠带宽** |
| K/V 关系 | K、V 是两个独立张量 | **V 就是 K 的前 512 维** |
| head 共享 | GQA，group=8 | 全共享，group=`num_heads` |
| page size | 16 | **1** |
| head dim | 128 / 256 | qk=576，vo=512（不对称） |
| 主要优化手段 | tile、MMA、流水 | KV 复用、split-KV、向量化访存 |

优化路线大致是：

```text
warp-per-(batch,head) 标量 kernel（baseline，KV 读 num_heads 遍）
    -> CTA 负责一个 batch 的所有 head，KV 只读一遍（shared memory 复用）
    -> 128-bit 向量化访存 + FP32 online softmax
    -> BF16 MMA 承担 QK 与 PV
    -> split-KV + 二次归并，解决小 batch 下 SM 打不满的问题
```

---

## 2. 题目要求计算什么

### 2.1 输入张量和固定参数

评测固定 `head_dim_ckv = 512`、`head_dim_kpe = 64`、`page_size = 1`、`causal = 0`，
`num_heads ∈ {64, 128}`，全部 BF16。

```text
q_nope: [batch_size, num_heads, 512]
q_pe:   [batch_size, num_heads, 64]
ckv:    [batch_size * seq_len, 1, 512]
kpe:    [batch_size * seq_len, 1, 64]
output: [batch_size, num_heads, 512]
```

注意 `ckv/kpe` 的中间那一维是 `1`，就是 page size。**page_size = 1 意味着一个 page 就是
一个 token**，页表 `kv_indices[i]` 直接给出该逻辑位置对应的物理 token 行号，不存在
20002 里的「页内 entry」这一层。

四组元数据：

| 元数据 | shape | 含义 |
|---|---|---|
| `q_indptr[b]` | `B+1` | 第 `b` 个 request 的 query 行起点，decode 下就是 `[0,1,...,B]` |
| `kv_indptr[b:b+2]` | `B+1` | 第 `b` 个 request 在 `kv_indices` 中占用的区间 |
| `kv_indices[i]` | `B*L` | 第 `i` 个逻辑 KV 位置对应的物理 token 行号 |
| `kv_lens[b]` | `B` | 第 `b` 个 request 实际参与 softmax 的 KV 长度 |

**`kv_lens` 和 `kv_indptr` 区间长度不是一回事。** 前者是真正的有效长度，后者是分配的
页数。评测里两者相等（都等于 `seq_len`），但正确实现应该以 `kv_lens[b]` 为准并
clamp 到区间长度内 —— 这正是 baseline 里那两行 clamp 的用途，等价于 20002 中
`last_page_len` 的角色。

### 2.2 页表寻址（比 20002 简单一档）

```text
page_table_pos = kv_indptr[b] + j          # j 是 request 内的第 j 个 KV
token          = kv_indices[page_table_pos]
ckv_row        = ckv + token * 512
kpe_row        = kpe + token * 64
```

没有 `>>4` / `&15`，没有 K/V 两个 plane 交错。代价是页表本身变大了：20002 每 16 个 token
读一次页表，这里 **每个 token 都要读一次**。所以 `kv_indices` 的读取带宽不再可以忽略，
它是 `4 / (2*576) ≈ 0.35%` 的额外流量 —— 仍然很小，但访问模式（每 token 一次 gather）
会影响后续 KV 读取能否合并。

### 2.3 Attention 数学

对 request `b`、head `h`，令 `t_j = kv_indices[kv_indptr[b] + j]`：

```text
s_j = ( dot(q_nope[b,h,:], ckv[t_j,:]) + dot(q_pe[b,h,:], kpe[t_j,:]) ) * sm_scale
p   = softmax(s)
out[b,h,:] = sum_j p_j * ckv[t_j,:]           # 注意 V 就是 ckv
```

其中

```text
sm_scale = 1 / sqrt(head_dim_ckv + head_dim_kpe) = 1 / sqrt(576)
```

**这个 scale 用的是 576 而不是 512。** 因为 QK 的实际维度是拼接后的 576。写错会让所有
输出偏离一个固定的 softmax 温度，`match_ratio` 掉到很低但形状看起来还「像」对的，是本题
最容易踩的坑之一。题目正文没写这个公式，它来自 benchmark 脚本
[`bench_batch_mla.py:30`](../benchmark/bench_batch_mla.py#L30)，OJ 后台参考实现用的也是这个值。

`causal` 固定为 0，且 decode 只有 1 个 query token，即使 `causal=1` 也是全可见，所以
本题不需要 mask —— 唯一的边界就是 `j < kv_lens[b]`。

### 2.4 FLOPs 与带宽口径

QK 有 576 维、PV 有 512 维，各含一次乘和一次加：

```text
FLOPs = 2 * B * num_heads * L * (2 * 512 + 64)
```

KV cache 的**理论最小**读取量（每个 token 只读一次）：

```text
Bytes_min = 2 * B * L * (512 + 64)
```

[`benchmark_mla.py`](benchmark_mla.py) 两个都记录。判断优化是否到位，应该看
`kv_bandwidth_GB_s` 是否接近 C500 的 HBM 峰值，而不是看 TFLOPS —— 后者在 decode 场景下
天然极低，不代表实现差。

顺便算一下算术强度：每读 1 个 KV token（1152 字节）要做
`num_heads * 2 * 1152 ≈ 147K` FLOPs（`num_heads=64`）。看起来强度不低，**但前提是这个
token 只被读一次**。baseline 每个 head 各读一遍，实际强度直接除以 `num_heads`，退化成
纯带宽受限，这就是它慢两个数量级的根本原因。

---

## 3. Baseline CUDA 代码分析

### 3.1 工作划分

[`mla_paged_baseline.cu`](mla_paged_baseline.cu) 沿用 20002 baseline 的组织方式：128 线程
CTA = 两个 64-lane warp，**每个 warp 负责一个 `(batch, head)` 输出行**，任务总数
`batch_size * num_heads`。

`head_dim_ckv = 512 = 8 * 64`，所以每个 lane 持有 8 个 `q_nope` 元素和 8 个输出累加器；
`head_dim_kpe = 64 = 1 * 64`，每个 lane 持有 1 个 `q_pe` 元素：

```text
lane l 负责 ckv 维度 {l, l+64, l+128, ..., l+448}
lane l 负责 kpe 维度 {l}
```

```cpp
float q_nope_frag[8];   // kMaxCkvPerLane
float q_pe_frag[1];     // kMaxPePerLane
float out_acc[8];
```

这三个数组的尺寸由 `kMaxCkvPerLane * kWarpSize = 512` 和 `kMaxPePerLane * kWarpSize = 64`
决定，`run_kernel` 里对应地拒绝超出该范围的 head dim，避免静默算错。

### 3.2 每个 KV token 的执行过程

```text
for j in [0, kv_len):
    token = kv_indices[kv_indptr[b] + j]        # page_size=1，页表即 token 号
    score = warp_reduce( dot(q_nope, ckv[token]) + dot(q_pe, kpe[token]) ) * sm_scale
    online_softmax_update(score, ckv[token])    # V 复用同一段 ckv
```

online softmax 与 20002 完全一致，FP32 累加：

```text
m_new = max(m, score)
alpha = exp(m - m_new)
beta  = exp(score - m_new)
o     = alpha * o + beta * V
d     = alpha * d + beta
m     = m_new
```

值得注意的一处 MLA 特性：`ckv_ptr` 在 QK 和 PV 中被复用了两次。理论上一次 load 就够，但
baseline 写成两个独立循环，编译器是否复用寄存器不受控 —— 这是优化版可以立刻拿到的收益。

### 3.3 Baseline 的主要瓶颈（按重要性排序）

1. **KV cache 被重复读 `num_heads` 遍。** `num_heads=64` 时实际 KV 流量是理论下限的
   64 倍。这是数量级瓶颈，其余问题加起来都不如它。
2. **标量访存。** 每个 lane 每次读 1 个 bf16（2 字节），远低于 128-bit 向量宽度。
3. **每个 token 一次 64-lane shuffle reduction。** `L` 次 `log2(64)=6` 级 shuffle，延迟
   完全串行暴露，无法与访存重叠。
4. **并行度不足。** 任务数只有 `B * num_heads`。`B=1, H=64` 时只有 64 个 warp，
   在 C500 上远远打不满，SM 大部分闲置。

### 3.4 Baseline 实测结果

MetaX C500，identity 页表，22/22 全部 `match_ratio = 1.0`、`severe_error_count = 0`，
最大绝对误差 `3.9e-3`（完整数据见
[`stage_a_baseline_results.csv`](stage_a_baseline_results.csv)）。摘录几个代表点：

| case | B | L | H | baseline (ms) | flashinfer (ms) | 倍数 | KV 带宽 (GB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 1 | 1024 | 64 | 5.390 | 0.036 | 150x | 0.2 |
| 05 | 1 | 4096 | 64 | 21.487 | 0.052 | 413x | 0.2 |
| 10 | 1 | 16384 | 64 | 91.160 | 0.086 | 1060x | 0.2 |
| 12 | 16 | 16384 | 64 | 108.649 | 0.396 | 274x | 2.8 |
| 19 | 1 | 16384 | 128 | 90.998 | 0.108 | 843x | 0.2 |
| 16 | 64 | 1024 | 128 | 20.645 | 0.325 | 64x | 3.7 |

三个可以直接读出瓶颈的现象：

1. **带宽只有 0.2~5.7 GB/s**，而 C500 的 HBM 带宽在 TB/s 量级，利用率不到千分之一 ——
   这是瓶颈 1（KV 重复读 `num_heads` 遍）加瓶颈 2（标量访存）的直接体现。
2. **`B=1` 的所有点带宽都恰好是 0.2 GB/s**，与 `L` 无关。说明这些点根本没在拼带宽，
   而是并行度不足（只有 `1*64=64` 个 warp）导致延迟完全暴露 —— 对应方向 4 的 split-KV。
3. **`num_heads` 从 64 翻到 128，延迟几乎不变**（case 10 vs 19：91.2 vs 91.0 ms）。因为
   head 数翻倍的同时可用并行度也翻倍，两者抵消。这反过来证明当前实现受限于访存延迟而
   非计算量。

这个 baseline 的价值不在性能，而在**它是精确的**：不截断 KV、不做近似、不缓存输出，
可以当作 correctness oracle 使用。

---

## 4. Benchmark 代码组织逻辑

[`benchmark_mla.py`](benchmark_mla.py) 与 20002 的 `benchmark_paged.py` 结构一一对应，
读懂一个就读懂另一个。它做四件事：

**(1) 按题目 ABI 精确加载。** `load_kernel` 声明 `9` 个指针 + `7` 个 `int64`，顺序与题目
`run_kernel` 完全一致。参数顺序错位在 ctypes 层不会报错，只会得到垃圾结果，所以这里
必须逐个对照题面。

**(2) 构造与 OJ 一致的元数据。** `make_metadata` 生成 decode 形态：

```python
q_indptr  = arange(0, B+1)               # 每个 request 恰好 1 个 query
kv_indptr = arange(0, (B+1)*L, L)        # 每个 request L 个 page
kv_lens   = full((B,), L)
kv_indices = arange(B*L)                 # 或 randperm(B*L)
```

`--permute-pages` 把 `kv_indices` 换成随机置换。**官方 benchmark 用的是 `arange`**
（见 [`bench_batch_mla.py:35`](../benchmark/bench_batch_mla.py#L35)），但 OJ 后台不保证如此，
所以任何依赖「页表连续」的优化都必须先检测、再 fallback，并用这个开关验证 fallback 路径。

**(3) 对拍 FlashInfer。** 用 `BatchMLAPagedAttentionWrapper` 算出 reference，逐元素比较：

```python
tolerance = ATOL + RTOL * |reference|        # 1.6e-2 / 1.6e-2
passed = finite and match_ratio >= 0.99 and severe_count == 0
```

`severe_count`（误差超过 8 倍容差的元素数）必须为 0。这条比 `match_ratio` 更严：它能抓出
「大部分元素对、少数元素完全错」的情况，例如漏算最后一个 KV tile。

**(4) 计时。** `elapsed_ms` 用 CUDA event 包住 `repeats` 次调用取均值；`choose_repeats`
让总时长稳定在 200 ms 附近，短用例自动多跑几次以压掉 launch 抖动。候选和 reference 用
同一套逻辑计时，跨版本比较才有意义。

测试用例覆盖三个维度：`num_heads ∈ {64,128}`（决定 KV 复用收益）、`batch_size ∈ {1..64}`
（决定并行度是否充足）、`seq_len ∈ {257..16384}`（决定 KV 是否放得下 cache）。用例 21/22
用非 2 的幂长度（1023、257），专门检查尾块处理 —— 只测 1024/4096 这类对齐长度会掩盖
边界 bug。

典型用法：

```bash
# 建立基线
python mla_paged_attention_code/benchmark_mla.py \
  --source mla_paged_attention_code/mla_paged_baseline.cu \
  --library mla_paged_attention_code/mla_paged_baseline.so \
  --cases all --max-repeats 3 --force-build \
  --output mla_paged_attention_code/stage_a_baseline_results.csv

# 快速回归（含非对齐长度）
python mla_paged_attention_code/benchmark_mla.py --cases 1,13,21,22 --max-repeats 3

# 验证随机页表 fallback
python mla_paged_attention_code/benchmark_mla.py --cases 1,13 --permute-pages
```

---

## 5. 优化方向

按预期收益从大到小排列。**建议严格按顺序做，每一步都跑一次回归再进入下一步** ——
20002 的经验是，跳步之后出现的错误极难定位到底是哪一层引入的。

### 方向 1：一个 CTA 承担一个 request 的所有 head（数量级收益）

这是本题的核心。既然所有 head 共享同一份 KV，就应该让 KV 只被读一次：

```text
grid  = (batch_size,)  或 (batch_size, num_kv_splits)
每个 CTA:
    把本 CTA 负责的所有 head 的 q_nope/q_pe 常驻寄存器或 shared memory
    for each KV tile:
        协同把 tile 的 ckv/kpe 载入 shared memory      <- 只读一次
        每个 head 用 shared 中的同一份 KV 算 QK / PV
```

`num_heads=64` 时理论 KV 流量下降 64 倍。这一步做完，性能应该从「比 reference 慢 1000 倍」
进到「同一量级」。

需要权衡的是 shared memory：一个 KV token 占 `1152` 字节，64 KiB shared 只能放约 56 个
token。所以 tile 大小要和 double buffering 一起规划，典型取 16 或 32 token 一个 tile。

### 方向 2：向量化访存与 KV 复用

- 用 `uint4`（128-bit = 8 个 bf16）搬运 `ckv`。512 维正好 64 个 `uint4`，一个 64-lane warp
  一次就能覆盖整行，访存完全合并。
- `ckv` 在 QK 和 PV 中是同一份数据，载入 shared 后两次使用，不要重复 global load。
- `kpe` 只有 64 维（128 字节），可以和 `ckv` 分开放在 shared 的不同区域，避免 bank conflict。

### 方向 3：QK/PV 交给 BF16 MMA

把 `num_heads` 个 head 的 query 视作一个 `[num_heads, 576]` 的矩阵，与 `[tile, 576]` 的 K
做矩阵乘，就得到 `[num_heads, tile]` 的 score。这天然是 MMA 的形状 —— MLA decode 虽然
「只有 1 个 query token」，但 **64/128 个 head 拼起来就是一个大 GEMM 的 M 维**。

这一步能消掉 baseline 中每 token 一次的 shuffle reduction。注意 PV 的 N 维是 512，
需要 `512/16 = 32` 个 MMA fragment，寄存器压力较大，可能要沿 ckv 维度切块累加。

### 方向 4：split-KV（小 batch 场景的关键）

`B=1` 时即使方向 1 做完，也只有 1 个 CTA 在跑，SM 利用率仍然极低。做法是把一个 request 的
KV 沿长度切成 `S` 段，每段一个 CTA 独立算出 **部分 output、部分 max `m`、部分 sum `d`**，
再用第二个 kernel 按 online softmax 的合并公式归并：

```text
m   = max(m_1, m_2)
d   = d_1 * exp(m_1 - m) + d_2 * exp(m_2 - m)
out = (o_1 * exp(m_1 - m) + o_2 * exp(m_2 - m)) / d
```

FlashInfer 内部的 `merge_state` 走的就是这条路。`S` 应该根据 `B * num_heads` 与 SM 数动态
选取：并行度已经够（如 `B=64`）时 `S=1`，`B=1, L=16384` 时 `S` 取 16~64 量级。

从测试用例设计上看，`B=1` 的长序列用例（case 5/8/10/17/19）就是专门用来暴露这个问题的。

### 方向 5：identity 页表特化

官方 benchmark 的 `kv_indices` 是 `arange`，此时 `kv_indices[i] == i`，KV 在物理上完全
连续。可以像 20002 那样在首次 plan 时把页表拷回 host 检测，若为 identity 则用模板参数
编译期消除页表读取，KV 读取变成纯连续流式访问。

**但必须保留 fallback。** 用 `--permute-pages` 验证随机页表路径仍然正确。20002 的教训是：
identity 快路径本身很容易写对，写错的往往是被忽略的 fallback。

### 方向 6：低优先级的细节

- 32 位 offset：本题 `B*L*512` 最大约 `4.2e8`，接近但未超过 `2^32`，用 `uint32_t` 前要
  先确认最大用例的边界，收益不大而风险不小。
- `q_pe` 只有 64 维，占 QK 的 11%，单独优化它的收益有限。
- online softmax 的 `exp` 可以用 `exp2` 配合把 `sm_scale` 预乘 `log2(e)` 来替代，是常见的
  小幅收益。

### 不应该做的事

- 不要截断 KV、不要用均值近似、不要缓存输出。这些能刷出漂亮数字但结果是错的，
  20002 的总结里明确记录了「异常快的结果必须先检查是否完整加载了所有 fragment」。
- 不要照搬 Paged Prefill 的 `O(L²)` tile 策略。decode 没有 query 维度可以 tile，
  盲目加大 Q tile 只会浪费寄存器。

---

## 6. 常见实现错误

- `sm_scale` 用 `1/sqrt(512)` 而不是 `1/sqrt(576)`；
- 忘记 `V == ckv`，去找一个并不存在的 V 张量；
- 忽略 `kv_indices`，直接用 `kv_indptr[b] + j` 当 token 号（identity 页表下能通过，随机页表下全错）；
- 用 `kv_indptr` 区间长度代替 `kv_lens[b]`；
- 以为 `page_size=1` 就可以忽略页表，实际每个 token 都要查一次；
- 每个 head 独立读一遍 KV，浪费掉 MLA 最大的结构优势；
- `B=1` 时不做 split-KV，SM 利用率极低却以为是访存瓶颈；
- 只测对齐长度（1024/4096），漏掉 1023/257 这类尾块；
- 用 TFLOPS 判断优化是否到位 —— decode 场景应该看有效带宽。

---

## 7. 复现

编译 baseline：

```bash
mxcc -O3 -std=c++17 --offload-arch=xcore1000 \
  -I/opt/maca/tools/cu-bridge/include -shared -fPIC \
  mla_paged_attention_code/mla_paged_baseline.cu \
  -o mla_paged_attention_code/mla_paged_baseline.so
```

跑全量基线：

```bash
python mla_paged_attention_code/benchmark_mla.py \
  --cases all --max-repeats 3 --force-build \
  --output mla_paged_attention_code/stage_a_baseline_results.csv
```

开发新版本时的最小回归集：先跑 case 21/22（非对齐长度）确认边界，再跑
`--permute-pages` 的 case 1/13 确认页表路径，最后跑 case 10/19（`B=1` 长序列）看性能。
这个顺序能最早暴露正确性问题，避免在错误的实现上继续调性能。

---

## 8. 总结

MLA Paged Attention 的难点不在 attention 公式本身 —— 它比 Paged Prefill 简单（page_size=1、
无 causal mask、无 GQA 分组寻址）。真正的难点在于**认清它是 memory-bound 的 decode 算子**，
以及**利用「所有 head 共享同一份 KV」这个 MLA 独有的结构**。

baseline 用一个 warp 算一行输出，把 KV cache 重复读了 `num_heads` 遍，因此在长序列上比
FlashInfer 慢约三个数量级（`B=1,L=16384,H=64`：`91.160 ms` vs `0.086 ms`）。优化的第一步
（也是收益最大的一步）就是让一个 CTA 承担一个 request 的全部 head，把 KV 读取次数降到
理论下限；之后才轮到向量化、MMA、split-KV 这些常规手段。
