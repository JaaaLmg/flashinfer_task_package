# FlashInfer Paged Decode 算子原理、Baseline 实现与优化方向

本文对应 XPU-OJ 20004，分析对象是 [`paged_decode_baseline.cu`](paged_decode_baseline.cu) 与
[`benchmark_decode.py`](benchmark_decode.py)。题目接口见
[`Agent 推理算子库优化 - FlashInfer Paged Decode.md`](../xpuoj_problem/problem_20004/Agent%20推理算子库优化%20-%20FlashInfer%20Paged%20Decode.md)，
组织方式对齐 [`batch_prefill_paged_code/explaination.md`](../batch_prefill_paged_code/explaination.md)。

**这道题是前三题的「交叉点」。** 它的数据布局和 Paged Prefill (20002) 完全一样（NHD 五维
`kv_data`、`page_size=16`、GQA），但它的计算形态和 MLA (20003) 一样是 **decode**（每 request
1 个 query token）。所以：

- 从 20002 拿走：页表寻址、`last_page_len` 语义、GQA 的 `kv_head = qo_head / group_size`。
- 从 20003 拿走：memory-bound 的判断标准、KV 复用思路、split-KV。
- 20002 的 tile/MMA 那一套 **不要照搬**，因为这里没有 query 维度可以 tile。

---

## 1. 先建立整体认识

Decode 是 LLM 推理的自回归阶段：每步只生成 1 个 token，所以 query 长度恒为 1，但要和历史
积累的全部 KV cache 做注意力。计算量是 `O(L)` 而不是 prefill 的 `O(L²)`。

一个 request 的工作量：读 `L` 个 token 的 K 和 V（各 `num_kv_heads * head_dim` 个 bf16），
做 `2 * num_qo_heads * L * head_dim * 2` 次浮点运算。以 `L=16384, D=128, H_kv=4, H_qo=32` 为例：

```text
KV 字节数 = 2(K/V) * 2(bf16) * 16384 * 4 * 128 = 33.6 MB
FLOPs     = 4 * 16384 * 32 * 128              = 268 MFLOP
算术强度  = 268e6 / 33.6e6 ≈ 8 FLOP/byte
```

C500 的 BF16 算力/带宽比远高于 8，所以**这是彻底的 memory-bound 算子**。判断优化是否到位应该看
`kv_bandwidth_GB_s` 是否逼近 HBM 峰值，而不是看 TFLOPS。

三道题的定位对比：

| 对比项 | Paged Prefill (20002) | MLA (20003) | **Paged Decode (20004)** |
|---|---|---|---|
| Q 长度 | 每 request `L` | 1 | **1** |
| 复杂度 | `O(L²)` | `O(L)` | **`O(L)`** |
| 瓶颈 | compute-bound | memory-bound | **memory-bound** |
| KV 布局 | NHD 五维 paged | 两个独立张量 | **NHD 五维 paged（同 20002）** |
| page size | 16 | 1 | **16** |
| KV 共享 | GQA group=8 | 全共享 | **GQA group=8（d=64 时 4）** |
| causal | 参数控制 | 无 | **无此参数** |

优化路线：

```text
warp-per-(batch,qo_head) 标量 kernel（baseline，KV 读 group_size 遍）
    -> CTA 承担一个 (batch, kv_head) 的整个 GQA group，KV 只读一遍
    -> 128-bit 向量化访存 + page 内 16 token 连续搬运
    -> BF16 MMA（把 group 内 8 个 qo_head 当作 GEMM 的 M 维）
    -> split-KV + merge_state，解决小 batch 打不满 SM 的问题
```

---

## 2. 题目要求计算什么

### 2.1 输入张量与固定参数

评测固定 `page_block_size = 16`、`num_qo_heads = 32`，`head_dim ∈ {64, 128, 256}`，
`num_kv_heads` 随 `head_dim` 变（见 2.2），全部 BF16。

```text
q:       [batch_size, num_qo_heads, head_dim]
kv_data: [num_blocks, 2, page_block_size, num_kv_heads, head_dim]
output:  [batch_size, num_qo_heads, head_dim]
```

注意 `q` 的第一维是 `batch_size` 而不是 `batch_size * seq_len` —— **decode 每个 request 只有一
个 query 行**。这与 20002 是不同的，从 20002 复制代码时最容易在这里翻车。

`kv_data` 的第 2 维长度为 2：`kv_data[:, 0]` 是 K，`kv_data[:, 1]` 是 V。所以同一个 page 内
K 和 V 相隔 `page_block_size * num_kv_heads * head_dim` 个元素，这就是 baseline 里的
`kv_plane_stride`。

三组元数据：

| 元数据 | shape | 含义 |
|---|---|---|
| `kv_indptr[b:b+2]` | `B+1` | 第 `b` 个 request 在 `kv_indices` 中占用的页区间 |
| `kv_indices[i]` | `num_blocks` | 第 `i` 个逻辑页对应的物理页号 |
| `last_page_len[b]` | `B` | 第 `b` 个 request 最后一页的有效 token 数（`1..16`） |

**没有 `seq_len_kv` 之外的长度信息可信。** 真正的 KV 长度必须由页表算出：

```text
num_pages = kv_indptr[b+1] - kv_indptr[b]
kv_len    = (num_pages - 1) * 16 + last_page_len[b]
```

直接用参数里的 `seq_len_kv` 在评测里碰巧能过（因为各 request 等长），但那是运气 —— 正确实现
应该无视这个参数。baseline 里 `(void)seq_len_kv;` 那一行就是在明确表态。

### 2.2 GQA 分组

`num_qo_heads = 32` 固定，`num_kv_heads` 按 `head_dim` 取值（对齐官方
[`bench_batch_decode.py:71`](../benchmark/bench_batch_decode.py#L71)）：

| head_dim | num_kv_heads | group_size |
|---:|---:|---:|
| 64 | 8 | 4 |
| 128 | 4 | **8** |
| 256 | 4 | **8** |

```text
kv_head = qo_head / group_size
```

`group_size` 个 query head **共享同一份 K/V**。这是本题第一优化杠杆：baseline 让每个 qo_head
独立读一遍 KV，实际流量是理论下限的 `group_size`（8）倍。

### 2.3 页表寻址

```text
logical_page  = kv_indptr[b] + (j >> 4)        # j 是 request 内第 j 个 KV token
physical_page = kv_indices[logical_page]
entry         = j & 15
K_ptr = kv_data + physical_page * (2*16*H_kv*D) + entry*(H_kv*D) + kv_head*D
V_ptr = K_ptr   + 16*H_kv*D
```

和 20002 逐字相同。好消息是页表读取分摊到 16 个 token 上，开销可忽略；坏消息是**页与页之间
物理不连续**，跨页时访存流会断开，这限制了单纯靠预取能拿到的收益。

### 2.4 Attention 数学

```text
s_j = dot(q[b,h,:], K[t_j, kv_head, :]) * sm_scale,   j < kv_len
p   = softmax(s)
out[b,h,:] = sum_j p_j * V[t_j, kv_head, :]

sm_scale = 1 / sqrt(head_dim)
```

注意这里 `sm_scale` 就是标准的 `1/sqrt(head_dim)`（20003 的 `1/sqrt(576)` 是 MLA 特有的
拼接维度，本题**不适用**）。

**没有 causal 参数**，因为 decode 的唯一 query 是序列最后一个 token，全部历史 KV 都可见。
唯一边界就是 `j < kv_len`。

### 2.5 FLOPs 与带宽口径

```text
FLOPs     = 4 * B * L * num_qo_heads * head_dim
Bytes_min = 2 * 2 * B * L * num_kv_heads * head_dim      # K 和 V 各读一次
```

[`benchmark_decode.py`](benchmark_decode.py) 两个都记录，看 `kv_bandwidth_GB_s`。

---

## 3. Baseline CUDA 代码分析

### 3.1 工作划分

[`paged_decode_baseline.cu`](paged_decode_baseline.cu) 沿用 20002/20003 baseline 的骨架：
128 线程 CTA = 两个 64-lane warp，**每个 warp 负责一个 `(batch, qo_head)` 输出行**，任务总数
`batch_size * num_qo_heads`。

`head_dim` 最大 256 = `4 * 64`，所以每个 lane 持有最多 4 个 query 元素和 4 个输出累加器：

```text
lane l 负责维度 {l, l+64, l+128, l+192}（超出 head_dim 的部分被 if 屏蔽）
```

```cpp
constexpr int kMaxPerLane = 4;   // 256 / kWarpSize
float q_frag[kMaxPerLane];
float out_acc[kMaxPerLane];
```

`run_kernel` 里 `head_dim > kMaxPerLane * kWarpSize` 会直接拒绝，避免静默算错。

### 3.2 每个 KV token 的执行过程

```text
for j in [0, kv_len):
    physical_page = kv_indices[kv_indptr[b] + (j>>4)]
    entry         = j & 15
    score = warp_reduce( dot(q, K[page, entry, kv_head, :]) ) * sm_scale
    online_softmax_update(score, V[page, entry, kv_head, :])
```

online softmax 与 20002/20003 完全一致，FP32 累加：

```text
m_new = max(m, score)
alpha = exp(m - m_new)
beta  = exp(score - m_new)
o     = alpha * o + beta * V
d     = alpha * d + beta
m     = m_new
```

`alpha` 那行的 `row_max > -1.0e19f` 判断是为了让第一次迭代时 `alpha = 0`，避免
`exp(-1e20 - score)` 产生 NaN。

### 3.3 Baseline 的主要瓶颈（按重要性排序）

1. **KV cache 被重复读 `group_size` 遍。** `head_dim=128` 时 `group_size=8`，实际 KV 流量是
   理论下限的 8 倍。
2. **标量访存。** 每个 lane 每次读 1 个 bf16（2 字节）。一个 warp 的一次 K 读只覆盖
   `64*2=128` 字节，而 `head_dim=128` 的一整行也才 256 字节 —— 完全没有利用 page 内 16 个
   token 连续排布的结构。
3. **每个 token 一次 64-lane shuffle reduction。** `L` 次 `log2(64)=6` 级 shuffle 串行暴露，
   无法与访存重叠。
4. **并行度不足。** 任务数只有 `B * 32`。`B=1` 时只有 32 个 warp，C500 上远远打不满。

### 3.4 Baseline 实测结果

MetaX C500，随机页表（默认），22/22 全部 `match_ratio = 1.0`、`severe_error_count = 0`，
最大绝对误差 `1.95e-3`（完整数据见
[`stage_a_baseline_results.csv`](stage_a_baseline_results.csv)）。摘录几个代表点：

| case | B | L | D | baseline (ms) | flashinfer (ms) | 倍数 | KV 带宽 (GB/s) |
|---|---:|---:|---:|---:|---:|---:|---:|
| 01 | 1 | 1024 | 128 | 2.311 | 0.033 | 70x | 0.9 |
| 09 | 1 | 16384 | 128 | 39.719 | 0.064 | 621x | 0.8 |
| 11 | 16 | 16384 | 128 | 42.292 | 0.536 | 79x | 12.7 |
| 05 | 128 | 1024 | 128 | 5.726 | 0.256 | 22x | 46.9 |
| 14 | 64 | 4096 | 64 | 10.244 | 0.593 | 17x | 52.4 |
| 20 | 4 | 16384 | 256 | 63.245 | 0.406 | 156x | 4.2 |

四个可以直接读出瓶颈的现象：

1. **带宽最高只到 56.7 GB/s**（case 18），而 C500 的 HBM 带宽在 TB/s 量级 —— 利用率不到 5%。
   这是瓶颈 1（KV 重复读 8 遍）加瓶颈 2（标量访存）的直接体现。
2. **带宽只随 `B` 增长，几乎不随 `L` 变化。** `B=1` 的三个点（case 01/06/09）无论 `L` 是 1024
   还是 16384，带宽都钉在 0.8~0.9 GB/s；而固定 `L=1024` 把 `B` 从 1 提到 128，带宽从 0.9 涨到
   46.9 GB/s。说明当前实现完全是**并行度受限、延迟暴露**，不是真的在拼带宽 —— 这正是方向 4
   split-KV 要解决的问题。
3. **延迟几乎只由 `L` 决定。** case 01→05 的 `B` 翻了 128 倍，延迟只从 2.311 涨到 5.726 ms；
   而 case 01→09 的 `L` 翻 16 倍，延迟涨了 17 倍。串行的 KV 循环是唯一的关键路径。
4. **`B=128` 时增速开始变陡**（case 04→05：3.215→5.726 ms，`B` 翻倍延迟涨 78%），说明
   到这个规模 SM 才真正被填满，带宽也逼近该实现的上限 47 GB/s。

这个 baseline 的价值不在性能，而在**它是精确的**：不截断 KV、严格按 `last_page_len` 处理尾页、
默认在随机页表下验证，可以当作 correctness oracle 使用。

---

## 4. Benchmark 代码组织逻辑

[`benchmark_decode.py`](benchmark_decode.py) 与 20002 的 `benchmark_paged.py`、20003 的
`benchmark_mla.py` 结构一一对应。它做四件事：

**(1) 按题目 ABI 精确加载。** `load_kernel` 声明 `6` 个指针 + `6` 个 `int64`，顺序与题目
`run_kernel` 完全一致（注意本题**没有 causal 参数**，比 20002 少一个）。参数顺序错位在
ctypes 层不会报错，只会得到垃圾结果。

**(2) 构造与 OJ 一致的元数据。** `make_metadata` 生成 decode 形态：

```python
pages_per_request = ceil(L / 16)
kv_indptr     = arange(0, (B+1)*pages_per_request, pages_per_request)
last_page_len = full((B,), (L-1) % 16 + 1)
kv_indices    = randperm(num_pages + 7)[:num_pages]   # 默认随机
```

**默认用随机页表**，并且物理页比逻辑页多 7 个 —— 这样「把 `kv_indices[i]` 当成 `i`」的错误实现
会立刻暴露。`--identity-pages` 切换成官方 benchmark 的 `arange`，用来验证「页表连续」快路径。
注意这里的默认值和 20003 的 benchmark 相反（那题官方用 `arange`，所以默认 identity、用
`--permute-pages` 验证 fallback），因为 20002 的页表本来就是乱的。

`num_qo_heads` 固定 32，`num_kv_heads` 由 `kv_heads_for(head_dim)` 决定，与官方
benchmark 保持一致。

**(3) 对拍 FlashInfer。** 用 `BatchDecodeWithPagedKVCacheWrapper(use_tensor_cores=True)` 算出
reference，逐元素比较：

```python
tolerance = ATOL + RTOL * |reference|        # 1.6e-2 / 1.6e-2
passed = finite and match_ratio >= 0.99 and severe_count == 0
```

`severe_count`（误差超过 8 倍容差的元素数）必须为 0。它能抓出「大部分元素对、少数完全错」的
情况，典型是漏算最后一个不满的 page。

**(4) 计时。** `elapsed_ms` 用 CUDA event 包住 `repeats` 次调用取均值；`choose_repeats` 让总
时长稳定在 200 ms 附近。候选与 reference 用同一套逻辑，跨版本比较才有意义。

测试用例覆盖三个维度：`head_dim ∈ {64,128,256}`（决定 `group_size` 与每行字节数）、
`batch_size ∈ {1..128}`（决定并行度是否充足）、`seq_len_kv ∈ {257..16384}`（决定 KV 是否放得下
cache）。用例 21/22 用非 16 倍数的长度（1023、257），专门检查 `last_page_len` 尾页处理 ——
`1023 = 63*16 + 15`、`257 = 16*16 + 1`，后者的最后一页只有 1 个有效 token，是最狠的边界。

典型用法：

```bash
# 建立基线
python paged_decode/benchmark_decode.py \
  --source paged_decode/paged_decode_baseline.cu \
  --library paged_decode/paged_decode_baseline.so \
  --cases all --max-repeats 3 --force-build \
  --output paged_decode/stage_a_baseline_results.csv

# 快速回归（含尾页边界）
python paged_decode/benchmark_decode.py --cases 1,21,22 --max-repeats 3

# 验证 identity 页表快路径
python paged_decode/benchmark_decode.py --cases 1,9 --identity-pages
```

---

## 5. 优化方向

按预期收益从大到小排列。**建议严格按顺序做，每一步都跑一次回归再进入下一步。**

### 方向 1：一个 CTA 承担一个 GQA group（数量级收益）

既然 `group_size` 个 qo_head 共享同一份 K/V，就应该让 KV 只被读一次：

```text
grid = (batch_size, num_kv_heads)   或 (batch_size, num_kv_heads, num_kv_splits)
每个 CTA:
    把本 group 的 group_size 个 qo_head 的 q 常驻寄存器
    for each page (16 token):
        协同把该 page 的 K/V 载入 shared memory        <- 只读一次
        group 内每个 head 用 shared 中的同一份 KV 算 QK / PV
```

`head_dim=128` 时理论 KV 流量下降 8 倍。这一步做完，性能应该从「比 reference 慢两三个数量级」
进到「同一量级」。

shared memory 预算：一个 page 的一个 kv_head 的 K+V 是 `2 * 16 * 128 * 2 = 8 KiB`
（`head_dim=256` 时 16 KiB）。放得下 double buffering，可以按 2~4 个 page 一个 tile 规划。

### 方向 2：向量化访存 + 利用 page 内连续性

- 用 `uint4`（128-bit = 8 个 bf16）搬运。`head_dim=128` 一行是 16 个 `uint4`，一个 64-lane warp
  一次可以搬 4 个 token 的 K。
- **page 内 16 个 token 的同一个 kv_head 在内存中是跨 stride 的**（stride = `H_kv * D`），
  不是完全连续。所以按「一个 page 的一个 kv_head」这个 `16 × head_dim` 的二维块来搬运，
  每行连续、行间等距，正好是合并访存的形状。
- K 和 V 相隔 `kv_plane_stride`，可以两条独立的 load 流水，不要串行等待。

### 方向 3：QK/PV 交给 BF16 MMA

把 group 内 `group_size` 个 head 的 query 视作 `[group_size, head_dim]` 的矩阵，与
`[tile_tokens, head_dim]` 的 K 做矩阵乘得到 `[group_size, tile_tokens]` 的 score。
decode 虽然「只有 1 个 query token」，但 **GQA group 内 8 个 head 拼起来就是 GEMM 的 M 维**。

`group_size=8` 偏小（MMA 的 M 通常是 16），可以考虑一个 CTA 同时处理 2 个 kv_head 的 group 把
M 凑到 16，或者接受 padding。这一步能消掉 baseline 中每 token 一次的 shuffle reduction。

### 方向 4：split-KV（小 batch 场景的关键）

`B=1` 时即使方向 1 做完，也只有 `1 * num_kv_heads = 4` 个 CTA，SM 利用率极低。把一个
`(batch, kv_head)` 的 KV 沿长度切成 `S` 段，每段一个 CTA 独立算出**部分 output、部分 max `m`、
部分 sum `d`**，再用第二个 kernel 归并：

```text
m   = max(m_1, m_2)
d   = d_1 * exp(m_1 - m) + d_2 * exp(m_2 - m)
out = (o_1 * exp(m_1 - m) + o_2 * exp(m_2 - m)) / d
```

FlashInfer 内部的 `merge_state` 走的就是这条路。`S` 应根据 `B * num_kv_heads` 与 SM 数动态选取：
`B=64/128` 时 `S=1`，`B=1, L=16384` 时 `S` 取 16~64 量级。

用例 9（`B=1, L=16384`）与用例 5（`B=128, L=1024`）是这条路径的两个极端，用它们对照能直接看出
split 策略是否选对。

### 方向 5：identity 页表特化（收益有限，谨慎）

`--identity-pages` 时 `kv_indices[i] == i`，KV 物理连续。但 **OJ 的页表大概率不是 identity**
（20002 的经验），而且本题页表读取本来就分摊到 16 个 token 上、开销很小。所以这条的性价比
远低于 20003，**建议放到最后，或者干脆不做**。若要做，必须保留 fallback 并用默认（随机页表）
模式验证。

### 方向 6：低优先级细节

- `exp` 换成 `exp2`（把 `sm_scale` 预乘 `log2(e)`），常见的小幅收益。
- `head_dim=64` 的 case（`group_size=4`、`num_kv_heads=8`）分组更小、CTA 更多，参数可能需要
  和 128/256 分开调，用模板特化而不是运行时分支。
- 32 位 offset：`num_blocks * 2 * 16 * H_kv * D` 在最大用例下约 `1.3e8`，安全，但收益很小。

### 不应该做的事

- 不要用参数 `seq_len_kv` 代替页表算出的 `kv_len`。评测里等长所以能过，但这是在赌。
- 不要照搬 Paged Prefill 的 `O(L²)` tile 策略 —— decode 没有 query 维度可以 tile，加大 Q tile
  只会浪费寄存器。
- 不要截断 KV、不要用均值近似。异常快的结果必须先怀疑是不是漏读了 KV。

---

## 6. 常见实现错误

- 把 `q` 当成 `[B*L, H, D]`（20002 的形状），实际是 `[B, H, D]`；
- 用参数 `seq_len_kv` 而不是 `(num_pages-1)*16 + last_page_len[b]` 算 KV 长度；
- 忘记最后一页是部分填充的，多算了 padding token（`match_ratio` 会掉但不会归零，很隐蔽）；
- 忽略 `kv_indices`，直接用逻辑页号当物理页号（identity 页表下能过，随机页表下全错）；
- K/V 的 plane 偏移写反或漏掉 `kv_plane_stride`；
- GQA 映射写成 `kv_head = qo_head % num_kv_heads`（应该是整除）；
- 每个 qo_head 独立读一遍 KV，浪费掉 GQA 的结构优势；
- `B=1` 时不做 split-KV，SM 利用率极低却以为是访存瓶颈；
- 只测 16 倍数长度，漏掉 1023/257 这类尾页；
- 用 TFLOPS 判断优化是否到位 —— decode 场景应该看有效带宽；
- 把 20003 的 `sm_scale = 1/sqrt(576)` 抄过来（本题是 `1/sqrt(head_dim)`）。

---

## 7. 复现

编译 baseline：

```bash
mxcc -O3 -std=c++17 --offload-arch=xcore1000 \
  -I/opt/maca/tools/cu-bridge/include -shared -fPIC \
  paged_decode/paged_decode_baseline.cu \
  -o paged_decode/paged_decode_baseline.so
```

跑全量基线：

```bash
python paged_decode/benchmark_decode.py \
  --cases all --max-repeats 3 --force-build \
  --output paged_decode/stage_a_baseline_results.csv
```

开发新版本时的最小回归集：先跑 case 21/22（尾页边界）确认正确性，再跑 case 12/16
（`head_dim` 的两个极端 64 和 256）确认维度分支，最后跑 case 9 与 case 5
（`B=1` 长序列 vs `B=128` 短序列）看两端性能。这个顺序能最早暴露正确性问题。

---

## 8. 总结

Paged Decode 的数学是三道题里最简单的：没有 causal mask、没有维度拼接、`sm_scale` 就是标准的
`1/sqrt(head_dim)`。它的难点全在**性能模型的转换**上 —— 数据布局看着和 20002 一模一样，
会诱导你复用 prefill 的 tile/MMA 思路，但 decode 是 memory-bound 的，那套思路在这里不成立。

baseline 用一个 warp 算一行输出，把 KV cache 重复读了 `group_size` 遍且访存全是标量，实测
带宽利用率不到 5%，在低并行度场景下比 FlashInfer 慢两个数量级以上
（`B=1,L=16384,D=128`：`39.719 ms` vs `0.064 ms`，621 倍）。优化的第一步（也是收益最大的一步）是让一个 CTA 承担一个 GQA
group，把 KV 读取次数降到理论下限；`B=1` 这类低并行度场景则必须靠 split-KV 才能填满 SM。
之后才轮到向量化、MMA 这些常规手段。
