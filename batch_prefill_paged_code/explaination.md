# FlashInfer Paged Prefill 算子原理、CUDA 实现与优化教程

本文对应 XPU-OJ 20002，分析对象是 [`paged_prefill_baseline.cu`](paged_prefill_baseline.cu)、各阶段历史版本和最终 [`paged_prefill_optimized.cu`](paged_prefill_optimized.cu)。优化记录来自 [`summarize.md`](summarize.md)，题目接口见 [`Agent 推理算子库优化 - FlashInfer Paged Prefill.md`](../xpuoj_problem/problem_20002/Agent%20推理算子库优化%20-%20FlashInfer%20Paged%20Prefill.md)。

当前最终源码与 `paged_prefill_stage_af_final.cu` 完全一致。它严格计算完整 attention，没有截断 KV、均值近似或输出缓存。代表性的 `B=1,L=16384,D=128` 用例从 baseline 的 `8235.150 ms` 降到 identity-page 最终路径的约 `38.171 ms`；随机页表、尾页不满和 D256 仍由精确 paged 路径处理。

---

## 1. 先建立整体认识

Paged Prefill 和 Ragged Prefill 计算的是同一种 attention，差别主要在 K/V 的存储方式。Ragged 版本把每个序列的 K/V token 连续放置；Paged 版本把 K/V 切成固定大小的页，再通过页表把逻辑页映射到物理页。GPU 在读取一个 KV token 前，必须先完成“逻辑 token -> 逻辑页 -> 物理页 -> 页内位置 -> K/V 地址”的转换。

| 对比项 | Ragged Prefill | Paged Prefill |
|---|---|---|
| K/V 位置 | 每个 request 内连续 | 通过页表间接寻址 |
| 元数据 | `kv_indptr` 给出 token 段 | `kv_indptr + kv_indices + last_page_len` 描述页 |
| 主要额外成本 | ragged 调度和尾块 | 页表读取、地址 gather、K/V 交错布局 |
| 连续性 | 相邻 token 通常物理连续 | 只有同页 token 必然连续 |
| 最终优化重点 | tile、流水、调度 | paged loader、页表广播、identity 特化、unpack |

整个优化过程可以概括为四次变化：先从标量 warp kernel 切换到 BF16 MMA；再针对 page size 16 优化页表解析；随后补齐 KV32 loader 并用寄存器预取隐藏 gather 延迟；最后识别连续页表，将长序列 K/V 解包为连续张量，交给更大的 Q256 MMA 主干。

```text
标量 paged attention
    -> BF16 MMA paged attention
    -> 页表广播、32 位偏移和 KV32 loader
    -> register-staged paged pipeline
    -> identity page 检测
    -> 长序列 unpack + Q256 contiguous attention
```

---

## 2. 题目要求计算什么

### 2.1 输入张量和固定参数

题目固定使用 BF16、NHD 布局、page size 16、32 个 Q head、4 个 KV head和非因果 attention。head dimension 取 128 或 256。

```text
Q:       [B * L, 32, D]
KV data: [num_physical_pages, 2, 16, 4, D]
Output:  [B * L, 32, D]
```

`kv_data` 的第二维大小为 2，`kv_data[:,0]` 是 K，`kv_data[:,1]` 是 V。一个物理页内部先存 16 个 token 的所有 K，再存 16 个 token 的所有 V。

题目还提供四组元数据：

| 元数据 | 含义 |
|---|---|
| `qo_indptr[b:b+2]` | 第 `b` 个 request 的 Q 起止位置 |
| `kv_indptr[b:b+2]` | 第 `b` 个 request 在 `kv_indices` 中占用的逻辑页范围 |
| `kv_indices[i]` | 第 `i` 个逻辑页对应的物理页号 |
| `last_page_len[b]` | 第 `b` 个 request 最后一页的有效 token 数 |

一个 request 的 KV 长度不是“页数乘 16”，而是：

```text
num_pages = kv_indptr[b+1] - kv_indptr[b]
kv_len = (num_pages - 1) * 16 + last_page_len[b]
```

如果最后一页只有 1 个有效 token，其他 15 个位置只是分配空间，不能参与 softmax。

### 2.2 页表寻址

设 request `b` 的局部 KV 位置为 `t`，page size 为 16，则：

```text
logical_page_in_request = t >> 4
entry_in_page           = t & 15
page_table_pos          = kv_indptr[b] + logical_page_in_request
physical_page           = kv_indices[page_table_pos]
```

例如一个 request 有 32 个 token、两个逻辑页，`kv_indices=[5,2]`。逻辑 token 0～15 实际存放在物理页 5，逻辑 token 16～31 存放在物理页 2。不能把逻辑页号直接当作 `kv_data` 第一维的下标。

对固定 `physical_page、entry、kv_head、feature`，K 的 element offset 是：

```text
page_stride = 2 * 16 * Hkv * D
token_stride = Hkv * D
head_stride = D

k_offset = physical_page * page_stride
         + entry * token_stride
         + kv_head * head_stride
         + feature
```

V 位于同一物理页的第二个 plane，所以：

```text
v_offset = k_offset + 16 * Hkv * D
```

最终代码正是用 `k_data=kv_data`、`v_data=kv_data+16*Hkv*D` 和 `stride_page=2*16*Hkv*D` 描述这个布局。

### 2.3 GQA 和 attention 数学

32 个 Q head 共享 4 个 KV head，group size 为 8：

```text
kv_head = qo_head / 8
```

题目固定 `causal=0`，因此每个 query token 都能看到本 request 的全部 KV token。对 request `b`、query 位置 `i`、Q head `hq`，令 `hk=floor(hq/8)`，则：

```text
s_j = dot(Q[b,i,hq,:], K[b,j,hk,:]) / sqrt(D)
p_j = softmax(s)_j
O[b,i,hq,:] = sum_j p_j * V[b,j,hk,:]
```

当 Q 和 KV 都有长度 L 时，QK 和 PV 各需要一次乘法和一次加法，主干 FLOPs 为：

```text
F = 4 * B * L * L * Hq * D
```

页表读取、softmax、shared memory 搬运和同步没有计入这个有效 FLOPs，因此 TFLOPS 适合比较版本，但不等于纯 GEMM 峰值。

### 2.4 非因果模式仍然需要边界 mask

`causal=0` 只表示所有真实 KV token 都可见，不代表 tile 中越过 `kv_len` 的填充位置有效。L=257 时，最后一个 KV tile 会包含大量不存在的 token；这些 score 必须设为负无穷，不能进入 softmax。

早期 MMA 版本忘记了这层边界，在 L=257 上只有约 84% 元素匹配。恢复 `kv_idx < chunk_end` 后才得到正确结果。这个问题很容易被长度恰好对齐 16、32 或 64 的用例掩盖。

---

## 3. Baseline CUDA 代码分析

### 3.1 工作划分

Baseline 使用 128-thread CTA，即两个 64-lane warp。每个 warp 负责一个 `(request,q_pos,qo_head)` 输出行。任务总数按 `batch_size * seq_len * num_qo_heads` 展开，再由 `qo_indptr` 判断当前 q_pos 是否真实存在。

head dimension 最大为 256，所以每个 lane 准备 4 个 Q 元素和 4 个输出累加器：

```text
lane l -> l, l+64, l+128, l+192
```

D=128 时后两个位置被边界判断屏蔽，D=256 时四个位置全部有效。

### 3.2 每个 KV token 的执行过程

Baseline 对每个可见 token 顺序执行：先用 `kv_pos>>4` 和 `kv_pos&15` 求逻辑页及页内位置；再从 `kv_indices` 读取物理页；之后计算 K/V 指针，完成标量 QK、64-lane shuffle reduction、online softmax 和标量 PV。

```text
for kv_pos in [0, kv_len):
    logical_page = page_begin + (kv_pos >> 4)
    physical_page = kv_indices[logical_page]
    entry = kv_pos & 15
    k_ptr, v_ptr = address(physical_page, entry, kv_head)
    score = warp_reduce(dot(q, k)) / sqrt(D)
    online_softmax_update(score, v)
```

online softmax 保存当前最大值 `m`、分母 `d` 和输出累加器 `o`：

```text
m_new = max(m, score)
alpha = exp(m - m_new)
beta  = exp(score - m_new)
o     = alpha * o + beta * V
d     = alpha * d + beta
m     = m_new
```

最终输出 `o/d`。这条路径结构简单、结果精确，适合作为 correctness oracle，但它没有利用矩阵单元，也没有复用相邻 query 的 K/V tile。

### 3.3 Baseline 的主要瓶颈

每个可见 `(query,key)` 对都执行一轮标量 128/256 维 QK、shuffle reduction、softmax 更新和标量 PV。长序列的工作量按 L² 增长，页表解析又处在最内层循环。前 12 个基准点总延迟为 `15175.446 ms`，最长 `B=1,L=16384,D=128` 用例为 `8235.150 ms`，有效吞吐不到 1 TFLOPS。

这里的数量级瓶颈首先是标量 QK/PV，其次才是页表地址计算。正确的优化顺序是先让 MMA 承担主体矩阵计算，再处理 paged gather 的额外成本。

---

## 4. 最终实现的四条执行路径

当前 `run_kernel` 先检查固定的 heads、page size 和非因果模式，然后按 head dimension、序列长度及页表类型分派。

| 输入条件 | 最终路径 | 核心配置 |
|---|---|---|
| D128，L<4096 | 直接 paged | Q64 / W4 / KV32 |
| D128，L>=4096，随机/置换页表 | 直接 paged | Q128 / W4 / KV32 |
| D128，L>=4096，identity 页表 | unpack 后连续 attention | Q256 / W8 / KV64 |
| D256 | 直接 paged | Q64 / W4 / KV64 |

Q64、Q128 和 Q256 表示 packed `(query token,GQA head)` 行数。因为一个 KV head 对应 8 个 Q head，它们分别覆盖 8、16 和 32 个 query token。

```text
run_kernel
  ├─ D128, short ───────────────> paged Q64/KV32
  ├─ D128, long, permuted pages -> paged Q128/KV32
  ├─ D128, long, identity pages -> unpack K/V -> ragged Q256/KV64
  └─ D256 ──────────────────────> paged Q64/KV64
```

这种分派的原因不是代码风格，而是资源边界不同。D128 可以用 KV32 降低 shared footprint，并用寄存器 staging 隐藏 gather；D256 的 fragment 和寄存器已经更大，继续 staging 会造成过高压力，因此保留较稳妥的 KV64 shared 直载。长 D128 的 attention 是 O(L²)，可以用一次 O(L) 的 unpack 换取更高效的连续主干。

---

## 5. 直接 Paged MMA 路径

### 5.1 Q/K/V tile 和 MMA

直接 paged kernel 把多个 packed Q 行和一组 KV token 放进一个 CTA。Q 先从 global memory 写入 swizzled shared memory，再装配为 MMA fragment；K/V 则通过页表 gather 后进入 shared memory。QK 使用 BF16 `16x16x16` MMA，score 和 online softmax 状态使用 FP32，softmax 权重再通过 MMA 与 V 相乘。

Q64/W4 表示 CTA 有 4 个 64-lane warp，每个 warp 负责一个 16-row Q fragment。Q128/W4 时每个 warp 负责两个 Q fragment。KV32 表示 `NUM_MMA_KV=2`，即两个 16-token MMA tile。

### 5.2 页内除法变成移位

page size 固定为 16，因此通用 `divmod` 可以严格替换为：

```cpp
page_iter = packed >> 4;
entry_idx = packed & 15;
```

这项优化本身不改变内存访问次数，但它位于每个 tile、每个线程的地址生成路径，能减少整数除法和通用 fast-div 状态。

### 5.3 页表 warp 广播

KV64 恰好覆盖四个 16-token page。原 loader 的许多线程会重复执行：

```text
physical_page = kv_indices[logical_page]
```

优化后，每个 warp 只让 lane 0～3 读取最多四个物理页号，其他 lane 根据自己的相对 page 编号通过 64-lane shuffle 获得结果。地址公式变为：

```text
logical_page_base = packed_page_iter_base >> 4
loaded_physical_page = lane 0..3 load page table
physical_page = shuffle(loaded_physical_page,
                        page_iter - logical_page_base)
```

它保留了任意 `kv_indices` permutation 的正确性，却把大量重复页表 load 压缩为每 warp 少量读取。最长第一轮点由约 `69.595 ms` 降到 `62.799 ms`，说明页表寻址确实是 paged 主路径的重要成本。

把页号进一步放进 CTA shared memory 看似可以让 4 个 warp 共用一次加载，但实测 D128 明显回退，D256 还会因为额外 16 字节超过 64 KiB shared 上限而无法 launch。页表很小不代表 shared cache 一定划算；barrier、shared load 和更长寄存器生命周期同样有成本。

### 5.4 32 位 element offset

比赛最大可分配形状的 K/V element offset 小于 `2^32`，所以热路径中的 offset 数组可以从 `size_t` 缩为 `uint32_t`，只有最终指针加法时再扩展。这样减少 64 位整数寄存器和地址运算，最长第一轮点进一步降到约 `60.411 ms`。

这是一项题目范围内的特化。通用库如果允许超过 4G 个 BF16 element，就不能直接使用 32 位 offset。

### 5.5 KV32 loader 修复

第一轮直接把 KV64 改成 KV32 时结果只有约 7% 匹配，不是 KV32 数学上不可行，而是上游 paged V loader 的数组尺寸依赖：

```text
NUM_MMA_KV / NUM_WARPS_Q
```

KV32/W4 下是 `2/4=0`，V fragment 实际没有被正确加载。最终代码为 KV32 增加一条 128-bit V gather 路径，尺寸改为 `NUM_MMA_KV*2/NUM_WARPS_Q`，并配套 `compute_sfm_v_with_perm` 与 128-bit 输出写回。修复后 D128 shared footprint 降低，长点从约 `60.4 ms` 降到 `47.8 ms`。

这个例子说明，tile 搜索不能只改 traits。loader、fragment 数、shared layout、PV selector 和输出 store 必须一起支持新的 tile。

### 5.6 Register-staged paged pipeline

原 paged 路径按 `global gather -> shared -> MMA` 串行执行。D128 的最终路径先把下一个 KV32 tile 的 K/V gather 到寄存器，再在当前 tile 执行 logits transform、mask 和 online softmax 时发起后续 global load；使用前才把寄存器 fragment 写入 shared。

```text
当前 tile:    QK -> transform/mask -> softmax -> PV
下一 tile:         global K/V gather -> register -> shared
```

`page_load_k_r` 和 `page_load_v_r` 负责 global-to-register，`produce_k_w` 和 `produce_v_w` 负责 register-to-shared。predicated load 允许最后一轮预取越过真实长度，越界数据不会进入有效计算。

D128 的 `NUM_MMA_D=8`，staging fragment 尚可接受，实测获得约 4%～8% 收益。D256 的 fragment 翻倍，若继续 staging 会显著增加寄存器压力，因此代码用编译期分支保留 shared 直载路径。

### 5.7 Identity-page 模板特化

官方 benchmark 使用顺序页表时，`kv_indices[i]==i`。`get_cached_plan` 在首次预热时把较小的 page table 拷回 host，检查是否为 identity，并把结果保存在计划中。

identity paged kernel 通过模板参数 `IDENTITY_PAGES=true` 编译，直接令：

```text
physical_page = logical_page
```

热循环不再读取 `kv_indices`，也不再做 shuffle 广播。若页表不是 identity，则调用普通 paged kernel，完整执行真实 page gather。随机 permutation 和 partial last page 测试用于保证 fallback 没有被破坏。

当前 plan cache 面向“少量固定输入、充分预热”的评测模型。identity 标记只在首次建 plan 时计算，cache key 主要使用 Q 指针、Q indptr 和形状，并未把 page table 的内容版本纳入键。如果生产系统会在相同指针和形状下原地修改 `kv_indices`，就必须把 page-table 指针或版本号加入 cache key，并为首次初始化增加并发保护。

---

## 6. Identity 长序列的 Unpack 路径

### 6.1 为什么先搬一次数据反而更快

identity page 省掉了页表跳转，但 `kv_data` 仍采用 `[page,2,16,Hkv,D]`，K 和 V plane 在每页内交错。attention 主循环每次仍需计算 page/entry 和跨页地址。

对长序列而言，unpack 的数据搬运量是 O(L)，attention 的 QK/PV 是 O(L²)。L 足够大时，先用带宽型 kernel 将 K/V 转成连续 NHD，只占总时间很小一部分，却能让后续所有 QK/PV tile 使用更简单、更高吞吐的连续 loader。

```text
paged identity KV
    -> unpack_identity_pages
    -> packed_k [B*L, Hkv, D]
    -> packed_v [B*L, Hkv, D]
    -> contiguous ragged MMA kernel
```

每次调用都会重新 unpack 当前 K/V，不缓存输入内容或输出，因此结果仍对应本次传入的数据。

### 6.2 Unpack kernel

`unpack_identity_pages` 以 16 字节 `uint4` 为搬运单位。对于连续 token `token`：

```text
page  = token >> 4
entry = token & 15
src_k = page * page_elems + entry * Hkv * D + vector_offset
src_v = src_k + 16 * Hkv * D
```

K 和 V 分别写入两块连续 scratch。`get_unpack_scratch` 按见过的最大 token 数一次性扩容，后续调用复用，避免在计时热路径反复分配。

这些 scratch 是进程级静态缓冲区，适合当前单流 benchmark，但不是完整的生产内存管理。多线程并发、多个 CUDA stream 或反复扩容时，应由上层显式提供 workspace，并负责旧缓冲区回收和生命周期同步。

### 6.3 从 Q128 到 Q256

初始 unpack 路径使用连续 Q128/KV32 主干，最长点约 `41.49 ms`。继续放大 Q tile 时，Q128/KV64 曾出现约 `32.68 ms` 的错误低时延，原因不是硬件突然更快，而是原 ctk64 路径只支持单个 Q fragment。

为了让 Q256/W8/KV64 正确工作，最终修复了三处单-fragment 假设：

1. `load_q_global_smem_64b` 为每个 `mma_q` 增加独立的 16-row shared 偏移，避免后一个 fragment 覆盖前一个；
2. `load_q_smem_reg_64b` 遍历所有 `mma_q`，并在每个 dimension 块后正确复位 shared 列偏移；
3. ctk64 `compute_qk` 遍历全部 `q_frag[mma_q]` 和 `s_frag[mma_q]`，不再写死 fragment 0。

最终配置为：

```text
CTA_TILE_Q  = 256 packed rows = 32 query tokens * GQA 8
NUM_WARPS_Q = 8
NUM_MMA_Q   = 2 per warp
CTA_TILE_KV = 64
NUM_MMA_KV  = 4
```

修复后的 Q256 路径在 `B=1,L=16384,D=128` 上约为 `38.171 ms`，比第一轮直接 paged 的 `60.407 ms` 下降约 36.8%。

按完整 QK/PV 工作量计算，该点的有效吞吐约为 `115.3 TFLOPS`。

### 6.4 Q256/KV128 的边界

Q256/W16/KV128 看起来能进一步扩大复用，但 shared footprint 已经到达 64 KiB 边界，现有 V selector/layout 又不支持该组合。测试出现错误结果，不能采用。最终 Q256/W8/KV64 是当前实现中经过完整修复并通过验证的最大合法组合。

---

## 7. 从 Baseline 到最终版本的迭代过程

### 阶段 1：精确 SIMT baseline

一个 warp 计算一行输出，每个 KV token 都解析页表、标量计算 QK、更新 FP32 online softmax并标量累加 PV。它建立了正确性锚点，也暴露了标量计算和逐 token 页表寻址两大问题。

### 阶段 2：迁移 BF16 MMA 主干

复用 xcore1000 FlashAttention 框架，建立 Q64/KV64、4 Q warp、1 KV warp 的 paged kernel，并缓存调度表。最长点从 8235 ms 降到约 71 ms。L=257 的错误进一步补齐了非因果尾 tile mask。

### 阶段 3：固定页大小和直接 launch

page size 16 允许用 shift/mask 替代通用 divmod；唯一正确的 traits 固定后，shared-memory attribute 只设置一次，并跳过每次 dispatcher/device 查询。长点小幅降到约 69.6 ms，短点受益更明显。

### 阶段 4：优化页表和地址生成

使用 warp lane 0～3 加载物理页号，再通过 shuffle 广播；offset 数组缩为 32 位。最长点依次降到约 62.8 和 60.4 ms。page-base 预乘、固定 stride 和 CTA shared 页缓存均未获益，说明编译器地址生成与 shared 同步之间存在细致权衡。

### 阶段 5：补齐 KV32 paged loader

定位并修复 paged V loader 的零长度 fragment 问题，增加 128-bit gather、PV permute 和写回路径。D128 shared footprint下降，长点约为 47.8 ms。Q128/KV32 正确但不总是更快，因此只用于长 D128 的非 identity fallback。

### 阶段 6：加入寄存器 staging 和 identity 快路径

D128 下一 KV32 tile 的 K/V 先 gather 到寄存器，与当前 tile 的 MMA/softmax 重叠；identity page 在首次预热时检测并编译期消除页表读取。直接 paged identity 长点进一步降到约 44 ms。

### 阶段 7：Unpack 后调用连续主干

对 `D=128,L>=4096` 的 identity page，每次先将交错 paged K/V 解包为连续 K/V。正确的连续 Q128/KV32 路径约为 41.5 ms，证明 O(L) unpack 能被 O(L²) attention 收益覆盖。

### 阶段 8：修复 Q256 ctk64

补齐 Q global-to-shared、shared-to-register 和 QK 三处多 Q fragment 循环，最终采用 Q256/W8/KV64。最长点约 38.17 ms。更大的 KV128 或不同 igroup 策略要么布局错误，要么小幅回退，因此停止继续扩张 tile。

| 数字阶段 | 核心变化 | B1 L16384 D128 | 决策 |
|---:|---|---:|---|
| 1 | 标量 warp-per-output | `8235.150 ms` | baseline |
| 2 | Q64/KV64 BF16 MMA | `71.316 ms` | 建立主干 |
| 3 | 固定 page 16、直接 launch | `69.595 ms` | 保留 |
| 4 | 页表广播、32 位 offset | `60.411 ms` | 保留 |
| 5 | 正确 KV32 paged loader | 约 `47.85 ms` | 保留 |
| 6 | register staging、identity 特化 | 约 `44.18 ms` | 保留 |
| 7 | unpack + 连续 Q128/KV32 | 约 `41.49 ms` | 过渡版本 |
| 8 | unpack + Q256/W8/KV64 | 约 `38.17 ms` | 当前最终版本 |

---

## 8. 关键失败实验及其意义

| 实验 | 观察 | 得出的结论 |
|---|---|---|
| 直接将 Q64 改 Q128 | 只有约 10% 匹配 | 多 Q fragment 的 loader/地址未同步扩展 |
| 第一版 KV32 | 只有约 7% 匹配 | 上游 paged V fragment 数退化为 0 |
| D128 改成 2 Q warp | 正确但长点回退约 27% | 每 warp 工作过重，寄存器和并行度失衡 |
| CTA shared 页号缓存 | D128 回退，D256 无法 launch | 少量页表 load 不足以覆盖 barrier 和 shared 成本 |
| 预乘 physical page base | 比 32 位 offset 版本慢约 0.57% | 更少乘法不一定产生更好的后端指令调度 |
| 固定 stride | 正确但回退 | 编译期常量不保证寄存器分配更优 |
| KV128 paged | shared 达 66560 B，超过 65536 B | 硬件 shared 上限直接否决该配置 |
| Q128/KV64 错误快版本 | 约 32.68 ms 但输出错误 | 漏算 Q fragment 会制造虚假吞吐 |
| Q256/W16/KV128 | 低时延但 selector/layout 错误 | 达到 shared 边界不代表软件布局合法 |
| igroup strategy 0/1 | 正确但均比默认略慢 | 后端调度策略必须以实测为准 |

这些实验共同说明，paged attention 的 tile 参数与地址生成、V fragment 布局、shared swizzle、寄存器压力和硬件容量紧密耦合。任何异常快的结果都必须先检查是否完整加载了所有 Q/K/V fragment。

---

## 9. 正确性与性能如何验证

### 9.1 随机物理页置换

测试器默认不会只使用顺序页表，而是从更多物理页中随机选择 permutation。这样能直接发现“忽略 `kv_indices`”或“把逻辑页当物理页”的错误。identity 快路径之外的所有版本都必须通过随机 permutation。

### 9.2 Partial last page

L=257 会创建 17 个 page，最后一页只有 1 个有效 token。该用例同时检查 `last_page_len`、非因果尾 tile mask 和页内 entry 计算。只测 1024、4096 这类对齐长度无法覆盖这个错误。

### 9.3 两种 head dimension

D128 和 D256 不只是循环次数不同，它们会改变 MMA fragment、shared footprint 和寄存器压力。D128 的 KV32/register-staged 路径不能直接假设适合 D256；最终 D256 明确保留 Q64/KV64 paged 路径。

### 9.4 Identity 与 fallback 分开验证

最终版本针对 identity page 做了编译期专化和长序列 unpack，因此必须分别测试：顺序页表的快路径、随机页表的精确 fallback、partial last page，以及 D256 fallback。总结中记录的最终回归为 identity 12/12 通过，随机页/尾页/D256 代表 fallback 4/4 通过，最大绝对误差在 `4.88e-4` 量级。

### 9.5 性能数字的解释

第一轮直接 paged 版本曾在 20 点上比本地 FlashInfer reference 总时间低约 15%，但线上 reference 的实现明显更快，因此本地 reference 不能视为硬件上限。第二轮的优化重点由此从少量页表整数指令转向 KV32 residency、gather 流水和长序列 unpack。教程中的性能结论应优先看同一环境下相邻版本的延迟比，而不是跨环境比较绝对值。

---

## 10. 最终源码阅读地图

`paged_prefill_optimized.cu` 约 500 KiB，包含内联的 FlashInfer/xcore1000 依赖。建议从文件底部向上阅读：

1. `run_kernel`：查看 D128/D256 和长度分派；
2. `launch_paged`：查看 identity 检测后的 paged/unpack 选择，以及 `BatchPrefillPagedParams` 的 stride 设置；
3. `get_cached_plan`：查看精确 Q task 构造、identity page 检测和计划缓存；
4. `unpack_identity_pages`、`get_unpack_scratch`、`launch_unpacked_ragged`：查看长序列连续化路径；
5. `batch_prefill_with_paged_kv_cache_kernel_xc1000`：查看直接 paged 主循环；
6. `page_load_k_r`、`page_load_v_r`、`page_produce_k/v`：查看 KV32 register-staged loader；
7. `batch_prefill_with_ragged_kv_cache_kernel_xc1000_ctk64_official`：查看 Q256/KV64 连续主干；
8. `load_q_global_smem_64b`、`load_q_smem_reg_64b`、ctk64 `compute_qk`：查看多 Q fragment 修复。

最终热路径可以压缩成下面的伪代码：

```cpp
run_kernel(args) {
    validate_fixed_shape();
    if (D == 128 && L < 4096)
        return paged_q64_kv32(args);

    if (D == 128) {
        plan = get_cached_plan_and_detect_identity(args);
        if (plan.identity_pages) {
            unpack_paged_kv_to_contiguous(args.kv_data);
            return ragged_q256_kv64(args.q, packed_k, packed_v);
        }
        return paged_q128_kv32(args);
    }

    return paged_q64_kv64_d256(args);
}

paged_attention_cta(params) {
    load_q_to_mma_fragments();
    for (each KV tile) {
        map_logical_pages_to_physical_pages();
        gather_next_kv_to_registers();
        qk_mma_current_tile();
        apply_partial_tile_mask();
        online_softmax_update();
        pv_mma_current_tile();
        store_next_kv_to_shared();
    }
    normalize_and_store_output();
}
```

---

## 11. 常见实现错误

- 把 `kv_indptr` 的页数当作 token 数，忽略 `last_page_len`；
- 直接用 logical page 访问 `kv_data`，忽略 `kv_indices`；
- 把 `[page,2,16,Hkv,D]` 误当成两个完全连续的大 K/V 张量；
- 因为 `causal=0` 就删除最后一个 partial KV tile 的边界 mask；
- 修改 KV tile 后没有同步修改 V loader、PV selector 和输出 store；
- 修改 Q tile/warp 后只计算 `q_frag[0]`，漏掉其余 MMA-Q fragment；
- 用 identity-page 快路径处理随机 permutation；
- 将 D128 的 register staging 无条件套到 D256，导致寄存器或 shared 超限；
- 只比较本地 reference，不检查同进程前后版本和实际页表路径；
- 把错误版本的低时延当作性能上限。

---

## 12. 复现与继续实验

编译最终源码：

```bash
mxcc -O3 -std=c++17 --offload-arch=xcore1000 \
  -I/opt/maca/tools/cu-bridge/include -shared -fPIC \
  batch_prefill_paged_code/paged_prefill_optimized.cu \
  -o batch_prefill_paged_code/paged_prefill_optimized.so
```

运行随机页表回归：

```bash
python batch_prefill_paged_code/benchmark_paged.py \
  --source batch_prefill_paged_code/paged_prefill_optimized.cu \
  --library batch_prefill_paged_code/paged_prefill_optimized.so \
  --cases 1,2,4,8,10,12 \
  --output batch_prefill_paged_code/paged_random_regression.csv \
  --max-repeats 3 --force-build
```

运行 identity-page 快路径：

```bash
python batch_prefill_paged_code/benchmark_paged.py \
  --source batch_prefill_paged_code/paged_prefill_optimized.cu \
  --library batch_prefill_paged_code/paged_prefill_optimized.so \
  --cases 2,4,7,12,14,15 \
  --identity-pages \
  --output batch_prefill_paged_code/paged_identity_regression.csv \
  --max-repeats 3
```

测试新配置时，应先跑 L=257 的 partial page、一个随机 permutation、一个 D256 用例，再跑长 D128 identity 性能。这样能尽早发现页表、V loader、shared 上限和 fragment 漏算问题。

---

## 13. 总结

Paged Prefill 的核心难点不在 softmax 公式，而在“如何让矩阵单元高吞吐地消费分页 K/V”。Baseline 每个 token 都做页表解析和标量计算；第一轮 MMA 解决了计算量级问题；页表 warp 广播、32 位 offset 和 KV32 loader继续压缩 paged gather；register staging 把下一 tile 的随机读取与当前 tile 的 MMA/softmax 重叠；identity 检测则让官方顺序页表完全绕过热循环页表访问。对于长 D128，O(L) unpack 的成本远小于 O(L²) attention，最终可以使用修复后的 Q256/W8/KV64 连续主干。

最终实现保留了严格 fallback：随机物理页、partial last page 和 D256 仍走真实 paged loader。它没有用输入或输出缓存换性能，而是根据页表性质和数据规模选择更合适的数据通路。整个优化过程最重要的经验是：先区分矩阵计算、页表寻址、shared residency 和流水停顿各自占多少，再选择对应手段；只改 tile 数字或只减少一次地址运算，往往无法解决真正的瓶颈。
