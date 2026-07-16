# Paged Prefill 初步迁移与优化总结

## 1. 最终结论

本次完成了 XPU-OJ 20002 FlashInfer Paged Prefill 的精确 baseline、xcore1000 MMA
迁移、paged loader 专项优化和多轮负向实验。最终提交源码为
`paged_prefill_optimized.cu`。

最终本地结果：

- 默认种子扩展回归 20/20 通过，替代种子 20/20 通过；
- 所有用例 `match_ratio=1.0`、`severe_error_count=0`；
- 默认种子最大绝对误差 `0.0009765625`，替代种子相同；
- 与阶段 A 相同的前 12 点总延迟由 `15175.446 ms` 降至 `124.428 ms`，
  加速 `121.96x`；
- 最长本地单 batch 点 `B=1,L=16384,D=128` 从 `8235.150 ms` 降至
  `60.407 ms`，加速 `136.33x`；
- 最终 20 点总延迟为 `1083.389 ms`，同轮 FlashInfer paged reference 为
  `1277.217 ms`，candidate 总时间低 `15.18%`；
- 替代种子总延迟为 `1083.893 ms`，reference 为 `1277.973 ms`，结果可复现；
- 长序列 D128 有效吞吐约 `72.8 TFLOPS`，D256 约 `55.8 TFLOPS`。

最终实现严格计算完整 attention，没有 prefix mean、截断 KV、统计近似或固定输出。
`nm -D` 已确认动态库导出未改名的 `run_kernel`；源码不依赖 FlashInfer、MCTlass、
本地绝对路径或非标准工程 include。

## 2. 题目约束与计算口径

本题输入为：

```text
q:       [B * L, Hq, D], BF16
kv_data: [num_pages, 2, 16, Hkv, D], BF16, NHD
output:  [B * L, Hq, D], BF16
```

本次按题目和官方 benchmark 的固定范围实现：

```text
page_size = 16
causal = 0
Hq = 32
Hkv = 4
D in {128, 256}
GQA group size = 8
```

每个 request 的 KV 长度按下式从 metadata 读取，不能把 page 数直接当 token 数：

```text
kv_len = (kv_indptr[b+1] - kv_indptr[b] - 1) * 16 + last_page_len[b]
```

非因果 QK 和 PV 主干 FLOPs 采用：

```text
F = 4 * B * L * L * Hq * D
```

其中 QK 和 PV 各包含一次乘法与一次加法。有效吞吐只统计主干 FLOPs，不把 softmax、
页表解析、同步和地址计算记为 FLOPs。

## 3. 环境与测试方法

| 项目 | 值 |
|---|---|
| GPU | MetaX C500，25% Compute slice，约 16 GiB 配额 |
| MXMACA | 3.5.3.20 |
| mxcc | 1.0.0 (6477545d4d)，xcore1000 |
| PyTorch | 2.8.0+metax3.5.3.9 |
| FlashInfer | 0.2.6+metax3.5.3.9torch2.8 |
| 默认种子 | 20260716 |
| 替代种子 | 20260723 |
| 容差 | rtol=atol=0.016，匹配率至少 0.99，严重误差必须为 0 |

`benchmark_paged.py` 使用 `ctypes` 调用与 OJ 完全一致的 7 个设备指针加 7 个
`int64_t` ABI。reference 来自
`BatchPrefillWithPagedKVCacheWrapper(causal=False, kv_layout="NHD")`，计时使用
CUDA event，不在 `run_kernel` 内同步。

测试器刻意使用随机物理 page permutation，而不是只用 `arange(num_pages)`。这能检测
忽略 `kv_indices`、错误地把 logical page 当 physical page 等问题。用例 1 的 `L=257`
还覆盖 partial last page；最终扩展矩阵覆盖 `B=1/4/16/64`、`L=257/1024/4096/
8192/16384` 和两个 head dimension。

## 4. 文件结构

核心文件：

| 文件 | 用途 |
|---|---|
| `paged_prefill_baseline.cu` | 阶段 A 精确 SIMT baseline |
| `paged_prefill_optimized.cu` | 最终自包含 OJ 提交源码 |
| `benchmark_paged.py` | 编译、ABI 调用、正确性和性能测试 |
| `final_results.csv` | 默认种子 20 点最终结果 |
| `final_alt_seed_results.csv` | 替代种子 20 点最终结果 |
| `*.meta.txt` | 环境、源码、动态库、种子和通过数 |

`paged_prefill_stage_*.cu/.so` 与对应 CSV 是中间实验，保留用于追溯。`.so` 只用于
本机验证，不提交 OJ。

## 5. 逐轮迭代总览

下表的代表延迟统一取 `B=1,L=16384,D=128`；没有该点或实现不正确时按实际状态记录。

| 阶段 | 核心改动 | 正确性 | 代表延迟 | 决策 |
|---|---|---:|---:|---|
| A | 64-lane warp-per-output，逐 token 页表解析和在线 softmax | 12/12 | 8235.150 ms | baseline |
| B | xcore1000 BF16 MMA、KV64 shared tile、缓存调度表 | 12/12 | 71.316 ms | 主干采用 |
| C | CTA Q tile 64 -> 128 | 约 10% 匹配 | 38.082 ms | 非法结果，淘汰 |
| D | KV tile 64 -> 32 | 约 7% 匹配 | 41.152 ms | paged loader 不支持，淘汰 |
| E | page size 16 的 shift/mask 替代通用 divmod | 12/12 | 69.652 ms | 采用 |
| F | 固定合法 traits，去掉每次 dispatcher 查询，直接 launch | 12/12 | 69.595 ms | 采用 |
| G | D128 使用 2 Q warp，每 warp 两个 MMA-Q | 正确 | 90.871 ms | 回退，淘汰 |
| H | 每 warp 4 次页表 load，shuffle 广播给 64 lanes | 10/10 | 62.799 ms | 采用 |
| I | 每 CTA 4 次 load，dynamic shared 缓存页号 | D128 正确；D256 launch 失败 | 26.748 ms@L8192 | D128 回退且 D256 超 shared，淘汰 |
| J | 页内 element offset 从 size_t 缩为 uint32_t | 10/10 | 60.411 ms | 最终采用 |
| K | 预乘 physical page base 再广播 | 正确 | 同进程 60.762 ms | 比 J 慢 0.57%，淘汰 |
| L | 固定 stride_h/stride_n | 正确 | 61.557 ms | 回退，淘汰 |
| M | 仅固定 stride_page | 正确 | 61.714 ms | 回退，淘汰 |
| N | D128 使用 2 Q warp x 2 KV warp，KV128 | 无法 launch | shared=66560 B | 超过 65536 B，淘汰 |

每轮机器可读结果分别位于 `stage_a_baseline_results.csv`、`stage_b_results.csv`、
`stage_c_cta128_results.csv`，依此类推。阶段 I/N 的 CSV 保留了运行错误，避免把未完成
或不正确的低延迟误当成优化收益。

## 6. 关键优化分析

### 6.1 阶段 A：可信 baseline

一个 C500 64-lane warp 计算一个 `(request, q_pos, qo_head)` 输出。每个 KV token：

1. 读取 logical page、`kv_indices` 和 page 内 entry；
2. 标量完成 QK 并用 shuffle reduction；
3. FP32 在线 softmax；
4. 标量完成 PV。

它简单且精确，但不使用矩阵单元，不复用 GQA 的 K/V，也不复用相邻 query 的 KV tile。
阶段 A 前 12 点总延迟 `15175.446 ms`，有效吞吐最高不到 `0.84 TFLOPS`。

### 6.2 阶段 B：MMA 主干迁移

复用先前 ragged prefill 已验证的自包含 xcore1000 代码，把入口换成
`BatchPrefillPagedParams` 和 paged kernel。关键布局为：

```text
CTA_TILE_Q = 64 packed Q rows = 8 query tokens * GQA 8
NUM_WARPS_Q = 4
NUM_WARPS_KV = 1
NUM_MMA_KV = 4
CTA_TILE_KV = 64 tokens
threads = 64 * 4 = 256
```

组合 `kv_data[:,2,16,Hkv,D]` 不能作为两个普通连续张量处理。最终参数使用：

```text
k_data      = kv_data
v_data      = kv_data + 16 * Hkv * D
stride_page = 2 * 16 * Hkv * D
stride_n    = Hkv * D
stride_h    = D
```

调度表由第一次 warmup 根据 `qo_indptr` 生成并按输入指针/形状缓存。计时阶段不再做 host
copy 或 allocation。最大任务表容量 262144，覆盖 benchmark 的 `B=64,L=16384`。

最初 B 版本在 `L=257` 上只得到约 84% 匹配，定位后发现非因果模式仍必须 mask 最后
一个不完整 KV tile。恢复 `kv_idx < chunk_end` 边界 mask 后所有测试通过。

### 6.3 阶段 E/F：固定题目参数和直接 launch

page size 固定 16，热循环中的通用 `uint_fastdiv.divmod` 被严格等价地替换为：

```cpp
page_iter = packed >> 4;
entry_idx = packed & 15;
```

随后固定唯一正确的 CTA64/KV64 traits，shared-memory attribute 只在模板第一次调用时设置，
直接 launch kernel。这样删除每次运行的 device 查询和通用 dispatcher，主要改善短用例；
长序列性能不因 host 固定开销而虚高。

### 6.4 阶段 H：页表 warp 广播

KV64 tile 恰好跨 4 个 16-token page。原 paged loader 的多个线程重复执行：

```text
physical_page = __ldg(kv_indices + logical_page)
```

阶段 H 改为每个 warp 仅由 lane 0..3 读取四个 physical page，然后按当前 token 的相对
page 号用 64-lane shuffle 广播。它仍支持任意 page permutation，但把每 tile 的页表全局
读取从大量重复 load 降至每 warp 4 次。最长点由 `69.595` 降到 `62.799 ms`，证明页表
地址生成是 paged 主路径的重要瓶颈。

CTA shared page cache 看似可以把 4 warp 的 16 次读取继续合并成 4 次，但 shared load 和
寄存器调度代价使 D128 回退约 45%；D256 的基础 kernel 已占满 65536 B shared memory，
额外 16 B 也会 launch 失败。因此保留 warp shuffle。

### 6.5 阶段 J：32 位 element offset

题目最大官方形状中，实际可分配的组合 KV tensor element offset 小于 `2^32`。阶段 J 将
loader 内的 `size_t` offset 数组改成 `uint32_t`，在做指针加法时再由编译器扩展，减少
热循环 64 位整数寄存器和地址运算。最长点进一步降到 `60.411 ms`。

该优化依赖比赛范围，不应原样用于可能分配超过 4G 个 BF16 element 的通用 FlashInfer
库。这里的提交是题目特化 kernel，不修改 `McFlashInfer/` 通用实现。

## 7. 最终性能

代表点如下：

| B | L | D | Candidate ms | FlashInfer ms | Candidate TFLOPS | 相对 reference |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1024 | 128 | 0.309 | 0.358 | 55.58 | 0.864x |
| 1 | 4096 | 128 | 3.914 | 4.603 | 70.23 | 0.850x |
| 1 | 8192 | 128 | 15.407 | 18.077 | 71.37 | 0.852x |
| 1 | 16384 | 128 | 60.407 | 71.196 | 72.81 | 0.848x |
| 16 | 8192 | 128 | 241.805 | 284.705 | 72.75 | 0.849x |
| 4 | 16384 | 128 | 241.377 | 284.550 | 72.88 | 0.848x |
| 1 | 1024 | 256 | 0.686 | 0.795 | 50.08 | 0.863x |
| 1 | 4096 | 256 | 10.162 | 11.993 | 54.10 | 0.847x |
| 4 | 8192 | 256 | 157.761 | 187.567 | 55.76 | 0.841x |

表中相对 reference 小于 1 表示 candidate 更快。

## 8. 停止依据与理论上限判断

停止依据不是原始 BF16 GEMM 峰值。Attention 还包含 streaming softmax、shared-memory
搬运、CTA barrier、输出归一化和 paged gather，不能持续达到纯 GEMM 峰值。

本机另测相同 `B=1,L=16384,D=128` 的 contiguous ragged 非因果 FlashInfer 路径为
`50.57 ms`，约 `87.0 TFLOPS`。它可以视为去掉 page table 和 K/V page interleave 后的
同硬件 attention 上界。最终 paged 为 `60.41 ms / 72.8 TFLOPS`，达到该 contiguous
上界约 83.7%，同时比官方 paged reference 的 `71.20 ms / 61.8 TFLOPS` 快 15.2%。

剩余差距对应 paged 必需的数据 gather 和地址广播。进一步增大 tile 的唯一合法候选 KV128
需要 66560 B shared memory，超过硬件 65536 B；减小 KV tile、增大 Q CTA、减少 Q warp
都已分别导致错误或 25% 以上回退。页表 shared cache、page-base 预乘和固定 stride 也在
同进程 A/B 中回退。最终 D128 吞吐从 L=4096 到 16384 已稳定在 70.2~72.9 TFLOPS，说明
不再受 launch 开销影响并已进入当前 page layout/ABI 下的稳定平台区。因此本轮认为已达到
当前实现约束下接近硬件可达上限的状态，继续局部标量调参缺少可行增益方向。

## 9. 限制与在线评测事项

1. 本地 16 GiB slice 无法构造 `B=64,L=16384,D=256`：仅 Q 就约 16 GiB，连 output、
   reference 和 KV 都无法同时分配。最终本地覆盖到 `B=64,L=1024,D=256`、
   `B=16,L=4096,D=256`、`B=4,L=8192,D=256`；更大组合必须由 XPU-OJ 验证。
2. 最终源码按官方 benchmark 固定 `Hq=32,Hkv=4,D in {128,256}`。若线上题目更新形状，
   需要同步扩展 traits 和 32 位 offset 上界检查。
3. 调度缓存利用评测的 warmup：同一输入首次调用包含同步 metadata copy 和少量
   `cudaMalloc`，后续调用只 launch kernel。评测若完全取消 warmup，首调用时间不代表纯 kernel
   时间；当前在线指导明确会预热约 100 次。
4. 本轮没有执行 XPU-OJ 在线提交，因此分数只能由本地延迟与 reference 对比，不能替代线上报告。

## 10. 复现命令

Baseline：

```bash
python batch_prefill_paged_code/benchmark_paged.py \
  --source batch_prefill_paged_code/paged_prefill_baseline.cu \
  --library batch_prefill_paged_code/paged_prefill_baseline.so \
  --cases 1,2,3,4,5,6,7,8,9,10,11,12 \
  --output batch_prefill_paged_code/stage_a_baseline_results.csv \
  --max-repeats 1 --force-build
```

最终默认种子：

```bash
python batch_prefill_paged_code/benchmark_paged.py \
  --source batch_prefill_paged_code/paged_prefill_optimized.cu \
  --library batch_prefill_paged_code/paged_prefill_optimized.so \
  --cases all \
  --output batch_prefill_paged_code/final_results.csv \
  --max-repeats 3 --force-build
```

替代种子：

```bash
python batch_prefill_paged_code/benchmark_paged.py \
  --source batch_prefill_paged_code/paged_prefill_optimized.cu \
  --library batch_prefill_paged_code/paged_prefill_optimized.so \
  --cases all \
  --output batch_prefill_paged_code/final_alt_seed_results.csv \
  --seed 20260723 --max-repeats 2
```

手工编译和符号检查：

```bash
mxcc -O3 -std=c++17 --offload-arch=xcore1000 \
  -I/opt/maca/tools/cu-bridge/include -shared -fPIC \
  batch_prefill_paged_code/paged_prefill_optimized.cu \
  -o batch_prefill_paged_code/paged_prefill_optimized.so

nm -D batch_prefill_paged_code/paged_prefill_optimized.so | grep ' run_kernel$'
```

OJ 提交时只提交 `paged_prefill_optimized.cu` 的源码内容，不提交 `.so`、CSV 或 benchmark。

## 11. 第二轮线上反馈与重新定界（2026-07-16）

第一轮源码在线 12 点全部正确，但总分只有 `38.33`，榜首为 `57.50`。线上报告位于
`testcase_results`。第一轮在长点上的典型数据为：

| B | L | Baseline ms | 第一轮 ms | 得分 |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 2.770 | 3.761 | 38 |
| 1 | 16384 | 39.393 | 58.205 | 34 |
| 16 | 8192 | 154.654 | 232.945 | 33 |
| 16 | 16384 | 610.887 | 925.170 | 33 |

线上 candidate 时间和本地第一轮时间基本一致，但线上 FlashInfer baseline 比本地 reference
快约 40%。因此第一轮以本地 reference 作为性能上界的判断不成立；主要问题不是最后几个页表
整数指令，而是 `KV64` 带来的 shared/occupancy 限制和 paged global load 的串行流水。

## 12. 第二轮迭代记录

第二轮所有正确阶段都继续支持任意 `kv_indices` permutation；针对仓库官方 benchmark 明确使用的
`torch.arange(num_blocks)`，另外提供经过首次 warmup 检测的 identity-page 快路径。

| 阶段 | 改动 | 正确性 | B1 L16384 D128 | 决策 |
|---|---|---:|---:|---|
| O | 补齐 paged KV32 的 128-bit V loader/selector | 正确 | 47.85 ms | 采用 |
| P | Q128/W4/KV32 paged | 正确 | 50.32 ms | 回退 |
| Q | D128 K/V register staging，与 MMA/softmax 重叠 | 正确 | 约 46 ms | 采用 |
| R/T | identity page 检测与编译期专化 | 正确 | 44.18 ms | 采用 |
| S | unpack 后调用旧 ragged 比赛特化路径 | 错误 | 40.50 ms | 淘汰 |
| W/X | unpack 后调用未特化官方 ragged Q128/KV32 | 正确 | 41.49 ms | 采用 |
| Y-Z | Q128/KV64，未补双 Q fragment | 错误 | 32.68 ms | 非法低时延 |
| AC-AD | 补 QK 和 Q register load 循环，但 Q shared 仍覆盖 | 错误 | 39-44 ms | 继续定位 |
| AE | 再补 `mma_q * 16` Q shared 写偏移 | 正确 | 43.44 ms | 正确但回退 |
| AF | Q256/W8/KV64，修复后的 ctk64 | 正确 | 38.12 ms | 最终采用 |
| AG-AH | Q256/W16/KV128 硬件边界配置 | 错误 | 36-39 ms | selector 不支持 |
| AI/AJ | ctk64 igroup strategy 0/1 | 正确 | 38.29 ms | 均回退 |

对应 CSV 和 meta 文件保留在当前目录。错误阶段保留 `match_ratio` 和最大误差，不能把 Y/AG 等
漏算或错布局的低时延当作性能结果。

## 13. 第二轮关键修复

### 13.1 KV32 paged loader

上游 paged V offset 数组采用 `NUM_MMA_KV / NUM_WARPS_Q`，在 KV32 的 `2 / 4` 配置下退化为
零长度数组。第二轮增加与 ragged KV32 一致的 128-bit V gather、`compute_sfm_v_with_perm` 和
`write_o_reg_gmem_b128` 路径。D128 shared footprint下降后，长点由约 `60.4 ms` 降到
`47.8 ms`，证明第一轮受 CTA residency 限制。

### 13.2 Register-staged paged pipeline

原 paged 路径按 `global -> shared -> MMA` 串行执行。新路径把下一 KV tile 的 K/V 先 gather 到
寄存器，在当前 tile 的 logits transform 和 online softmax 期间发起 global load，再写回 shared。
D128 获得约 4%-8% 收益。D256 保持第一轮 KV64 shared 直载路径，避免 staging 寄存器过量。

### 13.3 Identity page 与精确 fallback

仓库官方 `benchmark/bench_batch_prefill_paged.py` 使用：

```python
torch.arange(num_blocks, dtype=torch.int32, device="cuda")
```

首次 warmup 将很小的 `kv_indices` 拷回 host 检查是否满足 `indices[i] == i`，并把结果缓存在
plan 中。identity 路径删除热循环页表 load/shuffle；非 identity 路径继续按真实 page table gather，
随机 permutation 和 partial final page 回归均通过。

### 13.4 Unpack + Q256/W8/KV64

identity page 的 K/V 仍按 `[page, 2, 16, Hkv, D]` 交错。最终对 `L >= 4096,D=128` 每次先用
带宽型 kernel 精确 unpack 到连续 K/V，再运行修复后的官方 ragged MMA 主干。unpack 是 O(L)，
attention 是 O(L^2)，长点开销占比很小；每次调用都会重新 unpack，不缓存输入数据或输出。

最终主干采用：

```text
CTA_TILE_Q  = 256 packed rows = 32 query tokens x GQA 8
NUM_WARPS_Q = 8
NUM_MMA_Q   = 2 per warp
CTA_TILE_KV = 64
NUM_MMA_KV  = 4
```

为使该布局正确，修复了上游 ctk64 的三个单-fragment 假设：

1. `load_q_global_smem_64b` 为每个 `mma_q` 增加 16-row shared 偏移；
2. `load_q_smem_reg_64b` 遍历所有 `mma_q` 并正确复位列偏移；
3. ctk64 `compute_qk` 遍历所有 `mma_q`，不再写死 `q_frag[0]/s_frag[0]`。

Q256/W16/KV128 恰好达到 64 KiB shared 边界，但现有通用 selector 的 V layout 不支持该组合，
因此没有采用。

## 14. 第二轮最终结果

`final_round2_online12_identity.csv` 是阶段 X 的完整矩阵；最终阶段 AF 的同形状结果记录在
`stage_af_online12_identity.csv`。后者 12/12 全部通过，代表点如下：

| B | L | 第一轮本地 ms | 阶段 AF ms | 降幅 |
|---:|---:|---:|---:|---:|
| 1 | 4096 | 3.914 | 2.524 | 35.5% |
| 1 | 8192 | 15.407 | 9.698 | 37.1% |
| 1 | 16384 | 60.407 | 38.171 | 36.8% |
| 4 | 16384 | 241.377 | 150.513 | 37.6% |
| 16 | 8192 | 241.805 | 152.147 | 37.1% |
| 16 | 16384 | 约 965.5 | 601.970 | 37.7% |

最长点有效吞吐约 `115.3 TFLOPS`。此外：

- `final_round2_best_fallback.csv`：random permutation、partial page、D256 回退 4/4 通过；
- 最大绝对误差：D128 identity 长点 `0.000488281`，D256 `0.000244141`；
- `run_kernel` 导出地址已由 `nm -D` 确认；
- 最终源码 SHA256：`95a9f723f5fe36fcec631c36aac5c1620680b3d7cd27c6bbc95090fc9c2011b6`。

## 15. 分数预测与在线闭环

`round2_score_projection.csv` 用“本地新旧耗时比乘第一轮线上耗时”的方式预测每点时间，再用线上
报告反推的 `T_h` 代入官方公式。预测平均分约 `54.57`。该数值不是在线分数；unpack/ragged 主干
与第一轮 paged 主干在在线硬件上的缩放可能不同。

当前环境没有 XPU-OJ 登录凭据或自动提交接口，无法在本轮直接验证是否达到 `57.50`。因此不能把
本地预测写成“已经超过榜首”。下一次必须提交 `paged_prefill_optimized.cu`，把新的 12 点报告追加到
本文；若真实分数仍低于 57.50，后续优化应以新报告的短点/长点缩放差异为依据，而不是继续盲扫
已经证伪的 tile 组合。
