# FlashInfer Ragged Prefill 算子入门：从注意力公式到 CUDA 冒烟实现

## 1. 这份教程讲什么

本题要实现的是带有以下特征的注意力前向计算：

- 多个 batch 的序列长度可以不同，Q 与 KV 的长度也可以不同，即 **ragged/变长序列**；
- 张量使用 NHD 布局；
- 使用 GQA：32 个 Query/Output head 共享 4 个 KV head；
- 使用 bottom-right 对齐的 causal mask；
- 输入、输出为 BF16，关键中间计算应使用 FP32；
- 评测器调用参赛代码中的 `extern "C" void run_kernel(...)`。

本文分析两个不同层次的“基线”：

1. 题目中的正式 baseline 是 FlashInfer Python API，即 `wrapper.plan(...)` 和 `wrapper.run(...)`。仓库没有包含它所调用的底层高性能 CUDA kernel 源码，因此只能解释其接口和语义，不能对其内部指令逐行分析。
2. 仓库里真正可逐行阅读的 CUDA 代码是 `starter/示例冒烟代码.md`。它的目标是验证编译、接口和提交链路，其中的精确 kernel 可以帮助理解算法，但整体不是一个可用于比赛优化的正式方案。

题目规格以 `xpuoj_problem/problem_20001/Agent 推理算子库优化 - FlashInfer Ragged Prefill.md` 为准。旧提交教程中“每段长度都等于 `seq_len`”等描述已经不符合当前 15 个测试点，不能据此实现正式答案。

## 2. 一句话理解这个算子

对 batch 中的每一个 Query token、每一个 Query head：

1. 找出它对应的 KV head；
2. 找出同一 batch 段中因果规则允许看到的 Key/Value token；
3. 计算 Query 与这些 Key 的相似度；
4. 对相似度做 softmax；
5. 用 softmax 权重对 Value 加权求和，得到一个 128 维输出。

它本质上仍是 scaled dot-product attention，只是加入了 ragged、GQA 和特殊的 causal 对齐规则。

## 3. 数学定义

设第 `b` 个 batch 段的 Query 长度为 `Lq`，KV 长度为 `Lk`。对于局部 Query 位置 `t` 和 Query head `hq`：

```text
G  = num_qo_heads / num_kv_heads = 32 / 4 = 8
hk = floor(hq / G)
```

`hk` 是该 Query head 对应的 KV head。对可见的 KV 位置 `j`，先计算：

```text
score[t, hq, j] = dot(Q[t, hq, :], K[j, hk, :]) / sqrt(128)
```

再计算：

```text
p[t, hq, :] = softmax(score[t, hq, :])
O[t, hq, :] = sum_j p[t, hq, j] * V[j, hk, :]
```

输出维度仍为 128。完整的注意力矩阵无需写回显存；高性能实现应在片上完成分块 softmax 和 `P @ V`。

## 4. 输入张量与 NHD 布局

题目中的连续张量形状为：

| 张量 | 形状 | 含义 |
|---|---|---|
| `q` | `(total_q, 32, 128)` | 所有 batch 的 Query token 拼接在一起 |
| `k` | `(total_kv, 4, 128)` | 所有 batch 的 Key token 拼接在一起 |
| `v` | `(total_kv, 4, 128)` | 所有 batch 的 Value token 拼接在一起 |
| `output` | `(total_q, 32, 128)` | 每个 Query token、Query head 的结果 |

NHD 表示内存中的三个维度依次是：

```text
N: token 行
H: head
D: head dimension
```

因此，Q 元素的线性地址为：

```text
q[(q_row * num_qo_heads + qo_head) * head_dim_qk + d]
```

K/V 使用 `num_kv_heads` 和 `kv_head`，不能误用 Q 的 32 个 head：

```text
k[(kv_row * num_kv_heads + kv_head) * head_dim_qk + d]
v[(kv_row * num_kv_heads + kv_head) * head_dim_vo + d]
```

这是最常见的地址计算错误之一。

## 5. ragged 到底是什么意思

不同 batch 段没有补齐成相同长度，而是直接首尾相接。`qo_indptr` 和 `kv_indptr` 记录每段的边界。

例如：

```text
qo_indptr = [0, 3, 5, 9]
kv_indptr = [0, 4, 8, 10]
```

则：

| batch | Q 的全局行范围 | `Lq` | KV 的全局行范围 | `Lk` |
|---:|---|---:|---|---:|
| 0 | `[0, 3)` | 3 | `[0, 4)` | 4 |
| 1 | `[3, 5)` | 2 | `[4, 8)` | 4 |
| 2 | `[5, 9)` | 4 | `[8, 10)` | 2 |

对第 `b` 段：

```cpp
qo_begin = qo_indptr[b];
qo_len   = qo_indptr[b + 1] - qo_begin;
kv_begin = kv_indptr[b];
kv_len   = kv_indptr[b + 1] - kv_begin;
```

`seq_len` 只是所有段长度的上界，可用来构造 launch grid，但不能当作任一段的真实长度。真实的 `total_q` 和 `total_kv` 分别是 `qo_indptr[batch_size]` 与 `kv_indptr[batch_size]`。

## 6. GQA：为什么 32 个 Q head 只需要 4 个 KV head

本题：

```text
num_qo_heads = 32
num_kv_heads = 4
G = 8
```

每个 KV head 服务连续的 8 个 Q head：

| Q head | KV head |
|---|---:|
| 0～7 | 0 |
| 8～15 | 1 |
| 16～23 | 2 |
| 24～31 | 3 |

计算公式为：

```cpp
kv_head = qo_head / 8;
```

逻辑上 8 个 Q head 的 Query 不同，因而注意力权重和输出也不同；它们只是读取相同的 K/V。这种共享关系也是后续最重要的优化机会之一。

## 7. bottom-right causal mask

### 7.1 等长时

若 `Lq == Lk`，第 `t` 个 Query 可看见 `0..t`，与普通下三角 causal mask 相同：

```text
visible = t + 1
```

### 7.2 Q 比 KV 短时

本题使用 bottom-right 对齐。其可见 KV 数量为：

```text
visible = clamp(Lk - Lq + t + 1, 0, Lk)
```

等价条件是：

```text
kv_pos < t + 1 + (Lk - Lq)
```

例如 `Lq=2, Lk=4`：

| Query 局部位置 `t` | `visible` | 可见 KV |
|---:|---:|---|
| 0 | 3 | 0、1、2 |
| 1 | 4 | 0、1、2、3 |

直观上，较短的 Q 对齐在较长 KV 的右下角，常见于已有历史 KV、当前只追加若干 Query token 的情况。

不能直接写成 `kv_pos <= q_pos`，否则题目中的 `q_len < kv_len` 用例会全部错误。

## 8. 正式 FlashInfer baseline 做了什么

题目给出的 baseline 包含三个步骤。

### 8.1 创建 workspace

```python
workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=q.device)
```

FlashInfer 使用 workspace 保存计划信息或中间状态。参赛接口没有 workspace 参数，因此自己的单文件 CUDA 实现不能照搬这一组织方式，也不应在每次 `run_kernel` 中动态申请显存。

### 8.2 `wrapper.plan(...)`

`plan` 接收两个 indptr、head 数、head dimension、数据类型和 causal 标志。概念上它负责根据输入形状准备调度信息、选择实现以及配置后续 kernel。计划阶段不是注意力数学计算本身。

### 8.3 `wrapper.run(...)`

`run` 读取 Q/K/V，根据 plan 执行 ragged GQA causal attention，并把结果写入预分配的 `output`。

参赛代码不需要复现 FlashInfer 的软件结构，只需在 `run_kernel` 中产生数值等价的输出。

## 9. starter 精确 kernel 的线程组织

`ragged_prefill_smoke_kernel` 使用 128 threads/block，也就是每个 block 4 个 warp。代码假定 warp 宽度为 32。

每个 warp 独立处理一个三元组：

```text
(batch, q_pos, qo_head)
```

线性任务号的拆解顺序是：

```cpp
qo_head = work % num_qo_heads;
work /= num_qo_heads;
q_pos = work % exact_len;
batch = work / exact_len;
```

随后从 indptr 读取真实段长。如果 `q_pos >= qo_len`，说明这个任务只是由 `seq_len` 上界产生的 padding 工作，整个 warp 提前返回。

这种映射简单、容易验证，但同一段内相邻的 32 个 Q head 会各自重复遍历相同的 KV 序列。

## 10. 一个 warp 如何算出一个输出向量

本题维度固定为 128。warp 有 32 个 lane，因此每个 lane 负责 4 个维度：

```text
d = lane + 0 * 32
d = lane + 1 * 32
d = lane + 2 * 32
d = lane + 3 * 32
```

每个 lane 首先把自己负责的 4 个 Q 元素转为 FP32，放入 `qv[4]`；同时用 `acc[4]` 保存输出累加值。

对每一个可见 `kv_pos`：

1. 每个 lane 读取 4 个 K 元素并计算局部点积；
2. `warp_sum` 用 shuffle 将 32 个 lane 的局部点积相加，得到完整的 128 维点积；
3. 乘 `1/sqrt(128)` 得到 score；
4. 更新在线 softmax 状态；
5. 每个 lane 读取 4 个 V 元素，更新自己的 4 个输出累加量。

循环结束后，每个 lane 将自己的 `acc[4] / l` 转回 BF16 并写出。32 个 lane 合起来正好写出一个 128 维输出向量。

## 11. 为什么在线 softmax 不需要保存整个 score 向量

直接实现会先保存所有 score，再求最大值、指数和归一化；长序列会产生很大的中间张量。starter 使用在线 softmax，只维护：

- `m`：目前见过的最大 score；
- `l`：以 `m` 为基准的指数和；
- `acc`：同样缩放基准下的 Value 加权和。

读到新 score `s` 后：

```text
m_new = max(m, s)
alpha = exp(m - m_new)
beta  = exp(s - m_new)

l_new   = l * alpha + beta
acc_new = acc * alpha + beta * V
```

最终：

```text
output = acc / l
```

减去当前最大值可以避免 `exp(score)` 溢出。`m`、`l`、score 和 `acc` 都使用 FP32，是通过精度校验的基础。

## 12. `prefix_mean_kernel` 实际做了什么

当旧的启发式条件触发时，starter 不再对尾部位置计算真正的注意力，而是计算 V 的前缀平均：

```text
mean[t] = (V[0] + ... + V[t]) / (t + 1)
```

并把同一个 KV head 的均值广播给对应的 8 个 Q head。随后精确 kernel 只覆盖前 1024 个位置。

前缀均值不是 attention：它完全没有读取 Q 和 K，也没有 softmax 权重。它只是旧冒烟数据上利用均匀随机数和宽松误差设计的近似。当前题目使用标准正态输入，并增加了 ragged、`q_len != kv_len` 和严格边界用例；正式实现不能依赖这种近似。

此外，这个 kernel 的循环直接使用 `seq_len`，而不是每段真实的 `qo_len/kv_len`。一旦在真正 ragged 的形状上触发，可能越过某段边界，甚至越界读写。因此它只能被理解为链路验证代码，不能作为优化起点的正确性基础。

## 13. `run_kernel` 的主机侧逻辑

starter 的 `run_kernel`：

1. 设定每 block 128 threads；
2. 默认 `exact_len = seq_len`；
3. 对某些旧的大形状把 `exact_len` 改为 1024，并先启动前缀均值 kernel；
4. 再启动精确 attention kernel；
5. 不调用 `cudaDeviceSynchronize()`，让评测器负责同步和计时。

按当前题目公布的 15 个测试点，旧近似条件通常不会触发，但这不意味着近似是合法方案。正式代码应删除该路径，始终执行数学正确的 attention。

## 14. 端到端伪代码

下面的伪代码最能概括题意：

```text
for b in [0, batch_size):
    q_begin, q_end = qo_indptr[b], qo_indptr[b + 1]
    k_begin, k_end = kv_indptr[b], kv_indptr[b + 1]
    Lq, Lk = q_end - q_begin, k_end - k_begin

    for t in [0, Lq):
        visible = clamp(Lk - Lq + t + 1, 0, Lk)  # causal=1

        for hq in [0, 32):
            hk = hq / 8
            scores = []
            for j in [0, visible):
                scores[j] = dot(Q[q_begin+t, hq], K[k_begin+j, hk]) / sqrt(128)

            p = softmax(scores)
            O[q_begin+t, hq] = sum_j p[j] * V[k_begin+j, hk]
```

GPU 优化改变的是循环如何分块、数据放在哪里以及用什么指令计算，不应改变以上语义。

## 15. 当前测试范围中的关键边界

正式实现至少要覆盖：

- batch 为 1、2、4、15、16、27、33；
- `seq_len` 只是最大段长上界；
- 段长不相等；
- `q_len < kv_len`；
- 长度 1；
- 长度 65 等非 2 的幂尾块；
- BF16 输入输出，FP32 中间计算；
- bottom-right causal；
- 32/4 GQA 映射。

用例 14 和 15 要求逐元素通过，不能用“总体 99% 匹配”掩盖尾块、单 token 或 mask 错误。

## 16. 本地 benchmark 的一个重要口径问题

`benchmark/bench_batch_prefill_ragged.py` 中：

```python
qo_indptr = torch.arange(0, batch_size + 1, ...)
kv_indptr = torch.arange(0, batch_size + 1, ...) * seq_len
q = torch.rand(batch_size, 32, head_dim_qk, ...)
```

因此每段实际 `q_len=1`、`kv_len=seq_len`。这更像“一个新 Query 读取长 KV”，并不是 OJ 中的方形 prefill。

但脚本的 FLOPs 公式使用了 `batch_size * seq_len * seq_len`，相当于假定 `q_len=kv_len=seq_len`。对于当前实际输入，它把 FLOPs 高估了约 `seq_len/2` 倍，所以 CSV 中出现数万 TFLOPS 并不代表硬件真的达到该性能。

该 CSV 的延迟可用于观察原 FlashInfer 在“每段一个 Query”的特定输入上表现，但不能作为 OJ Ragged Prefill 的性能基线，更不能用其中的 `tflops` 指导优化结论。正式优化前应建立与 15 个 OJ 形状一致的本地正确性和计时脚本。

## 17. 阅读代码时应抓住的主线

可以把整个算子记成四层：

1. **段边界层**：indptr 决定当前 Q/KV 属于哪个 batch；
2. **head 映射层**：8 个 Q head 对应 1 个 KV head；
3. **mask 层**：bottom-right causal 决定能看到多少 KV；
4. **数值层**：缩放点积、在线 softmax、Value 加权和。

starter 已经把这四层的正确公式展示出来了；它慢的根本原因不是公式，而是“一条 warp 独自、标量地、重复地完成一个 Query head 的全部工作”。下一步优化的核心就是在保持四层语义不变的前提下，让多个 Query 位置和同组 Q head 共同复用 K/V，并让矩阵乘单元承担主要计算。
