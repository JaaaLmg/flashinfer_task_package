# Ragged Prefill 阶段 A Baseline 与 Benchmark 使用手册

## 1. 文档目的

本文说明阶段 A 中建立的 FlashInfer Ragged Prefill 精确 baseline，包括：

- `code/` 目录下各文件的用途；
- baseline CUDA MACA kernel 的代码结构；
- 本地 benchmark 如何编译、调用、校验和计时；
- 完整复现命令；
- CSV 每一列的含义与计算公式；
- 哪个文件可以提交到 XPU-OJ，以及提交时的注意事项。

题目规格仍以 `xpuoj_problem/problem_20001/Agent 推理算子库优化 - FlashInfer Ragged Prefill.md` 为准。

## 2. 文件结构

阶段 A 相关文件如下：

```text
code/
├── ragged_prefill_baseline.cu          # CUDA MACA baseline 源码，也是候选 OJ 提交源码
├── ragged_prefill_baseline.so          # mxcc 本地编译产生的动态库，不提交 OJ
├── benchmark_stage_a.py                # 本地编译、正确性校验和性能计时脚本
├── stage_a_benchmark_results.csv       # 完整 15 类形状的机器可读结果
├── stage_a_benchmark_results.meta.txt  # 本次运行环境和结果位置
├── stage_a_benchmark_report.md         # 本次结果的中文摘要与表格
├── stage_a_smoke_results.csv           # 小边界冒烟结果，可删除或重新生成
└── stage_a_smoke_results.meta.txt      # 小边界冒烟环境信息
```

这些文件分为三类：

| 类别 | 文件 | 是否需要提交 OJ |
|---|---|---|
| 算子源码 | `ragged_prefill_baseline.cu` | 提交其源码内容 |
| 本地辅助 | `.so`、`benchmark_stage_a.py` | 不提交 |
| 测试记录 | `.csv`、`.meta.txt`、报告 `.md` | 不提交，留作复现和对比 |

`.so` 是本机针对 MXMACA `xcore1000` 编译得到的二进制。它与当前运行环境绑定，OJ 会自行编译源码，因此不能用 `.so` 替代 `.cu` 提交。

## 3. Baseline kernel 的代码结构

源码位于 `code/ragged_prefill_baseline.cu`，主要分为三层。

### 3.1 设备常量与 warp 归约

MetaX C500 在当前环境中报告 64-lane warp：

```cpp
constexpr int kWarpSize = 64;
constexpr uint64_t kFullWarpMask = ~uint64_t{0};
```

`warp_sum(float)` 使用 64 位 active mask 和步长 `32,16,8,4,2,1` 的 shuffle，将 64 个 lane 的局部点积归约成完整 score。

这与 starter 中写死的 32-lane warp 不同。当前 kernel 针对比赛使用的 MetaX C500/MXMACA 环境实现，不能未经修改就当作普通 NVIDIA 32-lane CUDA kernel 使用。

### 3.2 `ragged_prefill_baseline_kernel`

每个 64-lane warp 处理一个：

```text
(batch, q_pos, qo_head)
```

主要步骤为：

1. 从线性任务编号拆出 batch、段内 Query 位置和 Query head；
2. 从 `qo_indptr`、`kv_indptr` 读取当前段的真实边界；
3. 对超出真实 `qo_len` 的上界 grid 任务直接返回；
4. 计算 bottom-right causal 可见长度：

   ```text
   visible = clamp(kv_len - qo_len + q_pos + 1, 0, kv_len)
   ```

5. 用 `kv_head = qo_head / 8` 完成 32/4 GQA head 映射；
6. 每个 lane 负责 128 维向量中的两个维度；
7. 顺序遍历可见 KV，计算缩放 QK 点积；
8. 使用 FP32 `row_max/row_sum/out_acc` 完成数值稳定的在线 softmax；
9. 将 FP32 结果转为 BF16，写入预分配的 `output`。

该 kernel 不包含 `prefix_mean` 近似，也不会只精确计算前 1024 个 token。每一个输出都来自完整、精确的 attention 计算。

### 3.3 `extern "C" void run_kernel(...)`

这是评测器需要查找和调用的唯一公开入口。其参数类型、数量和顺序与题目完全一致。

`run_kernel` 做以下工作：

1. 检查题目固定参数 `Hq=32`、`Hkv=4`、`Dqk=Dvo=128`；
2. 使用 `batch_size * seq_len * num_qo_heads` 构造任务上界；
3. 以 128 threads/block，即每 block 两个 C500 warp 启动设备 kernel；
4. 启动后立即返回，不在函数内部调用 `cudaDeviceSynchronize()`。

`seq_len` 仅用于构造上界 grid。真实段长始终来自 indptr。

## 4. Benchmark 脚本的结构

`code/benchmark_stage_a.py` 同时负责构造输入、编译、调用、正确性校验和计时。

### 4.1 固定题目参数

```python
H_Q = 32
H_KV = 4
D_QK = 128
D_VO = 128
RTOL = 1.6e-2
ATOL = 1.6e-2
```

输入 Q/K/V 使用标准正态分布和 BF16，符合当前题目数据类型与分布要求。

### 4.2 测试用例

`cases()` 构造题目公布的 15 类形状，包括：

- 等长 1024、2048、4096、16384；
- batch 1、2、4、15、16、27、33；
- ragged 长、中、短段；
- `q_len < kv_len`；
- 单 token；
- 长度 65/33 的非 2 的幂尾块。

题目只公开了 ragged 用例的 batch、总长度和最大长度，没有公开隐藏评测中每一段的精确 indptr。因此本地脚本构造确定性正整数段长，使 batch、`total_q/total_kv`、`max_q/max_kv` 与题目表格完全一致。它不能复刻未知的隐藏 indptr，但可以验证 kernel 不依赖等长假设。

### 4.3 编译过程

`compile_library()` 调用：

```bash
mxcc -O3 -std=c++17 \
  --offload-arch=xcore1000 \
  -I/opt/maca/tools/cu-bridge/include \
  -shared -fPIC \
  code/ragged_prefill_baseline.cu \
  -o code/ragged_prefill_baseline.so
```

若 `.so` 存在且修改时间不早于 `.cu`，默认跳过编译。使用 `--force-build` 可以强制重新编译。

脚本中 `/opt/maca/tools/cu-bridge/include` 是当前比赛镜像的路径。如果其他环境的 MXMACA 安装位置不同，需要相应修改。

### 4.4 调用 OJ C ABI

`load_kernel()` 用 `ctypes.CDLL` 加载 `.so`，再为 `run_kernel` 声明：

```text
6 个设备指针 + 7 个 int64 标量
```

`launch()` 通过 PyTorch CUDA tensor 的 `data_ptr()` 传递设备地址。这个过程验证的是与 OJ 相同的 C 符号和参数顺序，而不是另写一个 Python 算子入口。

### 4.5 FlashInfer 参考结果

每个用例都创建：

```python
flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    workspace, kv_layout="NHD", backend="auto"
)
```

然后使用相同的 Q/K/V、indptr、head 数、维度和 causal 参数生成 `reference`。

### 4.6 正确性规则

逐元素基础容差为：

```text
tolerance = atol + rtol * abs(reference)
atol = rtol = 1.6e-2
```

普通用例要求：

```text
match_ratio >= 0.99
severe_error_count == 0
```

其中严重误差定义为：

```text
abs(candidate - reference) > 8 * tolerance
```

用例 14 和 15 要求 `match_ratio == 1.0`。输出出现 NaN/Inf 也会直接失败。

### 4.7 计时规则

计时使用 `torch.cuda.Event`，单位为毫秒。`elapsed_ms()` 返回多次 launch 的平均设备时间。

重复次数由第一次计时自适应决定：目标测量约 100 ms，但不超过 `--max-repeats`。FlashInfer reference 至少允许最多 20 次，以减小短 kernel 的计时抖动。很慢的阶段 A 长序列通常只重复一次。

性能计时不包含 `.cu` 编译和 FlashInfer `plan()`；计时对象分别是 candidate `run_kernel` launch 和 FlashInfer `wrapper.run()`。

## 5. 环境要求

当前脚本至少需要：

- MetaX C500 或兼容的 MXMACA GPU 环境；
- 可执行的 `mxcc`；
- PyTorch MACA 版且 `torch.cuda.is_available()` 为 `True`；
- FlashInfer MACA 版；
- Python 3.12 附近版本；
- 足够显存。完整 15 类用例会同时持有 Q/K/V、candidate output、reference output 和 128 MiB workspace。

检查命令：

```bash
mx-smi
mxcc --version
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
python -c "import flashinfer; print(flashinfer.__version__)"
```

本次成功运行的主要版本为：

```text
GPU:         MetaX C500
MXMACA:      3.5.3.20
PyTorch:     2.8.0+metax3.5.3.9
FlashInfer:  0.2.6+metax3.5.3.9torch2.8
mxcc:        1.0.0 (6477545d4d)
```

## 6. 使用命令

以下命令均从仓库根目录 `/data/flashinfer_task_package` 执行。

### 6.1 查看帮助

```bash
python code/benchmark_stage_a.py --help
```

### 6.2 先跑最快的边界冒烟

```bash
python code/benchmark_stage_a.py \
  --cases 14,15 \
  --output code/stage_a_smoke_results.csv \
  --force-build \
  --max-repeats 3
```

预期：两个用例均显示 `passed=True`，并生成 CSV 和同名 `.meta.txt`。

### 6.3 跑完整 15 类形状

```bash
python code/benchmark_stage_a.py \
  --cases all \
  --output code/stage_a_benchmark_results.csv \
  --max-repeats 3
```

当前阶段 A 的 `B=1,L=16384` 用例约需 3.1 秒/次。完整运行还包含参考计算和其他用例，应耐心等待。

### 6.4 强制重新编译

```bash
python code/benchmark_stage_a.py \
  --cases 14,15 \
  --force-build
```

修改 `.cu` 后，只要其时间戳比 `.so` 新，脚本会自动重编；怀疑缓存或构建异常时使用 `--force-build`。

### 6.5 只跑指定用例

```bash
python code/benchmark_stage_a.py --cases 1,9,10,11,15
```

`--cases` 接受 `all` 或逗号分隔的 ID。

### 6.6 改变输出文件与随机种子

```bash
python code/benchmark_stage_a.py \
  --cases all \
  --output code/results_after_change.csv \
  --seed 20260716 \
  --max-repeats 5
```

每个 CSV 会伴随一个同基本名的 `.meta.txt`。为保留优化历史，建议每轮使用不同输出名，不要覆盖阶段 A 文件。

### 6.7 参数一览

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--cases` | `all` | 全部用例或逗号分隔的用例 ID |
| `--output` | `code/stage_a_benchmark_results.csv` | CSV 输出路径 |
| `--max-repeats` | `10` | candidate 自适应计时的最大重复次数 |
| `--seed` | `20260715` | 随机输入的基础种子；实际每例再加 case ID |
| `--force-build` | 关闭 | 忽略时间戳并强制调用 mxcc |

脚本所有已选用例都通过时退出码为 0；出现失败或异常时退出码为 1。

## 7. 输出文件的含义

### 7.1 `stage_a_benchmark_results.csv`

一行对应一个测试用例。它是后续自动比较、绘图和参数搜索应读取的主要文件。

### 7.2 `.meta.txt`

记录：

- 运行时间和主机；
- Python、PyTorch、FlashInfer 版本；
- GPU 名称；
- 源码、动态库和 CSV 路径；
- 通过用例数；
- ragged 长度是本地按公开统计构造，而非隐藏 OJ 原始 indptr。

不同机器或不同软件栈的结果不能只按 CSV 延迟直接比较，应同时保存 meta 文件。

### 7.3 `stage_a_benchmark_report.md`

这是当前阶段 A 结果的人工可读摘要，包含运行环境、15 个用例的简化表格和阶段结论。重新运行脚本不会自动更新这份静态报告；以新生成的 CSV 和 meta 为准。

### 7.4 `ragged_prefill_baseline.so`

这是本地 benchmark 的加载目标。源码更新后需要重新编译。它不是比赛提交物，也不保证能在其他 MXMACA 版本、其他目标架构或普通 CUDA 环境中加载。

## 8. CSV 各列详细解释

CSV 当前包含 19 列。

| 列名 | 单位/类型 | 含义 |
|---|---|---|
| `case_id` | 整数 | 本地测试用例编号，与题目表格的 1～15 对应 |
| `name` | 字符串 | 便于阅读的用例类型名称，如等长、ragged、`q<kv`、单 token |
| `batch_size` | 整数 | batch 段数，即 `len(q_lens)`，也等于两个 indptr 的长度减 1 |
| `seq_len` | token 数 | 所有 Q/KV 段真实长度的最大值；传给 `run_kernel` 作为 launch 上界，不是每段真实长度 |
| `total_q` | token 数 | 所有段 Query 长度之和，等于 `qo_indptr[batch_size]` |
| `total_kv` | token 数 | 所有段 KV 长度之和，等于 `kv_indptr[batch_size]` |
| `max_q` | token 数 | 所有段中最大的 `q_len` |
| `max_kv` | token 数 | 所有段中最大的 `kv_len` |
| `visible_pairs` | 配对数 | 所有 batch、所有 Query 位置在 bottom-right causal mask 下可见的 Query-token/KV-token 位置配对总数；尚未乘 32 个 Q head |
| `candidate_ms` | ms | 当前 `.cu` 中 `run_kernel` 的平均设备执行时间，不包含编译和数据生成 |
| `flashinfer_ms` | ms | 相同输入上 FlashInfer `wrapper.run()` 的平均设备执行时间，不包含 `plan()` |
| `slowdown_vs_flashinfer` | 倍数 | `candidate_ms / flashinfer_ms`。大于 1 表示 candidate 更慢；小于 1 才表示 candidate 比本地 FlashInfer 更快 |
| `effective_tflops` | TFLOPS | candidate 的有效吞吐；按真实 ragged 可见配对计算 QK 与 PV 主干 FLOPs，不包含 softmax 标量 FLOPs |
| `candidate_repeats` | 次数 | 计算 `candidate_ms` 时使用的重复次数 |
| `flashinfer_repeats` | 次数 | 计算 `flashinfer_ms` 时使用的重复次数 |
| `match_ratio` | `[0,1]` | 满足 `abs_error <= atol + rtol*abs(reference)` 的元素比例 |
| `max_abs_error` | 数值绝对值 | candidate 与 FlashInfer reference 之间最大的逐元素绝对误差 |
| `severe_error_count` | 元素数 | 满足 `abs_error > 8*(atol + rtol*abs(reference))` 的元素数量；必须为 0 |
| `passed` | 布尔值 | 是否同时通过有限值、匹配率和严重误差规则 |

### 8.1 `visible_pairs` 的计算

对第 `b` 段的局部 Query 位置 `t`：

```text
visible(b,t) = clamp(Lk[b] - Lq[b] + t + 1, 0, Lk[b])
```

因此：

```text
visible_pairs = sum_b sum_t visible(b,t)
```

若等长且 causal，单段结果为 `L*(L+1)/2`。若 `q_len < kv_len`，早期 Query 也能看到历史 KV 前缀，所以不能用普通左对齐三角形计算。

### 8.2 `effective_tflops` 的计算

主干有效 FLOPs 定义为：

```text
flops = 2 * visible_pairs * num_qo_heads * (head_dim_qk + head_dim_vo)
```

其中：

- 前面的 2 表示乘加按 2 FLOPs 计；
- QK 和 PV 分别贡献 `head_dim_qk` 与 `head_dim_vo`；
- `visible_pairs` 尚未含 head，因此乘 `num_qo_heads=32`。

最后：

```text
effective_tflops = flops / (candidate_ms * 1e9)
```

因为毫秒转换为秒并换算到 `10^12 FLOPs/s` 后，分母可合并为 `candidate_ms * 1e9`。

该值用于同一题目内比较，不包含 `exp/max/div` 等 softmax 操作，因此不是硬件所有指令的完整 FLOP 统计。

### 8.3 如何读一行结果

以用例 4 为例：

```text
candidate_ms              = 3120.756 ms
flashinfer_ms             = 26.594 ms
slowdown_vs_flashinfer    = 117.35
effective_tflops          = 0.705
match_ratio               = 1.0
severe_error_count        = 0
passed                    = True
```

这表示数值完全通过当前判定，但阶段 A 实现比本地 FlashInfer 慢约 117 倍。`passed=True` 只代表正确性通过，不代表性能已达到可打榜水平。

## 9. `.cu` 是否就是可以直接提交的代码

### 9.1 简短答案

`code/ragged_prefill_baseline.cu` 的**源码内容**已经具有题目要求的 CUDA Maca 提交形式，可以作为当前 baseline 提交；但不是“只要文件后缀是 `.cu`，任何 `.cu` 都能直接提交”。

当前 XPU-OJ 教程描述的流程是：

1. 在语言下拉框选择 **CUDA Maca**；
2. 打开 `code/ragged_prefill_baseline.cu`；
3. 复制文件的全部源码内容；
4. 粘贴到 OJ 的代码框中，不要带 Markdown 的三反引号；
5. 提交并查看每个测试点的编译、正确性和性能结果。

如果平台之后增加文件上传功能，也应上传 `.cu` 源码，而不是 `.so` 或 benchmark Python 文件。

### 9.2 为什么当前源码满足提交形式

- 包含题目需要的 CUDA/BF16 头文件；
- 在文件作用域定义了精确的 `extern "C" void run_kernel(...)`；
- 参数类型、顺序、数量与题目一致；
- `run_kernel` 内部自行 launch CUDA kernel；
- 没有 `main()`，因为 OJ 提供调用方；
- 没有在计时路径中调用 `cudaDeviceSynchronize()`；
- 没有动态显存申请；
- 不依赖本地 benchmark 的 Python/ctypes 代码；
- 已用 `nm -D` 确认动态库导出未改名的 `run_kernel` 符号；
- 已在本地 C500 上对 15 类形状与 FlashInfer 对照通过。

### 9.3 不能提交哪些文件

- `benchmark_stage_a.py`：它是本地测试驱动，不是 OJ kernel；
- `ragged_prefill_baseline.so`：它是本机编译产物，OJ 会自行编译；
- CSV/meta/Markdown：它们是记录文件；
- `starter/示例冒烟代码.md`：它包含 Markdown 和旧的非精确近似，不应当作正式答案提交。

### 9.4 “可提交”不等于“能获得好成绩”

当前 baseline 的正确性可信，但性能非常低。最长用例本地约 3120 ms，比 FlashInfer 慢约 117 倍。OJ 还可能设置单测试点时间限制，因此可能出现以下情况：

- 编译成功、正确性通过，但得分很低；
- 某些长用例因平台时限而 TLE；
- OJ 的软件版本、资源切片和隐藏 indptr 与本地不同，延迟与本地不完全一致。

所以该 `.cu` 的定位是“阶段 A 精确基线和后续优化的正确性锚点”，不是最终打榜版本。后续每轮优化都应保留同样的 `run_kernel` ABI，并用本 benchmark 回归后再提交 OJ。

## 10. 常见问题

### 10.1 `cuda_bf16.h file not found`

确认编译命令含：

```text
-I/opt/maca/tools/cu-bridge/include
```

若 MXMACA 安装在其他路径，先执行：

```bash
find /opt -name cuda_bf16.h -o -name cuda_runtime.h
```

再修改脚本中的 include 路径。

### 10.2 `invalid target ID: sm_80`

Standalone `mxcc` 不能使用 NVIDIA 风格的 `--cuda-gpu-arch=sm_80`。当前环境使用：

```text
--offload-arch=xcore1000
```

### 10.3 修改源码后结果没有变化

使用：

```bash
python code/benchmark_stage_a.py --cases 14,15 --force-build
```

并确认日志出现 `[build] mxcc ...`。

### 10.4 CSV 只有部分用例

如果设备 kernel 异常，CUDA context 可能已不可继续使用。脚本会保存已经收集的行并停止。检查终端中的 `ERROR`，修复后重新启动 Python 进程再跑。

### 10.5 多次运行延迟不完全一致

GPU 切片比例、频率、温度、后台任务和短 kernel 的事件计时都会带来波动。比较优化前后时应：

- 使用同一机器和软件环境；
- 保存 meta 文件；
- 使用同一 seed、用例和重复次数；
- 对短用例适当提高 `--max-repeats`；
- 至少重复整轮 benchmark 两到三次，再比较中位数。

## 11. 推荐的日后使用流程

每次优化建议遵循：

1. 保留阶段 A 的 `.cu` 和 CSV，不覆盖；
2. 复制源码形成新版本；
3. 先跑 `--cases 14,15 --force-build`；
4. 再跑 `q<kv` 用例 9、10、11；
5. 再跑 ragged 用例 1、12、13；
6. 全部正确后运行 `--cases all`；
7. 将新 CSV 与 `stage_a_benchmark_results.csv` 按 `case_id` 对比；
8. 确认没有性能严重回退后，再把新 `.cu` 源码内容粘贴到 XPU-OJ；
9. 保存 OJ 结果、源码版本、CSV 和 meta，形成可复现的优化记录。
