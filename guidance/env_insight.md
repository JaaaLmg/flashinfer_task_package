# MACA 环境洞察：CUDA 兼容机制与 Profile 工具实操

本文记录在赛事镜像（MetaX C500 / MACA 3.7.1.5）上实测得到的环境事实，重点是 **profile 工具的用法**。
所有命令和输出均在本机验证过，不是从文档推断的。

配套阅读：[`optimization.md`](optimization.md) 第 13 节列出了 profile 要回答的 9 个问题，
本文给出在 MACA 上**具体用哪个指标回答**。

---

## 0. 环境速查

```text
GPU              MetaX C500，1 卡，64 GB 显存
MACA Version     3.7.1.5      驱动 3.8.30      mx-smi 2.3.1
编译器           mxcc 1.0.0 (LLVM)   /opt/maca/mxgpu_llvm/bin/mxcc
CUDA 兼容层      /opt/maca-3.7.1/tools/cu-bridge/
Profiler         mcProfiler   /usr/local/bin/mcProfiler
```

实测 device properties（用 `cudaGetDeviceProperties` 探针取得）：

| 属性 | 值 | 与 NVIDIA 的差异 |
|---|---|---|
| **warpSize** | **64** | NVIDIA 是 32 —— **最容易踩的坑** |
| multiProcessorCount | 104 | — |
| sharedMemPerBlock | 65536 (64 KB) | — |
| clockRate | 1.6 GHz | — |
| memoryBusWidth | 4096 bit | HBM |
| 峰值带宽 | **1843.2 GB/s** | 由 profiler RoofLine 的 `MAX_Bandwith` 字段给出 |

> 1843.2 GB/s 是判断 memory-bound 算子（20003 MLA / 20004 Decode）优化是否到位的分母。
> [`paged_decode/explaination.md`](../paged_decode/explaination.md) 说"看 `kv_bandwidth_GB_s` 是否逼近 HBM 峰值"，
> 峰值就是这个数。

---

## 1. CUDA 兼容是怎么做到的

### 1.1 你 include 的"CUDA 头文件"是假的

`paged_decode_baseline.cu` 顶部写的 `#include <cuda_bf16.h>`，在这台机器上**没有 NVIDIA 的版本**。
实际被找到的是沐曦写的同名替身：

```
/opt/maca-3.7.1/tools/cu-bridge/include/cuda_bf16.h      ← 30 行
/opt/maca-3.7.1/tools/cu-bridge/include/cuda_runtime.h   ← 13 行
```

`cu-bridge/include/` 下有 **160 个**这样的 shim（`cuda.h`、`cublas_v2.h`、`cooperative_groups.h`、
`cuda_fp8.h`、`nvrtc.h`……）。`cuda_bf16.h` 的全部实质内容就是：

```cpp
#include <maca_bfloat16.h>
typedef maca_bfloat16            __nv_bfloat16;
typedef struct __maca_bfloat162  __nv_bfloat162;
typedef maca_bfloat16            nv_bfloat16;
```

运行时 API 同理，`bridge/runtime/cuda_to_maca_mcr_adaptor.h` 里是上千行宏改名：

```cpp
#define cudaError_t             mcError_t
#define cudaSuccess             mcSuccess
#define cudaMemcpyHostToDevice  mcMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  mcMemcpyDeviceToHost
```

它甚至伪装版本号 —— `bridge/runtime/cuda_benchmark_targets.h`：

```cpp
#define WCUDART_VERSION 11060     // 对外声称自己是 CUDA 11.6
```

这样上游库里 `#if CUDA_VERSION >= xxxx` 的条件编译才能正常走通。

### 1.2 完整链路

```text
你的 .cu 源码
   └─ cu-bridge 头文件：cuda* 符号 → typedef / #define → mc* 符号
        └─ mxcc（LLVM fork，自研 CUDA C++ 前端 + 沐曦 GPU 后端）
             └─ 链接 /opt/maca/lib：libruntime_cu.so / libmcruntime.so
                  └─ MetaX C500（自研 ISA）
```

关键结论：这是 **源码级兼容（编译期翻译）**，不是二进制兼容。
N 卡上编出来的 `.so` / PTX / cubin 在 C500 上跑不了，必须重新编译。

### 1.3 CUDA 生态并不开源，那兼容的是什么

NVIDIA 的驱动、PTX→SASS 后端、cuBLAS/cuDNN 二进制全部闭源。沐曦兼容的只有两层公开契约：

1. **语言语法** —— `__global__`、`<<<grid, block>>>`、`threadIdx`、`__shfl_down_sync`。
   这是 C++ 方言扩展，语法本身不受版权保护，任何人可以自己实现前端。
2. **API 函数签名** —— `cudaMalloc(void**, size_t)` 长什么样。
   API 接口在法律上不受版权保护（Google v. Oracle 先例），可自由重新实现。

**没有使用任何 NVIDIA 代码**，全部是按签名独立实现：

| 层次 | NVIDIA | MACA | 位置 |
|---|---|---|---|
| 头文件 | `cuda_runtime.h` | `mc_runtime_api.h` | `/opt/maca/include/mcr/` |
| 编译器 | nvcc | **mxcc** | `/opt/maca/mxgpu_llvm/bin/` |
| nvcc 命令行替身 | nvcc | **cucc**（bash 脚本） | `cu-bridge/bin/cucc` |
| 运行时 | libcudart | libruntime_cu / libmcruntime | `/opt/maca/lib/` |
| BLAS / DNN | cuBLAS / cuDNN | **mcblas / mcdnn** | `/opt/maca/include/` |
| CUTLASS | CUTLASS | **mctlass / mctlassEx** | `/opt/maca/include/` |
| CUB / Thrust | CUB / Thrust | **mccub / cub / thrust** | `/opt/maca/include/` |
| FlashInfer | flashinfer | **mcflashinfer** | `/opt/maca/include/mcflashinfer/` |
| CUPTI | CUPTI | **mcpti** | `/opt/maca/include/mcpti/` |
| 驱动 / ISA | 闭源 SASS | 自研 ISA | **无二进制兼容** |

镜像里还有 `/opt/maca/include/mcr/hip_to_maca_adaptor.h`，说明同时接了 HIP 一层，
整体思路和 AMD ROCm/HIP 是同一路数。

### 1.4 "写法没区别"只在语法层面成立

微架构差异是真实存在的，最直接的例子就在 baseline 里
（[`paged_decode_baseline.cu:23`](../paged_decode/paged_decode_baseline.cu#L23)）：

```cpp
constexpr int      kWarpSize     = 64;              // 不是 32
constexpr uint64_t kFullWarpMask = ~uint64_t{0};    // 64 位 mask，不是 0xffffffff
```

warpSize=64 会连锁影响：

- `__shfl_*_sync` 的 mask 宽度和 offset 归约层数（`for (off=32; off>0; off>>=1)` 而非 `off=16`）
- `__ballot_sync` 返回 64 位
- 每 lane 分摊的 head_dim 元素数（`kMaxPerLane = 256/64 = 4`）
- shared memory bank conflict 的模式

照抄 32-lane 假设的 CUDA kernel 会**静默算错**，不报错。同理，MMA fragment 形状、
async copy 支持情况、swizzle 最优布局都必须实测，
不能靠 NVIDIA 经验断言（[`optimization.md:257`](optimization.md) 已就此警告）。

---

## 2. 手工编译 .cu（实测可用）

`mxcc` **不会**自动带上 cu-bridge 的头文件和宏，直接编译会失败：

```console
$ mxcc probe.cu -o probe
probe.cu:1:10: fatal error: 'cuda_runtime.h' file not found
```

`cucc` 包装脚本在本机也有环境问题（`'__macro_mxcc.h' file not found`）。
**实测可用的完整命令**：

```bash
mxcc your_kernel.cu -o your_kernel \
  -I/opt/maca-3.7.1/tools/cu-bridge/include \
  -I/opt/maca/include \
  -Xclang -imacros -Xclang /opt/maca-3.7.1/tools/cu-bridge/include/__macro_mxcc.h \
  -L/opt/maca/lib -lruntime_cu -lmcruntime
```

三个易错点：

1. `-imacros` 必须用 `-Xclang` 转发，写成 `-imacros xxx.h` 会报 `unknown argument`；
2. 链接库是 `-lruntime_cu -lmcruntime`，**不是** `cu-bridge/lib/libcuda.so`
   （那个会导致 `DSO missing from command line`）；
3. 编 `.so` 供 Python `ctypes` 调用时再加 `-shared -fPIC`。

> OJ 平台自己处理编译，这套命令是**本地对拍和 profile 用的**。

---

## 3. Profile 工具：mcProfiler

### 3.1 工具全景

| 工具 | 路径 | 对标 | 状态 |
|---|---|---|---|
| **mcProfiler** | `/usr/local/bin/mcProfiler` | **Nsight Compute** | ✅ 实测可用，本文重点 |
| gui-profiler | `/opt/mcProfiler-ubuntu18.04/gui-profiler-0.1.0.AppImage` | ncu-ui | AppImage，需要 FUSE |
| mxvs | `/opt/maca/bin/mxvs` | Visual Profiler | ⚠️ AppImage，本机缺 `libfuse.so.2`，需 `--appimage-extract` |
| mcpti | `/opt/maca/include/mcpti/` | CUPTI | 头文件，可自写 tracing 工具 |
| mx-smi | `/usr/bin/mx-smi` | nvidia-smi | ✅ 可用 |

**没有直接对标 Nsight Systems 的时间线工具。** `/opt/maca/bin/mcTracer` 存在但不在 PATH，
未验证可用性。要做跨 kernel / CPU-GPU 时间线分析，目前只能靠 CUDA event 计时手工埋点，
或基于 mcpti 自己写。

### 3.2 第一个坑：必须在 mcProfiler 目录下运行

在项目目录直接跑会失败：

```console
$ mcProfiler show_metrics
ERROR: [Errno 2] No such file or directory: '/data/flashinfer_task_package/config'
cleaning server..
```

它要求 cwd 下有 `config/` 目录（里面是 `summary.pcd` 等指标定义文件）。**正确做法**：

```bash
cd /opt/mcProfiler-ubuntu18.04 && mcProfiler show_metrics
```

被 profile 的程序用 `--cwd` 单独指定，两者互不影响。

> 如果上一次运行中断，可能残留 `profiler_server` 进程（会看到 `clean server failed`），
> 需要手工 `pkill profiler_server`。

### 3.3 基本用法

```bash
# 列出所有可用指标
cd /opt/mcProfiler-ubuntu18.04 && mcProfiler show_metrics

# profile 一个可执行文件
cd /opt/mcProfiler-ubuntu18.04 && mcProfiler perf_exec \
    --cmdline "/tmp/probe" \
    --casename probe_test \
    --cwd /tmp \
    --per-kernel

# profile 一个 Python benchmark
cd /opt/mcProfiler-ubuntu18.04 && mcProfiler perf_exec \
    --cmdline "python benchmark_decode.py" \
    --casename decode_v1 \
    --cwd /data/flashinfer_task_package/paged_decode \
    --kernelnames paged_decode_baseline_kernel \
    --counts 5
```

常用参数：

| 参数 | 作用 | 建议 |
|---|---|---|
| `--cmdline` | 被测命令（必填） | 用绝对路径 |
| `--casename` | 本次 case 名（必填） | 带上版本号，便于比对多轮优化 |
| `--cwd` | 被测程序工作目录 | benchmark 脚本要读写 CSV 时必须设 |
| `--per-kernel` | 每次 kernel 调用独立出报告 | 更精细但**慢很多** |
| `--kernelnames` | 只抓指定 kernel | **强烈建议**，否则 PyTorch 的上百个 kernel 全被抓 |
| `--exclude` | 反转 `--kernelnames` 语义 | — |
| `--counts N` | 限制采集的 kernel 数，默认 50，`0` 为不限 | 设 3~5 足够，避免报告爆炸 |
| `--single-pass` | 单遍推断全局计数器，显著提速 | 要求 workload 足够大能打满所有 AP |
| `--custom` / `--profile-from-start 0` | 只统计 `mcProfilerStart/Stop` 之间 | 跳过 warmup 时用 |

**耗时提醒：** 实测一个只有 20 次 kernel 调用的最小程序，`--per-kernel` 全指标跑了约
**2 分钟**（profiler 要多遍重放程序来轮换硬件计数器）。真实 benchmark 务必配合
`--kernelnames` + `--counts` 收窄范围，否则可能几十分钟。

### 3.4 输出产物

报告落在 `/opt/mcProfiler-ubuntu18.04/output<时间戳>/`，命令结束时会打印路径：

```text
perf done, please check report file /opt/mcProfiler-ubuntu18.04/output20260730083111
```

每次 kernel 调用生成 4 个文件（`<序号>_<mangled kernel 名>.*`）：

| 文件 | 内容 | 用途 |
|---|---|---|
| `*.txt` | 人类可读全指标报告 | **主要看这个** |
| `*.txt.csv` | 同内容 CSV | 脚本化提取、多轮对比 |
| `*.txt.json` | 同内容 JSON | 程序化处理 |
| `*_dumped_result.json` | 原始硬件计数器 | 一般不用 |

另外会生成 PNG 图表，其中最有价值的是 **`RoofLine<时间戳>.png`**。

### 3.5 报告结构与关键指标

`.txt` 报告分为 7 个 Sub-module。以下是实测输出片段（一个简单的 `out[i]=a[i]*2` kernel）：

```text
##############################
Sub-module: Summary
------------------------------
Name: Total Cycles              Value: 10,475.94(Kcycles)
Name: Memory Access per Second  Value: 433,553.51(MB/s)
Name: AP busy Duty              Value: 0.63%
Name: RoofLine                  Value: {
    "data": {
        "MAX_Bandwith":   1843.2,                  ← C500 峰值带宽 GB/s
        "case_bandwith":  568.267,                 ← 本 kernel 实测带宽
        "MAX_I":          260.0,                   ← roofline 拐点算术强度
        "case_I":         6.409,                   ← 本 kernel 算术强度
        "case_flops":     3.642,
        "all_memacs":     33560096.0,              ← 实际访存字节
        "block_num":      104
    },
    "filename": ".../RoofLine20260730083237043.png"
}
```

**`RoofLine` 这一项是整份报告最有价值的东西**，它一次给全：

- `MAX_Bandwith = 1843.2` GB/s —— 优化目标的分母
- `case_bandwith` —— 你的实测带宽，除以上面就是带宽利用率
- `case_I` vs `MAX_I = 260` —— `case_I` 远小于 260 即确认 memory-bound
- `all_memacs` —— 实际访存字节数，**用于验证 KV 重复读取**（见 §3.7）

其余 6 个 Sub-module 的对标关系：

| Sub-module | 关键指标 | Nsight Compute 对应 |
|---|---|---|
| **Summary** | `RoofLine`、`Memory Access per Second`、`AP busy Duty`、`Total Cycles` | Speed of Light / Roofline |
| **Memory Statistics** | `Global Memory Read bytes`、`VL1 Hit Rate`、`L2C Hit Rate`、`Dnoc Read Average Latency` + Histogram | Memory Workload Analysis |
| **Workgroup Memory** | `shared memory access efficiency`、`average conflict cycles per instruction` | Shared Memory bank conflict |
| **Occupancy** | `Achieved waves` / `Dispatched waves`、`AP MMA Duty ratio`、`instruction per cycle` | Occupancy / Compute Workload |
| **ISU Statistics** | `ISU stall cycles layout`、`Instructions Comparison` | Warp State / Stall Reasons |
| **CE Statistics** | `WORKGROUPS`、`WAVES`、`Average Wave life cycles` | Launch Statistics |
| **Compute Workload** | 指令分类计数 | Instruction Statistics |

实测 Memory Statistics 段的样子：

```text
Name: Global Read Instructions     Value: 65,536
Name: Global Memory Read bytes     Value: 16,782,464.0byte     ← 关键
Name: Global Memory Write bytes    Value: 16,777,632.0byte
Name: VL1 Hit Rate                 Value: 95.31%
Name: L2C Hit Rate                 Value: 0.42%
Name: Dnoc Read Average Latency    Value: 237.83
```

Occupancy 段：

```text
Name: Achieved waves        Value: 65,536.0
Name: Dispatched waves      Value: 65,536.0
Name: AP MMA Duty ratio     Value: 0.0%          ← BF16 MMA 单元是否真被用到
Name: instruction per cycle Value: 18.74
```

> `AP MMA Duty ratio` 直接回答 [`optimization.md`](optimization.md) 第 13 节的问题 3
> "BF16 矩阵乘加单元是否真的被使用" —— 上面这个标量 kernel 是 0.0%，符合预期。

### 3.6 快速提取指标

`.txt` 是纯文本，直接 grep 即可；多轮对比建议用 CSV：

```bash
OUT=/opt/mcProfiler-ubuntu18.04/output20260730083111

# 看单个指标（注意是 -A2，中间隔着 Description 行）
grep -A2 "Name: Global Memory Read bytes" $OUT/*.txt | head

# Name/Value 配对成表（镜像里没有 column 命令，直接看 paste 输出即可）
grep -E "^Name:|^Value:" $OUT/10_*.txt | paste - -

# 从 RoofLine 里抠带宽利用率
python3 -c "
import json,re,glob
t = open(glob.glob('$OUT/10_*.txt')[0]).read()
d = json.loads(re.search(r'Name: RoofLine.*?Value: (\{.*?\n\})', t, re.S).group(1))['data']
print(f\"带宽 {d['case_bandwith']:.1f} / {d['MAX_Bandwith']} GB/s = {d['case_bandwith']/d['MAX_Bandwith']*100:.1f}%\")
print(f\"算术强度 {d['case_I']:.2f} (拐点 {d['MAX_I']}) -> {'memory-bound' if d['case_I'] < d['MAX_I'] else 'compute-bound'}\")
"
```

### 3.7 针对 20004 Paged Decode 的实用套路

Decode 是彻底的 memory-bound，**别一上来看一堆 stall 指标**。最省事的路径是两个数：

**① 带宽利用率**（来自 `RoofLine`）

```text
case_bandwith / 1843.2
```

这是唯一真正的优化进度条。

**② KV 重复读取倍数**（来自 `Global Memory Read bytes`）

baseline 是 warp-per-`(batch, qo_head)`，同一个 GQA group 里 8 个 qo_head 各读一遍 KV。
所以：

```text
重复倍数 = Global Memory Read bytes / 理论最小字节
理论最小字节 = 2(K和V) × 2(bf16) × total_kv × num_kv_heads × head_dim
```

- baseline 应该 ≈ **8**（= `group_size`）
- 改成 CTA 承担整个 GQA group 后，应该掉到 ≈ **1**

这一个比值就能验证"KV 只读一遍"这条最大的优化是否真的生效，比看 stall 直观得多。

后续再按需要看：

| 想验证什么 | 看哪个指标 |
|---|---|
| 向量化访存是否生效 | `Global Read Instructions` 应随 128-bit 化而下降 |
| shared layout / swizzle 是否有 bank conflict | Workgroup Memory 的 `average conflict cycles per instruction` |
| 小 batch 是否打不满 SM（要不要 split-KV） | Occupancy 的 `Achieved waves` vs `Dispatched waves`；`WORKGROUPS` 对比 104 个 SM |
| BF16 MMA 是否真被用上 | `AP MMA Duty ratio` |
| 访存延迟是否被掩盖 | `Dnoc Read Average Latency` + `Dnoc Read Latency Histogram` |

### 3.8 profiler 不可用时的兜底

如果 profiler 太慢或指标语义拿不准，用 **逐层 ablation**
（[`optimization.md:473`](optimization.md) 的建议）：固定同一形状，每次只切换一个开关
（是否共享 K/V、是否 MMA、是否双缓冲、是否向量化），从延迟变化反推瓶颈。
在指标含义不确定时，这个办法反而更可靠。

---

## 4. 踩坑清单

| 现象 | 原因 | 解法 |
|---|---|---|
| `mxcc: fatal error: 'cuda_runtime.h' file not found` | mxcc 不自动带 cu-bridge include | 加 `-I/opt/maca-3.7.1/tools/cu-bridge/include -I/opt/maca/include` |
| `fatal error: '__macro_mxcc.h' file not found` | 缺 imacros | `-Xclang -imacros -Xclang /opt/maca-3.7.1/tools/cu-bridge/include/__macro_mxcc.h` |
| `mxcc: error: unknown argument: '-imacros xxx'` | mxcc 不直接认 `-imacros` | 必须用 `-Xclang` 转发 |
| `undefined reference to wcudaMalloc` | 没链接运行时 | `-L/opt/maca/lib -lruntime_cu -lmcruntime` |
| `libruntime_cu.so: DSO missing from command line` | 链了 `cu-bridge/lib/libcuda.so` | 换成 `-lruntime_cu -lmcruntime` |
| `mcProfiler: No such file or directory: 'config'` | cwd 不对 | `cd /opt/mcProfiler-ubuntu18.04` 后再运行 |
| `clean server failed` | 残留 profiler 进程 | `pkill profiler_server` |
| mcProfiler 跑很久不结束 | 全指标多遍重放 | 加 `--kernelnames` + `--counts 5`，或 `--single-pass` |
| `mxvs`: `dlopen(): libfuse.so.2` | AppImage 缺 FUSE | `/opt/maca/bin/mxvs --appimage-extract` 后跑解出来的可执行文件 |
| kernel 结果静默错误 | 按 warpSize=32 写的归约 / mask | C500 是 **64**，检查 shuffle offset 和 mask 位宽 |
