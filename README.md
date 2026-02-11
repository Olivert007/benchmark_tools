# Benchmark Tools

Triton GPU Kernel 性能基准测试与深度剖析工具集。

---

## What — 这是什么

本仓库提供两大核心能力：

1. **Triton Kernel 实现与基准测试**（`Kernels/`）
   - 基于 [Triton](https://github.com/triton-lang/triton) 的高性能 GPU Kernel 实现（当前包含 Conv3D）
   - 自动化性能基准测试框架，支持 Triton vs PyTorch 对比、正确性验证、多格式结果导出

2. **GPU Profiling 工具链**（`profile/`）
   - 封装 ROCm `rocprofv3` 的 ATT（Advanced Thread Trace）和 PMC（Performance Monitoring Counter）追踪流程
   - Triton 编译器中间表示（MLIR/LLVM IR/AMDGCN）导出工具
   - 自动记录 Triton 版本、LLVM commit、环境信息，方便实验复现

## When — 什么时候用

| 场景 | 推荐工具 |
|------|----------|
| 想快速评估 Triton Conv3D 与 PyTorch 的性能差距 | `Kernels/conv3d/benchmark_conv3d.py` |
| 在自己的代码中集成 Triton Conv3D | `from conv3d import triton_conv3d` |
| 需要逐指令级别分析 Kernel 热点（Stall/Latency） | `profile/rocprofv3_att_trace.sh` + ATT YAML |
| 需要硬件计数器数据（Cache Hit、Occupancy、LDS 冲突等） | `profile/rocprofv3_pmc_trace.sh` + PMC YAML |
| 需要查看 Triton 编译器生成的 MLIR / LLVM IR / AMDGCN 汇编 | `profile/triton_ir_asm_dump.sh` |

## How — 项目结构与原理

```text
benchmark_tools/
├── README.md                          # 本文档
├── .gitignore
├── Kernels/
│   ├── README.md                      # Kernel 子模块文档
│   └── conv3d/
│       ├── __init__.py                # 包入口，导出 triton_conv3d
│       ├── triton_conv3d.py           # Triton 3D 卷积 Kernel 实现
│       └── benchmark_conv3d.py        # 性能基准测试脚本
└── profile/
    ├── conv3d_att_trace.yaml          # ATT 追踪配置（线程级指令追踪）
    ├── conv3d_pmc_trace.yaml          # PMC 追踪配置（10 pass 硬件计数器）
    ├── rocprofv3_att_trace.sh         # ATT 追踪执行脚本（含 PC Sampling 支持）
    ├── rocprofv3_pmc_trace.sh         # PMC 追踪执行脚本（含 GPU 频率锁定）
    └── triton_ir_asm_dump.sh          # Triton 编译器 IR/ASM 导出脚本
```

### Kernels — Triton Conv3D

- **triton_conv3d.py**：独立可用的 Triton 3D 卷积实现，支持 `stride`、`padding`、`dilation`、`groups`，提供 Fixed Config（低延迟）和 Autotune（自动寻优）两种模式。通过环境变量 `USE_AMD_TRITON_CONFIGS=1` 可切换到 AMD 优化配置空间。
- **benchmark_conv3d.py**：内置 10 个真实场景 shape 配置（源自实际模型调用统计），使用 `triton.testing.do_bench` 进行基准测试，支持导出 JSON / CSV / Markdown 报告。

### Profile — ROCm 深度剖析

- **ATT 追踪**：指令级线程追踪，可获取每条汇编指令的 Hitcount、Latency、Stall、Idle，用于定位 Kernel 热点。
- **PMC 追踪**：通过 10 个 pass 采集约 50+ 项硬件计数器（涵盖 VALU/SALU 指令数、L2 Cache 命中率、LDS 冲突、Occupancy、显存读写带宽等）。执行前自动将 GPU 设为 `stable_std` 模式锁定频率，结束后自动恢复。
- **IR/ASM 导出**：通过设置 `MLIR_ENABLE_DUMP`、`LLVM_IR_ENABLE_DUMP`、`AMDGCN_ENABLE_DUMP` 环境变量，导出 Triton 编译全链路的中间表示，方便分析编译器优化行为。

---

## Quick Start

### 环境依赖

```bash
pip install torch triton
# PMC/ATT 追踪还需要 ROCm 环境（rocprofv3、amd-smi）
```

### 1. 运行基准测试

```bash
cd Kernels/conv3d

# 基本测试：Fixed Config 模式，对比 Triton vs PyTorch
python benchmark_conv3d.py

# 启用 Autotune（会测试 Fixed + Autotune 两种模式）
python benchmark_conv3d.py --autotune

# 使用 bfloat16，跳过 PyTorch 基线
python benchmark_conv3d.py --dtype bfloat16 --no-torch-baseline

# 保存结果到文件（JSON + CSV + Markdown）
python benchmark_conv3d.py --save-results --output-dir ./results

# 查看全部选项
python benchmark_conv3d.py --help
```

### 2. 在代码中使用 Triton Conv3D

```python
from conv3d import triton_conv3d
import torch

input_tensor = torch.randn(1, 64, 8, 64, 64, device='cuda')
weight = torch.randn(128, 64, 3, 3, 3, device='cuda')
bias = torch.randn(128, device='cuda')

# Fixed Config（推理推荐，启动快）
output = triton_conv3d(input_tensor, weight, bias, use_autotune=False)

# Autotune（训练推荐，首次慢但后续更快）
output = triton_conv3d(input_tensor, weight, bias, use_autotune=True)
```

### 3. ATT 追踪（指令级热点分析）

```bash
cd profile

# 对 Conv3D 验证脚本进行 ATT 追踪
./rocprofv3_att_trace.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -c ./conv3d_att_trace.yaml \
    -o ./output_att

# 指定 GPU、启用 Triton 详细日志
./rocprofv3_att_trace.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -c ./conv3d_att_trace.yaml \
    -o ./output_att \
    -d 0 -v 1

# 启用 PC Sampling（Beta）
./rocprofv3_att_trace.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -c ./conv3d_att_trace.yaml \
    -o ./output_att \
    -p -m host_trap -u time -i 1

# 传递参数给 Python 脚本（用 -- 分隔）
./rocprofv3_att_trace.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -c ./conv3d_att_trace.yaml \
    -o ./output_att \
    -- --no-torch-baseline --repeat-runs 30
```

### 4. PMC 追踪（硬件计数器分析）

```bash
cd profile

# 运行 PMC 追踪（自动锁定 GPU 频率）
./rocprofv3_pmc_trace.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -c ./conv3d_pmc_trace.yaml \
    -o ./output_pmc

# 指定 GPU 索引
./rocprofv3_pmc_trace.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -c ./conv3d_pmc_trace.yaml \
    -o ./output_pmc \
    -g 0

# 清除已有输出后重新追踪
./rocprofv3_pmc_trace.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -c ./conv3d_pmc_trace.yaml \
    -o ./output_pmc \
    --clear
```

### 5. Triton IR/ASM 导出

```bash
cd profile

# 导出编译器中间表示（MLIR / LLVM IR / AMDGCN）
./triton_ir_asm_dump.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -o ./output_dump

# 指定 GPU 和详细级别
./triton_ir_asm_dump.sh \
    -f ../Kernels/conv3d/benchmark_conv3d.py \
    -o ./output_dump \
    -d 0 -v 4
```

---

## 环境变量参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `USE_AMD_TRITON_CONFIGS` | 启用 AMD 优化 Autotune 配置空间 | `0` |
| `TRITON_PRINT_AUTOTUNING` | 打印 Autotune 调优过程 | `0` |
| `TRITON_VERBOSE` | Triton 日志级别（1-4） | 未设置 |
| `ROCR_VISIBLE_DEVICES` | 选择可见的 GPU 设备 | 未设置 |
| `TRITON_ALWAYS_COMPILE` | 强制重新编译 Kernel（IR/ASM 导出时推荐） | `0` |
| `TRITON_CACHE_DIR` | Triton 编译缓存目录 | 默认 `~/.triton` |

## License

MIT
