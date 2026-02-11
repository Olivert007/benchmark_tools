# Benchmark Tools

Triton GPU Kernel 性能基准测试与深度剖析工具集。

---

## 项目简介

本仓库提供两大核心能力：

1. **Triton Kernel 实现与基准测试**（`Kernels/`）
   - 基于 [Triton](https://github.com/triton-lang/triton) 的高性能 GPU Kernel 实现
   - 自动化性能基准测试框架，支持 Triton vs PyTorch 对比、正确性验证、多格式结果导出

2. **GPU Profiling 工具链**（`profile/`）
   - 封装 ROCm `rocprofv3` 的 ATT（Advanced Thread Trace）和 PMC（Performance Monitoring Counter）追踪流程
   - Triton 编译器中间表示（MLIR / LLVM IR / AMDGCN）导出工具
   - 自动记录 Triton 版本、LLVM commit、环境信息，方便实验复现

## 项目结构

```text
benchmark_tools/
├── README.md
├── Kernels/
│   ├── README.md
│   ├── conv3d/                    # Triton 3D 卷积 Kernel
│   │   ├── triton_conv3d.py       # Kernel 实现
│   │   ├── benchmark_conv3d.py    # 基准测试脚本
│   │   └── __init__.py
│   └── fuse_moe/                  # 融合专家混合 (Fused MOE) Kernel
│       ├── README.md
│       ├── test_fusemoe.py        # 主测试脚本（benchmark / tune / correctness / profile）
│       ├── run_fusemoe.sh         # 快捷运行脚本
│       ├── benchmarks/            # 性能基准测试与调优模块
│       └── utils/                 # 辅助工具
└── profile/
    ├── README.md
    ├── rocprofv3_att_trace.sh     # ATT 追踪执行脚本
    ├── rocprofv3_pmc_trace.sh     # PMC 追踪执行脚本
    ├── triton_ir_asm_dump.sh      # Triton 编译器 IR/ASM 导出脚本
    ├── conv3d_att_trace.yaml      # ATT 追踪配置
    └── conv3d_pmc_trace.yaml      # PMC 追踪配置
```

## 已支持的 Kernel

| Kernel | 路径 | 说明 |
|--------|------|------|
| **Conv3D** | `Kernels/conv3d/` | Triton 3D 卷积，支持 stride / padding / dilation / groups，提供 Fixed Config 和 Autotune 两种模式 |
| **Fused MOE** | `Kernels/fuse_moe/` | 融合专家混合操作，支持 FP16 / FP8 / INT8 量化，提供 benchmark、tune、correctness、profile 四种运行模式 |

> 各 Kernel 的详细用法请参阅对应目录下的 README。

## Quick Start

### 环境依赖

```bash
pip install torch triton
# PMC/ATT 追踪还需要 ROCm 环境（rocprofv3、amd-smi）
```

### Kernel 基准测试

```bash
# Conv3D 基准测试
cd Kernels/conv3d
python benchmark_conv3d.py                # Fixed Config 模式
python benchmark_conv3d.py --autotune     # Autotune 模式

# Fused MOE 基准测试
cd Kernels/fuse_moe
python test_fusemoe.py --mode benchmark --type fp16
python test_fusemoe.py --mode tune --batch-sizes "512,1024"   # 配置调优
```

### GPU Profiling

```bash
cd profile

# ATT 追踪（指令级热点分析）
./rocprofv3_att_trace.sh -f <script.py> -c <trace.yaml> -o ./output_att

# PMC 追踪（硬件计数器分析，自动锁定 GPU 频率）
./rocprofv3_pmc_trace.sh -f <script.py> -c <trace.yaml> -o ./output_pmc

# Triton IR/ASM 导出
./triton_ir_asm_dump.sh -f <script.py> -o ./output_dump
```

## 环境变量参考

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `USE_AMD_TRITON_CONFIGS` | 启用 AMD 优化 Autotune 配置空间 | `0` |
| `TRITON_PRINT_AUTOTUNING` | 打印 Autotune 调优过程 | `0` |
| `TRITON_VERBOSE` | Triton 日志级别（1-4） | 未设置 |
| `ROCR_VISIBLE_DEVICES` | 选择可见的 GPU 设备 | 未设置 |
| `TRITON_ALWAYS_COMPILE` | 强制重新编译 Kernel | `0` |
| `TRITON_CACHE_DIR` | Triton 编译缓存目录 | `~/.triton` |

## License

MIT
