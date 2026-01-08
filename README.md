# Benchmark Tools

Triton Conv3D 性能基准测试工具集

## 项目简介

本项目提供了基于 Triton 的 3D 卷积高性能实现和完整的性能基准测试工具。

## 主要功能

- **高性能 3D 卷积**: 基于 Triton 的优化实现,支持 NVIDIA 和 AMD GPU
- **性能基准测试**: 对比 Triton 和 PyTorch 原生实现的性能
- **真实场景测试**: 基于实际模型运行数据的测试配置
- **详细报告生成**: 支持 JSON、CSV 和 Markdown 格式的性能报告

## 项目结构

```
benchmark_tools/
├── conv3d/                    # Conv3D Kernel 实现和测试
│   ├── __init__.py           # 包初始化文件
│   ├── triton_conv3d.py      # Triton 3D 卷积核心实现
│   └── benchmark_conv3d.py   # 性能基准测试脚本
├── .gitignore
└── README.md
```

### 文件说明

- `conv3d/triton_conv3d.py`: Triton 3D 卷积核心实现
- `conv3d/benchmark_conv3d.py`: 性能基准测试脚本
- 更多 Kernel 实现即将添加...

## 依赖要求

```bash
torch
triton
```

## 使用方法

### 基本测试(使用固定配置)

```bash
python -m conv3d.benchmark_conv3d
```

或者直接运行:

```bash
cd conv3d
python benchmark_conv3d.py
```

### 启用 Autotune 测试

```bash
python -m conv3d.benchmark_conv3d --autotune
```

### 指定输出目录

```bash
python -m conv3d.benchmark_conv3d --output-dir ./results
```

### 更多选项

```bash
python -m conv3d.benchmark_conv3d --help
```

### 作为 Python 包使用

```python
from conv3d import triton_conv3d
import torch

# 创建输入数据
input_tensor = torch.randn(1, 64, 8, 64, 64, device='cuda')
weight = torch.randn(128, 64, 3, 3, 3, device='cuda')
bias = torch.randn(128, device='cuda')

# 使用固定配置(推理时推荐)
output = triton_conv3d(input_tensor, weight, bias, use_autotune=False)

# 使用 autotune(训练时推荐)
output = triton_conv3d(input_tensor, weight, bias, use_autotune=True)
```

## 环境变量

- `TRITON_LOG_LEVEL`: 日志级别 (0=关闭, 1=基本, 2=详细, 3=调试)
- `USE_AMD_TRITON_CONFIGS`: 启用 AMD 优化配置 (0/1)
- `COLLECT_CONV3D_DATA`: 启用数据收集 (0/1)
- `TRITON_PRINT_AUTOTUNING`: 显示 autotune 过程 (0/1)

## 性能特性

- 支持 Fixed Config 和 Autotune 两种模式
- 支持分组卷积 (groups)
- 支持自定义 stride、padding、dilation
- 自动内存优化和计算优化

## 输出报告

测试完成后会生成以下文件:
- `benchmark_real_world_YYYYMMDD_HHMMSS.json`: 详细的 JSON 数据
- `benchmark_real_world_YYYYMMDD_HHMMSS.csv`: CSV 格式数据
- `benchmark_real_world_YYYYMMDD_HHMMSS.md`: Markdown 格式报告


## 贡献

欢迎提交 Issue 和 Pull Request!
