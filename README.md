# Benchmark Tools

Triton Conv3D 高性能实现与基准测试工具

## 功能

- 基于 Triton 的 3D 卷积实现，支持 NVIDIA/AMD GPU
- 性能基准测试，对比 Triton vs PyTorch
- 真实场景测试配置（10个典型 case）
- 可选的结果导出（JSON/CSV/Markdown）

## 项目结构

```text
benchmark_tools/
└── conv3d/
    ├── triton_conv3d.py      # Triton 3D 卷积实现
    ├── benchmark_conv3d.py   # 性能测试脚本
    └── __init__.py
```

## 快速开始

### 依赖

```bash
pip install torch triton
```

### 基准测试

```bash
# 基本测试（固定配置）
python benchmark_conv3d.py

# 启用 Autotune（显示调优过程）
python benchmark_conv3d.py --autotune

# 保存结果到文件
python benchmark_conv3d.py --save-results --output-dir ./results

# 更多选项
python benchmark_conv3d.py --help
```

### 在代码中使用

```python
from conv3d import triton_conv3d
import torch

input_tensor = torch.randn(1, 64, 8, 64, 64, device='cuda')
weight = torch.randn(128, 64, 3, 3, 3, device='cuda')
bias = torch.randn(128, device='cuda')

# 固定配置（推理推荐）
output = triton_conv3d(input_tensor, weight, bias, use_autotune=False)

# Autotune（训练推荐）
output = triton_conv3d(input_tensor, weight, bias, use_autotune=True)
```

## 环境变量

- `USE_AMD_TRITON_CONFIGS`: 启用 AMD 优化配置 (0/1)
- `TRITON_PRINT_AUTOTUNING`: 显示 autotune 过程 (0/1)

## 特性

- Fixed Config / Autotune 模式
- 支持 groups、stride、padding、dilation
- 使用 `torch.allclose` 验证正确性
- 动态生成配置名称
