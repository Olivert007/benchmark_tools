# Kernels

Triton GPU Kernel 实现与基准测试集合。

## 已支持的 Kernel

| Kernel | 目录 | 说明 |
|--------|------|------|
| **Conv3D** | `conv3d/` | Triton 3D 卷积实现，支持 stride / padding / dilation / groups，提供 Fixed Config 和 Autotune 两种模式 |
| **Fused MOE** | `fuse_moe/` | 融合专家混合操作，支持 FP16 / FP8 / INT8 量化，提供 benchmark、tune、correctness、profile 多种运行模式 |

## 依赖

```bash
pip install torch triton
```

## 快速开始

### Conv3D

```bash
cd conv3d

# 基本测试（Fixed Config）
python benchmark_conv3d.py

# 启用 Autotune
python benchmark_conv3d.py --autotune

# 保存结果
python benchmark_conv3d.py --save-results --output-dir ./results
```

### Fused MOE

```bash
cd fuse_moe

# 正确性验证
python test_fusemoe.py --mode correctness --type fp16

# 性能基准测试
python test_fusemoe.py --mode benchmark --type fp8_w8a8

# 配置调优
python test_fusemoe.py --mode tune --batch-sizes "512,1024"
```

> 各 Kernel 的详细文档请参阅对应目录下的 README。
