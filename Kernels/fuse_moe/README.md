# Fused MOE Performance Testing

为ROCm/AMD GPU优化的融合专家混合(MOE)操作测试、基准测试和调优的测试脚本。

## Overview

该脚本针对FusedMoE 提供kernel性能分析、正确性验证和调优功能。支持多种量化方案（FP16、FP8、INT8）。

## Architecture

```
fuse_moe/
├── __init__.py              # 主模块导出
├── README.md               # 本文档
├── test_moe.py             # 主测试脚本
├── benchmarks/             # 性能基准测试和调优
│   ├── __init__.py
│   ├── kernel_runner.py     # 运行 kernel + CUDA Graph 计时
│   ├── tuner.py             # 搜索空间生成 + 最优 config 搜索
│   └── perf_analyzer.py     # 理论性能指标分析（TFLOPS、带宽、效率）
└── utils/                   # 辅助函数
    ├── __init__.py
    ├── correctness_checker.py # 正确性验证
    ├── helper.py              # 辅助函数
    ├── perf_metrics.py        # 性能指标计算（FLOPs + 内存估算）
    └── inputs_builder.py      # 输入生成和数据准备
```

## Modules

### Main Script
- **test_moe.py**: 所有测试模式的命令行接口，提供基准测试、调优、正确性检查和性能分析功能的统一访问入口。

### Benchmark Modules

- **kernel_runner.py**: 运行 kernel 并通过 CUDA Graph 计时，处理不同配置和量化方案。
- **tuner.py**: 搜索空间生成、config 配置排除规则和最优配置搜索。
- **perf_analyzer.py**: 理论性能指标分析，包括 TFLOPS、带宽和效率计算。

### Utility Modules

- **correctness_checker.py**: 正确性验证
- **perf_metrics.py**: 性能指标计算，包括TFLOPS估算和内存读写估算
- **inputs_builder.py**: 模型输入生成和数据准备工具
- **helper.py**: 辅助函数和工具

## 使用方法

### 命令行接口

主入口是`test_moe.py`：

```bash
# 基础用法 - 运行正确性测试
python test_moe.py --mode correctness --type fp16

# 多batch 的profile
python test_moe.py --mode profile --batch-sizes "128,256,512,1024" --type fp8_w8a8

# autotune best config
python test_moe.py --mode tune --batch-sizes "512,1024" --save-dir my_tuned_configs

# 使用现有配置进行benchmark
python test_moe.py --mode benchmark --config-dir my_tuned_configs --type int8_w8a16

# 高级选项
python test_moe.py --mode tune \
    --batch-sizes "1024,2048" \
    --num-experts 32 \
    --hidden-size 7168 \
    --shard-intermediate-size 4096 \
    --topk 8 \
    --type fp8_w8a8 \
    --block-shape 128,128 \
    --device 0
```

### 编程接口

```python
from fuse_moe.benchmarks import BenchmarkWorker, ConfigTuner
from fuse_moe.utils import correctness_checker, profile_moe

# 初始化组件
worker = BenchmarkWorker(seed=42)
tuner = ConfigTuner(seed=42)

# 运行性能基准测试
config, kernel_time = worker.benchmark(
    num_tokens=1024,
    num_experts=32,
    shard_intermediate_size=4096,
    hidden_size=7168,
    topk=8,
    dtype=torch.float16,
    use_fp8_w8a8=True,
    block_shape=[128, 128]
)

# 配置调优
search_space = tuner.get_configs_compute_bound()
best_config = tuner.tune(
    num_tokens=1024,
    hidden_size=7168,
    num_experts=32,
    shard_intermediate_size=4096,
    topk=8,
    dtype=torch.float16,
    search_space=search_space,
    use_fp8_w8a8=True
)

# 正确性验证
correctness_checker.correctness(
    num_tokens=512,
    num_experts=16,
    shard_intermediate_size=2048,
    hidden_size=4096,
    topk=4,
    dtype=torch.float16,
    use_fp8_w8a8=True
)
```
