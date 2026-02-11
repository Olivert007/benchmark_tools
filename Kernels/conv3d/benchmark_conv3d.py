#!/usr/bin/env python3
"""
Triton Conv3D 性能基准测试脚本

功能：
- 批量测试不同的 shape 配置
- 对比 fixed config vs autotune 性能
- 可选：生成详细的性能报告（JSON/CSV/Markdown）

使用方法：
1. 基本测试（使用真实场景数据，默认 Fixed Config）：
   python benchmark_conv3d.py

2. 启用 Autotune 测试（会测试 Fixed + Autotune 两种模式，并显示 tune 过程）：
   python benchmark_conv3d.py --autotune

3. 保存测试结果到文件：
   python benchmark_conv3d.py --save-results

4. 指定输出目录和保存结果：
   python benchmark_conv3d.py --save-results --output-dir ./results
"""

import os
import sys
import argparse
import json
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from triton.testing import do_bench

# 添加当前目录到路径（独立运行）
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

try:
    # 尝试从包导入
    from .triton_conv3d import (
        triton_conv3d, 
        is_triton_available,
    )
except ImportError:
    # 独立运行时的导入
    from triton_conv3d import (
        triton_conv3d, 
        is_triton_available,
    )


# ========== 测试配置预设 ==========

def get_real_world_shapes() -> List[Dict[str, Any]]:
    """
    从实际运行数据提取的真实 Conv3D 场景
    基于模型实际执行过程中的调用统计
    
    数据来源: conv3d_call_data.json (1237次调用, 27种唯一配置)
    选择标准:
    - 高频调用场景 (前3名)
    - 大空间尺寸场景 (>=640x640)
    - 大通道数场景 (>=192)
    - 多样化的 depth (1, 2, 3, 4, 5, 6, 21)
    - 不同的 kernel size (1x1x1, 3x3x3, 3x1x1)
    - Stride=2 下采样场景
    """
    return [
        # Case 1: 最高频场景 - 378次调用, 平均1.56ms
        # 中等通道, 小空间, 小depth
        {
            'batch_size': 1,
            'in_channels': 384,
            'out_channels': 384,
            'depth': 3,
            'height': 162,
            'width': 92,
            'kernel_size': (3, 3, 3),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 378次调用, 1.56ms, 114.66 GFLOPs
        },
        
        # Case 2: 第二高频 - 200次调用, 平均3.26ms
        # 大空间尺寸 (1280x720), 中等通道
        {
            'batch_size': 1,
            'in_channels': 96,
            'out_channels': 96,
            'depth': 6,
            'height': 1282,
            'width': 722,
            'kernel_size': (3, 3, 3),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 200次调用, 3.26ms, 1834.59 GFLOPs, 1.69 GB
        },
        
        # Case 3: 第三高频 - 180次调用, 平均3.44ms
        # 大通道数 (192), 大空间 (640x360)
        {
            'batch_size': 1,
            'in_channels': 192,
            'out_channels': 192,
            'depth': 6,
            'height': 642,
            'width': 362,
            'kernel_size': (3, 3, 3),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 180次调用, 3.44ms, 1834.59 GFLOPs, 850 MB
        },
        
        # Case 4: 输入通道很小 - 20次调用, 平均16.99ms
        # 典型的第一层 Conv (3 channels -> 96)
        {
            'batch_size': 1,
            'in_channels': 3,
            'out_channels': 96,
            'depth': 6,
            'height': 1282,
            'width': 722,
            'kernel_size': (3, 3, 3),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 20次调用, 16.99ms, 57.33 GFLOPs
        },
        
        # Case 5: 1x1x1 卷积 - 通道扩张
        # Depth=1, 大空间, 1x1x1 kernel
        {
            'batch_size': 1,
            'in_channels': 192,
            'out_channels': 384,
            'depth': 1,
            'height': 320,
            'width': 180,
            'kernel_size': (1, 1, 1),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 2次调用, 164.60ms, 8.49 GFLOPs
        },
        
        # Case 6: 大 depth - 1次调用, 平均315.85ms
        # Depth=21, 小通道, 1x1x1 kernel
        {
            'batch_size': 1,
            'in_channels': 32,
            'out_channels': 32,
            'depth': 21,
            'height': 160,
            'width': 90,
            'kernel_size': (1, 1, 1),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 1次调用, 315.85ms, 0.62 GFLOPs
        },
        
        # Case 7: Stride=2 下采样 - 20次调用, 平均18.55ms
        # 3x1x1 kernel, stride 2 in depth
        {
            'batch_size': 1,
            'in_channels': 192,
            'out_channels': 192,
            'depth': 5,
            'height': 320,
            'width': 180,
            'kernel_size': (3, 1, 1),
            'stride': (2, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 20次调用, 18.55ms, 25.48 GFLOPs
        },
        
        # Case 8: 高通道 3x3x3 - 160次调用, 平均3.52ms
        # 384 channels, 中等空间
        {
            'batch_size': 1,
            'in_channels': 384,
            'out_channels': 384,
            'depth': 4,
            'height': 322,
            'width': 182,
            'kernel_size': (3, 3, 3),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 160次调用, 3.52ms, 917.29 GFLOPs
        },
        
        # Case 9: 通道扩张 192->384 - 40次调用, 平均9.64ms
        # 3x3x3 kernel, 中等空间
        {
            'batch_size': 1,
            'in_channels': 192,
            'out_channels': 384,
            'depth': 4,
            'height': 322,
            'width': 182,
            'kernel_size': (3, 3, 3),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 40次调用, 9.64ms, 458.65 GFLOPs
        },
        
        # Case 10: 小depth 1x1x1通道扩张 - 40次调用, 平均10.19ms
        # Depth=2, 1x1x1 kernel, 192->384
        {
            'batch_size': 1,
            'in_channels': 192,
            'out_channels': 384,
            'depth': 2,
            'height': 320,
            'width': 180,
            'kernel_size': (1, 1, 1),
            'stride': (1, 1, 1),
            'padding': (0, 0, 0),
            'groups': 1,
            # 统计: 40次调用, 10.19ms, 16.99 GFLOPs
        },
    ]


# ========== 测试执行 ==========

def generate_config_name(config: Dict[str, Any]) -> str:
    """根据配置动态生成名称"""
    return (f"N{config['batch_size']}_"
            f"C{config['in_channels']}->{config['out_channels']}_"
            f"DHW{config['depth']}x{config['height']}x{config['width']}_"
            f"K{config['kernel_size'][0]}x{config['kernel_size'][1]}x{config['kernel_size'][2]}_"
            f"S{config['stride'][0]}x{config['stride'][1]}x{config['stride'][2]}_"
            f"G{config.get('groups', 1)}")


def create_test_tensors(config: Dict[str, Any], device: str, dtype: torch.dtype):
    """创建测试用的输入张量"""
    batch_size = config['batch_size']
    in_channels = config['in_channels']
    out_channels = config['out_channels']
    depth = config['depth']
    height = config['height']
    width = config['width']
    kernel_size = config['kernel_size']
    groups = config.get('groups', 1)
    
    # 创建输入
    input_tensor = torch.randn(
        batch_size, in_channels, depth, height, width,
        device=device, dtype=dtype
    )
    
    # 创建权重
    weight = torch.randn(
        out_channels, in_channels // groups, 
        kernel_size[0], kernel_size[1], kernel_size[2],
        device=device, dtype=dtype
    )
    
    # 创建偏置
    bias = torch.randn(out_channels, device=device, dtype=dtype)
    
    return input_tensor, weight, bias


def run_single_test(
    config: Dict[str, Any],
    device: str,
    dtype: torch.dtype,
    use_autotune: bool,
    warmup_runs: int = 3,
    repeat_runs: int = 10,
    test_torch_baseline: bool = True,
) -> Dict[str, Any]:
    """
    运行单个配置的性能测试
    
    Args:
        config: 测试配置
        device: 设备 ('cuda' 或 'cpu')
        dtype: 数据类型
        use_autotune: 是否使用 autotune
        warmup_runs: 预热运行次数（do_bench 自动处理，此参数仅用于记录）
        repeat_runs: 重复运行次数（传递给 do_bench 的 rep 参数）
        test_torch_baseline: 是否测试 PyTorch 基准
        
    Returns:
        测试结果字典
    """
    config_name = generate_config_name(config)
    
    print(f"\n{'='*80}")
    print(f"Testing: {config_name}")
    print(f"  Shape: N={config['batch_size']}, C={config['in_channels']}->{config['out_channels']}, "
          f"THW={config['depth']}x{config['height']}x{config['width']}")
    print(f"  Kernel: {config['kernel_size']}, Stride: {config['stride']}, "
          f"Padding: {config['padding']}, Groups: {config.get('groups', 1)}")
    print(f"  Mode: {'Autotune' if use_autotune else 'Fixed Config'}")
    print(f"{'='*80}")
    
    # 创建测试数据
    input_tensor, weight, bias = create_test_tensors(config, device, dtype)
    
    stride = config.get('stride', (1, 1, 1))
    padding = config.get('padding', (0, 0, 0))
    dilation = config.get('dilation', (1, 1, 1))
    groups = config.get('groups', 1)
    
    result = {
        'config_name': config_name,
        'config': config,
        'device': device,
        'dtype': str(dtype),
        'use_autotune': use_autotune,
        'warmup_runs': warmup_runs,
        'repeat_runs': repeat_runs,
    }
    
    # ========== Triton 测试 ==========
    print(f"⏱️  Running Triton benchmark (using do_bench with automatic warmup)...")
    
    # 使用 Triton 的 do_bench 进行基准测试
    # do_bench 自动处理预热、同步和多次运行
    triton_mean = do_bench(lambda: triton_conv3d(
        input_tensor, weight, bias,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
        use_autotune=use_autotune
    ), rep=repeat_runs)
    
    # 获取输出用于验证
    output_triton = triton_conv3d(
        input_tensor, weight, bias,
        stride=stride, padding=padding, dilation=dilation, groups=groups,
        use_autotune=use_autotune
    )
    
    result['triton'] = {
        'mean_ms': round(triton_mean, 3),
    }
    
    print(f"✓ Triton: {triton_mean:.3f} ms")
    
    # ========== PyTorch 基准测试 ==========
    if test_torch_baseline:
        print(f"⏱️  Running PyTorch benchmark (using do_bench with automatic warmup)...")
        
        # 使用 Triton 的 do_bench 进行基准测试
        # do_bench 自动处理预热、同步和多次运行
        torch_mean = do_bench(lambda: F.conv3d(
            input_tensor, weight, bias,
            stride=stride, padding=padding, dilation=dilation, groups=groups
        ), rep=repeat_runs)
        
        # 获取输出用于验证
        output_torch = F.conv3d(
            input_tensor, weight, bias,
            stride=stride, padding=padding, dilation=dilation, groups=groups
        )
        
        result['pytorch'] = {
            'mean_ms': round(torch_mean, 3),
        }
        
        print(f"✓ PyTorch: {torch_mean:.3f} ms")
        
        # 计算加速比
        speedup = triton_mean / torch_mean
        result['speedup'] = round(speedup, 2)
        print(f"📊 Speedup (T/P): {speedup:.2f}x {'🚀 Triton faster' if speedup < 1 else '⚠️ PyTorch faster'}")
        
        # 验证正确性 - 使用 torch.allclose
        # rtol: 相对容差, atol: 绝对容差
        # 公式: |output_triton - output_torch| <= atol + rtol * |output_torch|
        outputs_match = torch.allclose(output_triton, output_torch, rtol=1e-5, atol=1e-5)
        
        # 计算差异统计信息（用于调试）
        max_diff = torch.abs(output_triton - output_torch).max().item()
        mean_diff = torch.abs(output_triton - output_torch).mean().item()
        
        result['max_difference'] = max_diff
        result['mean_difference'] = mean_diff
        result['outputs_match'] = outputs_match
        
        status = "✅ PASS" if outputs_match else "❌ FAIL"
        print(f"✓ Correctness: {status} (max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e})")
    
    # 计算额外指标
    output_elements = output_triton.numel()
    throughput = output_elements / (triton_mean / 1000) / 1e9  # G elements/sec
    result['output_elements'] = output_elements
    result['throughput_gelements_per_sec'] = round(throughput, 2)
    
    # 计算 FLOPs
    flops = (
        2 * config['batch_size'] * config['out_channels']
        * (output_triton.shape[2] * output_triton.shape[3] * output_triton.shape[4])
        * (config['in_channels'] // config.get('groups', 1))
        * (config['kernel_size'][0] * config['kernel_size'][1] * config['kernel_size'][2])
    )
    result['flops'] = flops
    result['gflops'] = round(flops / 1e9, 2)
    result['tflops_per_sec'] = round(flops / (triton_mean / 1000) / 1e12, 2)
    
    print(f"📈 Throughput: {throughput:.2f} G elements/sec")
    print(f"📊 Performance: {result['tflops_per_sec']:.2f} TFLOPS/s")
    
    return result


def run_benchmark_suite(
    configs: List[Dict[str, Any]],
    device: str = 'cuda',
    dtype: torch.dtype = torch.float32,
    test_both_modes: bool = True,
    test_torch_baseline: bool = True,
    warmup_runs: int = 3,
    repeat_runs: int = 10,
) -> List[Dict[str, Any]]:
    """
    运行一系列基准测试
    
    Args:
        configs: 测试配置列表
        device: 测试设备
        dtype: 数据类型
        test_both_modes: 是否测试 fixed 和 autotune 两种模式
        test_torch_baseline: 是否测试 PyTorch 基准
        warmup_runs: 预热次数（do_bench 自动处理，此参数仅用于记录）
        repeat_runs: 重复运行次数（传递给 do_bench 的 rep 参数）
        
    Returns:
        所有测试结果列表
    """
    results = []
    total_tests = len(configs) * (2 if test_both_modes else 1)
    current_test = 0
    
    print(f"\n{'='*80}")
    print(f"🎯 Starting Benchmark Suite")
    print(f"{'='*80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Test modes: {'Fixed + Autotune' if test_both_modes else 'Fixed only'}")
    print(f"Device: {device}")
    print(f"Dtype: {dtype}")
    print(f"Warmup runs: {warmup_runs}")
    print(f"Repeat runs: {repeat_runs}")
    print(f"Total tests: {total_tests}")
    print(f"{'='*80}\n")
    
    for config in configs:
        config_name = generate_config_name(config)
        
        # 测试 fixed config 模式
        current_test += 1
        print(f"\n[{current_test}/{total_tests}] Testing {config_name} (Fixed Config)")
        
        try:
            result_fixed = run_single_test(
                config, device, dtype,
                use_autotune=False,
                warmup_runs=warmup_runs,
                repeat_runs=repeat_runs,
                test_torch_baseline=test_torch_baseline,
            )
            results.append(result_fixed)
        except Exception as e:
            print(f"❌ Error in fixed config test: {e}")
            import traceback
            traceback.print_exc()
        
        # 测试 autotune 模式
        if test_both_modes:
            current_test += 1
            print(f"\n[{current_test}/{total_tests}] Testing {config_name} (Autotune)")
            
            try:
                result_autotune = run_single_test(
                    config, device, dtype,
                    use_autotune=True,
                    warmup_runs=warmup_runs,
                    repeat_runs=repeat_runs,
                    test_torch_baseline=False,  # 只在 fixed 模式测试一次
                )
                results.append(result_autotune)
            except Exception as e:
                print(f"❌ Error in autotune test: {e}")
                import traceback
                traceback.print_exc()
    
    return results


# ========== 结果导出 ==========

def export_results_json(results: List[Dict[str, Any]], filepath: str):
    """导出结果为 JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✓ Exported results to {filepath}")


def export_results_csv(results: List[Dict[str, Any]], filepath: str):
    """导出结果为 CSV"""
    import csv
    
    if not results:
        return
    
    # 展平结果
    flat_results = []
    for r in results:
        flat = {
            'config_name': r['config_name'],
            'device': r['device'],
            'dtype': r['dtype'],
            'use_autotune': r['use_autotune'],
            'batch_size': r['config']['batch_size'],
            'in_channels': r['config']['in_channels'],
            'out_channels': r['config']['out_channels'],
            'depth': r['config']['depth'],
            'height': r['config']['height'],
            'width': r['config']['width'],
            'kernel_size': str(r['config']['kernel_size']),
            'stride': str(r['config']['stride']),
            'padding': str(r['config']['padding']),
            'groups': r['config'].get('groups', 1),
            'triton_mean_ms': r['triton']['mean_ms'],
            'output_elements': r['output_elements'],
            'throughput_gelements_per_sec': r['throughput_gelements_per_sec'],
            'gflops': r['gflops'],
            'tflops_per_sec': r['tflops_per_sec'],
        }
        
        if 'pytorch' in r:
            flat['pytorch_mean_ms'] = r['pytorch']['mean_ms']
            flat['speedup'] = r.get('speedup', None)
            flat['max_difference'] = r.get('max_difference', None)
            flat['outputs_match'] = r.get('outputs_match', None)
        
        flat_results.append(flat)
    
    fieldnames = flat_results[0].keys()
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flat_results)
    
    print(f"✓ Exported results to {filepath}")


def export_results_markdown(results: List[Dict[str, Any]], filepath: str):
    """导出结果为 Markdown 报告"""
    lines = []
    lines.append("# Triton Conv3D Benchmark Results\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append(f"Total tests: {len(results)}\n")
    
    # 汇总统计
    lines.append("\n## Summary Statistics\n")
    
    if any('speedup' in r for r in results):
        speedups = [r['speedup'] for r in results if 'speedup' in r]
        lines.append(f"- Average speedup: **{sum(speedups)/len(speedups):.2f}x**\n")
        lines.append(f"- Best speedup: **{max(speedups):.2f}x**\n")
        lines.append(f"- Worst speedup: **{min(speedups):.2f}x**\n")
    
    avg_tflops = sum(r['tflops_per_sec'] for r in results) / len(results)
    lines.append(f"- Average performance: **{avg_tflops:.2f} TFLOPS/s**\n")
    
    # 详细结果表格
    lines.append("\n## Detailed Results\n")
    lines.append("| Config | Mode | Shape | Triton (ms) | PyTorch (ms) | Speedup | TFLOPS/s |\n")
    lines.append("|--------|------|-------|-------------|--------------|---------|----------|\n")
    
    for r in results:
        config_name = r['config_name']
        mode = "Autotune" if r['use_autotune'] else "Fixed"
        shape = (f"N{r['config']['batch_size']}_"
                f"C{r['config']['in_channels']}->{r['config']['out_channels']}_"
                f"THW{r['config']['depth']}x{r['config']['height']}x{r['config']['width']}")
        triton_time = f"{r['triton']['mean_ms']:.3f}"
        pytorch_time = f"{r['pytorch']['mean_ms']:.3f}" if 'pytorch' in r else "N/A"
        speedup = f"{r.get('speedup', 'N/A'):.2f}x" if 'speedup' in r else "N/A"
        tflops = f"{r['tflops_per_sec']:.2f}"
        
        lines.append(f"| {config_name} | {mode} | {shape} | {triton_time} | {pytorch_time} | {speedup} | {tflops} |\n")
    
    # 配置详情
    lines.append("\n## Configuration Details\n")
    for r in results:
        lines.append(f"\n### {r['config_name']} ({'Autotune' if r['use_autotune'] else 'Fixed'})\n")
        lines.append(f"- **Shape**: N={r['config']['batch_size']}, "
                    f"C={r['config']['in_channels']}→{r['config']['out_channels']}, "
                    f"THW={r['config']['depth']}×{r['config']['height']}×{r['config']['width']}\n")
        lines.append(f"- **Kernel**: {r['config']['kernel_size']}\n")
        lines.append(f"- **Stride**: {r['config']['stride']}\n")
        lines.append(f"- **Padding**: {r['config']['padding']}\n")
        lines.append(f"- **Groups**: {r['config'].get('groups', 1)}\n")
        lines.append(f"- **Triton Time**: {r['triton']['mean_ms']:.3f} ms\n")
        
        if 'pytorch' in r:
            lines.append(f"- **PyTorch Time**: {r['pytorch']['mean_ms']:.3f} ms\n")
            lines.append(f"- **Speedup**: {r.get('speedup', 'N/A'):.2f}x\n")
            lines.append(f"- **Correctness**: {'✅ PASS' if r.get('outputs_match') else '❌ FAIL'} "
                        f"(max_diff={r.get('max_difference', 0):.6f})\n")
        
        lines.append(f"- **Throughput**: {r['throughput_gelements_per_sec']:.2f} G elements/sec\n")
        lines.append(f"- **Compute**: {r['gflops']:.2f} GFLOPs, {r['tflops_per_sec']:.2f} TFLOPS/s\n")
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print(f"✓ Exported results to {filepath}")


# ========== 命令行接口 ==========

def main():
    parser = argparse.ArgumentParser(
        description="Triton Conv3D Performance Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to test on (default: cuda)'
    )
    
    parser.add_argument(
        '--dtype',
        type=str,
        default='bfloat16',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type (default: bfloat16)'
    )
    
    parser.add_argument(
        '--autotune',
        action='store_true',
        help='Enable autotune tests (test both fixed and autotune modes, will show tuning process)'
    )
    
    parser.add_argument(
        '--no-torch-baseline',
        action='store_true',
        help='Skip PyTorch baseline tests'
    )
    
    parser.add_argument(
        '--warmup-runs',
        type=int,
        default=3,
        help='Number of warmup runs for recording purposes (do_bench handles warmup automatically, default: 3)'
    )
    
    parser.add_argument(
        '--repeat-runs',
        type=int,
        default=10,
        help='Number of repeat runs (passed to do_bench rep parameter, default: 10)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='../benchmark_results',
        help='Output directory for results (default: ../benchmark_results)'
    )
    
    parser.add_argument(
        '--save-results',
        action='store_true',
        help='Save results to JSON/CSV/Markdown files (default: False)'
    )
    
    args = parser.parse_args()
    
    # 如果启用 autotune，自动显示 tune 过程
    if args.autotune:
        os.environ['TRITON_PRINT_AUTOTUNING'] = '1'
        os.environ['TRITON_LOG_LEVEL'] = '2'
        print("🔍 Autotune 模式已启用，将显示配置测试过程")
        print("  • TRITON_PRINT_AUTOTUNING=1")
        print("  • TRITON_LOG_LEVEL=2")
        print()
    
    # 检查环境
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("❌ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    if not is_triton_available():
        print("❌ Triton not available")
        return 1
    
    # 解析 dtype
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    # 使用真实场景测试配置
    configs = get_real_world_shapes()
    
    print(f"\n{'='*80}")
    print(f"🚀 Triton Conv3D Benchmark")
    print(f"{'='*80}")
    print(f"Preset: real_world")
    print(f"Device: {args.device}")
    print(f"Dtype: {args.dtype}")
    print(f"Configurations: {len(configs)}")
    print(f"Test modes: {'Fixed + Autotune (with tuning process)' if args.autotune else 'Fixed only'}")
    print(f"PyTorch baseline: {'No' if args.no_torch_baseline else 'Yes'}")
    print(f"Save results: {'Yes' if args.save_results else 'No'}")
    print(f"Warmup/Repeat runs: {args.warmup_runs}/{args.repeat_runs}")
    print(f"{'='*80}\n")
    
    # 运行测试
    results = run_benchmark_suite(
        configs,
        device=args.device,
        dtype=dtype,
        test_both_modes=args.autotune,
        test_torch_baseline=not args.no_torch_baseline,
        warmup_runs=args.warmup_runs,
        repeat_runs=args.repeat_runs,
    )
    
    # 导出结果（仅在指定 --save-results 时）
    if args.save_results:
        # 创建输出目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 生成时间戳
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 导出结果
        print(f"\n{'='*80}")
        print(f"📊 Exporting Results")
        print(f"{'='*80}")
        
        json_path = output_dir / f"benchmark_real_world_{timestamp}.json"
        csv_path = output_dir / f"benchmark_real_world_{timestamp}.csv"
        md_path = output_dir / f"benchmark_real_world_{timestamp}.md"
        
        export_results_json(results, str(json_path))
        export_results_csv(results, str(csv_path))
        export_results_markdown(results, str(md_path))
    
    # 打印总结
    print(f"\n{'='*80}")
    print(f"✅ Benchmark Complete!")
    print(f"{'='*80}")
    print(f"Tests completed: {len(results)}")
    if args.save_results:
        print(f"Results saved to: {output_dir}")
    else:
        print(f"Results not saved (use --save-results to save)")
    print(f"{'='*80}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

