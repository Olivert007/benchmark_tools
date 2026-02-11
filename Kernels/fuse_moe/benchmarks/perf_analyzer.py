"""
MOE Performance Analyzer Module

This module analyzes MOE kernel performance by calculating theoretical metrics
including TFLOPS, memory bandwidth, and computational efficiency.
"""

from typing import Dict, Optional, List
import torch
from utils.perf_metrics import calculate_moe_flops, estimate_moe_memory
from benchmarks.kernel_runner import BenchmarkWorker


def get_peak_tflops_for_precision(precision_type: str) -> float:
    """
    Get the theoretical peak TFLOPS for a given precision type on GPU.

    Args:
        precision_type: The precision type string

    Returns:
        Peak TFLOPS value for the specified precision
    """
    # Peak performance values for modern GPU
    peak_tflops_map = {
        # FP precision
        "fp16": 123.0,  # FP16 peak TFLOPS
        "bf16": 123.0,  # BF16 uses same hardware as FP16
        "fp32": 61.4,  # FP32 peak TFLOPS
        "fp8": 123.0,  # FP8 uses FP16 hardware
        # INT precision
        "int8": 123.0,  # INT8 peak TOPS
        "int4": 246.0,  # INT4 peak TOPS
    }

    return peak_tflops_map.get(precision_type, 123.0)  # Default to FP16 if not found


def profile_moe(
    num_tokens: int,
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    topk: int,
    dtype: torch.dtype,
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    block_shape: Optional[List[int]],
    kernel_time: Optional[float] = None,
    device: str = "cuda",
    config_dir: Optional[str] = None,
    seed: int = 42,
) -> Dict:
    """
    Analyze MOE operation performance, calculating TFLOPS and memory bandwidth.

    Args:
        num_tokens: Number of tokens
        num_experts: Number of experts
        shard_intermediate_size: Size of intermediate layer per shard
        hidden_size: Size of hidden dimension
        topk: Number of top experts per token
        dtype: Data type for computation
        use_fp8_w8a8: Whether using FP8 quantization
        use_int8_w8a8: Whether using INT8 quantization
        use_int8_w8a16: Whether using INT8/FP16 mixed quantization
        block_shape: Optional block shape for block-wise quantization
        kernel_time: Optional kernel execution time in microseconds
        device: Device to run on (default: 'cuda')
        config_dir: Optional path to configuration file for custom MOE configs
        seed: Random seed for benchmark reproducibility (default: 42)

    Returns:
        Dictionary containing performance metrics
    """

    if config_dir:
        print(f"  - Config Directory: {config_dir}")
    else:
        print(f"  - Config Directory: Using default configurations")

    # Determine quantization type and precision from flags
    # Mapping: (quant_type, precision_type, display_name)
    _quant_map = {
        "fp8_w8a8":  ("fp8_w8a8",  "fp8",  "FP8 (weights and activations)"),
        "int8_w8a8": ("int8_w8a8", "int8", "INT8 (weights and activations)"),
        "int8_w8a16": ("int8_w8a16", "int8", "INT8 (weights) / FP16 (activations)"),
    }

    # Find which flag is set
    active_quant = None
    for flag, key in [(use_fp8_w8a8, "fp8_w8a8"), (use_int8_w8a8, "int8_w8a8"), (use_int8_w8a16, "int8_w8a16")]:
        if flag:
            active_quant = key
            break

    if active_quant:
        quant_type, precision_type, display_name = _quant_map[active_quant]
        print(f"  - Quantization: {display_name}")
    else:
        quant_type = "none"
        precision_type = "bf16" if dtype == torch.bfloat16 else "fp16"
        print(f"  - Quantization: None ({precision_type.upper()})")

    if block_shape is not None:
        print(f"  - Block Quantization Shape: {block_shape}")

    # Get theoretical peak TFLOPS for this precision
    peak_tflops = get_peak_tflops_for_precision(precision_type)
    print(f"  - Peak Performance: {peak_tflops} TFLOPS ({precision_type.upper()})")

    # If kernel_time wasn't provided, try to benchmark
    best_config = None
    if kernel_time is None:
        try:
            worker = BenchmarkWorker(seed=seed)
            best_config, kernel_time = worker.benchmark(
                num_tokens=num_tokens,
                num_experts=num_experts,
                shard_intermediate_size=shard_intermediate_size,
                hidden_size=hidden_size,
                topk=topk,
                dtype=dtype,
                use_fp8_w8a8=use_fp8_w8a8,
                use_int8_w8a8=use_int8_w8a8,
                use_int8_w8a16=use_int8_w8a16,
                block_shape=block_shape,
                device=device,
                config_dir=config_dir,  # Pass config_dir to benchmark worker
            )
        except Exception as e:
            print(f"Failed to benchmark kernel: {e}")
            kernel_time = None
    else:
        print(f"\nUsing provided kernel time: {kernel_time:.2f} μs")

    # Calculate theoretical performance metrics if we have timing information
    perf_data = {}
    if kernel_time is not None:
        print(f"\nCalculating performance metrics...")
        runtime_seconds = kernel_time / 1e6  # Convert microseconds to seconds

        # Calculate FLOPs
        flops_breakdown = calculate_moe_flops(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=shard_intermediate_size,
            num_experts=num_experts,
            top_k=topk,
            quant_type=quant_type,
            per_act_token_quant=False,
            block_shape=block_shape,
        )

        # Calculate memory usage
        memory_breakdown = estimate_moe_memory(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            intermediate_size=shard_intermediate_size,
            num_experts=num_experts,
            top_k=topk,
            dtype_bytes=(
                2
                if dtype in [torch.float16, torch.bfloat16]
                else (1 if quant_type in ["int8_w8a8", "fp8_w8a8", "int8_w8a16"] else 4)
            ),
        )

        # Calculate TFLOPS
        total_flops = flops_breakdown["total"]
        tflops = total_flops / (runtime_seconds * 1e12)

        # Calculate efficiency percentage
        efficiency_percentage = (tflops / peak_tflops) * 100

        # Calculate memory bandwidth
        total_memory_bytes = memory_breakdown["total"]
        bandwidth_gbps = total_memory_bytes / (runtime_seconds * 1e9)

        # Calculate perTokenTime percentage
        per_token_time = kernel_time / num_tokens

        perf_data = {
            "BestConfig": f"{best_config}",
            "KernelTime": f"{kernel_time:.2f}",
            "TotalFLOPs": f"{total_flops:,}",
            "Performance": f"{tflops:.2f}",
            "Efficiency": f"{efficiency_percentage:.2f}",
            "Throughput": f"{bandwidth_gbps:.2f}",
            "PerTokenTime": f"{per_token_time:.2f}",
        }
        print(f"\nProfiling completed successfully!")

    return perf_data
