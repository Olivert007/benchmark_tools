"""
Refactored MOE Testing Script

This script demonstrates the usage of the modular MOE testing framework.
"""

import os
import argparse
import csv
import time
from datetime import datetime
from typing import List, Optional

try:
    from prettytable import PrettyTable
except ImportError:
    raise ImportError(
        "prettytable is required but not installed. "
        "Please install it with: pip install prettytable"
    )

import torch
from benchmarks.kernel_runner import BenchmarkWorker
from benchmarks.tuner import ConfigTuner
from benchmarks.perf_analyzer import profile_moe
from utils.correctness_checker import correctness
from utils.helper import get_vllm_version, get_triton_version

DEFAULT_SEED = 42


def get_device_info() -> str:
    """Get device information string. Must be called after setup_environment()."""
    device = torch.cuda.current_device()
    return (
        f"Using {torch.cuda.get_device_name(device)}_{device}: "
        f"{torch.cuda.get_device_properties(device).total_memory / (1024**3):.2f} GB memory, "
        f"torch={torch.__version__}, triton={get_triton_version()}, vllm={get_vllm_version()}"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MOE Performance Testing")

    # Model configuration
    parser.add_argument(
        "--num-tokens",
        type=int,
        default=128,
        help="Number of tokens/batch size (default: 128)",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=7168,
        help="Hidden dimension size (default: 7168)",
    )
    parser.add_argument(
        "--num-experts", type=int, default=256, help="Number of experts (default: 256)"
    )
    parser.add_argument(
        "--shard-intermediate-size",
        type=int,
        default=512,
        help="Size of intermediate dimension per shard (default: 512)",
    )
    parser.add_argument(
        "--topk", type=int, default=8, help="Number of experts per token (default: 8)"
    )

    # Quantization options
    parser.add_argument(
        "--type",
        choices=["fp16", "int8_w8a8", "int8_w8a16", "fp8_w8a8"],
        default="int8_w8a8",
        help="Quantization type",
    )

    # Block shape
    parser.add_argument(
        "--block-shape",
        type=str,
        default="128,128",
        help="Block shape as n,k (default: 128,128)",
    )

    # Compute data type
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
        help="Computation data type (default: float16)",
    )

    # Test modes
    parser.add_argument(
        "--mode",
        choices=["correctness", "tune", "benchmark", "profile"],
        default="profile",
        help="Test mode",
    )

    # Batch sizes
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="2",
        help="Comma-separated list of batch sizes to test (default: 2)",
    )

    # Device selection
    parser.add_argument(
        "--device", type=int, default=2, help="GPU device ID (default: 2)"
    )

    # Additional options
    parser.add_argument(
        "--save-dir",
        default="tuned_configs",
        help="Directory to save tuned configurations (default: ./tuned_configs)",
    )

    # Configuration directory
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help="Directory path to load MOE best configurations from (default: None)",
    )

    return parser.parse_args()


def setup_environment(device_id: int):
    """Setup environment variables and device."""
    os.environ["ROCR_VISIBLE_DEVICES"] = str(device_id)


def parse_quantization_flags(quant_type: str):
    """Parse quantization type into boolean flags."""
    use_int8_w8a8 = quant_type == "int8_w8a8"
    use_int8_w8a16 = quant_type == "int8_w8a16"
    use_fp8_w8a8 = quant_type == "fp8_w8a8"
    return use_int8_w8a8, use_int8_w8a16, use_fp8_w8a8


def parse_block_shape(block_shape_str: str) -> List[int]:
    """Parse block shape string into list of integers."""
    try:
        block_shape = [int(x) for x in block_shape_str.split(",")]
        if len(block_shape) != 2:
            print(f"Invalid block shape: {block_shape_str}. Using default [128, 128]")
            return [128, 128]
        return block_shape
    except ValueError:
        print(f"Invalid block shape: {block_shape_str}. Using default [128, 128]")
        return [128, 128]


def run_correctness_tests(
    batch_sizes: List[int],
    hidden_size: int,
    num_experts: int,
    shard_intermediate_size: int,
    topk: int,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_fp8_w8a8: bool = False,
    block_shape: Optional[List[int]] = None,
    compute_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
):
    """Run correctness tests for given batch sizes."""
    print("\n=== Correctness Test on GPU ===")
    for batch_size in batch_sizes:
        print(f"Testing batch size {batch_size}...")
        correctness(
            num_tokens=batch_size,
            hidden_size=hidden_size,
            num_experts=num_experts,
            shard_intermediate_size=shard_intermediate_size,
            topk=topk,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            block_shape=block_shape,
            compute_dtype=compute_dtype,
            device=device,
        )


def run_tuning(
    batch_sizes: List[int],
    save_dir: str,
    hidden_size: int,
    num_experts: int,
    shard_intermediate_size: int,
    topk: int,
    dtype: torch.dtype,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_fp8_w8a8: bool = False,
    block_shape: Optional[List[int]] = None,
    device: str = "cuda",
    seed: int = DEFAULT_SEED,
):
    """Run configuration tuning for given batch sizes."""
    print("\n=== Tune on GPU ===")

    tuner = ConfigTuner(seed=seed)
    search_space = tuner._get_configs_compute_bound_opt()

    print(f"Start tuning over {len(search_space)} configurations...")

    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    full_save_dir = os.path.join(current_script_dir, save_dir)
    print(f"Saving tuned configurations to: {full_save_dir}")
    os.makedirs(full_save_dir, exist_ok=True)

    all_configs = {}
    for batch_size in batch_sizes:
        print(f"Tuning for batch size {batch_size}...")
        batch_start_time = time.time()
        best_config = tuner.tune(
            num_tokens=batch_size,
            search_space=search_space,
            hidden_size=hidden_size,
            num_experts=num_experts,
            shard_intermediate_size=shard_intermediate_size,
            topk=topk,
            dtype=dtype,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            block_shape=block_shape,
            device=device,
        )
        batch_duration = time.time() - batch_start_time
        print(f"  Completed in {batch_duration:.1f} seconds")
        # Save individual config
        configs = {batch_size: best_config}
        tuner.save_configs(
            configs,
            save_dir=full_save_dir,
            hidden_size=hidden_size,
            num_experts=num_experts,
            shard_intermediate_size=shard_intermediate_size,
            topk=topk,
            dtype=dtype,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            block_shape=block_shape,
        )
        if best_config:
            all_configs[batch_size] = best_config

    # Save all collected configs into a single file
    if all_configs:
        print(f"Saving best configs for batch sizes {list(all_configs.keys())}...")
        tuner.save_configs(
            all_configs,
            save_dir=full_save_dir,
            hidden_size=hidden_size,
            num_experts=num_experts,
            shard_intermediate_size=shard_intermediate_size,
            topk=topk,
            dtype=dtype,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            block_shape=block_shape,
        )


def run_benchmarks(
    batch_sizes: List[int],
    hidden_size: int,
    num_experts: int,
    shard_intermediate_size: int,
    topk: int,
    dtype: torch.dtype,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_fp8_w8a8: bool = False,
    block_shape: Optional[List[int]] = None,
    device: str = "cuda",
    config_dir: Optional[str] = None,
    device_info: str = "",
    seed: int = DEFAULT_SEED,
):
    """Run benchmarks for given batch sizes."""
    print("\n=== Benchmark on GPU ===")

    worker = BenchmarkWorker(seed=seed)

    # Create PrettyTable for displaying results
    results_table = PrettyTable()
    results_table.title = f"Fused MoE Benchmark: {device_info}"
    results_table.field_names = [
        "Batch Size",
        "Shard Size",
        "Hidden Size",
        "Best Config",
        "Kernel Time (μs)",
    ]

    # Set column alignment
    results_table.align["Batch Size"] = "r"
    results_table.align["Shard Size"] = "r"
    results_table.align["Hidden Size"] = "r"
    results_table.align["Best Config"] = "r"
    results_table.align["Kernel Time (μs)"] = "l"

    # Simple results storage
    results = []
    for batch_size in batch_sizes:
        config, kernel_time = worker.benchmark(
            num_tokens=batch_size,
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
            config_dir=config_dir,
        )
        results.append(
            {
                "batch_size": batch_size,
                "kernel_time": kernel_time,
                "config": config,
                "shard_intermediate_size": shard_intermediate_size,
                "hidden_size": hidden_size,
            }
        )

        # Add row to PrettyTable
        results_table.add_row(
            [
                batch_size,
                shard_intermediate_size,
                hidden_size,
                config,
                f"{kernel_time:.2f}",
            ]
        )

        print(f"Batch size: {batch_size}, Kernel time: {kernel_time:.2f} μs")

    # Display the results table
    print("BENCHMARK RESULTS:")
    print(results_table)

    # Save to CSV if we have results
    if results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = f"benchmark_results_{timestamp}.csv"

        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Batch Size",
                    "Shard Intermediate Size",
                    "Hidden Size",
                    "Best Config",
                    "Kernel Time (μs)",
                ]
            )

            for result in results:
                writer.writerow(
                    [
                        result["batch_size"],
                        result["shard_intermediate_size"],
                        result["hidden_size"],
                        str(result["config"]),
                        f"{result['kernel_time']:.2f}",
                    ]
                )

        print(f"\nResults saved to: {csv_filename}")


def run_profiling(
    batch_sizes: List[int],
    hidden_size: int,
    num_experts: int,
    shard_intermediate_size: int,
    topk: int,
    dtype: torch.dtype,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_fp8_w8a8: bool = False,
    block_shape: Optional[List[int]] = None,
    device: str = "cuda",
    config_dir: Optional[str] = None,
    device_info: str = "",
    seed: int = DEFAULT_SEED,
):
    """Run profiling for given batch sizes."""
    print("\n=== Profile on GPU ===")
    all_results = []

    for batch_size in batch_sizes:
        print(f"Profiling for batch size {batch_size}...")
        result = profile_moe(
            num_tokens=batch_size,
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
            config_dir=config_dir,
            seed=seed,
        )
        if result:
            result["BatchSize"] = batch_size
            all_results.append(result)

    moe_params = f"shard_intermediate_size: {shard_intermediate_size} -- hidden_size: {hidden_size} -- num_experts: {num_experts} -- topk: {topk}"

    # Create a summary table with all batch sizes
    if all_results:
        summary_table = PrettyTable()
        summary_table.title = f"Fused MoE Benchmark: {device_info}"
        summary_table.field_names = [
            "Batch Size",
            f"Config: {moe_params}",
            "Kernel Time (μs)",
            "Performance (TFLOPS)",
            "Efficiency (%)",
            # "Throughput (GB/s)", # Disabled for now
            "PerTokenTime (μs/token)",
        ]

        for result in all_results:
            summary_table.add_row(
                [
                    result["BatchSize"],
                    result["BestConfig"],
                    result["KernelTime"],
                    result["Performance"],
                    result["Efficiency"],
                    # result["Throughput"],
                    result["PerTokenTime"],
                ]
            )

        print("\nSummary Results:")
        print(summary_table)

        # Save combined results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"moe_profile_summary_{timestamp}.csv"

        with open(filename, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Batch Size",
                    "Best Config",
                    "Kernel Time (μs)",
                    "Performance (TFLOPS)",
                    "Efficiency (%)",
                    # "Throughput (GB/s)",
                    "Per Token Time (token/s)",
                ]
            )

            for result in all_results:
                csv_writer.writerow(
                    [
                        result["BatchSize"],
                        result["BestConfig"],
                        result["KernelTime"],
                        result["Performance"],
                        result["Efficiency"],
                        # result["Throughput"],
                        result["PerTokenTime"],
                    ]
                )

        print(f"\nSummary results saved to {filename}")


def main():
    args = parse_args()

    # Setup environment
    setup_environment(args.device)

    # Get device info after environment setup
    device_info = get_device_info()

    # Parse configuration
    use_int8_w8a8, use_int8_w8a16, use_fp8_w8a8 = parse_quantization_flags(args.type)
    block_shape = parse_block_shape(args.block_shape)
    compute_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    # Improved batch size parsing - handle spaces and other common formats
    batch_sizes_str = args.batch_sizes.replace(" ", "")  # Remove spaces
    batch_sizes = [int(bs) for bs in batch_sizes_str.split(",") if bs]

    # Print configuration
    print("\n" + "=" * 80)
    print("MOE Performance Testing Configuration:")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"Hidden Size: {args.hidden_size}")
    print(f"Number of Experts: {args.num_experts}")
    print(f"Shard Intermediate Size: {args.shard_intermediate_size}")
    print(f"Top-k: {args.topk}")
    print(f"Quantization Type: {args.type}")
    print(f"Block Shape: {block_shape}")
    print(f"Compute Data Type: {args.dtype}")
    print(f"Batch Sizes: {batch_sizes}")
    # print(f"Device: {args.device}")
    if args.config_dir:
        print(f"Config Directory: {args.config_dir}")
    print("=" * 80)

    # Common keyword arguments for all modes
    common_kwargs = dict(
        hidden_size=args.hidden_size,
        num_experts=args.num_experts,
        shard_intermediate_size=args.shard_intermediate_size,
        topk=args.topk,
        use_int8_w8a8=use_int8_w8a8,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        block_shape=block_shape,
        device="cuda",
    )

    # Run the selected mode
    if args.mode == "correctness":
        run_correctness_tests(
            batch_sizes,
            compute_dtype=compute_dtype,
            **common_kwargs,
        )
    elif args.mode == "tune":
        run_tuning(
            batch_sizes,
            save_dir=args.save_dir,
            dtype=compute_dtype,
            **common_kwargs,
        )
    elif args.mode == "benchmark":
        run_benchmarks(
            batch_sizes,
            dtype=compute_dtype,
            config_dir=args.config_dir,
            device_info=device_info,
            **common_kwargs,
        )
    elif args.mode == "profile":
        run_profiling(
            batch_sizes,
            dtype=compute_dtype,
            config_dir=args.config_dir,
            device_info=device_info,
            **common_kwargs,
        )


if __name__ == "__main__":
    main()
