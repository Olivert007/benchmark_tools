"""
MOE Configuration Tuner Module

This module provides functionality to tune MOE configurations for optimal performance.
"""

import os
import json
import torch
import triton
from typing import List, Dict
from itertools import product
from datetime import datetime
from tqdm import tqdm
from vllm.platforms import current_platform
from utils.helper import get_config_dtype_str
from .kernel_runner import BenchmarkWorker, BenchmarkConfig


class ConfigTuner:
    """Tuner for MOE configurations."""

    def __init__(self, seed: int = 42):
        self.worker = BenchmarkWorker(seed)

    def get_configs_compute_bound(
        self, use_fp16: bool, block_quant_shape: List[int]
    ) -> List[Dict]:
        """Get configurations for compute-bound tuning."""
        configs = []

        if current_platform.is_rocm():
            param_ranges = self._get_rocm_tuning_space(use_fp16)
        else:
            # Reduced search space for faster tuning
            block_m_range = [16, 32, 64, 128, 256]
            block_n_range = [32, 64, 128, 256]
            block_k_range = [64, 128, 256]
            num_warps_range = [4, 8]
            group_m_range = [1, 16, 32, 64]
            num_stage_range = [2, 3, 4, 5]

            param_ranges = {
                "BLOCK_SIZE_M": block_m_range,
                "BLOCK_SIZE_N": block_n_range,
                "BLOCK_SIZE_K": block_k_range,
                "GROUP_SIZE_M": group_m_range,
                "num_warps": num_warps_range,
                "num_stages": num_stage_range,
            }

        keys, values = zip(*param_ranges.items())
        for config_values in product(*values):
            config = dict(zip(keys, config_values))
            configs.append(config)

        # Remove configs that are not compatible with fp8 block quantization
        if block_quant_shape is not None and not use_fp16:
            block_n, block_k = block_quant_shape[0], block_quant_shape[1]
            configs = [
                c for c in configs
                if c["BLOCK_SIZE_K"] % block_k == 0 and c["BLOCK_SIZE_N"] % block_n == 0
            ]
        return configs

    def _get_rocm_tuning_space(self, use_fp16: bool) -> Dict[str, List]:
        """Get ROCm-specific tuning space."""
        block_mn_range = [16, 32, 64, 128, 256]
        block_k_range = [16, 32, 64, 128, 256]
        if not use_fp16:
            block_k_range.remove(16)  # BLOCK_K=16 not supported for fp8
        num_warps_range = [1, 2, 4, 8]
        group_m_range = [1, 4, 8, 16, 32]
        num_stage_range = [2]
        waves_per_eu_range = [0]
        matrix_instr_nonkdim_range = [16, 32] if use_fp16 else []
        kpack_range = [1, 2] if use_fp16 else []

        param_ranges = {
            "BLOCK_SIZE_M": block_mn_range,
            "BLOCK_SIZE_N": block_mn_range,
            "BLOCK_SIZE_K": block_k_range,
            "GROUP_SIZE_M": group_m_range,
            "num_warps": num_warps_range,
            "num_stages": num_stage_range,
            "waves_per_eu": waves_per_eu_range,
        }
        if use_fp16:
            param_ranges["matrix_instr_nonkdim"] = matrix_instr_nonkdim_range
            param_ranges["kpack"] = kpack_range

        return param_ranges

    def _get_rocm_configs_compute_bound_opt(self) -> List[Dict[str, int]]:
        """Get optimized ROCm configurations with exclusion rules."""
        configs: List[BenchmarkConfig] = []

        # Define exclusion rules as tuples for easier management
        exclusion_rules = [
            # (block_m, block_n, block_k, num_warps, num_stages, group_sizes)
            (256, 128, 32, 2, 2, [1, 2, 4, 8, 16, 32]),
            (256, 256, 32, 4, 2, [1, 2, 4, 8]),
            (256, 256, 64, 1, 2, [1, 2, 4, 8, 16, 32]),
            (256, 256, 32, 1, 2, [1, 2, 4, 8, 16, 32]),
            (256, 256, 128, [1, 4], 2, [1, 2, 4, 8, 16, 32]),  # Multiple num_warps
            (256, 32, 128, 1, 2, [1, 2, 4, 8, 16, 32]),
            # Comprehensive exclusion for all BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128 configs
            # These configs consistently fail with "Out of resources" error
            (256, 256, 128, [1, 2, 4, 8], 2, [1, 2, 4, 8, 16, 32]),  # All num_warps and GROUP_SIZE_M combinations
            # Comprehensive exclusion for all BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=256 configs
            # These configs also consistently fail with "Out of resources" error
            (256, 256, 256, [1, 2, 4, 8], 2, [1, 2, 4, 8, 16, 32]),  # All num_warps and GROUP_SIZE_M combinations  
            # Comprehensive exclusion for all BLOCK_SIZE_M=256, BLOCK_SIZE_N=16, BLOCK_SIZE_K=256 configs
            # These configs cause memory access faults
            (256, 16, 256, [1, 2, 4, 8], 2, [1, 2, 4, 8, 16, 32]),  # All num_warps and GROUP_SIZE_M combinations    
            (256, 32, 256, [1, 2, 4, 8], 2, [1, 2, 4, 8, 16, 32]),  # All num_warps and GROUP_SIZE_M combinations    
            # Comprehensive exclusion for all BLOCK_SIZE_M=256, BLOCK_SIZE_N=64, BLOCK_SIZE_K=128 configs
            # These configs also cause memory access faults
            (256, 64, 128, [1, 2, 4, 8], 2, [1, 2, 4, 8, 16, 32]),  # All num_warps and GROUP_SIZE_M combinations    
            # Comprehensive exclusion for all BLOCK_SIZE_M=256, BLOCK_SIZE_N=128, BLOCK_SIZE_K=128 configs
            # These configs also cause memory access faults
            (256, 128, 128, [1, 2, 4, 8], 2, [1, 2, 4, 8, 16, 32]),  # All num_warps and GROUP_SIZE_M combinations 
            # Comprehensive exclusion for all BLOCK_SIZE_M=256, BLOCK_SIZE_N=256, BLOCK_SIZE_K=128 configs
            # These configs also cause memory access faults
            (256, 256, 128, [1, 2, 4, 8], 2, [1, 2, 4, 8, 16, 32]),  # All num_warps and GROUP_SIZE_M combinations      
        ]

        def should_exclude_config(
            block_m, block_n, block_k, num_warps, num_stages, group_size
        ):
            """Check if a configuration should be excluded based on rules."""
            for rule in exclusion_rules:
                rule_m, rule_n, rule_k, rule_warps, rule_stages, rule_groups = rule

                # Handle multiple num_warps values in rules
                if isinstance(rule_warps, list):
                    warps_match = num_warps in rule_warps
                else:
                    warps_match = num_warps == rule_warps

                if (
                    block_m == rule_m
                    and block_n == rule_n
                    and block_k == rule_k
                    and warps_match
                    and num_stages == rule_stages
                    and group_size in rule_groups
                ):
                    return True
            return False

        # Generate configurations using itertools.product
        block_m_range = [16, 32, 64, 128, 256]
        block_k_range = [32, 64, 128, 256]
        block_n_range = [16, 32, 64, 128, 256]
        num_warps_range = [1, 2, 4, 8]
        num_stages_range = [2]
        group_size_range = [1, 2, 4, 8, 16, 32]

        for num_stages, block_m, block_k, block_n, num_warps, group_size in product(
            num_stages_range, block_m_range, block_k_range,
            block_n_range, num_warps_range, group_size_range,
        ):
            if not should_exclude_config(
                block_m, block_n, block_k, num_warps, num_stages, group_size
            ):
                configs.append({
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": group_size,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                })

        print(f"Generated {len(configs)} configurations for ROCm tuning")
        return configs

    def _get_configs_compute_bound_opt(self) -> List[Dict[str, int]]:
        # Reduced search space for faster tuning.
        # TODO(woosuk): Increase the search space and use a performance model to
        # prune the search space.
        configs: List[BenchmarkConfig] = []
        if current_platform.is_rocm():
            configs = self._get_rocm_configs_compute_bound_opt()
        else:
            for num_stages, block_m, block_k, block_n, num_warps, group_size in product(
                [2, 3, 4, 5],       # num_stages
                [16, 32, 64, 128, 256],  # block_m
                [64, 128, 256],      # block_k
                [32, 64, 128, 256],  # block_n
                [4, 8],              # num_warps
                [1, 16, 32, 64],     # group_size
            ):
                configs.append({
                    "BLOCK_SIZE_M": block_m,
                    "BLOCK_SIZE_N": block_n,
                    "BLOCK_SIZE_K": block_k,
                    "GROUP_SIZE_M": group_size,
                    "num_warps": num_warps,
                    "num_stages": num_stages,
                })
        return configs

    def tune(
        self,
        num_tokens: int,
        hidden_size: int,
        num_experts: int,
        shard_intermediate_size: int,
        topk: int,
        dtype: torch.dtype,
        search_space: List[Dict],
        use_int8_w8a8: bool = False,
        use_int8_w8a16: bool = False,
        use_fp8_w8a8: bool = False,
        block_shape: List[int] = None,
        device="cuda",
    ) -> Dict:
        """Tune configurations for optimal performance."""
        best_config = None
        best_time = float("inf")

        # Keep track of all results and failures
        all_results = []
        failed_configs = []
        with torch.cuda.device(torch.cuda.current_device()):
            for config_idx, config in enumerate(tqdm(search_space)):
                print(
                    f"Testing config {config_idx + 1}/{len(search_space)}: {self._sort_config(config)}"
                )
                try:
                    kernel_time = self.worker.benchmark_config(
                        config,
                        num_tokens,
                        num_experts,
                        shard_intermediate_size,
                        hidden_size,
                        topk,
                        dtype,
                        use_fp8_w8a8,
                        use_int8_w8a8,
                        use_int8_w8a16,
                        block_shape,
                        device=device,
                        num_iters=10,
                    )

                    result = {"config": config, "kernel_time": kernel_time}
                    all_results.append(result)
                    if kernel_time < best_time:
                        best_time = kernel_time
                        best_config = config
                        print(
                            f"New best config: {best_config}, Time: {kernel_time:.2f} μs"
                        )

                except triton.runtime.autotuner.OutOfResources:
                    # Some configurations may be invalid and fail to compile.
                    failed_configs.append({"config": config, "error": "OutOfResources"})
                    print(f"Config: {config} failed: Out of resources")
                    continue

                except torch.cuda.OutOfMemoryError:
                    # Handle CUDA OOM errors
                    failed_configs.append(
                        {"config": config, "error": "OutOfMemoryError"}
                    )
                    print(f"Config {config} failed: CUDA out of memory")
                    torch.cuda.empty_cache()  # Try to recover
                    continue

                except RuntimeError as e:
                    # Specifically catch memory access fault errors from ROCm
                    error_message = str(e)
                    error_type = (
                        "Memory access fault"
                        if "Memory access fault" in error_message
                        else "RuntimeError"
                    )
                    failed_configs.append(
                        {
                            "config": config,
                            "error": error_type,
                            "message": error_message[:500],
                        }
                    )
                    print(f"Config {config} failed: {error_type} (total failures: {len(failed_configs)})")
                    continue

                except Exception as e:
                    # Catch any other exceptions
                    error_message = str(e)
                    failed_configs.append({"config": config, "error": error_message})
                    print(f"Config: {config} failed: {error_message[:100]}...")
                    continue

        now = datetime.now()
        print(f"{now.ctime()} Completed tuning for batch_size={num_tokens}")
        # Check if we found any valid configuration
        if not best_config:
            raise RuntimeError(
                "No valid configuration found during tuning. All configurations failed."
            )
        return best_config

    def save_configs(
        self,
        configs: Dict[int, Dict],
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        block_shape: List[int],
        save_dir: str,
    ) -> None:
        """Save tuned configurations to file."""
        dtype_str = get_config_dtype_str(
            dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
        )

        batch_sizes = sorted(configs.keys())
        filename = self._get_config_file_name(
            num_experts, 
            shard_intermediate_size // 2, 
            dtype_str, 
            block_shape,
            batch_sizes
        )
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, filename)
        print(f"Writing best config to {filename}...")
        with open(filename, "w") as f:
            json.dump(configs, f, indent=4)
            f.write("\n")

    def _get_config_file_name(
        self, 
        E: int, 
        N: int, 
        dtype: str, 
        block_shape: List[int] = None,
        batch_sizes: List[int] = None,
    ) -> str:
        """Generate configuration file name."""
        device_name = current_platform.get_device_name().replace(" ", "_")
        dtype_selector = "" if not dtype else f",dtype={dtype}"
        block_shape_selector = (
            ""
            if not block_shape or not all(block_shape)
            else f",block_shape={block_shape}"
        ).replace(" ", "")
        batch_size_selector = ""
        if batch_sizes:
            bs_str = ",".join(map(str, batch_sizes))
            batch_size_selector = f",bs={bs_str}"        
        return f"E={E},N={N}{batch_size_selector},device_name={device_name}{dtype_selector}{block_shape_selector}.json"
    
    
    def _sort_config(self, config: BenchmarkConfig) -> BenchmarkConfig:
        return {
            "BLOCK_SIZE_M": config["BLOCK_SIZE_M"],
            "BLOCK_SIZE_N": config["BLOCK_SIZE_N"],
            "BLOCK_SIZE_K": config["BLOCK_SIZE_K"],
            "GROUP_SIZE_M": config["GROUP_SIZE_M"],
            "num_warps": config["num_warps"],
            "num_stages": config["num_stages"],
            **(
                {"waves_per_eu": config["waves_per_eu"]}
                if "waves_per_eu" in config
                else {}
            ),
            **(
                {"matrix_instr_nonkdim": config["matrix_instr_nonkdim"]}
                if "matrix_instr_nonkdim" in config
                else {}
            ),
            **({"kpack": config["kpack"]} if "kpack" in config else {}),
        }
