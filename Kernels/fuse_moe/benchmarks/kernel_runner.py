"""
MOE Benchmark Worker Module

This module provides functionality to benchmark MOE operations with different configurations.
"""

import os
import json
import torch
from typing import Dict, List, Optional, Tuple, TypedDict
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe import fused_moe, override_config
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_default_config,
    get_moe_configs,
)

from utils.inputs_builder import create_model_inputs
from utils.helper import get_config_dtype_str


class BenchmarkConfig(TypedDict):
    BLOCK_SIZE_M: int
    BLOCK_SIZE_N: int
    BLOCK_SIZE_K: int
    GROUP_SIZE_M: int
    num_warps: int
    num_stages: int


def get_moe_configs_from_file(config_path: str) -> Optional[Dict[int, BenchmarkConfig]]:
    """Load MOE configurations from a specified JSON file."""

    if os.path.exists(config_path):
        print(f"Loading MOE configuration from {config_path}")
        try:
            with open(config_path, "r") as f:
                config_dict = json.load(f)
            # Convert string keys (from JSON) to integers
            return {int(k): v for k, v in config_dict.items()}
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading configuration from {config_path}: {e}")

    print(f"No configuration file found at {config_path}")
    return None


class BenchmarkWorker:
    """Worker class for benchmarking MOE operations."""

    def __init__(self, seed: int, device: str = "cuda") -> None:
        torch.set_default_device(device)
        current_platform.seed_everything(seed)
        self.seed = seed
        self.device = device

    def benchmark_config(
        self,
        config: Dict,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        block_shape: List[int] = None,
        device: str = None,
        num_iters: int = 100,
    ) -> float:
        """Benchmark a specific configuration."""

        # Use provided device or default to the one set in __init__
        device = device or self.device

        # Create model inputs
        x, w1, w2, w1_scale, w2_scale, a1_scale, a2_scale, _, _ = create_model_inputs(
            num_tokens=num_tokens,
            hidden_size=hidden_size,
            num_experts=num_experts,
            shard_intermediate_size=shard_intermediate_size,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            block_shape=block_shape,
            compute_dtype=dtype,
            device=device,
        )

        # Generate gating outputs for multiple iterations
        gating_output = torch.randn(
            num_iters, num_tokens, num_experts, dtype=torch.float32, device=device
        )
        input_gating = torch.empty(
            num_tokens, num_experts, dtype=torch.float32, device=device
        )

        def prepare(i: int):
            input_gating.copy_(gating_output[i])

        def run():
            with override_config(config):
                fused_moe(
                    x,
                    w1,
                    w2,
                    input_gating,
                    topk,
                    renormalize=True,
                    inplace=True,
                    use_fp8_w8a8=use_fp8_w8a8,
                    use_int8_w8a8=use_int8_w8a8,
                    use_int8_w8a16=use_int8_w8a16,
                    w1_scale=w1_scale,
                    w2_scale=w2_scale,
                    a1_scale=a1_scale,
                    a2_scale=a2_scale,
                    block_shape=block_shape,
                )

        # JIT compilation & warmup
        run()
        torch.cuda.synchronize()

        # Capture 10 invocations with CUDA graph
        capture_num = 10
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(capture_num):
                run()
        torch.cuda.synchronize()

        # Warmup
        warmup_nums = 10
        for _ in range(warmup_nums):
            graph.replay()
        torch.cuda.synchronize()

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        latencies = []
        for i in range(num_iters):
            prepare(i)
            torch.cuda.synchronize()

            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            latencies.append(start_event.elapsed_time(end_event))

        avg = (
            sum(latencies) / (num_iters * capture_num) * 1000
        )  # Convert to microseconds
        graph.reset()
        return avg

    def benchmark(
        self,
        num_tokens: int,
        num_experts: int,
        shard_intermediate_size: int,
        hidden_size: int,
        topk: int,
        dtype: torch.dtype,
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        block_shape: List[int],
        device: str = None,
        config_dir: str = None,
    ) -> Tuple[Dict[str, int], float]:
        """Benchmark using the best available configuration."""
        current_platform.seed_everything(self.seed)

        # Use provided device or default to the one set in __init__
        device = device or self.device

        # Resolve dtype string early so it's always available
        dtype_str = get_config_dtype_str(
            dtype,
            use_int8_w8a16=use_int8_w8a16,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
        )

        op_config = None
        if config_dir:
            op_config = get_moe_configs_from_file(config_dir)

        if op_config is None:
            # Try to get configs from vllm's built-in system
            block_n = block_shape[0] if block_shape else 0
            block_k = block_shape[1] if block_shape else 0
            op_config = get_moe_configs(
                num_experts, shard_intermediate_size // 2, dtype_str, block_n, block_k
            )

        if op_config is None:
            config = get_default_config(
                num_tokens,
                num_experts,
                shard_intermediate_size,
                hidden_size,
                topk,
                dtype_str,
                block_shape,
            )
        else:
            config = op_config[min(op_config.keys(), key=lambda x: abs(x - num_tokens))]

        kernel_time = self.benchmark_config(
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
        )

        return config, kernel_time
