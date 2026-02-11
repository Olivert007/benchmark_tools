"""
MOE Performance Metrics Module

This module provides functionality to calculate theoretical performance metrics
for Mixture of Experts (MOE) layers, including FLOPs estimation and memory usage analysis.
"""

from typing import Dict, List, Optional


def _quantize_input_flops(
    num_tokens: int,
    hidden_size: int,
    quant_type: str,
    per_act_token_quant: bool,
    block_shape: Optional[List[int]] = None,
) -> int:
    """Calculate FLOPs for input quantization."""
    if quant_type == "none":
        return 0

    total_elements = num_tokens * hidden_size

    if block_shape is not None and block_shape[0] > 0 and block_shape[1] > 0:
        block_n, block_k = block_shape[0], block_shape[1]
        num_blocks = (
            ((num_tokens + block_n - 1) // block_n)
            * ((hidden_size + block_k - 1) // block_k)
        )
        ops_per_block = 2 * block_n * block_k + 1
        return ops_per_block * num_blocks + total_elements

    elif per_act_token_quant:
        return num_tokens * 2 * hidden_size + num_tokens + total_elements

    elif quant_type in ["fp8_w8a8", "int8_w8a8"]:
        return 2 * total_elements + 1 + total_elements

    elif quant_type in ["int8_w8a16", "int4_w4a16"]:
        return 2 * total_elements + hidden_size + total_elements

    return 0


def calculate_moe_flops(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    quant_type: str = "none",
    per_act_token_quant: bool = False,
    block_shape: Optional[List[int]] = None,
    include_routing: bool = True,
) -> Dict[str, int]:
    """
    Calculate total theoretical FLOPs for an MOE layer, broken down by component.

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size per shard
        num_experts: Number of experts
        top_k: Number of experts per token
        quant_type: Quantization type ('none', 'fp8_w8a8', 'int8_w8a8', etc.)
        per_act_token_quant: Whether using per-token quantization
        block_shape: Optional block shape for block-wise quantization
        include_routing: Whether to include routing (top-k selection) FLOPs

    Returns:
        Dictionary with FLOPs breakdown and total
    """
    results = {
        "topk": num_tokens * num_experts if include_routing else 0,
        "first_quant": _quantize_input_flops(
            num_tokens, hidden_size, quant_type, per_act_token_quant, block_shape
        ),
        "first_matmul": 2 * num_tokens * top_k * hidden_size * intermediate_size,
        "activation": 5 * num_tokens * top_k * intermediate_size,
        "second_quant": _quantize_input_flops(
            num_tokens * top_k, intermediate_size // 2,
            quant_type, per_act_token_quant, block_shape,
        ),
        "second_matmul": 2 * num_tokens * top_k * (intermediate_size // 2) * hidden_size,
        "final_reduction": 2 * num_tokens * top_k * hidden_size,
    }
    results["total"] = sum(results.values())
    return results


def estimate_moe_memory(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    dtype_bytes: int = 2,
) -> Dict[str, float]:
    """
    Estimate memory usage for MOE computation in bytes.

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension size
        intermediate_size: Intermediate dimension size
        num_experts: Number of experts
        top_k: Number of experts per token
        dtype_bytes: Bytes per element (default: 2 for FP16/BF16)

    Returns:
        Dictionary with memory estimates in bytes
    """
    memory = {}

    ## total read memory ##
    # Input reads
    memory["input"] = num_tokens * hidden_size * dtype_bytes

    # For each token, we need to read topk experts from w1
    memory["w1_weights"] = num_experts * hidden_size * intermediate_size * dtype_bytes
    # For each token, we need to read topk experts from w2
    memory["w2_weights"] = (
        num_experts * (intermediate_size // 2) * hidden_size * dtype_bytes
    )

    # Gating output read
    memory["gating_read"] = hidden_size * num_experts * 4

    # Intermediate data transfers
    memory["topk_data"] = (
        num_tokens * top_k * dtype_bytes * 4 * 2
    )  # weights and ids is float32

    ##  total write memory ##
    # Output write
    memory["output"] = num_tokens * hidden_size * dtype_bytes

    # Total
    memory["total"] = sum(memory.values())

    return memory
