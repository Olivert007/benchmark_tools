"""
MOE Model Factory Module

This module provides functionality to create model inputs and initialize data types
for MOE testing and benchmarking.
"""

import torch
from typing import List, Optional, Tuple


def initialize_dtype(
    use_fp8_w8a8: bool,
    use_int8_w8a8: bool,
    use_int8_w8a16: bool,
    compute_dtype: torch.dtype,
) -> Tuple[torch.dtype, torch.dtype, torch.dtype]:
    """
    Initialize data types based on quantization flags.

    Args:
        use_fp8_w8a8: Whether to use FP8 quantization for both weights and activations
        use_int8_w8a8: Whether to use INT8 quantization for both weights and activations
        use_int8_w8a16: Whether to use INT8 for weights and FP16 for activations
        compute_dtype: Base computation data type (usually torch.float16)

    Returns:
        Tuple of (quant_dtype, compute_dtype, init_dtype)
    """
    if use_fp8_w8a8:
        quant_dtype = torch.float8_e4m3fn
    elif use_int8_w8a8 or use_int8_w8a16:
        quant_dtype = torch.int8
    else:
        quant_dtype = torch.float16

    compute_dtype = torch.float16
    init_dtype = torch.float16 if use_fp8_w8a8 else compute_dtype
    return quant_dtype, compute_dtype, init_dtype


def create_scale_factors(
    num_experts: int,
    shard_intermediate_size: int,
    hidden_size: int,
    use_int8_w8a16: bool,
    use_int8_w8a8: bool,
    use_fp8_w8a8: bool,
    block_shape: Optional[List[int]],
    device: str = "cuda",
) -> Tuple:
    """Create scaling factors for quantized weights and activations."""
    w1_scale = None
    w2_scale = None
    a1_scale = None
    a2_scale = None

    if use_int8_w8a16:
        # For INT8/FP16 mixed precision, create per-column scaling
        w1_scale = torch.randn(
            (num_experts, 2 * shard_intermediate_size),
            dtype=torch.float32,
            device=device,
        )
        w2_scale = torch.randn(
            (hidden_size, num_experts), dtype=torch.float32, device=device
        )

    if use_fp8_w8a8 or use_int8_w8a8:
        if use_int8_w8a8 and block_shape is None:
            # For INT8 non-block quantization
            w1_scale = torch.randn(
                num_experts, shard_intermediate_size, dtype=torch.float32, device=device
            )
            w2_scale = torch.randn(
                num_experts, hidden_size, dtype=torch.float32, device=device
            )
        elif block_shape is None:
            # For FP8 tensor-wise quantization
            w1_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            w2_scale = torch.randn(num_experts, dtype=torch.float32, device=device)
            a1_scale = torch.randn(1, dtype=torch.float32, device=device)
            a2_scale = torch.randn(1, dtype=torch.float32, device=device)
        else:  # block-wise quantization
            # Create block-wise scaling factors
            block_n, block_k = block_shape[0], block_shape[1]
            N = shard_intermediate_size // 2  # Half of intermediate size for the second matrix
            K = hidden_size
            factor_for_scale = 1e-2  # Small factor to avoid overflow

            # Calculate the number of tiles in each dimension
            n_tiles_w1 = (2 * N + block_n - 1) // block_n
            n_tiles_w2 = (K + block_n - 1) // block_n
            k_tiles_w1 = (K + block_k - 1) // block_k
            k_tiles_w2 = (N + block_k - 1) // block_k

            w1_scale = (
                torch.rand((num_experts, n_tiles_w1, k_tiles_w1), dtype=torch.float32, device=device)
                * factor_for_scale
            )
            w2_scale = (
                torch.rand((num_experts, n_tiles_w2, k_tiles_w2), dtype=torch.float32, device=device)
                * factor_for_scale
            )

    return w1_scale, w2_scale, a1_scale, a2_scale


def create_model_inputs(
    num_tokens: int,
    hidden_size: int,
    num_experts: int,
    shard_intermediate_size: int,
    use_int8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    use_fp8_w8a8: bool = False,
    block_shape: Optional[List[int]] = None,
    compute_dtype: torch.dtype = torch.float16,
    device: str = "cuda",
) -> Tuple:
    """
    Create model inputs and parameters for MOE benchmarking.

    Args:
        num_tokens: Number of tokens (batch size)
        hidden_size: Hidden dimension size for the model
        num_experts: Number of experts in the MOE layer
        shard_intermediate_size: Size of intermediate dimension per expert shard
        use_int8_w8a8: Whether to use INT8 quantization for weights and activations
        use_int8_w8a16: Whether to use INT8 for weights and FP16 for activations
        use_fp8_w8a8: Whether to use FP8 quantization for weights and activations
        block_shape: Optional shape for block-wise quantization, as [block_n, block_k]
        compute_dtype: Data type for computation (default: torch.float16)
        device: Device to create tensors on (default: "cuda")

    Returns:
        Tuple of tensors and parameters
    """

    quant_dtype, compute_dtype, init_dtype = initialize_dtype(
        use_fp8_w8a8, use_int8_w8a8, use_int8_w8a16, compute_dtype
    )

    # Create input tensors
    x = torch.randn(num_tokens, hidden_size, dtype=compute_dtype, device=device)

    # Create weights
    if use_int8_w8a16 or use_int8_w8a8:
        w1 = torch.randint(
            -127,
            127,
            (num_experts, shard_intermediate_size, hidden_size),
            dtype=torch.int8,
            device=device,
        )
        w2 = torch.randint(
            -127,
            127,
            (num_experts, hidden_size, shard_intermediate_size // 2),
            dtype=torch.int8,
            device=device,
        )
    else:
        w1 = torch.randn(
            num_experts,
            shard_intermediate_size,
            hidden_size,
            dtype=init_dtype,
            device=device,
        )
        w2 = torch.randn(
            num_experts,
            hidden_size,
            shard_intermediate_size // 2,
            dtype=init_dtype,
            device=device,
        )

    # Create scaling factors
    w1_scale, w2_scale, a1_scale, a2_scale = create_scale_factors(
        num_experts,
        shard_intermediate_size,
        hidden_size,
        use_int8_w8a16,
        use_int8_w8a8,
        use_fp8_w8a8,
        block_shape,
        device,
    )

    # Convert weights to FP8 if needed
    if use_fp8_w8a8:
        FP8_DTYPE = torch.float8_e4m3fn
        w1 = w1.to(FP8_DTYPE)
        w2 = w2.to(FP8_DTYPE)

    # Create gating input
    input_gating = torch.randn(
        num_tokens, num_experts, dtype=torch.float32, device=device
    )

    return x, w1, w2, w1_scale, w2_scale, a1_scale, a2_scale, input_gating, quant_dtype
