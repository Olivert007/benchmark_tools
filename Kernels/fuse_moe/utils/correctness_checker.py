"""
MOE Correctness Checker Module

This module provides functionality to verify the correctness of MOE implementations
by comparing against baseline implementations.
"""

import torch
from typing import List, Optional
from vllm.model_executor.layers.fused_moe import fused_moe, override_config
from vllm.model_executor.layers.fused_moe.fused_moe import (
    get_default_config,
)
from vllm.config import VllmConfig, set_current_vllm_config
from utils.inputs_builder import create_model_inputs
from utils.helper import get_config_dtype_str


def baseline_fused_moe(
    a,
    score,
    topk,
    renormalize,
    w1,
    w2,
    w1_s=None,
    w2_s=None,
    a1_scale=None,
    a2_scale=None,
    quant_dtype=torch.float16,
    per_act_token_quant=False,
    block_shape=None,
):
    """Baseline implementation for correctness comparison."""
    from tests.kernels.utils import torch_experts
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk

    topk_weight, topk_ids, _ = fused_topk(a, score, topk, renormalize)

    baseline_output = torch_experts(
        a,
        w1,
        w2,
        topk_weight,
        topk_ids,
        w1_scale=w1_s,
        w2_scale=w2_s,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        quant_dtype=quant_dtype,
        per_act_token_quant=per_act_token_quant,
        block_shape=block_shape,
    )

    return baseline_output


def correctness(
    num_tokens: int,
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
    """
    Test correctness of MOE implementation against baseline.

    Args:
        num_tokens: Number of tokens
        hidden_size: Hidden dimension size
        num_experts: Number of experts
        shard_intermediate_size: Intermediate dimension size per shard
        topk: Number of top experts per token
        use_int8_w8a8: Whether to use INT8 quantization
        use_int8_w8a16: Whether to use INT8/FP16 mixed quantization
        use_fp8_w8a8: Whether to use FP8 quantization
        block_shape: Optional block shape for quantization
        compute_dtype: Data type for computation
        device: Device to run on
    """

    if use_fp8_w8a8:
        quant_dtype = torch.float8_e4m3fn
    elif use_int8_w8a8:
        quant_dtype = torch.int8
    elif use_int8_w8a16:
        quant_dtype = torch.int8
    else:
        quant_dtype = torch.float16

    dtype_str = get_config_dtype_str(
        compute_dtype,
        use_int8_w8a16=use_int8_w8a16,
        use_fp8_w8a8=use_fp8_w8a8,
        use_int8_w8a8=use_int8_w8a8,
    )

    moe_config = get_default_config(
        num_tokens,
        num_experts,
        shard_intermediate_size,
        hidden_size,
        topk,
        dtype_str,
    )

    vllm_config = VllmConfig()
    vllm_config.scheduler_config.max_num_seqs = 128
    vllm_config.scheduler_config.max_model_len = 8192

    x, w1, w2, w1_scale, w2_scale, a1_scale, a2_scale, input_gating, quant_dtype = (
        create_model_inputs(
            num_tokens,
            hidden_size,
            num_experts,
            shard_intermediate_size,
            use_int8_w8a8,
            use_int8_w8a16,
            use_fp8_w8a8,
            block_shape,
            compute_dtype,
        )
    )

    # need to optimze
    renormalize = False
    per_act_token_quant = False

    # with set_current_vllm_config(vllm_config), override_config(moe_config):
    with override_config(moe_config):
        torch_out = baseline_fused_moe(
            x.clone(),
            input_gating.clone(),
            topk,
            renormalize,
            w1.clone(),
            w2.clone(),
            w1_s=w1_scale,
            w2_s=w2_scale,
            a1_scale=a1_scale,
            a2_scale=a2_scale,
            quant_dtype=quant_dtype,
            per_act_token_quant=per_act_token_quant,
            block_shape=block_shape,
        )

        vllm_out = fused_moe(
            x,
            w1,
            w2,
            input_gating,
            topk,
            renormalize=renormalize,
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

        print(f"[VLLM] vllm_out: {vllm_out.shape}: \n{vllm_out.detach().cpu()}")
        print(f"[torch] torch_output: {torch_out.shape}: \n{torch_out.detach().cpu()}")

        vllm_flat = vllm_out.flatten()
        torch_flat = torch_out.flatten()

        # Calculate the cosine similarity and handle special cases
        vllm_norm = torch.norm(vllm_flat)
        torch_norm = torch.norm(torch_flat)

        if vllm_norm == 0 or torch_norm == 0:
            print("Warning: One or both tensors have zero norm!")
            cos_sim = float("nan")
        elif torch.isnan(vllm_flat).any() or torch.isnan(torch_flat).any():
            print("Warning: NaN values detected in tensors!")
            cos_sim = float("nan")
        elif torch.isinf(vllm_flat).any() or torch.isinf(torch_flat).any():
            print("Warning: Inf values detected in tensors!")
            cos_sim = float("nan")
        else:
            cos_sim = torch.nn.functional.cosine_similarity(
                vllm_flat, torch_flat, dim=0
            ).item()

        print(f"Cosine similarity: {cos_sim:.6f}")
        # torch.testing.assert_close(vllm_out, torch_out, atol=0.065, rtol=0.065)
