"""
独立的 Triton 3D 卷积实现

原始来源: FlagGems/src/flag_gems/ops/conv3d.py
Copyright: FlagGems 项目
修改: 完全解耦，可独立使用

本模块提供高性能的 3D 卷积 Triton kernel 实现，无需任何框架依赖。
仅依赖: torch, triton
"""

import logging
import math
import os
import time
import json
from typing import Optional, Union, Tuple, List, Dict, Any
from datetime import datetime

import torch
import triton
import triton.language as tl

logger = logging.getLogger(__name__)

# 日志级别控制
# TRITON_LOG_LEVEL:
#   0 = 关闭 (默认)
#   1 = 基本信息 (首次调用、配置选择)
#   2 = 详细信息 (每次调用的参数、时间)
#   3 = 调试信息 (kernel细节、内存布局)
TRITON_LOG_LEVEL = int(os.getenv('TRITON_LOG_LEVEL', '0'))

# AMD显卡优化控制
# USE_AMD_TRITON_CONFIGS:
#   0 = 使用默认配置（适用于NVIDIA，默认）
#   1 = 使用AMD优化配置（适用于AMD ROCm）
USE_AMD_TRITON_CONFIGS = os.getenv('USE_AMD_TRITON_CONFIGS', '0') == '1'

# 性能统计
_call_count = 0
_total_time = 0.0
_autotune_count = 0
_fixed_count = 0

# 数据收集（用于分析）
# 环境变量 COLLECT_CONV3D_DATA=1 启用数据收集
_COLLECT_DATA = os.getenv('COLLECT_CONV3D_DATA', '0') == '1'
_conv3d_call_data: List[Dict[str, Any]] = []


def conv3d_output_size(
    in_size: int,
    kernel_size: int,
    stride: int,
    padding: int,
    dilation: int,
) -> int:
    """
    计算3D卷积输出尺寸
    
    公式: output = (input + 2*padding - dilation*(kernel_size-1) - 1) // stride + 1
    
    Args:
        in_size: 输入尺寸
        kernel_size: 卷积核尺寸
        stride: 步长
        padding: 填充
        dilation: 膨胀率
        
    Returns:
        输出尺寸
    """
    return (in_size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


# 默认 Triton 配置搜索空间参数
def _generate_default_configs():
    """生成默认的 Triton 配置搜索空间（NVIDIA GPU 优化）"""
    # 定义搜索空间参数（较保守，适合大多数场景）
    waves_per_eu_list = [1, 2]                   # waves per execution unit
    block_ni_do_ho_wo_list = [32, 64, 128]      # 空间维度块大小
    block_ci_list = [16, 32]                     # 输入通道块大小
    block_co_list = [16, 32]                     # 输出通道块大小
    num_warps_list = [4, 8]                      # warp 数量
    num_stages_list = [3, 4]                     # pipeline stages
    
    configs = []
    
    # 生成所有组合
    for waves_per_eu in waves_per_eu_list:
        for block_ni_do_ho_wo in block_ni_do_ho_wo_list:
            for block_ci in block_ci_list:
                for block_co in block_co_list:
                    for num_warps in num_warps_list:
                        for num_stages in num_stages_list:
                            # 过滤不合理的组合
                            # 1. 大块配置需要更多 warps
                            if block_ni_do_ho_wo >= 128 and num_warps < 8:
                                continue
                            # 2. 小块配置不需要太多 warps
                            if block_ni_do_ho_wo <= 32 and num_warps > 4:
                                continue
                            
                            configs.append(
                                triton.Config(
                                    {
                                        "waves_per_eu": waves_per_eu,
                                        "BLOCK_NI_DO_HO_WO": block_ni_do_ho_wo,
                                        "BLOCK_CI": block_ci,
                                        "BLOCK_CO": block_co,
                                    },
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                )
                            )
    
    return configs

# 生成默认配置
DEFAULT_TRITON_CONFIGS = _generate_default_configs()

# AMD显卡优化配置（ROCm/CDNA/RDNA架构）
# 特点：
# 1. 扩展num_warps范围（AMD支持更多warp配置）
# 2. 更多num_stages选项（AMD的pipeline调度不同）
# 3. 优化块大小以匹配AMD的LDS（Local Data Share）容量
# 4. 考虑AMD的wavefront大小（64 vs NVIDIA的32）
# AMD 配置搜索空间参数
def _generate_amd_configs():
    """生成 AMD 优化的 Triton 配置搜索空间"""
    # 定义搜索空间参数
    waves_per_eu_list = [1, 2, 4]               # waves per execution unit (AMD 优化)
    block_ni_do_ho_wo_list = [32, 64, 128]      # 空间维度块大小
    block_ci_list = [16, 32]                     # 输入通道块大小
    block_co_list = [16, 32, 64]                 # 输出通道块大小
    num_warps_list = [4, 8, 16]                  # warp 数量
    num_stages_list = [2, 3, 4]                  # pipeline stages
    
    configs = []
    
    # 生成所有组合
    for waves_per_eu in waves_per_eu_list:
        for block_ni_do_ho_wo in block_ni_do_ho_wo_list:
            for block_ci in block_ci_list:
                for block_co in block_co_list:
                    for num_warps in num_warps_list:
                        for num_stages in num_stages_list:
                            # 过滤不合理的组合
                            # 1. 大块配置需要更多 warps
                            if block_ni_do_ho_wo >= 128 and num_warps < 8:
                                continue
                            # 2. 小块配置不需要太多 warps
                            if block_ni_do_ho_wo <= 32 and num_warps > 8:
                                continue
                            # 3. 大通道块需要更多 warps
                            if (block_ci >= 32 or block_co >= 64) and num_warps < 4:
                                continue
                            # 4. 小通道块不需要太多 stages
                            if block_ci <= 16 and block_co <= 16 and num_stages > 3:
                                continue
                            
                            configs.append(
                                triton.Config(
                                    {
                                        "waves_per_eu": waves_per_eu,
                                        "BLOCK_NI_DO_HO_WO": block_ni_do_ho_wo,
                                        "BLOCK_CI": block_ci,
                                        "BLOCK_CO": block_co,
                                    },
                                    num_warps=num_warps,
                                    num_stages=num_stages,
                                )
                            )
    
    return configs

# 生成 AMD 配置
AMD_TRITON_CONFIGS = _generate_amd_configs()

# 选择配置列表（根据环境变量）
_ACTIVE_TRITON_CONFIGS = AMD_TRITON_CONFIGS if USE_AMD_TRITON_CONFIGS else DEFAULT_TRITON_CONFIGS

# 固定配置（不使用autotune，直接运行）
# 这个配置是通用平衡配置，适合大多数场景
FIXED_CONFIG = {
    "waves_per_eu": 1,
    "BLOCK_NI_DO_HO_WO": 64,
    "BLOCK_CI": 16, 
    "BLOCK_CO": 32,
}
FIXED_NUM_WARPS = 4
FIXED_NUM_STAGES = 3


@triton.jit
def conv3d_forward_kernel_impl(
    # 指针参数
    input_pointer,
    weight_pointer,
    output_pointer,
    bias_pointer,
    # 形状参数
    in_n,
    input_depth,
    input_height,
    input_width,
    out_c,
    out_depth,
    out_height,
    out_width,
    # 步幅参数 (stride in memory, not convolution stride)
    input_n_stride,
    input_c_stride,
    input_depth_stride,
    input_height_stride,
    input_width_stride,
    weight_n_stride,
    weight_c_stride,
    weight_depth_stride,
    weight_height_stride,
    weight_width_stride,
    output_n_stride,
    output_c_stride,
    output_depth_stride,
    output_height_stride,
    output_width_stride,
    # 卷积参数 (constexpr for compilation optimization)
    weight_c: tl.constexpr,
    weight_depth: tl.constexpr,
    weight_height: tl.constexpr,
    weight_width: tl.constexpr,
    stride_depth: tl.constexpr,
    stride_height: tl.constexpr,
    stride_width: tl.constexpr,
    padding_depth: tl.constexpr,
    padding_height: tl.constexpr,
    padding_width: tl.constexpr,
    dilation_depth: tl.constexpr,
    dilation_height: tl.constexpr,
    dilation_width: tl.constexpr,
    groups: tl.constexpr,
    # 块大小参数 (由autotune配置)
    BLOCK_NI_DO_HO_WO: tl.constexpr,
    BLOCK_CI: tl.constexpr,
    BLOCK_CO: tl.constexpr,
):
    """
    Triton 3D卷积前向传播kernel
    
    并行化策略:
    - program_id(0): batch × out_depth × out_height × out_width (展平)
    - program_id(1): 输出通道
    - program_id(2): 分组维度
    
    内存访问模式:
    - 分块加载输入和权重
    - 使用矩阵乘法累加结果
    """
    # 获取当前程序块ID
    pid_ni_do_ho_wo = tl.program_id(0)  # batch和空间维度
    pid_co = tl.program_id(1)            # 输出通道
    pid_group = tl.program_id(2)         # 分组

    # ========== 计算输出位置索引 ==========
    # 将线性索引转换为4D索引 (batch, depth, height, width)
    ni_do_ho_wo_offset = pid_ni_do_ho_wo * BLOCK_NI_DO_HO_WO + tl.arange(
        0, BLOCK_NI_DO_HO_WO
    )
    ni_do_ho_offset = ni_do_ho_wo_offset // out_width
    ni_do_offset = ni_do_ho_offset // out_height
    in_n_point_value = ni_do_offset // out_depth
    output_depth_point_value = ni_do_offset % out_depth
    output_height_point_value = ni_do_ho_offset % out_height
    output_width_point_value = ni_do_ho_wo_offset % out_width

    # ========== 设置输入和权重指针 ==========
    # 输入形状: [in_n, groups, in_c, input_depth, input_height, input_width]
    # 权重形状: [groups, out_c, in_c, weight_depth, weight_height, weight_width]
    out_per_group_c = out_c // groups
    output_c_offset = pid_co * BLOCK_CO + tl.arange(0, BLOCK_CO)
    
    # 更新输入指针到当前batch和组
    input_pointer += (
        input_n_stride * in_n_point_value + input_c_stride * pid_group * weight_c
    )[:, None]
    
    # 更新权重指针到当前组和输出通道
    weight_pointer += (
        weight_n_stride * output_c_offset
        + weight_n_stride * pid_group * out_per_group_c
    )[None, :]

    # ========== 主卷积循环 ==========
    # 初始化累加器
    accum = tl.zeros((BLOCK_NI_DO_HO_WO, BLOCK_CO), dtype=tl.float32)
    
    # 计算输入通道的块数
    BLOCK_CI_COUNT = (weight_c + BLOCK_CI - 1) // BLOCK_CI
    
    # 遍历所有卷积核位置和输入通道块
    for dhwc in range(weight_depth * weight_height * weight_width * BLOCK_CI_COUNT):
        # 解析当前迭代的维度索引
        c = (dhwc % BLOCK_CI_COUNT) * BLOCK_CI  # 输入通道块
        dhw = dhwc // BLOCK_CI_COUNT
        dh = dhw // weight_width
        d = dh // weight_height   # depth维度的卷积核位置
        h = dh % weight_height    # height维度的卷积核位置
        w = dhw % weight_width    # width维度的卷积核位置

        # 计算输入通道偏移
        input_c_offset = c + tl.arange(0, BLOCK_CI)
        
        # 计算输入空间位置（考虑padding, dilation, stride）
        input_depth_offset = (
            d * dilation_depth - padding_depth + stride_depth * output_depth_point_value
        )
        input_height_offset = (
            h * dilation_height
            - padding_height
            + stride_height * output_height_point_value
        )
        input_width_offset = (
            w * dilation_width - padding_width + stride_width * output_width_point_value
        )

        # 计算当前迭代的输入和权重指针
        curr_input_pointer = (
            input_pointer
            + (input_c_stride * input_c_offset)[None, :]
            + (input_depth_stride * input_depth_offset)[:, None]
            + (input_height_stride * input_height_offset)[:, None]
            + (input_width_stride * input_width_offset)[:, None]
        )
        curr_weight_pointer = (
            weight_pointer
            + (weight_c_stride * input_c_offset)[:, None]
            + (weight_depth_stride * d)
            + (weight_height_stride * h)
            + (weight_width_stride * w)
        )

        # ========== 边界检查和掩码 ==========
        # 输入掩码：检查batch、通道和空间边界
        input_mask = (
            (in_n_point_value < in_n)[:, None]
            & (input_c_offset < weight_c)[None, :]
            & (0 <= input_depth_offset)[:, None]
            & (input_depth_offset < input_depth)[:, None]
            & (0 <= input_height_offset)[:, None]
            & (input_height_offset < input_height)[:, None]
            & (0 <= input_width_offset)[:, None]
            & (input_width_offset < input_width)[:, None]
        )
        # 权重掩码：检查通道和输出通道边界
        weight_mask = (input_c_offset < weight_c)[:, None] & (
            output_c_offset < out_per_group_c
        )[None, :]

        # ========== 加载数据并累加 ==========
        input_block = tl.load(curr_input_pointer, mask=input_mask, other=0.0)
        weight_block = tl.load(curr_weight_pointer, mask=weight_mask, other=0.0)

        # 矩阵乘法累加 (禁用TF32以保证精度)
        accum += tl.dot(input_block, weight_block, allow_tf32=False)

    # ========== 添加偏置 ==========
    bias_pointer += (pid_group[None] * out_per_group_c)[None, :] + output_c_offset[
        None, :
    ]
    mask_bias = (output_c_offset < out_per_group_c)[None, :]
    bias = tl.load(bias_pointer, mask_bias, other=0.0).to(tl.float32)
    accum += bias

    # ========== 写出结果 ==========
    # 计算输出指针
    output_pointer += (
        (output_n_stride * in_n_point_value)[:, None]
        + (output_c_stride * (pid_group * out_per_group_c + output_c_offset))[None, :]
        + (output_depth_stride * output_depth_point_value)[:, None]
        + (output_height_stride * output_height_point_value)[:, None]
        + (output_width_stride * output_width_point_value)[:, None]
    )
    
    # 输出掩码
    output_mask = (
        (in_n_point_value < in_n)[:, None]
        & (output_c_offset < out_per_group_c)[None, :]
        & (output_depth_point_value < out_depth)[:, None]
        & (output_height_point_value < out_height)[:, None]
        & (output_width_point_value < out_width)[:, None]
    )

    # 存储结果
    tl.store(output_pointer, accum, mask=output_mask)


# 带autotune的版本（自动选择最优配置，首次运行会慢）
conv3d_forward_kernel_autotune = triton.autotune(
    configs=_ACTIVE_TRITON_CONFIGS,
    key=[
        "in_n",
        "weight_c",
        "input_depth",
        "input_height",
        "input_width",
        "out_c",
        "out_depth",
        "out_height",
        "out_width",
        "weight_depth",
        "weight_height",
        "weight_width",
        "stride_depth",
        "stride_height",
        "stride_width",
        "padding_depth",
        "padding_height",
        "padding_width",
        "groups",
    ],
)(conv3d_forward_kernel_impl)

# 固定配置版本（不autotune，直接运行）
conv3d_forward_kernel_fixed = conv3d_forward_kernel_impl


def triton_conv3d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int, int]] = 1,
    padding: Union[int, Tuple[int, int, int], str] = 0,
    dilation: Union[int, Tuple[int, int, int]] = 1,
    groups: int = 1,
    use_autotune: bool = False,
) -> torch.Tensor:
    """
    Triton加速的3D卷积实现
    
    Args:
        input: 输入张量 [N, C_in, D, H, W]
        weight: 权重张量 [C_out, C_in/groups, kD, kH, kW]
        bias: 可选偏置 [C_out]
        stride: 卷积步长，可以是int或(depth, height, width)元组
        padding: 填充，可以是int、元组或'same'/'valid'字符串
        dilation: 膨胀率，可以是int或元组
        groups: 分组卷积的组数
        use_autotune: 是否使用autotune（默认False，使用固定配置更快）
        
    Returns:
        输出张量 [N, C_out, D_out, H_out, W_out]
        
    Note:
        - use_autotune=False: 使用固定配置，启动快，适合推理
        - use_autotune=True: 自动调优选择最优配置，首次慢但之后更快
    """
    global _call_count, _total_time, _autotune_count, _fixed_count
    _call_count += 1
    
    start_time = time.perf_counter() if TRITON_LOG_LEVEL >= 2 else None
    
    if TRITON_LOG_LEVEL >= 1:
        logger.info(f"\n{'='*80}")
        logger.info(f"Triton Conv3D Call #{_call_count}")
        logger.info(f"{'='*80}")
        logger.info(f"📥 Input Info:")
        logger.info(f"  ├─ Input shape: {input.shape}")
        logger.info(f"  ├─ Weight shape: {weight.shape}")
        logger.info(f"  ├─ Bias: {'Yes' if bias is not None else 'No'} {f'(shape={bias.shape})' if bias is not None else ''}")
        logger.info(f"  ├─ Input dtype: {input.dtype}")
        logger.info(f"  ├─ Weight dtype: {weight.dtype}")
        logger.info(f"  ├─ Input device: {input.device}")
        logger.info(f"  ├─ stride: {stride}")
        logger.info(f"  ├─ padding: {padding}")
        logger.info(f"  ├─ dilation: {dilation}")
        logger.info(f"  └─ groups: {groups}")
        
        # Calculate memory usage
        input_mem = input.numel() * input.element_size() / 1024 / 1024
        weight_mem = weight.numel() * weight.element_size() / 1024 / 1024
        bias_mem = bias.numel() * bias.element_size() / 1024 / 1024 if bias is not None else 0
        
        logger.info(f"💾 Memory Usage:")
        logger.info(f"  ├─ Input: {input_mem:.2f} MB ({input.numel():,} elements)")
        logger.info(f"  ├─ Weight: {weight_mem:.2f} MB ({weight.numel():,} elements)")
        logger.info(f"  └─ Bias: {bias_mem:.2f} MB" if bias is not None else "  └─ Bias: None")
        
        config_type = "AMD Optimized" if USE_AMD_TRITON_CONFIGS else "Default"
        logger.info(f"⚙️  Config Mode: {'Autotune' if use_autotune else 'Fixed'} ({config_type})")
        if use_autotune:
            logger.info(f"  └─ Testing {len(_ACTIVE_TRITON_CONFIGS)} configurations")
        
        if use_autotune:
            _autotune_count += 1
        else:
            _fixed_count += 1
    
    # ========== 参数验证 ==========
    assert weight.ndim == 5, f"Weight must be a 5D tensor, actual shape: {weight.shape}"
    assert (
        bias is None or bias.ndim == 1
    ), f"Bias must be a 1D tensor, actual shape: {bias.shape if bias is not None else None}"
    assert (
        input.shape[1] == groups * weight.shape[1]
    ), f"Input channels ({input.shape[1]}) must equal groups ({groups}) × weight channels ({weight.shape[1]})"
    assert (
        bias is None or weight.shape[0] == bias.shape[0]
    ), f"Weight output channels ({weight.shape[0]}) must equal bias length ({bias.shape[0] if bias is not None else None})"

    # ========== 解析参数 ==========
    # stride参数
    if isinstance(stride, (list, tuple)):
        stride_depth, stride_height, stride_width = stride
    else:
        stride_depth = stride_height = stride_width = stride

    # dilation参数
    if isinstance(dilation, (list, tuple)):
        dilation_depth, dilation_height, dilation_width = dilation
    else:
        dilation_depth = dilation_height = dilation_width = dilation

    # padding参数
    if isinstance(padding, str):
        if padding == "same":
            # 'same' padding要求stride=1
            assert (
                stride_depth == 1 and stride_height == 1 and stride_width == 1
            ), f"'same' padding mode does not support stride != 1, current stride: {stride}"
            
            id = input.shape[-3]
            ih = input.shape[-2]
            iw = input.shape[-1]
            kernel_size_d = weight.shape[-3]
            kernel_size_h = weight.shape[-2]
            kernel_size_w = weight.shape[-1]
            
            # 计算需要的padding
            padding_depth = math.ceil(
                (
                    stride_depth * (id - 1)
                    + 1
                    + dilation_depth * (kernel_size_d - 1)
                    - id
                )
                / 2
            )
            padding_height = math.ceil(
                (
                    stride_height * (ih - 1)
                    + 1
                    + dilation_height * (kernel_size_h - 1)
                    - ih
                )
                / 2
            )
            padding_width = math.ceil(
                (
                    stride_width * (iw - 1)
                    + 1
                    + dilation_width * (kernel_size_w - 1)
                    - iw
                )
                / 2
            )
            
            # 计算输出尺寸
            od = int(
                (id + 2 * padding_depth - dilation_depth * (kernel_size_d - 1) - 1)
                / stride_depth
                + 1
            )
            oh = int(
                (ih + 2 * padding_height - dilation_height * (kernel_size_h - 1) - 1)
                / stride_height
                + 1
            )
            ow = int(
                (iw + 2 * padding_width - dilation_width * (kernel_size_w - 1) - 1)
                / stride_width
                + 1
            )
        elif padding == "valid":
            padding_depth = padding_height = padding_width = 0
        else:
            raise ValueError(
                f"Unsupported padding string: {padding}, only 'valid' or 'same' are supported"
            )
    elif isinstance(padding, (list, tuple)):
        padding_depth, padding_height, padding_width = padding
    else:
        padding_depth = padding_height = padding_width = padding

    # ========== 计算输出尺寸 ==========
    in_n, _, input_depth, input_height, input_width = input.shape
    out_c, weight_c, weight_depth, weight_height, weight_width = weight.shape
    
    out_depth = conv3d_output_size(
        input_depth, weight_depth, stride_depth, padding_depth, dilation_depth
    )
    out_height = conv3d_output_size(
        input_height, weight_height, stride_height, padding_height, dilation_height
    )
    out_width = conv3d_output_size(
        input_width, weight_width, stride_width, padding_width, dilation_width
    )

    # ========== 准备输出张量 ==========
    output_dtype = input.dtype
    output = torch.empty(
        (in_n, out_c, out_depth, out_height, out_width),
        device=input.device,
        dtype=output_dtype,
    )

    # ========== 配置grid ==========
    # grid维度: (batch×depth×height×width的块数, 输出通道块数, groups数)
    def grid(META):
        return (
            triton.cdiv(
                in_n * out_depth * out_height * out_width, META["BLOCK_NI_DO_HO_WO"]
            ),
            triton.cdiv(out_c // groups, META["BLOCK_CO"]),
            groups,
        )

    # ========== 准备偏置 ==========
    if bias is None:
        bias_pointer = torch.zeros(out_c, device=input.device, dtype=output_dtype)
    else:
        bias_pointer = bias

    # ========== 选择kernel版本 ==========
    if use_autotune:
        # 使用autotune版本（首次运行会测试多个配置）
        kernel = conv3d_forward_kernel_autotune
        kernel[grid](
            input,
            weight,
            output,
            bias_pointer,
            in_n,
            input_depth,
            input_height,
            input_width,
            out_c,
            out_depth,
            out_height,
            out_width,
            *input.stride(),
            *weight.stride(),
            *output.stride(),
            weight_c,
            weight_depth,
            weight_height,
            weight_width,
            stride_depth,
            stride_height,
            stride_width,
            padding_depth,
            padding_height,
            padding_width,
            dilation_depth,
            dilation_height,
            dilation_width,
            groups=groups,
        )
    else:
        # 使用固定配置版本（直接运行，不autotune）
        kernel = conv3d_forward_kernel_fixed
        grid_size = (
            triton.cdiv(in_n * out_depth * out_height * out_width, FIXED_CONFIG["BLOCK_NI_DO_HO_WO"]),
            triton.cdiv(out_c // groups, FIXED_CONFIG["BLOCK_CO"]),
            groups,
        )
        kernel[grid_size](
            input,
            weight,
            output,
            bias_pointer,
            in_n,
            input_depth,
            input_height,
            input_width,
            out_c,
            out_depth,
            out_height,
            out_width,
            *input.stride(),
            *weight.stride(),
            *output.stride(),
            weight_c,
            weight_depth,
            weight_height,
            weight_width,
            stride_depth,
            stride_height,
            stride_width,
            padding_depth,
            padding_height,
            padding_width,
            dilation_depth,
            dilation_height,
            dilation_width,
            groups=groups,
            BLOCK_NI_DO_HO_WO=FIXED_CONFIG["BLOCK_NI_DO_HO_WO"],
            BLOCK_CI=FIXED_CONFIG["BLOCK_CI"],
            BLOCK_CO=FIXED_CONFIG["BLOCK_CO"],
            num_warps=FIXED_NUM_WARPS,
            num_stages=FIXED_NUM_STAGES,
        )

    # ========== 处理'same' padding的裁剪 ==========
    if padding == "same":
        output = output[..., (od - id) :, (oh - ih) :, (ow - iw) :]

    # ========== Log output ==========
    if TRITON_LOG_LEVEL >= 1:
        output_mem = output.numel() * output.element_size() / 1024 / 1024
        logger.info(f"📤 Output Info:")
        logger.info(f"  ├─ Output shape: {output.shape}")
        logger.info(f"  ├─ Output dtype: {output.dtype}")
        logger.info(f"  └─ Output memory: {output_mem:.2f} MB ({output.numel():,} elements)")
        
        if TRITON_LOG_LEVEL >= 3:
            # Detailed kernel configuration
            if use_autotune:
                logger.info(f"🔧 Kernel Config (Autotune):")
                logger.info(f"  └─ Will test {len(DEFAULT_TRITON_CONFIGS)} configs to find optimal")
            else:
                logger.info(f"🔧 Kernel Config (Fixed):")
                logger.info(f"  ├─ BLOCK_NI_DO_HO_WO: {FIXED_CONFIG['BLOCK_NI_DO_HO_WO']}")
                logger.info(f"  ├─ BLOCK_CI: {FIXED_CONFIG['BLOCK_CI']}")
                logger.info(f"  ├─ BLOCK_CO: {FIXED_CONFIG['BLOCK_CO']}")
                logger.info(f"  ├─ num_warps: {FIXED_NUM_WARPS}")
                logger.info(f"  ├─ num_stages: {FIXED_NUM_STAGES}")
                logger.info(f"  └─ grid_size: {grid_size if not use_autotune else 'dynamic'}")
                
                # Calculate theoretical FLOPS
                flops = (
                    2 * in_n * out_c * out_depth * out_height * out_width
                    * weight_c * weight_depth * weight_height * weight_width
                )
                logger.info(f"📊 Compute:")
                logger.info(f"  └─ FLOPs: {flops:,} ({flops / 1e9:.2f} GFLOPs)")
    
    # ========== Performance logging ==========
    elapsed = None
    if TRITON_LOG_LEVEL >= 2:
        elapsed = time.perf_counter() - start_time
        _total_time += elapsed
        
        logger.info(f"⏱️  Performance:")
        logger.info(f"  ├─ This call: {elapsed*1000:.3f} ms")
        logger.info(f"  ├─ Average: {_total_time*1000/_call_count:.3f} ms")
        logger.info(f"  └─ Total calls: {_call_count}")
        
        # Calculate throughput
        if elapsed > 0:
            output_elements_per_sec = output.numel() / elapsed
            logger.info(f"📈 Throughput:")
            logger.info(f"  └─ {output_elements_per_sec / 1e9:.2f} G elements/sec")
        
        logger.info(f"{'='*80}\n")
    
    # ========== Data collection for analysis ==========
    if _COLLECT_DATA:
        # Calculate elapsed time if not already calculated
        if elapsed is None and start_time is not None:
            elapsed = time.perf_counter() - start_time
        
        # Prepare stride, padding, dilation as tuples
        stride_tuple = (stride_depth, stride_height, stride_width)
        padding_tuple = (padding_depth, padding_height, padding_width)
        dilation_tuple = (dilation_depth, dilation_height, dilation_width)
        
        # Calculate memory usage
        input_mem_mb = input.numel() * input.element_size() / (1024 ** 2)
        weight_mem_mb = weight.numel() * weight.element_size() / (1024 ** 2)
        output_mem_mb = output.numel() * output.element_size() / (1024 ** 2)
        bias_mem_mb = bias.numel() * bias.element_size() / (1024 ** 2) if bias is not None else 0
        
        # Calculate FLOPs
        flops = (
            2 * in_n * out_c * out_depth * out_height * out_width
            * weight_c * weight_depth * weight_height * weight_width
        )
        
        # Calculate total parameters
        total_params = weight.numel() + (bias.numel() if bias is not None else 0)
        
        call_data = {
            'call_id': _call_count,
            'timestamp': datetime.now().isoformat(),
            # Shape information
            'input_shape': list(input.shape),
            'weight_shape': list(weight.shape),
            'output_shape': list(output.shape),
            'bias_shape': list(bias.shape) if bias is not None else None,
            # Data type information
            'input_dtype': str(input.dtype),
            'weight_dtype': str(weight.dtype),
            'output_dtype': str(output.dtype),
            # Convolution parameters
            'stride': stride_tuple,
            'padding': padding_tuple,
            'dilation': dilation_tuple,
            'groups': groups,
            # Configuration
            'use_autotune': use_autotune,
            'config_mode': 'autotune' if use_autotune else 'fixed',
            # Memory usage (MB)
            'input_memory_mb': round(input_mem_mb, 2),
            'weight_memory_mb': round(weight_mem_mb, 2),
            'output_memory_mb': round(output_mem_mb, 2),
            'bias_memory_mb': round(bias_mem_mb, 2) if bias is not None else 0,
            'total_memory_mb': round(input_mem_mb + weight_mem_mb + output_mem_mb + bias_mem_mb, 2),
            # Compute metrics
            'total_params': total_params,
            'flops': flops,
            'gflops': round(flops / 1e9, 2),
            # Performance
            'execution_time_ms': round(elapsed * 1000, 3) if elapsed is not None else None,
            'throughput_gelements_per_sec': round(output.numel() / elapsed / 1e9, 2) if elapsed is not None and elapsed > 0 else None,
        }
        
        _conv3d_call_data.append(call_data)

    return output


def is_triton_available() -> bool:
    """检查Triton是否可用"""
    try:
        import triton
        return torch.cuda.is_available()
    except ImportError:
        return False


def print_triton_stats():
    """Print Triton Conv3D statistics"""
    if _call_count == 0:
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Triton Conv3D Statistics Summary")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Call Statistics:")
    logger.info(f"  ├─ Total calls: {_call_count}")
    logger.info(f"  ├─ Autotune mode: {_autotune_count} calls")
    logger.info(f"  └─ Fixed config mode: {_fixed_count} calls")
    
    if TRITON_LOG_LEVEL >= 2 and _total_time > 0:
        logger.info(f"⏱️  Performance Statistics:")
        logger.info(f"  ├─ Total time: {_total_time:.3f} s")
        logger.info(f"  ├─ Average time: {_total_time*1000/_call_count:.3f} ms")
        logger.info(f"  └─ Throughput: {_call_count/_total_time:.2f} calls/sec")
    
    logger.info(f"{'='*80}\n")


def reset_triton_stats():
    """Reset statistics"""
    global _call_count, _total_time, _autotune_count, _fixed_count
    _call_count = 0
    _total_time = 0.0
    _autotune_count = 0
    _fixed_count = 0


# ========== Data Collection Functions ==========

def get_conv3d_call_data() -> List[Dict[str, Any]]:
    """
    Get all collected Conv3D call data
    
    Returns:
        List of dictionaries containing call information
    """
    return _conv3d_call_data.copy()


def export_conv3d_data_json(filepath: str) -> None:
    """
    Export collected Conv3D data to JSON file
    
    Args:
        filepath: Path to output JSON file
    """
    if not _conv3d_call_data:
        logger.warning("No Conv3D data collected. Set COLLECT_CONV3D_DATA=1 to enable data collection.")
        return
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(_conv3d_call_data, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ Exported {len(_conv3d_call_data)} Conv3D calls to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export Conv3D data: {e}")


def export_conv3d_data_csv(filepath: str) -> None:
    """
    Export collected Conv3D data to CSV file
    
    Args:
        filepath: Path to output CSV file
    """
    if not _conv3d_call_data:
        logger.warning("No Conv3D data collected. Set COLLECT_CONV3D_DATA=1 to enable data collection.")
        return
    
    try:
        import csv
        
        if not _conv3d_call_data:
            return
        
        # Get all possible keys
        fieldnames = set()
        for item in _conv3d_call_data:
            fieldnames.update(item.keys())
        fieldnames = sorted(fieldnames)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for item in _conv3d_call_data:
                # Convert lists/tuples to strings for CSV
                row = {}
                for key, value in item.items():
                    if isinstance(value, (list, tuple)):
                        row[key] = str(value)
                    else:
                        row[key] = value
                writer.writerow(row)
        
        logger.info(f"✓ Exported {len(_conv3d_call_data)} Conv3D calls to {filepath}")
    except Exception as e:
        logger.error(f"Failed to export Conv3D data to CSV: {e}")


def print_conv3d_data_summary() -> None:
    """Print summary statistics of collected Conv3D data"""
    if not _conv3d_call_data:
        logger.info("No Conv3D data collected. Set COLLECT_CONV3D_DATA=1 to enable data collection.")
        return
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Conv3D Data Collection Summary")
    logger.info(f"{'='*80}")
    logger.info(f"📊 Total calls collected: {len(_conv3d_call_data)}")
    
    # Analyze shape distributions
    input_shapes = {}
    output_shapes = {}
    dtypes = {}
    config_modes = {'autotune': 0, 'fixed': 0}
    
    total_flops = 0
    total_params = 0
    total_memory = 0
    total_time = 0
    
    for item in _conv3d_call_data:
        # Count shapes
        input_shape_key = str(item['input_shape'])
        input_shapes[input_shape_key] = input_shapes.get(input_shape_key, 0) + 1
        
        output_shape_key = str(item['output_shape'])
        output_shapes[output_shape_key] = output_shapes.get(output_shape_key, 0) + 1
        
        # Count dtypes
        dtype_key = item['input_dtype']
        dtypes[dtype_key] = dtypes.get(dtype_key, 0) + 1
        
        # Count config modes
        config_modes[item['config_mode']] += 1
        
        # Sum metrics
        total_flops += item['flops']
        total_params += item['total_params']
        total_memory += item['total_memory_mb']
        if item['execution_time_ms'] is not None:
            total_time += item['execution_time_ms']
    
    logger.info(f"\n📐 Shape Distribution:")
    logger.info(f"  Input shapes (top 5):")
    for shape, count in sorted(input_shapes.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"    {shape}: {count} calls")
    
    logger.info(f"  Output shapes (top 5):")
    for shape, count in sorted(output_shapes.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"    {shape}: {count} calls")
    
    logger.info(f"\n🔢 Data Types:")
    for dtype, count in dtypes.items():
        logger.info(f"  {dtype}: {count} calls")
    
    logger.info(f"\n⚙️  Configuration:")
    logger.info(f"  Autotune mode: {config_modes['autotune']} calls")
    logger.info(f"  Fixed config mode: {config_modes['fixed']} calls")
    
    logger.info(f"\n📊 Aggregate Metrics:")
    logger.info(f"  Total FLOPs: {total_flops:,} ({total_flops/1e12:.2f} TFLOPs)")
    logger.info(f"  Total parameters processed: {total_params:,}")
    logger.info(f"  Total memory usage: {total_memory:.2f} MB")
    if total_time > 0:
        logger.info(f"  Total execution time: {total_time:.2f} ms ({total_time/1000:.2f} s)")
        logger.info(f"  Average time per call: {total_time/len(_conv3d_call_data):.2f} ms")
    
    logger.info(f"{'='*80}\n")


def clear_conv3d_data() -> None:
    """Clear all collected Conv3D data"""
    global _conv3d_call_data
    _conv3d_call_data.clear()
    logger.info("✓ Cleared all Conv3D call data")


# Register to print statistics on exit
import atexit
atexit.register(print_triton_stats)

# Register to print data summary and export if data collection is enabled
if _COLLECT_DATA:
    def _export_on_exit():
        print_conv3d_data_summary()
        # Auto-export to default location
        default_json = "conv3d_call_data.json"
        default_csv = "conv3d_call_data.csv"
        export_conv3d_data_json(default_json)
        export_conv3d_data_csv(default_csv)
    
    atexit.register(_export_on_exit)


if __name__ == "__main__":
    # Simple test
    print("Triton Conv3D module loaded successfully")
    print(f"Triton available: {is_triton_available()}")
    print(f"Data collection enabled: {_COLLECT_DATA}")
    print(f"AMD optimized configs: {'ENABLED' if USE_AMD_TRITON_CONFIGS else 'DISABLED'}")
    print(f"Active config count: {len(_ACTIVE_TRITON_CONFIGS)}")
    
    if USE_AMD_TRITON_CONFIGS:
        print("\n🚀 AMD Optimization Enabled:")
        print("  - Using extended warp configurations (2-16)")
        print("  - Extended pipeline stages (2-5)")
        print("  - Optimized for RDNA/CDNA architectures")
        print("  - More block size variations for LDS optimization")
    
    if _COLLECT_DATA:
        print("\n📊 Data Collection Guide:")
        print("  1. Run your model with COLLECT_CONV3D_DATA=1")
        print("  2. Data will be auto-exported on exit to:")
        print("     - conv3d_call_data.json")
        print("     - conv3d_call_data.csv")
        print("  3. Or manually export using:")
        print("     - export_conv3d_data_json('custom_path.json')")
        print("     - export_conv3d_data_csv('custom_path.csv')")
        print("  4. View summary with: print_conv3d_data_summary()")
