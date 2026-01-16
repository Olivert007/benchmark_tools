"""
独立的 Triton 3D 卷积实现

原始来源: FlagGems/src/flag_gems/ops/conv3d.py
Copyright: FlagGems 项目
修改: 完全解耦，可独立使用

本模块提供高性能的 3D 卷积 Triton kernel 实现，无需任何框架依赖。
仅依赖: torch, triton
"""

import math
import os
from typing import Optional, Union, Tuple

import torch
import triton
import triton.language as tl

# AMD显卡优化控制
# USE_AMD_TRITON_CONFIGS:
#   0 = 使用默认配置（适用于NVIDIA，默认）
#   1 = 使用AMD优化配置（适用于AMD ROCm）
USE_AMD_TRITON_CONFIGS = os.getenv('USE_AMD_TRITON_CONFIGS', '0') == '1'


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

    return output


def is_triton_available() -> bool:
    """检查Triton是否可用"""
    try:
        import triton
        return torch.cuda.is_available()
    except ImportError:
        return False
