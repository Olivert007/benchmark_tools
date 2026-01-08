"""
Conv3D Kernel 实现和基准测试工具

本包包含多个卷积核心的实现:
- triton_conv3d: 基于 Triton 的 3D 卷积实现
- 更多 kernel 实现即将添加...
"""

from .triton_conv3d import (
    triton_conv3d,
    is_triton_available,
    print_triton_stats,
    reset_triton_stats,
    get_conv3d_call_data,
    export_conv3d_data_json,
    export_conv3d_data_csv,
    print_conv3d_data_summary,
    clear_conv3d_data,
)

__all__ = [
    'triton_conv3d',
    'is_triton_available',
    'print_triton_stats',
    'reset_triton_stats',
    'get_conv3d_call_data',
    'export_conv3d_data_json',
    'export_conv3d_data_csv',
    'print_conv3d_data_summary',
    'clear_conv3d_data',
]

__version__ = '0.1.0'
