"""
Conv3D Kernel 实现和基准测试工具

提供基于 Triton 的高性能 3D 卷积实现
"""

from .triton_conv3d import (
    triton_conv3d,
    is_triton_available,
)

__all__ = [
    'triton_conv3d',
    'is_triton_available',
]

__version__ = '0.1.0'
