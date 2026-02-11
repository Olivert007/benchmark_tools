import torch
from typing import Optional, TypedDict


def get_vllm_version():
    from vllm import __version__, __version_tuple__

    if __version__ == "dev":
        return "N/A (dev)"
    version_str = __version_tuple__[-1]
    if isinstance(version_str, str) and version_str.startswith("g"):
        # it's a dev build
        if "." in version_str:
            # it's a dev build containing local changes
            git_sha = version_str.split(".")[0][1:]
            date = version_str.split(".")[-1][1:]
            return f"{__version__} (git sha: {git_sha}, date: {date})"
        else:
            # it's a dev build without local changes
            git_sha = version_str[1:]  # type: ignore
            return f"{__version__} (git sha: {git_sha})"
    return __version__


def get_triton_version():
    """
    Get the installed Triton version using Python's package metadata.

    Returns:
        str: The version string if Triton is installed, "unknown" otherwise.
    """
    import importlib.metadata

    try:
        return importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        try:
            import pkg_resources
            return pkg_resources.get_distribution("triton").version
        except Exception:
            return "unknown"


def get_config_dtype_str(
    dtype: torch.dtype,
    use_int4_w4a16: Optional[bool] = False,
    use_int8_w8a16: Optional[bool] = False,
    use_fp8_w8a8: Optional[bool] = False,
    use_int8_w8a8: Optional[bool] = False,
    use_mxfp4_w4a4: Optional[bool] = False,
) -> Optional[str]:
    if use_fp8_w8a8:
        return "fp8_w8a8"
    elif use_int8_w8a8:
        return "int8_w8a8"
    elif use_int8_w8a16:
        return "int8_w8a16"
    elif use_int4_w4a16:
        return "int4_w4a16"
    elif use_mxfp4_w4a4:
        return "mxfp4_w4a4"
    elif dtype == torch.float16 or dtype == torch.half:
        return "float16"
    elif dtype == torch.bfloat16:
        return "bfloat16"
    return None
