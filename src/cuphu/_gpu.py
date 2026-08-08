"""GPU device query helpers."""

from cuphu._ext import _cuphu_ext


def gpu_count() -> int:
    """Return the number of CUDA-capable GPUs available on this machine."""
    return _cuphu_ext.gpu_count()


def gpu_name(device_id: int = 0) -> str:
    """Return the name string of the specified CUDA device."""
    return _cuphu_ext.gpu_name(device_id)


def gpu_info(device_id: int = 0) -> dict:
    """Return a dictionary of properties for the specified CUDA device."""
    return _cuphu_ext.gpu_info(device_id)
