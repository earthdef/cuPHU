"""Lazy import for the compiled extension module."""

try:
    import cuphu._cuphu_ext as _cuphu_ext
except ImportError as e:
    raise ImportError(
        "The cuPHU CUDA extension module could not be imported. "
        "Make sure the package was built with CUDA support: "
        "pip install -e . --no-build-isolation"
    ) from e

__all__ = ["_cuphu_ext"]
