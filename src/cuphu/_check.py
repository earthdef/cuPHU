"""Input validation helpers."""

import numpy as np


def check_2d_shape(arr: object, name: str) -> tuple[int, int]:
    """Return (nrow, ncol) of *arr*, raising ValueError if it is not 2-D."""
    arr = np.asarray(arr)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got ndim={arr.ndim}")
    return int(arr.shape[0]), int(arr.shape[1])


def check_shape_match(ref_shape: tuple[int, int], **kwargs: object) -> None:
    """Raise ValueError if any kwarg array does not match ref_shape."""
    for name, arr in kwargs.items():
        arr = np.asarray(arr)
        if arr.shape != ref_shape:
            raise ValueError(
                f"{name} shape {arr.shape} does not match expected {ref_shape}"
            )


def check_complex_dtype(**kwargs: object) -> None:
    for name, arr in kwargs.items():
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.complexfloating):
            raise TypeError(f"{name} must be complex, got dtype={arr.dtype}")


def check_float_dtype(**kwargs: object) -> None:
    for name, arr in kwargs.items():
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.floating):
            raise TypeError(f"{name} must be floating-point, got dtype={arr.dtype}")


def check_integer_dtype(**kwargs: object) -> None:
    for name, arr in kwargs.items():
        arr = np.asarray(arr)
        if not np.issubdtype(arr.dtype, np.integer):
            raise TypeError(f"{name} must be integer, got dtype={arr.dtype}")


def check_bool_or_byte_dtype(**kwargs: object) -> None:
    for name, arr in kwargs.items():
        arr = np.asarray(arr)
        if arr.dtype not in (np.dtype("bool"), np.dtype("uint8"), np.dtype("int8")):
            raise TypeError(
                f"{name} must be bool or 8-bit integer, got dtype={arr.dtype}"
            )


def check_cost_mode(cost: str) -> None:
    valid = {"smooth", "defo", "topo"}
    if cost not in valid:
        raise ValueError(f"cost must be one of {valid}, got {cost!r}")


def check_init_method(init: str) -> None:
    valid = {"mst", "mcf", "laplace"}
    if init not in valid:
        raise ValueError(f"init must be one of {valid}, got {init!r}")
