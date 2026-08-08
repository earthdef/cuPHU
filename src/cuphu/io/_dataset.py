"""Dataset protocols used by cuphu.unwrap()."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class InputDataset(Protocol):
    """Read-only array-like with a shape attribute.

    A numpy ndarray, a rasterio DatasetReader, or any object that
    implements ``__array__`` and has ``shape`` and ``ndim``.
    """

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray: ...


@runtime_checkable
class OutputDataset(Protocol):
    """Writable array-like with a shape attribute."""

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray: ...
