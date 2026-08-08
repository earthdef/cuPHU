"""
cuphu — GPU-accelerated phase unwrapping.

Quick start::

    import numpy as np
    import cuphu

    igram   = ...   # complex64 interferogram
    corr    = ...   # float32 coherence
    nlooks  = 23.8  # equivalent number of independent looks

    unw, conncomp = cuphu.unwrap(igram, corr, nlooks)

"""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from cuphu._unwrap import unwrap
from cuphu._conncomp import regrow_conncomp
from cuphu._gpu import gpu_count, gpu_info, gpu_name
from cuphu._looks import get_effective_looks

try:
    __version__ = _pkg_version("cuphu")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    "get_effective_looks",
    "gpu_count",
    "gpu_info",
    "gpu_name",
    "regrow_conncomp",
    "unwrap",
]
