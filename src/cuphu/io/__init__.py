"""I/O dataset protocols for cuphu.

These follow the same interface as snaphu-py's io module so that
raster files (GeoTIFF, NetCDF, …) can be passed directly to
cuphu.unwrap() without conversion.
"""

from cuphu.io._dataset import InputDataset, OutputDataset

__all__ = ["InputDataset", "OutputDataset"]
