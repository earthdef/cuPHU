# cuPHU

**CU**DA **PH**ase **U**nwrapping — GPU-accelerated InSAR phase unwrapping with multiple algorithms.

## Algorithms

cuPHU provides three unwrapping methods selectable via the `init` parameter.
**MCF** (Minimum Cost Flow) and **MST** (Minimum Spanning Tree) are GPU ports
of SNAPHU's own CPU algorithms — same cost model, same network-flow/spanning-
tree solver, same statistical formulation, just with the cost computation,
phase integration, and connected-component labeling moved to the GPU while
the inherently sequential solve itself stays on CPU. **Laplace** is a
different, newer approach written specifically for cuPHU: a weighted
least-squares formulation (Jacobi-preconditioned CG, refined by iteratively
reweighted least squares toward the same statistical-cost optimum) that runs
entirely on GPU.

| Method | `init=` | Solver | Results | When to use |
|---|---|---|---|---|
| **MCF** | `'mcf'` | GPU cost + CPU network-flow | Match SNAPHU-MCF |  **Recommended default.** Fastest option once tiled. |
| **MST** | `'mst'` | GPU cost + CPU spanning tree | To be tested | Not recommended until validated. |
| **Laplace PCG** | `'laplace'` | Runs entirely on GPU | Match MCF to within noise on most scenes |  Recommended for simplicity — no CPU thread/tile planning needed. Fastest option for one-tile. |


## Requirements

- NVIDIA GPU, compute capability ≥ 7.0 (Volta or newer)
- CUDA Toolkit ≥ 12.1
- CMake ≥ 3.18
- Python ≥ 3.9, pybind11 ≥ 2.12

## Build & install

Install into an active conda environment (uses CMake + Ninja under the hood
via scikit-build-core, and assembles the full Python package — compiled
extension plus pure-Python wrapper modules):

```bash
git clone --recursive https://github.com/earthdef/cuPHU
cd cuPHU
pip install -e . --no-build-isolation
```

> The SNAPHU source is included as a git submodule under `ext/snaphu/`.
> If you forgot `--recursive`, run `git submodule update --init --recursive`.

> A conda-forge package is planned; until then, install from source as above.

### Standalone CMake + Ninja install

You may also use CMake + Ninja install method, which allows more flexibility to control compiling options.

```bash
git clone --recursive https://github.com/earthdef/cuPHU
cd cuPHU
cmake -G Ninja -B build \
    -DCMAKE_INSTALL_PREFIX=/path/to/install \
    -DCMAKE_CUDA_ARCHITECTURES=native
ninja -C build install
```

> If `CMAKE_INSTALL_PREFIX` is the active conda environment `$CONDA_PREFIX`, cuPHU installs
> into that environment's real `site-packages` instead; otherwise it installs
> under `<prefix>/packages/cuphu` - remember to add `/path/to/install` to `PYTHONPATH`.

> If you would like to specify the GPU architectures, change `-DCMAKE_CUDA_ARCHITECTURES=120`
> for a compute capability 12.0 device, or, for a list of GPUs with different compute
> capabilities, change to, e.g., `-DCMAKE_CUDA_ARCHITECTURES="80;89"` (quoted, since `;` is a
> shell separator).

## Python API

cuPHU's Python API (`unwrap()`'s signature, the `io.InputDataset`/
`OutputDataset` protocols) is deliberately modeled on
[snaphu-py](https://github.com/isce-framework/snaphu-py)'s wrapper
design, so that code written against snaphu-py's CPU solver mostly just
works by swapping the import — cuPHU is not affiliated with that project,
but credit for the interface design belongs there.

### Quick start

Here are some examples of how to use cuPHU.

#### A simulated diagonal phase ramp

```python
import numpy as np
import cuphu

# Simulate a 256x256 interferogram containing a diagonal phase ramp.
y, x = np.ogrid[-3:3:256j, -3:3:256j]
igram = np.exp(1j * np.pi * (x + y)).astype(np.complex64)
corr  = np.ones(igram.shape, dtype=np.float32)  # noise-free coherence

unw, conncomp = cuphu.unwrap(igram, corr, nlooks=1.0, init="mcf")
```

Swap in `np.load(...)`/an HDF5 dataset/etc for real
data — `igram`/`corr` just need to be array-likes (see [`InputDataset`](#python-api)).

#### NISAR / ISCE3 — load from a RIFG product

```python
import h5py
import numpy as np
import cuphu

freq, pol = "A", "HH"
with h5py.File("RIFG.h5", "r") as f:
    ifg = f[f"science/LSAR/RIFG/swaths/frequency{freq}/interferogram"]
    igram = ifg[pol]["wrappedInterferogram"][()]
    corr  = ifg[pol]["coherenceMagnitude"][()]

    # Spacing of the wrapped interferogram itself (i.e. already reflects
    # whatever range_looks/azimuth_looks were used during crossmul -- read
    # the spacing directly rather than the looks factors + single-look
    # spacing separately, since RIFG doesn't record single-look spacing).
    rg_spacing = ifg["slantRangeSpacing"][()]
    az_spacing = ifg["sceneCenterAlongTrackSpacing"][()]

with h5py.File("reference_RSLC.h5", "r") as f:
    swath = f[f"science/LSAR/RSLC/swaths/frequency{freq}"]
    rg_bw = swath["processedRangeBandwidth"][()]
    az_bw = swath["processedAzimuthBandwidth"][()]
    v_mid = np.linalg.norm(
        f["science/LSAR/RSLC/metadata/orbit/velocity"][()].mean(axis=0)
    )

c = 299_792_458.0
nlooks = cuphu.get_effective_looks(
    range_looks=1, azimuth_looks=1,  # already baked into rg/az_spacing below
    range_spacing=rg_spacing, azimuth_spacing=az_spacing,
    range_resolution=c / (2 * rg_bw), azimuth_resolution=v_mid / az_bw,
)

unw, conncomp = cuphu.unwrap(igram, corr, nlooks, init="mcf")
```

**Work in progress:** a `cuphu` branch of
[isce3](https://github.com/earthdef/isce3) is wiring
cuPHU directly into NISAR's ISCE3 InSAR workflow.

#### ISCE2 / Sentinel-1 — from a stackSentinel product

```python
import numpy as np
import cuphu

D = "stack/merged/interferograms/20200511_20200517"
NROW, NCOL = 1847, 3498

igram = np.fromfile(f"{D}/filt_fine.int", dtype=np.complex64).reshape(NROW, NCOL)
corr  = np.fromfile(f"{D}/filt_fine.cor",  dtype=np.float32 ).reshape(NROW, NCOL)

nlooks = cuphu.get_effective_looks(
    range_looks, azimuth_looks,
    range_spacing, azimuth_spacing,
    range_resolution, azimuth_resolution,
)
unw, conncomp = cuphu.unwrap(igram, corr, nlooks, init="mcf")
```

### Full signature

```python
cuphu.unwrap(
    igram,                    # complex64 interferogram (nrow × ncol)
    corr,                     # float32 coherence in [0, 1]
    nlooks,                   # equivalent number of independent looks (>= 1)
    cost="smooth",            # 'smooth' | 'defo'  (statistical cost mode)
    init="mcf",               # 'laplace' | 'mcf' | 'mst'
    mask=None,                # uint8/bool mask — 0 means invalid pixel
    mag=None,                 # float32 amplitude (derived from igram if None)
    min_conncomp_frac=0.01,   # minimum connected component as fraction of total
    phase_grad_window=(7, 7), # boxcar averaging window for wrapped gradients
    ntiles=(1, 1),            # (row, col) tile count for large scenes
    tile_overlap=0,           # pixel overlap between adjacent tiles
    nproc=1,                  # CPU threads for parallel tile network-flow solves
    tile_cost_thresh=500,
    min_region_size=100,
    gpu_id=0,                 # CUDA device index
    unw=None,                 # pre-allocated float32 output array
    conncomp=None,            # pre-allocated uint32 output array
)
# returns: (unw: float32 ndarray, conncomp: uint32 ndarray)
```

### GPU utilities

```python
cuphu.gpu_count()     # number of available CUDA devices
cuphu.gpu_name(0)     # e.g. "NVIDIA A100-SXM4-80GB"
cuphu.gpu_info(0)     # dict: name, total_memory, sm_count, compute_capability
```

### Tiling (large scenes)

```python
# 9 CPU threads solving 3×3 = 9 tiles in parallel
unw, conncomp = cuphu.unwrap(
    igram, corr, nlooks,
    init="mcf",
    ntiles=(3, 3),
    tile_overlap=64,
    nproc=9,
)
```

Tiling splits the cost computation and solve across tiles; GPU streams overlap with CPU solver work. Each tile gets its own connected-component labels; boundary stitching re-registers the tiles' absolute phase levels afterward.

## Copyright

Copyright (c) 2026 California Institute of Technology ("Caltech").

All rights reserved.

## License

The cuPHU CUDA and Python code is released under the Apache 2.0 license.
SNAPHU (`ext/snaphu/`, incorporated directly into cuPHU's MCF/MST path) carries
Stanford's own copyright — free for any purpose, not just noncommercial.
**However**, the CS2 minimum-cost-flow solver bundled inside SNAPHU (used by
`init='mcf'`/`'mst'`) is separately copyrighted and restricted to
**noncommercial use only**. `init='laplace'` does not use CS2 and is not
subject to that restriction. See `LICENSE` for full terms.
