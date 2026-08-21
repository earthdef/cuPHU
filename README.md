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
| **MCF** | `'mcf'` | GPU cost + CPU network-flow | Exact match to SNAPHU-MCF | **Recommended default.** Best when you need an exact match to SNAPHU (e.g. validation baselines). Auto-tiles with `nproc` for speed at scale. |
| **MST** | `'mst'` | GPU cost + CPU spanning tree | To be tested | Not recommended until validated. |
| **Laplace PCG** | `'laplace'` | Runs entirely on GPU | Match MCF to within noise on most scenes | Fastest option, especially on large scenes (auto-tiles, no CPU threads needed). Prefer when speed matters most; pair with `single_tile_reoptimize`/`fix_cycle_spikes` to clean up occasional PCG-specific artifacts on very large scenes. |


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
> capabilities, change to, e.g., `-DCMAKE_CUDA_ARCHITECTURES="80;89"`.

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

cuPHU is wired directly into NISAR's ISCE3 InSAR workflow on the
[`cuphu` branch of isce3](https://github.com/earthdef/isce3/tree/cuphu).
Select it as the `phase_unwrap` algorithm in `runconfig.yaml`:

```yaml
runconfig:
    groups:
        worker:
            # cuphu requires GPU processing to be enabled.
            gpu_enabled: True
            gpu_id: 0

        processing:
            phase_unwrap:
                # Choose 'cuphu' as algorithm; other options 'icu', 'phass', 'snaphu'
                algorithm: cuphu
                cuphu:
                    # Unwrapping algorithm: 'mcf' (follow snaphu-mcf) or 'laplace'
                    init: mcf
                    # (row, col) tile count for large scenes.
                    # Leave unset to auto-compute (toward target_tile_size px/edge)
                    ntiles: [4, 4]
                    tile_overlap: [64, 64]
                    # CPU threads for parallel tile network-flow solves (mcf only)
                    nproc: 16
                    gpu_id: 0
```

The `cuphu:` block mirrors `cuphu.unwrap()`'s keyword arguments directly
(see [Full signature](#full-signature)); any option left out falls back to
isce3's own default (see `share/nisar/defaults/insar.yaml`'s `cuphu:`
section for the full list and descriptions).

This branch also includes an option to use a different algorithm/settings for
phase unwrapping in ionosphere correction — `processing.ionosphere_phase_correction.phase_unwrap`
overrides any field of `processing.phase_unwrap` for that unwrap pass only;
fields left unset there still inherit from the main `phase_unwrap:` block above.
For example, to unwrap the main RUNW igram with `cuphu`/`mcf` but use a single-tile `snaphu` on
CPU for the ionosphere sub-band igrams, which are smoother and smaller:

```yaml
                ionosphere_phase_correction:
                    enabled: True
                    phase_unwrap:
                        algorithm: snaphu
                        snaphu:
                            ntiles: [1, 1]
                            nproc: 1
```

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
    mask_pad_distance=0,      # grow valid mask by N px before solving (see notes)
    mag=None,                 # float32 amplitude (derived from igram if None)
    min_conncomp_frac=0.01,   # minimum connected component as fraction of total
    phase_grad_window=(7, 7), # boxcar averaging window for wrapped gradients
    ntiles=None,              # (row, col) tile count; auto-computed if None
    tile_overlap=None,        # pixel overlap between adjacent tiles
    target_tile_size=1024,    # target tile edge (px), used to auto-compute ntiles
    nproc=1,                  # CPU threads for parallel tile network-flow solves
    tile_cost_thresh=500,
    min_region_size=100,
    single_tile_reoptimize=False,  # CPU re-solve of the whole scene after tiling
    fix_cycle_spikes=False,   # detect/correct isolated row/column 2π spikes
    bridge=False,             # reconcile disconnected regions' whole-cycle offsets
    bridge_radius=500,
    bridge_min_num_pixel=14,
    bridge_erosion_size=2,
    bridge_max_boundary_samples=4096,
    bridge_ramp_type=None,    # ramp to remove before bridging, add back after
    bridge_ramp_max_num_sample=1000000,
    reference_pixel=(0, 0),   # shift output to match this pixel's raw wrapped phase
    gpu_id=0,                 # CUDA device index
    unw=None,                 # pre-allocated float32 output array
    conncomp=None,            # pre-allocated uint32 output array
)
# returns: (unw: float32 ndarray, conncomp: uint32 ndarray)
```

See each parameter's docstring (`help(cuphu.unwrap)`) for full details.

### GPU utilities

```python
cuphu.gpu_count()     # number of available CUDA devices
cuphu.gpu_name(0)     # e.g. "NVIDIA A100-SXM4-80GB"
cuphu.gpu_info(0)     # dict: name, total_memory, sm_count, compute_capability
```

### Tiling (large scenes)

`ntiles` is optional — leave it unset and cuPHU auto-computes a tiling
(toward `target_tile_size` px per tile edge): always for `init='laplace'`,
and for `init='mcf'`/`'mst'` when `nproc > 1` (so the parallel tile solve
has tiles to distribute). Pass it explicitly to override:

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

Tiling splits the cost computation and solve across tiles; GPU streams overlap with CPU solver work. Each tile gets its own connected-component labels; boundary stitching re-registers the tiles' absolute phase levels afterward. Optionally follow up with `single_tile_reoptimize=True` or `bridge=True` for further cleanup.

### Water masking and bridging

A mask (0 = invalid) that excludes a water body, or any other feature,
can split the scene's valid area into separate disconnected regions. Each
region gets solved with its own independent absolute-phase offset, so two
land areas on either side of a lake can end up on different whole-cycle
levels even though they're both physically part of the same continuous
deformation field. `bridge=True` reconciles this after the solve, so the
whole scene comes out on one consistent level:

```python
unw, conncomp = cuphu.unwrap(
    igram, corr, nlooks,
    init="mcf",
    mask=water_mask,   # uint8/bool, 0 = invalid (e.g. water)
    bridge=True,
)
```

Bridging compares a median phase value on either side of each gap (within
`bridge_radius` px) to pick the whole-cycle offset between regions, so it
needs enough real signal near each gap to be reliable — tune
`bridge_min_num_pixel`/`bridge_erosion_size` down for small or thin
regions. If the scene has a steep phase gradient (e.g. a strong ramp)
across a narrow gap, the raw median comparison can pick the wrong
whole-cycle jump; `bridge_ramp_type` (e.g. `'linear'`, `'quadratic'`)
fits and removes a ramp before comparing, then adds it back after.

Combine with `mask_pad_distance` if the mask also has narrow/isolated
valid slivers near a boundary that need a bit of padding to solve
reliably.

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
