/**
 * cuphu_bridge_gpu.cu
 *
 * GPU support for native phase-bridging (cuphu_bridge): reconciling whole-2pi
 * offsets between disconnected regions of an unwrapped-phase array, matching
 * isce3's bridge_unwrapped_phase() (python/packages/isce3/unwrap/bridge_phase.py).
 *
 * Region labeling here is deliberately a *different* connectivity notion than
 * cuphu_conncomp.cu's SNAPHU-confidence labeling (phase agreement + coherence
 * + incremental cost): bridging only cares whether a pixel was assigned any
 * unwrapped value at all. A pixel is valid iff unw != 0, and two valid pixels
 * are connected under full 8-connectivity -- exactly
 * scipy.ndimage.label(unw != 0, structure=np.ones((3,3))) on the Python side.
 * That's a simpler predicate than cuphu_conncomp.cu's, so this file defines
 * its own small label-propagation kernel family rather than reusing
 * cuphu_conncomp.cu's (which has the SNAPHU predicate baked into the neighbor
 * loop). The *algorithm* -- single-buffered atomicMin merge + pointer-jumping
 * compression, invariant labels[x] <= x -- is the same proven idiom from that
 * file; see its header comment for why this converges in O(log diameter)
 * rounds and why single-buffered races are benign.
 */

#include "cuphu.h"
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <utility>

/* ── kernel: initialize labels to own pixel index ────────────────────────── */
__global__ void BridgeInitLabelsKernel(uint32_t *labels, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) labels[idx] = (uint32_t)idx;
}

/* ── kernel: mask unw==0 pixels with the sentinel (not part of any region) ── */
__global__ void BridgeMaskZeroKernel(
    uint32_t    *labels,
    const float *unw,
    int          n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (unw[idx] == 0.0f) labels[idx] = 0xFFFFFFFFu;
}

/* ── kernel: one local min-merge pass over full 8-connectivity ──────────── */
__global__ void BridgeMergeLabels8Kernel(
    uint32_t *labels,
    int       nrow,
    int       ncol,
    int      *d_changed
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow || col >= ncol) return;

    int idx = row * ncol + col;
    uint32_t cur = labels[idx];
    if (cur == 0xFFFFFFFFu) return;   /* unw==0 here — not part of any region */

    uint32_t best = cur;

    /* full 8-connectivity, matching scipy's structure=np.ones((3,3)) */
    static const int drow[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    static const int dcol[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        int nr = row + drow[d];
        int nc = col + dcol[d];
        if (nr < 0 || nr >= nrow || nc < 0 || nc >= ncol) continue;
        int nidx = nr * ncol + nc;
        uint32_t nlbl = labels[nidx];
        if (nlbl == 0xFFFFFFFFu) continue;
        if (nlbl < best) best = nlbl;
    }

    if (best < cur) {
        atomicMin(&labels[idx], best);
        atomicExch(d_changed, 1);
    }
}

/* ── kernel: pointer jumping / path compression ───────────────────────────── */
__global__ void BridgeJumpLabelsKernel(
    uint32_t *labels,
    int       n,
    int      *d_changed
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t l = labels[idx];
    if (l == 0xFFFFFFFFu) return;
    uint32_t ll = labels[l];
    if (ll < l) {
        labels[idx] = ll;
        atomicExch(d_changed, 1);
    }
}

/* ── kernel: count component sizes ───────────────────────────────────────── */
__global__ void BridgeCountComponentsKernel(
    uint32_t       *counts,
    const uint32_t *labels,
    int             n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t root = labels[idx];
    if (root == 0xFFFFFFFFu) return;
    atomicAdd(counts + root, 1u);
}

/* ── kernel: assign final compact labels ─────────────────────────────────── */
__global__ void BridgeRelabelKernel(
    uint32_t       *labels,
    const uint32_t *map,
    int             n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    uint32_t root = labels[idx];
    if (root == 0xFFFFFFFFu) {
        labels[idx] = 0u;
        return;
    }
    labels[idx] = map[root];
}

/* ── host helper: compact label map, keeping every nonempty root (no size
 * filtering here -- min_num_pixel pruning is a separate CPU-side stage,
 * matching the plan's stage table) ──────────────────────────────────────── */
static void bridge_build_label_map(
    const uint32_t *h_counts,
    uint32_t       *h_map,
    int             npix
) {
    uint32_t next_label = 1;
    for (int i = 0; i < npix; ++i)
        h_map[i] = (h_counts[i] > 0) ? next_label++ : 0u;
}

/* ── kernel: binary erosion survival test over (labels != 0) ────────────── */
/*
 * A foreground pixel "survives" iff every SE-offset neighbor is also
 * foreground (out-of-bounds counts as background) -- matches
 * scipy.ndimage.binary_erosion(labels > 0, structure=se, border_value=0).
 * SE offsets are passed as a small flat device array (square or circular,
 * built host-side -- erosion_size is tiny (NISAR default 2), so the offset
 * count is at most a few dozen).
 */
__global__ void BridgeErodeKernel(
    const uint32_t *labels,
    uint8_t         *eroded_out,
    int              nrow,
    int              ncol,
    const int2      *se,
    int              se_n
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow || col >= ncol) return;

    int idx = row * ncol + col;
    if (labels[idx] == 0u) { eroded_out[idx] = 0; return; }

    bool survives = true;
    for (int k = 0; k < se_n; ++k) {
        int nr = row + se[k].x;
        int nc = col + se[k].y;
        if (nr < 0 || nr >= nrow || nc < 0 || nc >= ncol ||
            labels[nr * ncol + nc] == 0u) { survives = false; break; }
    }
    eroded_out[idx] = survives ? 1 : 0;
}

/* ── kernel: OR-reduce per-label survival flags ──────────────────────────── */
__global__ void BridgeSurvivesReduceKernel(
    const uint32_t *labels,
    const uint8_t   *eroded,
    uint8_t         *survives_at_all,   /* size num_label+1 */
    int              n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (eroded[idx]) survives_at_all[labels[idx]] = 1;
}

/* ── kernel: apply a per-label remap in place (0 stays 0) ────────────────── */
__global__ void BridgeRemapLabelsKernel(
    uint32_t       *labels,
    const uint32_t *remap,   /* size num_label+1, remap[0] must be 0 */
    int             n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    labels[idx] = remap[labels[idx]];
}

/* ── public entry point: erosion-based small/thin-region pruning ────────── */
/*
 * Pure survival test, matching isce3's label_conn_comp (square SE) /
 * label_boundary (circular SE) two-stage behavior: a region's PIXELS keep
 * their original (non-eroded) shape; only a region with zero surviving
 * pixels under erosion is dropped entirely. labels is modified in place
 * (compact 1..num_regions_out after dropping); erosion_size<=0 or
 * num_label_in==0 is a no-op.
 */
extern "C"
void cuphu_bridge_erode_prune_gpu(
    uint32_t *h_labels,     /* nrow*ncol, in/out                          */
    int       nrow,
    int       ncol,
    int       num_label_in,
    int       erosion_size,
    int       circular,     /* 0 = square SE, nonzero = circular SE       */
    int       gpu_id,
    int      *num_label_out
) {
    if (erosion_size <= 0 || num_label_in == 0) {
        *num_label_out = num_label_in;
        return;
    }

    CUDA_CHECK(cudaSetDevice(gpu_id));
    int npix = nrow * ncol;

    /* build SE offsets host-side (tiny: erosion_size is a handful of px) */
    std::vector<int2> h_se;
    for (int dr = -erosion_size; dr <= erosion_size; ++dr) {
        for (int dc = -erosion_size; dc <= erosion_size; ++dc) {
            if (circular && (dr * dr + dc * dc) > erosion_size * erosion_size) continue;
            h_se.push_back(make_int2(dr, dc));
        }
    }

    DevArray<uint32_t> d_labels(h_labels, (size_t)npix);
    DevArray<int2>      d_se(h_se.data(), h_se.size());
    DevArray<uint8_t>   d_eroded((size_t)npix);

    int  threads  = 256;
    int  blocks1d = (npix + threads - 1) / threads;
    dim3 block2d(16, 16);
    dim3 grid2d(
        (ncol + block2d.x - 1) / block2d.x,
        (nrow + block2d.y - 1) / block2d.y
    );

    BridgeErodeKernel<<<grid2d, block2d>>>(
        d_labels.get(), d_eroded.get(), nrow, ncol, d_se.get(), (int)h_se.size());

    DevArray<uint8_t> d_survives((size_t)num_label_in + 1);
    d_survives.fill_zero();
    BridgeSurvivesReduceKernel<<<blocks1d, threads>>>(
        d_labels.get(), d_eroded.get(), d_survives.get(), npix);

    std::vector<uint8_t> h_survives((size_t)num_label_in + 1);
    d_survives.to_host(h_survives.data());

    std::vector<uint32_t> h_remap((size_t)num_label_in + 1, 0u);
    uint32_t next_label = 1;
    for (int i = 1; i <= num_label_in; ++i)
        if (h_survives[i]) h_remap[i] = next_label++;

    DevArray<uint32_t> d_remap(h_remap.data(), h_remap.size());
    BridgeRemapLabelsKernel<<<blocks1d, threads>>>(d_labels.get(), d_remap.get(), npix);

    d_labels.to_host(h_labels);
    *num_label_out = (int)next_label - 1;
}

/* ── kernel: 3x3 max/min-filter disagreement boundary predicate ─────────── */
/*
 * Matches isce3's fixed get_all_bridge() exactly: a labeled pixel is a
 * boundary pixel iff its 3x3 neighborhood (scipy default 'reflect' border
 * mode) spans more than one label value. For a radius-1 filter, 'reflect'
 * at an out-of-bounds offset is identical to using the center pixel's own
 * value, so that's what's used here for out-of-range neighbors -- this
 * also correctly avoids flagging a homogeneous region's image-edge pixels
 * as boundary just because they're at the image edge.
 */
__global__ void BridgeBoundaryKernel(
    const uint32_t *labels,
    uint8_t         *boundary_out,
    int              nrow,
    int              ncol
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= nrow || col >= ncol) return;

    int idx = row * ncol + col;
    uint32_t center = labels[idx];

    uint32_t mx = center, mn = center;
    #pragma unroll
    for (int dr = -1; dr <= 1; ++dr) {
        #pragma unroll
        for (int dc = -1; dc <= 1; ++dc) {
            if (dr == 0 && dc == 0) continue;
            int nr = row + dr, nc = col + dc;
            uint32_t v = (nr < 0 || nr >= nrow || nc < 0 || nc >= ncol)
                             ? center : labels[nr * ncol + nc];
            mx = max(mx, v);
            mn = min(mn, v);
        }
    }

    boundary_out[idx] = (mx != mn) && (center > 0);
}

/* ── public entry point: 3x3 max/min-filter boundary extraction ─────────── */
extern "C"
void cuphu_bridge_boundary_gpu(
    const uint32_t *h_labels,
    int             nrow,
    int             ncol,
    int             gpu_id,
    uint8_t        *h_boundary_out   /* nrow*ncol, 1 = boundary pixel */
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));
    int npix = nrow * ncol;

    DevArray<uint32_t> d_labels(h_labels, (size_t)npix);
    DevArray<uint8_t>  d_boundary((size_t)npix);

    dim3 block2d(16, 16);
    dim3 grid2d(
        (ncol + block2d.x - 1) / block2d.x,
        (nrow + block2d.y - 1) / block2d.y
    );

    BridgeBoundaryKernel<<<grid2d, block2d>>>(
        d_labels.get(), d_boundary.get(), nrow, ncol);

    d_boundary.to_host(h_boundary_out);
}

/* ── grid nearest-opposite-region-boundary search ─────────────────────────
 *
 * Replaces the Python reference's O(N^2) region-pair KD-tree search (N =
 * number of regions) with a uniform spatial grid over ALL boundary points
 * pooled together, searched by an expanding ring per point. Cost per point
 * tracks local point density within a capped search radius (max_ring cells),
 * not the number of regions -- this is what actually fixes the scaling
 * problem the Python side still has after its boundary-only-points fix.
 *
 * Distance+endpoint recovery uses a single packed 64-bit atomicMin per
 * (label_p, label_q) table slot: quantized_dist in the high 40 bits (so
 * distance is always the primary sort key -- the packing is monotonic in
 * distance, which is what atomicMin needs to be correct) and the two
 * points' PER-REGION local indices (capped at max_boundary_samples,
 * enforced <=4096 = 12 bits each) in the low 24 bits. This makes the
 * winning key self-describing -- no separate racy side-channel write is
 * needed to recover which two points achieved the winning distance.
 */

static constexpr int BRIDGE_MAX_LOCAL_IDX_BITS = 12;   /* 4096 pts/region cap */
static constexpr int BRIDGE_MAX_LOCAL_IDX = 1 << BRIDGE_MAX_LOCAL_IDX_BITS;

__host__ __device__ __forceinline__ uint64_t bridge_pack_key(
    float dist, int local_p, int local_q
) {
    uint64_t qdist = (uint64_t)(dist * 64.0f);   /* 1/64 px sub-pixel precision */
    uint64_t idxbits = ((uint64_t)local_p << BRIDGE_MAX_LOCAL_IDX_BITS) |
                        (uint64_t)local_q;
    return (qdist << (2 * BRIDGE_MAX_LOCAL_IDX_BITS)) | idxbits;
}

__host__ __device__ __forceinline__ void bridge_unpack_key(
    uint64_t key, float *dist, int *local_p, int *local_q
) {
    uint64_t idxbits = key & ((1ull << (2 * BRIDGE_MAX_LOCAL_IDX_BITS)) - 1);
    *local_q = (int)(idxbits & (BRIDGE_MAX_LOCAL_IDX - 1));
    *local_p = (int)(idxbits >> BRIDGE_MAX_LOCAL_IDX_BITS);
    uint64_t qdist = key >> (2 * BRIDGE_MAX_LOCAL_IDX_BITS);
    *dist = (float)qdist / 64.0f;
}

/* one thread per boundary point; searches an expanding ring of grid cells
 * (up to max_ring) for candidate points in a DIFFERENT region, updating
 * distmat[label_p][label_q] (both orderings, for a symmetric dense table)
 * via atomicMin on the packed key. */
__global__ void BridgeNNSearchKernel(
    const int      *px, const int *py, const int *plabel, const int *plocal,
    int             num_points,
    const int      *cell_start,      /* size num_cells+1, CSR offsets */
    const int      *cell_points,     /* size num_points, point indices sorted by cell */
    int             gnrow, int gncol,
    float           cell_size,
    int             max_ring,
    unsigned long long *distmat_packed,   /* (num_label+1)*(num_label+1) */
    int                 num_label_p1      /* num_label+1, row stride     */
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) return;

    int   xi = px[i], yi = py[i], li = plabel[i], loci = plocal[i];
    int   cx = (int)(xi / cell_size), cy = (int)(yi / cell_size);

    /* NOTE: an early-stop based on "this point's single nearest
     * opposite-label neighbor" was tried and reverted -- it is unsound
     * for this kernel's actual goal (the true minimum distance for EVERY
     * (li, lj) pair, not just each point's single nearest neighbor):
     * a third region sitting geometrically "between" two others can be
     * uniformly closer from every boundary point of both, which
     * permanently blocks discovery of the farther pair from ANY point,
     * not just a few -- confirmed via a repro where a central region
     * blocked every cross-pair among 4 surrounding regions. Reachability
     * is instead guaranteed by sizing the grid (see the host-side
     * cell_size computation) so max_ring cells comfortably cover the
     * full image diagonal, and cost is bounded by keeping cells large
     * enough that the ring count needed stays small -- not by stopping
     * a given point's search early. */
    for (int ring = 0; ring <= max_ring; ++ring) {
        int cx0 = cx - ring, cx1 = cx + ring;
        int cy0 = cy - ring, cy1 = cy + ring;
        for (int gy = cy0; gy <= cy1; ++gy) {
            if (gy < 0 || gy >= gnrow) continue;
            bool edge_row = (gy == cy0 || gy == cy1);
            for (int gx = cx0; gx <= cx1; ++gx) {
                if (gx < 0 || gx >= gncol) continue;
                /* only the newly-exposed ring shell, not the interior
                 * already covered by a smaller ring */
                if (!edge_row && gx != cx0 && gx != cx1) continue;

                int cell = gy * gncol + gx;
                int s = cell_start[cell], e = cell_start[cell + 1];
                for (int k = s; k < e; ++k) {
                    int j = cell_points[k];
                    int lj = plabel[j];
                    if (lj == li) continue;
                    float dx = (float)(px[j] - xi);
                    float dy = (float)(py[j] - yi);
                    float dist = sqrtf(dx * dx + dy * dy);
                    int locj = plocal[j];

                    uint64_t key_pq = bridge_pack_key(dist, loci, locj);
                    atomicMin((unsigned long long *)&distmat_packed[li * num_label_p1 + lj],
                              (unsigned long long)key_pq);
                    uint64_t key_qp = bridge_pack_key(dist, locj, loci);
                    atomicMin((unsigned long long *)&distmat_packed[lj * num_label_p1 + li],
                              (unsigned long long)key_qp);
                }
            }
        }
    }
}

/* ── public entry point: nearest-opposite-region-boundary distmat + endpoints ── */
extern "C"
void cuphu_bridge_nn_distmat_gpu(
    const uint32_t *h_labels,
    const uint8_t  *h_boundary,
    int             nrow,
    int             ncol,
    int             num_label,
    int             max_boundary_samples,
    int             max_ring,
    int             gpu_id,
    float          *h_distmat_out,      /* (num_label+1)*(num_label+1), row-major */
    int            *h_endpoint_out      /* (num_label+1)*(num_label+1)*4: y0,x0,y1,x1 */
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    if (max_boundary_samples > BRIDGE_MAX_LOCAL_IDX)
        max_boundary_samples = BRIDGE_MAX_LOCAL_IDX;

    int n1 = num_label + 1;
    for (int i = 0; i < n1 * n1; ++i) {
        h_distmat_out[i] = -1.0f;
        h_endpoint_out[4 * i + 0] = -1;
        h_endpoint_out[4 * i + 1] = -1;
        h_endpoint_out[4 * i + 2] = -1;
        h_endpoint_out[4 * i + 3] = -1;
    }
    if (num_label <= 1) return;

    /* ── host-side compaction: per-region boundary points, uniformly
     * subsampled to max_boundary_samples -- same philosophy as isce3's own
     * deramp_max_num_sample, bounding total work to N x cap regardless of
     * any single region's true perimeter ───────────────────────────────── */
    std::vector<std::vector<std::pair<int,int>>> region_pts(n1);  /* [label] -> (row,col) */
    for (int r = 0; r < nrow; ++r) {
        for (int c = 0; c < ncol; ++c) {
            int idx = r * ncol + c;
            if (!h_boundary[idx]) continue;
            uint32_t lbl = h_labels[idx];
            if (lbl == 0 || (int)lbl > num_label) continue;
            region_pts[lbl].push_back({r, c});
        }
    }

    std::vector<int> px, py, plabel, plocal;
    for (int lbl = 1; lbl <= num_label; ++lbl) {
        auto &pts = region_pts[lbl];
        int stride = std::max(1, (int)pts.size() / max_boundary_samples);
        int local = 0;
        for (size_t k = 0; k < pts.size() && local < max_boundary_samples; k += stride) {
            px.push_back(pts[k].second);
            py.push_back(pts[k].first);
            plabel.push_back(lbl);
            plocal.push_back(local++);
        }
    }
    int num_points = (int)px.size();
    if (num_points == 0) return;

    /* ── host-side uniform grid build (CSR bucket sort) ──────────────────
     * cell_size balances two needs that pull in opposite directions:
     *  - density-based: ~2 points/cell keeps each ring's candidate list
     *    small (cheap per-ring cost).
     *  - coverage-based: max_ring cells must reach across the WHOLE image,
     *    or a region whose nearest opposite-label point happens to be
     *    farther than that is silently unreachable (found via a real
     *    full-scene test: 6 of 405 regions on a real 96M-pixel scene had
     *    no candidate within a too-small max_ring, silently getting zero
     *    correction instead of the correct one). Taking the max of both
     *    guarantees reachability without shrinking cells (and inflating
     *    per-ring cost) beyond what density alone would need. */
    float cell_size_density = std::sqrt(2.0f * (float)nrow * ncol /
                                        std::max(1, num_points));
    float image_diag = std::sqrt((float)nrow * nrow + (float)ncol * ncol);
    float cell_size_coverage = image_diag / std::max(1, max_ring);
    float cell_size = std::max(1.0f, std::max(cell_size_density, cell_size_coverage));
    int gnrow = (int)(nrow / cell_size) + 1;
    int gncol = (int)(ncol / cell_size) + 1;
    int num_cells = gnrow * gncol;

    std::vector<int> cell_of_point(num_points);
    std::vector<int> cell_count(num_cells + 1, 0);
    for (int i = 0; i < num_points; ++i) {
        int cx = (int)(px[i] / cell_size), cy = (int)(py[i] / cell_size);
        int cell = cy * gncol + cx;
        cell_of_point[i] = cell;
        cell_count[cell + 1]++;
    }
    for (int i = 0; i < num_cells; ++i) cell_count[i + 1] += cell_count[i];
    std::vector<int> cell_points(num_points);
    std::vector<int> fill_pos = cell_count;
    for (int i = 0; i < num_points; ++i)
        cell_points[fill_pos[cell_of_point[i]]++] = i;

    /* ── upload + search ──────────────────────────────────────────────── */
    DevArray<int> d_px(px.data(), num_points);
    DevArray<int> d_py(py.data(), num_points);
    DevArray<int> d_plabel(plabel.data(), num_points);
    DevArray<int> d_plocal(plocal.data(), num_points);
    DevArray<int> d_cell_start(cell_count.data(), cell_count.size());
    DevArray<int> d_cell_points(cell_points.data(), num_points);

    DevArray<unsigned long long> d_distmat_packed((size_t)n1 * n1);
    std::vector<unsigned long long> h_init((size_t)n1 * n1,
                                            ~0ull);
    CUDA_CHECK(cudaMemcpy(d_distmat_packed.get(), h_init.data(),
                          h_init.size() * sizeof(unsigned long long),
                          cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (num_points + threads - 1) / threads;
    BridgeNNSearchKernel<<<blocks, threads>>>(
        d_px.get(), d_py.get(), d_plabel.get(), d_plocal.get(), num_points,
        d_cell_start.get(), d_cell_points.get(), gnrow, gncol, cell_size,
        max_ring, d_distmat_packed.get(), n1);

    std::vector<unsigned long long> h_packed((size_t)n1 * n1);
    d_distmat_packed.to_host(h_packed.data());

    /* ── unpack winners: local index -> global pixel coords via each
     * region's flattened point list ──────────────────────────────────── */
    std::vector<std::vector<std::pair<int,int>>> region_flat_pts(n1); /* [label][local]=(row,col) */
    for (int i = 0; i < num_points; ++i)
        region_flat_pts[plabel[i]].push_back({py[i], px[i]});
    /* plocal[] was assigned in ascending order per region during the
     * compaction loop above, so region_flat_pts[lbl][local] is already
     * correctly indexed by construction */

    for (int li = 1; li <= num_label; ++li) {
        for (int lj = 1; lj <= num_label; ++lj) {
            if (li == lj) continue;
            unsigned long long key = h_packed[li * n1 + lj];
            if (key == ~0ull) continue;   /* no candidate found within max_ring */
            float dist; int loc_i, loc_j;
            bridge_unpack_key(key, &dist, &loc_i, &loc_j);
            if (loc_i >= (int)region_flat_pts[li].size() ||
                loc_j >= (int)region_flat_pts[lj].size()) continue;
            h_distmat_out[li * n1 + lj] = dist;
            auto &pi = region_flat_pts[li][loc_i];
            auto &pj = region_flat_pts[lj][loc_j];
            h_endpoint_out[4 * (li * n1 + lj) + 0] = pi.first;
            h_endpoint_out[4 * (li * n1 + lj) + 1] = pi.second;
            h_endpoint_out[4 * (li * n1 + lj) + 2] = pj.first;
            h_endpoint_out[4 * (li * n1 + lj) + 3] = pj.second;
        }
    }
}

/* ── public entry point: 8-connectivity labeling of unw != 0 ────────────── */
extern "C"
void cuphu_bridge_label_gpu(
    const float *h_unw,
    int          nrow,
    int          ncol,
    int          gpu_id,
    uint32_t    *h_labels_out,   /* nrow*ncol, compact labels 1..N, 0=unw==0 */
    int         *num_regions_out
) {
    CUDA_CHECK(cudaSetDevice(gpu_id));

    int npix = nrow * ncol;

    DevArray<float>    d_unw(h_unw, (size_t)npix);
    DevArray<uint32_t> d_labels((size_t)npix);

    int  threads  = 256;
    int  blocks1d = (npix + threads - 1) / threads;
    dim3 block2d(16, 16);
    dim3 grid2d(
        (ncol + block2d.x - 1) / block2d.x,
        (nrow + block2d.y - 1) / block2d.y
    );

    BridgeInitLabelsKernel<<<blocks1d, threads>>>(d_labels.get(), npix);
    BridgeMaskZeroKernel<<<blocks1d, threads>>>(d_labels.get(), d_unw.get(), npix);

    int *d_changed = nullptr;
    CUDA_CHECK(cudaMalloc(&d_changed, sizeof(int)));
    CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));

    static const int BATCH = 16;
    int max_iters = 2 * (nrow + ncol);

    for (int it = 0; it < max_iters; ) {
        int batch_end = std::min(it + BATCH, max_iters);
        for ( ; it < batch_end; ++it) {
            BridgeMergeLabels8Kernel<<<grid2d, block2d>>>(
                d_labels.get(), nrow, ncol, d_changed);
            BridgeJumpLabelsKernel<<<blocks1d, threads>>>(
                d_labels.get(), npix, d_changed);
        }

        int h_changed = 0;
        CUDA_CHECK(cudaMemcpy(&h_changed, d_changed, sizeof(int),
                              cudaMemcpyDeviceToHost));
        if (!h_changed) break;

        CUDA_CHECK(cudaMemset(d_changed, 0, sizeof(int)));
    }
    CUDA_CHECK(cudaFree(d_changed));

    DevArray<uint32_t> d_counts((size_t)npix);
    d_counts.fill_zero();
    BridgeCountComponentsKernel<<<blocks1d, threads>>>(
        d_counts.get(), d_labels.get(), npix);

    std::vector<uint32_t> h_counts((size_t)npix);
    d_counts.to_host(h_counts.data());

    std::vector<uint32_t> h_map((size_t)npix, 0u);
    bridge_build_label_map(h_counts.data(), h_map.data(), npix);

    DevArray<uint32_t> d_map(h_map.data(), (size_t)npix);
    BridgeRelabelKernel<<<blocks1d, threads>>>(d_labels.get(), d_map.get(), npix);

    d_labels.to_host(h_labels_out);

    uint32_t next_label = 1;
    for (int i = 0; i < npix; ++i)
        if (h_counts[i] > 0) ++next_label;
    *num_regions_out = (int)next_label - 1;
}
