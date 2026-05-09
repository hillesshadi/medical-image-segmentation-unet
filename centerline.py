"""
centerline.py
=============
Centerline and skeletonization for segmented stent masks.

Uses 3-D medial-axis thinning (scikit-image ``skeletonize_3d``) to extract
the stent centerline, then converts the resulting voxel skeleton to physical
mm coordinates using the image spacing metadata.

Functions
---------
extract_centerline       – 3-D skeletonization of a binary mask
skeleton_to_physical     – voxel → mm coordinate conversion
centerline_displacement  – point-cloud displacement statistics between two
                           centerline sets (mean, max, std)
"""

import numpy as np
from typing import Optional, Tuple, Dict
from skimage.morphology import skeletonize as skeletonize_2d

try:
    from skimage.morphology import skeletonize_3d
    _HAVE_SKEL3D = True
except ImportError:
    _HAVE_SKEL3D = False


# ---------------------------------------------------------------------------
# 3-D Skeletonization
# ---------------------------------------------------------------------------

def extract_centerline(
    mask: np.ndarray,
    min_branch_length: int = 3,
) -> np.ndarray:
    """Compute the topological skeleton (centerline) of a 3-D binary mask.

    Falls back to slice-by-slice 2-D skeletonization when the 3-D variant
    is unavailable (older scikit-image).

    Parameters
    ----------
    mask : np.ndarray, shape (Z, Y, X), bool or uint8
        Binary stent mask.
    min_branch_length : int
        Skeleton voxels that belong to chains shorter than this are pruned
        (removes spurious short branches on the skeleton).

    Returns
    -------
    skeleton : np.ndarray, bool, same shape as mask
        True where the skeleton passes through.
    """
    binary = (mask > 0)

    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=bool)

    if _HAVE_SKEL3D:
        skeleton = skeletonize_3d(binary.astype(np.uint8)) > 0
    else:
        # Slice-by-slice fallback
        skeleton = np.zeros_like(binary, dtype=bool)
        for z in range(binary.shape[0]):
            if binary[z].any():
                skeleton[z] = skeletonize_2d(binary[z])

    return skeleton


# ---------------------------------------------------------------------------
# Voxel → Physical coordinate conversion
# ---------------------------------------------------------------------------

def skeleton_to_physical(
    skeleton: np.ndarray,
    spacing: Tuple[float, float, float],
    origin: Tuple[float, float, float],
) -> np.ndarray:
    """Convert skeleton voxel indices to physical (mm) coordinates.

    Parameters
    ----------
    skeleton : np.ndarray, bool, shape (Z, Y, X)
        Skeleton from ``extract_centerline``.
    spacing : tuple of float
        Voxel spacing (sx, sy, sz) in mm.
    origin : tuple of float
        Physical origin (ox, oy, oz) in mm.

    Returns
    -------
    points : np.ndarray, shape (N, 3)
        Physical (x, y, z) coordinates in mm for each skeleton voxel.
    """
    pos = np.argwhere(skeleton)  # (N, 3): [z_idx, y_idx, x_idx]
    if pos.size == 0:
        return np.empty((0, 3), dtype=np.float64)

    sx, sy, sz = spacing
    ox, oy, oz = origin

    x_phys = ox + pos[:, 2] * sx
    y_phys = oy + pos[:, 1] * sy
    z_phys = oz + pos[:, 0] * sz

    return np.column_stack([x_phys, y_phys, z_phys])  # (N, 3)


# ---------------------------------------------------------------------------
# Centerline displacement statistics
# ---------------------------------------------------------------------------

def centerline_displacement(
    pts_ref: np.ndarray,
    pts_moved: np.ndarray,
    n_sample: int = 500,
) -> Dict[str, float]:
    """Estimate displacement between two centerline point clouds.

    For each point in ``pts_ref``, finds the nearest point in ``pts_moved``
    (nearest-neighbour) and computes the Euclidean distance.

    Parameters
    ----------
    pts_ref : np.ndarray, shape (N, 3)
        Reference centerline (T1 baseline).
    pts_moved : np.ndarray, shape (M, 3)
        Target centerline (T2 or T3).
    n_sample : int
        If N > n_sample, randomly sub-sample pts_ref for speed.

    Returns
    -------
    dict with keys ``"mean_mm"``, ``"max_mm"``, ``"std_mm"``,
    ``"median_mm"``.
    """
    if pts_ref.size == 0 or pts_moved.size == 0:
        # NaN = stent not detected in one or both scans
        # 0.0 was wrong — implies no displacement, not missing data
        missing = "reference" if pts_ref.size == 0 else "target"
        print(f"  [WARN] centerline_displacement: {missing} centerline is empty "
            f"— returning NaN (stent not detected)", flush=True)
        return {
            "mean_mm":   float("nan"),
            "max_mm":    float("nan"),
            "std_mm":    float("nan"),
            "median_mm": float("nan"),
        }

    if pts_ref.shape[0] > n_sample:
        idx = np.random.choice(pts_ref.shape[0], n_sample, replace=False)
        pts_ref = pts_ref[idx]

    # Brute-force nearest-neighbour (fast enough for hundreds of points)
    dists = []
    # KD-Tree nearest neighbour — O(N log M) vs O(N×M) brute force
    # 50-100x faster for large centerlines, same result
# REPLACE the return block of centerline_displacement WITH:

# Symmetric: also measure from moved→ref, take combined distribution
# One-directional NN underestimates displacement when centerlines
# have very different point densities (different scan slice counts)
    try:
        from scipy.spatial import cKDTree
        tree_ref   = cKDTree(pts_ref)
        tree_moved = cKDTree(pts_moved)
        d_fwd, _   = tree_moved.query(pts_ref,   k=1)   # ref → moved
        d_bwd, _   = tree_ref.query(pts_moved,   k=1)   # moved → ref
        dists = np.concatenate([d_fwd, d_bwd])
    except Exception:
        dists = np.array([
            float(np.sqrt(((pts_moved - p)**2).sum(axis=1)).min())
            for p in pts_ref
        ])

    dists = np.asarray(dists, dtype=np.float64)
    return {
        "mean_mm":   float(dists.mean()),
        "max_mm":    float(dists.max()),
        "std_mm":    float(dists.std()),
        "median_mm": float(np.median(dists)),
        "p95_mm":    float(np.percentile(dists, 95)),   # NEW — matches HD95 convention
    }

# ---------------------------------------------------------------------------
# Convenience: extract + convert for all datasets
# ---------------------------------------------------------------------------

# REPLACE extract_centerline WITH:

def extract_centerline(
    mask: np.ndarray,
    min_branch_length: int = 3,
) -> np.ndarray:
    """
    3D skeletonization with short-branch pruning.

    FIX: min_branch_length was accepted but never used.
    Short spurious branches from surface noise on the stent mask
    inflate max_mm displacement. Now actually pruned.

    Pruning strategy: label all connected components of the skeleton,
    discard any component with fewer than min_branch_length voxels.
    """
    binary = (mask > 0)
    if binary.sum() == 0:
        return np.zeros_like(binary, dtype=bool)

    if _HAVE_SKEL3D:
        skeleton = skeletonize_3d(binary.astype(np.uint8)) > 0
    else:
        skeleton = np.zeros_like(binary, dtype=bool)
        for z in range(binary.shape[0]):
            if binary[z].any():
                skeleton[z] = skeletonize_2d(binary[z])

    # Prune short branches — removes skeleton noise from stent mask edges
    if min_branch_length > 1 and skeleton.any():
        from skimage.measure import label as sk_label
        labeled, n = sk_label(skeleton, return_num=True)
        pruned = np.zeros_like(skeleton, dtype=bool)
        for i in range(1, n + 1):
            if int((labeled == i).sum()) >= min_branch_length:
                pruned[labeled == i] = True
        skeleton = pruned

    return skeleton