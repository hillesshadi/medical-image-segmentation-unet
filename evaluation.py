"""
evaluation.py
=============
Segmentation and cross-dataset evaluation metrics.

Metrics
-------
- Dice Similarity Coefficient (DSC)
- Intersection over Union / Jaccard Index (IoU)
- Hausdorff Distance at 95th percentile (HD95)  – optional, slow

Cross-dataset
-------------
``cross_dataset_dsc``  – pairwise DSC between stent masks of T1, T2, T3
                         (quantifies how much the stent has physically shifted
                          out of its original spatial footprint)
"""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label as sk_label
from typing import Dict, Optional, Tuple


# ---------------------------------------------------------------------------
# Voxel-level metrics
# ---------------------------------------------------------------------------

# REPLACE dice_coef WITH:

def dice_coef(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1.0,        # FIXED: was 1e-6 — caused fake DSC=1.0 on empty masks
) -> float:
    """
    DSC with empty-mask guard.

    FIX 1: smooth=1e-6 → both empty masks returned DSC=1.0 (1e-6/1e-6).
            smooth=1.0 → both empty masks return DSC=1.0 still, but see guard.
    FIX 2: Explicit guard — if BOTH masks empty, return NaN (missed detection).
            If only pred is empty (model missed it), return 0.0 (correct penalty).
    """
    p = (pred   > 0.5).astype(np.float32).ravel()
    t = (target > 0.5).astype(np.float32).ravel()

    p_sum = float(p.sum())
    t_sum = float(t.sum())

    # Both empty: ambiguous — report NaN, not 1.0
    if p_sum == 0 and t_sum == 0:
        return float("nan")

    inter = float((p * t).sum())
    return float((2.0 * inter + smooth) / (p_sum + t_sum + smooth))


# REPLACE iou_coef WITH:
def iou_coef(
    pred: np.ndarray,
    target: np.ndarray,
    smooth: float = 1.0,        # FIXED: was 1e-6
) -> float:
    p = (pred   > 0.5).astype(np.float32).ravel()
    t = (target > 0.5).astype(np.float32).ravel()

    if float(p.sum()) == 0 and float(t.sum()) == 0:
        return float("nan")

    inter = float((p * t).sum())
    union = float(p.sum() + t.sum() - inter)
    return float((inter + smooth) / (union + smooth))

def hausdorff95(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
) -> float:
    """95th-percentile Hausdorff distance.

    Uses the EDT (Euclidean Distance Transform) for efficiency.

    Parameters
    ----------
    pred, target : np.ndarray
        Binary masks.
    spacing : tuple of float or None
        Voxel spacing for physical-mm distances.  If None, distances are
        in voxels.

    Returns
    -------
    float (mm or voxels).  Returns NaN if either mask is empty.
    """
    # FIX: distance_transform_edt(sampling=...) expects (Z, Y, X) order
    # matching array axes. SimpleITK spacing is (X, Y, Z).
    # Must reverse to match array axis order.
    p_bin = (pred   > 0.5).astype(bool)
    t_bin = (target > 0.5).astype(bool)

    if not p_bin.any() or not t_bin.any():
        return float("nan")

    if spacing is not None:
        sx, sy, sz = spacing[0], spacing[1], spacing[2]
        sp = (sz, sy, sx)   # FIXED: was tuple(spacing) = (X,Y,Z) — wrong axis order
    else:
        sp = (1.0, 1.0, 1.0)

    # Distance from every pred voxel to nearest target voxel
    edt_t = distance_transform_edt(~t_bin, sampling=sp)
    dists_p2t = edt_t[p_bin]

    # Distance from every target voxel to nearest pred voxel
    edt_p = distance_transform_edt(~p_bin, sampling=sp)
    dists_t2p = edt_p[t_bin]

    all_dists = np.concatenate([dists_p2t, dists_t2p])
    return float(np.percentile(all_dists, 95))


def evaluate_segmentation(
    pred: np.ndarray,
    target: np.ndarray,
    spacing: Optional[Tuple[float, ...]] = None,
    include_hausdorff: bool = True,
) -> Dict[str, float]:
    """Compute DSC, IoU, and optionally HD95 for a single prediction.

    Returns
    -------
    dict with keys ``"DSC"``, ``"IoU"``, and (optionally) ``"HD95"``.
    """
    metrics: Dict[str, float] = {
        "DSC": dice_coef(pred, target),
        "IoU": iou_coef(pred, target),
    }
    if include_hausdorff:
        metrics["HD95"] = hausdorff95(pred, target, spacing=spacing)
    return metrics


# ---------------------------------------------------------------------------
# Cross-dataset pairwise DSC
# ---------------------------------------------------------------------------

def cross_dataset_dsc(
    masks: Dict[str, np.ndarray],
) -> Dict[Tuple[str, str], float]:
    """Compute pairwise DSC between all stent masks.

    This quantifies how much the stent has shifted out of its T1 spatial
    footprint across the three time-points.

    Parameters
    ----------
    masks : dict[name → np.ndarray]
        Binary stent masks for each dataset.

    Returns
    -------
    dict[(name_a, name_b) → DSC]
    """
    names = list(masks.keys())
    result: Dict[Tuple[str, str], float] = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            result[(a, b)] = dice_coef(masks[a], masks[b])
    return result


# ---------------------------------------------------------------------------
# Batch evaluation for all models × all datasets
# ---------------------------------------------------------------------------

def evaluate_all_segmentations(
    preds: Dict[str, Dict[str, np.ndarray]],
    stent_masks: Dict[str, np.ndarray],
    spacings: Optional[Dict[str, Tuple[float, ...]]] = None,
    include_hausdorff: bool = True,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Evaluate every model on every dataset.

    Parameters
    ----------
    preds : dict[model_name → dict[dataset_name → pred_vol]]
    stent_masks : dict[dataset_name → gt_vol]
    spacings : dict[dataset_name → tuple] or None
    include_hausdorff : bool

    Returns
    -------
    dict[model_name → dict[dataset_name → {metric: value}]]
    """
    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for mname, pred_by_ds in preds.items():
        results[mname] = {}
        for ds_name, pred_vol in pred_by_ds.items():
            gt = stent_masks[ds_name]
            sp = spacings[ds_name] if spacings else None
            results[mname][ds_name] = evaluate_segmentation(
                pred_vol, gt, spacing=sp, include_hausdorff=include_hausdorff
            )
    return results

# ADD after evaluate_all_segmentations:

def evaluate_all_structures(
    aorta_preds:  Dict[str, Dict[str, np.ndarray]],
    stent_preds:  Dict[str, Dict[str, np.ndarray]],
    aorta_masks:  Dict[str, np.ndarray],
    stent_masks:  Dict[str, np.ndarray],
    spacings:     Optional[Dict[str, Tuple[float, ...]]] = None,
    include_hausdorff: bool = True,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """
    Evaluate aorta and stent models separately.

    Returns
    -------
    {
      "aorta": {model_name: {dataset_name: {DSC, IoU, HD95}}},
      "stent": {model_name: {dataset_name: {DSC, IoU, HD95}}}
    }

    Why separate?
    Aorta DSC ~0.85-0.95 is expected. Stent DSC ~0.4-0.7 is expected due to
    thin structure. Mixing them in one dict hides stent failures behind good
    aorta numbers.
    """
    return {
        "aorta": evaluate_all_segmentations(
            aorta_preds, aorta_masks, spacings, include_hausdorff
        ),
        "stent": evaluate_all_segmentations(
            stent_preds, stent_masks, spacings, include_hausdorff
        ),
    }