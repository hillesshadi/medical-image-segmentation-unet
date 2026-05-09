"""
inference.py
============
Slice-by-slice inference + shape-based post-processing.

``predict_volume``   – run a trained 2-D model over every aorta-cropped slice
                       of a 3-D volume and reassemble the probability mask.
``apply_shape_filter`` – lightweight CCA filter to discard obvious noise; the
                       full morphological pipeline lives in postprocessing.py.
"""

import numpy as np
import torch
import torch.nn as nn
from skimage.transform import resize as sk_resize
from skimage.measure import label as sk_label
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict_volume(
    model:       nn.Module,
    volume:      np.ndarray,
    crops:       List[Optional[Tuple[int, int, int, int]]],
    device:      torch.device,
    image_size:  Tuple[int, int] = (512, 512),     # FIXED: 256→512
    threshold:   float = 0.5,
    metal_prior: Optional[np.ndarray] = None,      # NEW: for stent model
) -> np.ndarray:
    """
    Slice-by-slice inference with optional dual-channel support.

    For aorta model:  pass volume=aorta_channel, metal_prior=None
    For stent model:  pass volume=stent_channel, metal_prior=metal_mask

    FIX: Original always built (1,1,H,W) input — stent model (in_ch=2)
    crashed immediately. Now stacks metal_prior as channel 1 when provided.

    FIX: image_size raised from 256→512 to preserve stent strut detail.
    At 256px, a 1-voxel strut in a 512px CT is sub-pixel and disappears.
    """
    """Run slice-by-slice inference on a 3-D volume.

    For each axial slice that has a valid aorta bounding-box crop, the
    relevant region is extracted, resized to ``image_size``, passed through
    the model, and the resulting probability map is pasted back into the
    full-resolution prediction volume.

    Parameters
    ----------
    model : nn.Module
        Trained segmentation model (outputs logits, shape B×1×H×W).
    volume : np.ndarray, shape (Z, Y, X), float32
        Pre-processed and enhanced CT volume.
    crops : list
        Per-slice bounding boxes from ``mask_generation.get_bbox_2d_per_slice``.
        None → skip that slice.
    device : torch.device
        Inference device.
    image_size : tuple
        Spatial input size expected by the model.
    threshold : float
        Probability threshold for binary prediction (used in shape filter).

    Returns
    -------
    pred_vol : np.ndarray, float32, shape (Z, Y, X)
        Per-voxel sigmoid probability  (0–1; NOT thresholded yet).
    """
    model = model.to(device)
    model.eval()

    Z, H, W  = volume.shape
    prob_vol = np.zeros((Z, H, W), dtype=np.float32)
    is_dual  = metal_prior is not None

    with torch.no_grad():
        for z, bbox in enumerate(crops):
            if bbox is None:
                continue

            y0, y1, x0, x1 = bbox
            crop = volume[z, y0:y1, x0:x1]
            if crop.size == 0:
                continue

            crop_r = sk_resize(crop.astype(np.float32), image_size,
                               order=1, anti_aliasing=False,
                               preserve_range=True).astype(np.float32)

            if is_dual:
                # Stent model: stack stent-windowed crop + metal prior crop
                prior_crop = metal_prior[z, y0:y1, x0:x1].astype(np.float32)
                prior_r    = sk_resize(prior_crop, image_size,
                                       order=0,   # nearest-neighbour for binary prior
                                       anti_aliasing=False,
                                       preserve_range=True).astype(np.float32)
                # (1, 2, H, W)
                inp = torch.from_numpy(
                    np.stack([crop_r, prior_r], axis=0)[np.newaxis]
                ).to(device)
            else:
                # Aorta model: single channel (1, 1, H, W)
                inp = torch.from_numpy(crop_r[np.newaxis, np.newaxis]).to(device)

            logit     = model(inp)
            prob      = torch.sigmoid(logit).cpu().numpy()[0, 0]
            prob_crop = sk_resize(prob, (y1 - y0, x1 - x0),
                                  order=1, anti_aliasing=False,
                                  preserve_range=True).astype(np.float32)

            prob_vol[z, y0:y1, x0:x1] = np.maximum(
                prob_vol[z, y0:y1, x0:x1], prob_crop
            )

    return prob_vol


# ---------------------------------------------------------------------------
# Shape filter (quick CCA on thresholded prediction)
# ---------------------------------------------------------------------------

def apply_shape_filter(
    prob_vol: np.ndarray,
    threshold: float = 0.5,
    min_voxels: int = 20,
) -> np.ndarray:
    """Threshold a probability volume and discard tiny connected components.

    A quick, lightweight filter applied immediately after inference.
    The full morphological pipeline (closing, opening, tubularity test)
    is in ``postprocessing.py``.

    Parameters
    ----------
    prob_vol : np.ndarray, float32, (Z, Y, X)
        Raw probability output from the model.
    threshold : float
        Binary threshold (default 0.5).
    min_voxels : int
        Components smaller than this (in 3-D voxels) are discarded.

    Returns
    -------
    np.ndarray, float32, (Z, Y, X)
        Binary mask (values 0.0 or 1.0).
    """
    binary = (prob_vol >= threshold).astype(np.uint8)
    labeled, n = sk_label(binary, return_num=True)
    out = np.zeros_like(binary, dtype=np.float32)
    for i in range(1, n + 1):
        if int((labeled == i).sum()) >= min_voxels:
            out[labeled == i] = 1.0
    return out
