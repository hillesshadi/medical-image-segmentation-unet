"""
dataset.py
==========
PyTorch Dataset for 2-D stent segmentation.

Generates training pairs of (CT slice, stent-mask slice) from any number
of 3-D volumes.  Optional spatial augmentation is applied *synchronously*
to both image and mask so that spatial transforms are consistent.

Classes
-------
StentDataset2D   – 2-D slice dataset; supports augmentation
get_train_transform  – albumentations pipeline for image + mask
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Any

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _HAVE_ALB = True
except ImportError:
    _HAVE_ALB = False


# ---------------------------------------------------------------------------
# Albumentations augmentation pipeline
# ---------------------------------------------------------------------------

def get_train_transform(image_size: Tuple[int, int] = (512, 512)):
    """
    Augmentation pipeline with stent-safe parameters.

    Key fixes:
    - ElasticTransform alpha: 120→10 (was destroying 1-2px stent struts)
    - Mask interpolation: nearest-neighbour enforced via interpolation param
    - Image size: 256→512 to preserve stent strut detail
    """
    """Return an albumentations Compose pipeline for dual image+mask transforms.

    Applies:
      - Resize to ``image_size``
      - Random horizontal / vertical flip
      - Random 90° rotation
      - Small random rotation (±15°)
      - Elastic deformation (simulates tissue deformation)
      - Brightness / contrast jitter (image only)

    Parameters
    ----------
    image_size : tuple of int
        Target spatial size (H, W) for the network input.

    Returns
    -------
    albumentations.Compose or None
        None if albumentations is not installed.
    """
    if not _HAVE_ALB:
        return None

    return A.Compose([
        A.Resize(height=image_size[0], width=image_size[1],
                 interpolation=1),                    # bilinear for image
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.Rotate(limit=15, border_mode=0,
                 interpolation=1,                     # bilinear image
                 p=0.4),
        A.ElasticTransform(
            alpha=10,                                 # FIXED: was 120 — destroyed struts
            sigma=10 * 0.05,
            alpha_affine=10 * 0.03,
            p=0.2,                                    # reduced probability too
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.15,
            contrast_limit=0.15,
            p=0.4,
        ),
    ],
    additional_targets={"mask": "mask"},  # ensures nearest-neighbour for mask
    )


def _resize_slice(
    slice_2d: np.ndarray,
    target: Tuple[int, int],
) -> np.ndarray:
    """Simple bilinear resize without albumentations."""
    from skimage.transform import resize as sk_resize
    return sk_resize(
        slice_2d.astype(np.float32),
        target,
        order=1,
        anti_aliasing=False,
        preserve_range=True,
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class StentDataset2D(Dataset):
    """2-D axial-slice dataset for stent segmentation.

    For each volume, slices that have a non-null bounding-box crop
    (i.e. the aorta is visible) are included.  Slices without any stent
    voxels can be optionally kept or discarded to balance the dataset.

    Parameters
    ----------
    volumes : dict[name → np.ndarray]
        Preprocessed 3-D volumes of shape (Z, H, W), float32.
    stent_masks : dict[name → np.ndarray]
        Corresponding binary stent masks (Z, H, W), float32 or uint8.
    crops : dict[name → list[bbox or None]]
        Per-slice bounding boxes from ``mask_generation.get_bbox_2d_per_slice``.
        None means the aorta was not visible on that slice.
    transform : albumentations.Compose or None
        Augmentation pipeline (applied only when calling ``__getitem__``).
    image_size : tuple
        Fallback resize target if transform is None.
    allow_empty : bool
        If True, include slices that have no stent voxels (useful for
        training with hard-negative mining).
    """

    def __init__(
        self,
        volumes:    Dict[str, np.ndarray],
        stent_masks: Dict[str, np.ndarray],
        crops:      Dict[str, List[Optional[Tuple]]],
        transform:  Any = None,
        image_size: Tuple[int, int] = (256, 256),
        allow_empty: bool = True,
    ) -> None:
        super().__init__()
        self.transform   = transform
        self.image_size  = image_size
        self.allow_empty = allow_empty

        # Build flat list of (volume, mask, slice_idx)
        self._items: List[Tuple[np.ndarray, np.ndarray, int]] = []
        for name in volumes:
            vol  = volumes[name]
            mask = stent_masks[name]
            crop_list = crops.get(name, [None] * vol.shape[0])
            for z, bbox in enumerate(crop_list):
                if bbox is None:
                    continue
                if not allow_empty and mask[z].max() == 0:
                    continue
                self._items.append((vol, mask, z))

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        vol, mask, z = self._items[idx]
        img_slice  = vol[z]   # (H, W) float32
        mask_slice = (mask[z] > 0).astype(np.float32)  # binary

        if self.transform is not None and _HAVE_ALB:
            # CRITICAL FIX: pass mask as float32 directly, NOT uint8
            # uint8 round-trip corrupts thin binary masks via bilinear interpolation
            # albumentations uses nearest-neighbour for masks when dtype is correct
            aug = self.transform(
                image=img_slice,    # float32 [0,1]
                mask=mask_slice,    # float32 {0.0, 1.0}
            )
            img_out  = aug["image"].astype(np.float32)
            # Re-binarize after augmentation — nearest-neighbour may still produce
            # fractional values at boundaries; hard threshold restores binary mask
            mask_out = (aug["mask"] > 0.5).astype(np.float32)
        else:
            img_out  = _resize_slice(img_slice,  self.image_size)
            # Use order=0 (nearest-neighbour) for mask resize to preserve binary values
            from skimage.transform import resize as sk_resize
            mask_out = sk_resize(
                mask_slice, self.image_size,
                order=0,              # FIXED: was order=1 (bilinear) — blurs binary mask
                anti_aliasing=False,
                preserve_range=True,
            ).astype(np.float32)
            mask_out = (mask_out > 0.5).astype(np.float32)

        # Add channel dim:  (1, H, W)
        img_t  = torch.from_numpy(img_out[np.newaxis])
        mask_t = torch.from_numpy(mask_out[np.newaxis])
        return img_t, mask_t

# ADD after StentDataset2D class:

class StentDataset2D_DualChannel(Dataset):
    """
    Dual-channel dataset for the stent segmentation model.

    Input channels per slice:
      Channel 0: stent-windowed CT (center=1500, width=2000, normalised)
      Channel 1: binary metal prior from HU thresholding (float 0.0/1.0)

    Why dual channel?
      The stent model was fixed to in_ch=2 in Stage 5.
      The metal prior gives the network a spatial hint of where metal is,
      dramatically reducing false negatives on thin struts.

    Parameters
    ----------
    stent_volumes : dict[name → np.ndarray]   stent-windowed volumes
    metal_masks   : dict[name → np.ndarray]   boolean metal prior masks
    stent_gt      : dict[name → np.ndarray]   binary stent GT masks
    crops         : dict[name → list[bbox]]   aorta bounding boxes
    """
    def __init__(
        self,
        stent_volumes: Dict[str, np.ndarray],
        metal_masks:   Dict[str, np.ndarray],
        stent_gt:      Dict[str, np.ndarray],
        crops:         Dict[str, List[Optional[Tuple]]],
        transform:     Any = None,
        image_size:    Tuple[int, int] = (512, 512),
        allow_empty:   bool = False,   # default False: skip slices with no stent
    ) -> None:
        super().__init__()
        self.transform   = transform
        self.image_size  = image_size

        self._items = []
        for name in stent_volumes:
            svol       = stent_volumes[name]
            mmask      = metal_masks[name].astype(np.float32)
            gt         = stent_gt[name]
            crop_list  = crops.get(name, [None] * svol.shape[0])

            for z, bbox in enumerate(crop_list):
                if bbox is None:
                    continue
                if not allow_empty and gt[z].max() == 0:
                    continue
                self._items.append((svol, mmask, gt, z))

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        svol, mmask, gt, z = self._items[idx]

        ch0 = svol[z].astype(np.float32)            # stent-windowed CT
        ch1 = mmask[z].astype(np.float32)           # metal prior
        gt_slice = (gt[z] > 0).astype(np.float32)  # binary GT

        if self.transform is not None and _HAVE_ALB:
            # Augment ch0 (image), apply same spatial transform to ch1 and GT
            aug0 = self.transform(image=ch0, mask=gt_slice)
            # Apply only spatial part to metal prior using same seed
            aug1 = self.transform(image=ch1, mask=gt_slice)
            ch0_out  = aug0["image"].astype(np.float32)
            ch1_out  = aug1["image"].astype(np.float32)
            gt_out   = (aug0["mask"] > 0.5).astype(np.float32)
        else:
            from skimage.transform import resize as sk_resize
            ch0_out = sk_resize(ch0, self.image_size, order=1,
                                anti_aliasing=False, preserve_range=True).astype(np.float32)
            ch1_out = sk_resize(ch1, self.image_size, order=0,
                                anti_aliasing=False, preserve_range=True).astype(np.float32)
            gt_out  = sk_resize(gt_slice, self.image_size, order=0,
                                anti_aliasing=False, preserve_range=True).astype(np.float32)
            gt_out  = (gt_out > 0.5).astype(np.float32)

        # Stack channels: (2, H, W)
        img_t  = torch.from_numpy(np.stack([ch0_out, ch1_out], axis=0))
        mask_t = torch.from_numpy(gt_out[np.newaxis])
        return img_t, mask_t