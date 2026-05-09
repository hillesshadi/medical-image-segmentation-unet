"""
training.py
===========
Training loop for 2-D stent segmentation models.

Implements:
  - ``DiceLoss``  – differentiable Dice loss (stable for highly imbalanced
    foreground/background like a thin stent inside a large volume)
  - ``CombinedLoss`` – Dice + Binary Cross-Entropy
  - ``train_model``  – full training loop with tqdm progress bar and
    optional GPU support
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional

try:
    from tqdm import tqdm as _tqdm
    def _progress(iterable, **kw):
        return _tqdm(iterable, **kw)
except ImportError:
    def _progress(iterable, **kw):
        return iterable


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """Soft Dice loss for binary segmentation.

    Dice = (2 · |P ∩ T|) / (|P| + |T|)
    Loss = 1 − Dice

    A sigmoid is applied internally so the model can output raw logits.

    Parameters
    ----------
    smooth : float
        Laplace smoothing constant to avoid division by zero.
    """

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs     = torch.sigmoid(logits)
        probs_f   = probs.view(probs.size(0), -1)
        targets_f = targets.view(targets.size(0), -1).float()
        inter     = (probs_f * targets_f).sum(dim=1)
        denom     = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        dice      = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """0.5 · Dice + 0.5 · BCE (with logits).

    Combining Dice with BCE improves gradient flow early in training when
    the stent mask is nearly empty (Dice gradient ≈ 0 in that regime).

    Parameters
    ----------
    dice_weight, bce_weight : float
        Relative weights of each loss term.
    pos_weight : float or None
        Positive-class weight for ``BCEWithLogitsLoss``; pass
        ``total_negative / total_positive`` to handle class imbalance.
    """
    """
    Dice + BCE combined loss.

    FIX: pos_weight was stored as CPU tensor at init, causing device mismatch
    crash when model is on GPU. Now registered as buffer so it moves with .to(device).

    For stent: dice_weight=0.7 (dominant) — BCE alone collapses to all-background.
    For aorta: dice_weight=0.5 (balanced).
    """

    def __init__(
        self,
        dice_weight: float = 0.7,    # CHANGED from 0.5 — more Dice for imbalance
        bce_weight:  float = 0.3,    # CHANGED from 0.5
        pos_weight:  Optional[float] = None,
    ) -> None:
        super().__init__()
        self.dice   = DiceLoss()
        self.dice_w = dice_weight
        self.bce_w  = bce_weight

        # FIX: register as buffer so .to(device) moves it with the model
        if pos_weight is not None:
            self.register_buffer("pw", torch.tensor([pos_weight]))
        else:
            self.pw = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Build BCE with pos_weight on correct device at forward time
        bce_fn = nn.BCEWithLogitsLoss(
            pos_weight=self.pw.to(logits.device) if self.pw is not None else None
        )
        return (self.dice_w * self.dice(logits, targets) +
                self.bce_w  * bce_fn(logits, targets.float()))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model:           nn.Module,
    train_loader:    DataLoader,
    val_loader:      Optional[DataLoader] = None,
    device:          torch.device = torch.device("cpu"),
    epochs:          int   = 100,
    lr:              float = 3e-4,     # CHANGED from 1e-3
    pos_weight:      Optional[float] = None,
    checkpoint_path: Optional[str]  = None,
    warmup_epochs:   int   = 5,        # NEW — prevents collapse on imbalanced data
    is_stent_model:  bool  = False,    # NEW — selects loss weights per structure
) -> nn.Module:
    """Train a segmentation model for a given number of epochs.

    Parameters
    ----------
    model : nn.Module
        Un-trained (or pre-trained) PyTorch model.
    loader : DataLoader
        Training data loader producing (image, mask) pairs.
    device : torch.device
        ``cuda`` or ``cpu``.
    epochs : int
        Number of full passes through the dataset.
    lr : float
        Adam learning rate.
    pos_weight : float or None
        See ``CombinedLoss``.
    checkpoint_path : str or None
        If provided, save the best checkpoint here.

    Returns
    -------
    nn.Module
        The trained model (moved to CPU for portability).
    """
    """
    Training loop with validation, LR warmup, and gradient clipping.

    Key fixes:
    - smooth=1.0 in DiceLoss (was 1e-6 — caused exploding gradients)
    - pos_weight buffer on correct device (was CPU tensor — crashed on GPU)
    - dice_weight=0.7 for stent (was 0.5 — BCE alone cannot handle imbalance)
    - Validation loop to detect overfitting (was absent)
    - Linear LR warmup for first {warmup_epochs} epochs (was absent)
    - Best checkpoint saved on val_loss, not train_loss (was train_loss)
    """
    model = model.to(device)


    dice_w = 0.8 if is_stent_model else 0.5   # stent: Dice-dominant
    bce_w  = 0.2 if is_stent_model else 0.5
    criterion = CombinedLoss(
        dice_weight=dice_w,
        bce_weight=bce_w,
        pos_weight=pos_weight,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # Linear warmup then cosine decay
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)   # linear ramp up
        progress = (epoch - warmup_epochs) / max(epochs - warmup_epochs, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))        # cosine decay

    import numpy as np
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Training ────────────────────────────────────────────────────────
        model.train()
        train_loss = 0.0
        n_train    = 0

        pbar = _progress(train_loader, desc=f"Epoch {epoch}/{epochs} [train]", leave=False)
        for imgs, masks in pbar:
            imgs  = imgs.to(device,  non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            # Inline augmentation — random H/V flip (works on batch tensors)
            if torch.rand(1).item() > 0.5:
                imgs  = torch.flip(imgs,  dims=[-1])
                masks = torch.flip(masks, dims=[-1])
            if torch.rand(1).item() > 0.5:
                imgs  = torch.flip(imgs,  dims=[-2])
                masks = torch.flip(masks, dims=[-2])

            # Random 90° rotation
            k = torch.randint(0, 4, (1,)).item()
            if k > 0:
                imgs  = torch.rot90(imgs,  k=k, dims=[-2, -1])
                masks = torch.rot90(masks, k=k, dims=[-2, -1])

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = criterion(logits, masks)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            n_train    += 1
            if hasattr(pbar, "set_postfix"):
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_train = train_loss / max(n_train, 1)

        # ── Validation ──────────────────────────────────────────────────────
        avg_val = float("nan")
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val    = 0
            with torch.no_grad():
                for imgs, masks in val_loader:
                    imgs  = imgs.to(device,  non_blocking=True)
                    masks = masks.to(device, non_blocking=True)
                    logits = model(imgs)
                    loss   = criterion(logits, masks)
                    val_loss += loss.item()
                    n_val    += 1
            avg_val = val_loss / max(n_val, 1)

        # Log
        val_str = f"  val_loss={avg_val:.4f}" if not np.isnan(avg_val) else ""
        print(
            f"  Epoch {epoch:3d}/{epochs}  "
            f"train_loss={avg_train:.4f}{val_str}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}",
            flush=True,
        )

        # Save best checkpoint using val_loss if available, else train_loss
        monitor_loss = avg_val if not np.isnan(avg_val) else avg_train
        if checkpoint_path and monitor_loss < best_loss:
            best_loss = monitor_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
            }, checkpoint_path)

    return model.cpu()
