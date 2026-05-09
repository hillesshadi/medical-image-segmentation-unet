"""
main.py  (FULLY CORRECTED — integrates all Stage 1-13 fixes)
=======
Complete Longitudinal Aortic Stent Displacement Analysis Pipeline

BUGS FIXED vs. original main.py
---------------------------------
BUG-M1  preprocess_all_datasets called with old single-window signature
         → now uses dual-channel (aorta_volumes, stent_volumes, metal_masks)
BUG-M2  enhance_all_datasets called with wrong parameter names
         → now passes aorta_volumes + metal_masks, correct weight/iter names
BUG-M3  register_all_to_baseline called with old 6-arg signature
         → now passes aorta_volumes, stent_volumes, raw_volumes, aorta_masks
BUG-M4  generate_all_masks called with stent_dilate_iters=8 (bone contamination)
         → now uses corrected config values (dilate=3, closing_radius=1)
BUG-M5  build_all_models used — returns single-channel aorta models only
         → now uses build_segmentation_models, gets separate aorta+stent dicts
BUG-M6  StentDataset2D (1 channel) used for stent model training
         → now uses StentDataset2D_DualChannel for stent, StentDataset2D for aorta
BUG-M7  train_model called without is_stent_model / val_loader flags
         → now passes correct flags and splits val set from training data
BUG-M8  predict_volume called without metal_prior (stent model crashes)
         → now passes metal_prior for stent inference
BUG-M9  postprocess_mask (aorta params) applied to stent masks
         → now calls postprocess_stent_mask for stent, postprocess_aorta_mask for aorta
BUG-M10 voxel_coords_all built as {"X":arr,"Y":arr,"Z":arr} but fixed histogram
         function expects tuple (xs, ys, zs)
         → now stores tuples to match corrected plot_displacement_histograms
BUG-M11 spacing passed to 3D visualization functions directly from SimpleITK
         (X,Y,Z order) — _build_image_data fix in Stage 9 handles this, but
         spacing passed to centroid/displacement functions also needs (X,Y,Z)
         → validated via validate_spacing_consistency before displacement
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from typing import Dict

# ---------------------------------------------------------------------------
# Add project root to path
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import config as cfg

# All imports — using corrected function names from Stages 1-13
from preprocessing     import preprocess_all_datasets          # returns 6-tuple now
from eadtv_enhancement import enhance_all_datasets             # takes metal_masks now
from registration      import register_all_to_baseline         # takes aorta_masks now
from mask_generation   import (
    generate_all_masks,
    get_bbox_2d_per_slice,
)
from dataset           import (
    StentDataset2D,              # aorta model — single channel
    StentDataset2D_DualChannel,  # stent model — dual channel  [BUG-M6 FIX]
    get_train_transform,
)
from models            import build_segmentation_models        # [BUG-M5 FIX]
from training          import train_model
from inference         import predict_volume, apply_shape_filter
from postprocessing    import (
    postprocess_aorta_mask,  # [BUG-M9 FIX] separate pipelines
    postprocess_stent_mask,  # [BUG-M9 FIX]
)
from evaluation        import (
    evaluate_all_structures,   # evaluates aorta + stent separately
    cross_dataset_dsc,
)
from displacement      import (
    compute_all_displacements,
    voxel_coords_physical,
    validate_spacing_consistency,   # [BUG-M11 FIX] guard before displacement
    centroid_per_slice,
)
from visualization     import (
    plot_enhancement_comparison,
    plot_segmentation_overlay,
    plot_aorta_and_stent_3d,
    plot_aorta_and_stent_3d_composite,
    plot_displacement_histograms,
    plot_longitudinal_displacement,
    plot_metrics_comparison,
)
from report import export_csv, generate_pdf_report


# ---------------------------------------------------------------------------
# Directory setup
# ---------------------------------------------------------------------------

def setup_dirs() -> None:
    for path in [cfg.OUTPUT_DIR, cfg.CACHE_DIR, cfg.CHECKPOINT_DIR]:
        os.makedirs(path, exist_ok=True)
    for sub in ["enhancement", "segmentation", "3d", "histograms"]:
        os.makedirs(os.path.join(cfg.OUTPUT_DIR, sub), exist_ok=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    epochs:           int  = None,
    batch_size:       int  = None,
    skip_training:    bool = False,
    run_registration: bool = True,
) -> dict:

    epochs     = epochs     or cfg.EPOCHS
    batch_size = batch_size or cfg.BATCH_SIZE
    device     = torch.device(cfg.DEVICE)

    print(f"\n{'=' * 60}")
    print(f"  Aortic Stent Displacement Pipeline  (All Stages Corrected)")
    print(f"  Device : {device}")
    print(f"  Epochs : {epochs}   Batch : {batch_size}")
    print(f"{'=' * 60}\n")

    setup_dirs()
    dataset_names = list(cfg.DATASETS.keys())

    # ================================================================== #
    # STAGE 1 — PREPROCESSING  (BUG-M1 FIX)                             #
    # ================================================================== #
    # BEFORE (wrong): preprocess_all_datasets(cfg.DATASETS, center=300, width=1500)
    #   → single window clips stent HU, returns 4-tuple
    # AFTER (correct): dual-window call, returns 6-tuple including stent channel
    # ================================================================== #
    print("\n[1/12] Preprocessing — dual-channel HU windowing")
    (
        raw_volumes,    # raw HU  (Z,Y,X) float32
        aorta_volumes,  # aorta-windowed + normalised  [NEW]
        stent_volumes,  # stent-windowed + normalised  [NEW]
        metal_masks,    # bool mask HU>800  [NEW — used by EADTV + stent model]
        spacings,       # (sx, sy, sz) mm per dataset
        origins,        # (ox, oy, oz) mm per dataset
    ) = preprocess_all_datasets(
        cfg.DATASETS,
        aorta_center     = cfg.HU_AORTA_CENTER,       # 350 HU  [Stage 1 fix]
        aorta_width      = cfg.HU_AORTA_WIDTH,        # 600 HU
        stent_center     = cfg.HU_STENT_CENTER,       # 1500 HU [Stage 1 fix]
        stent_width      = cfg.HU_STENT_WIDTH,        # 2000 HU
        norm_mode        = cfg.NORM_MODE,
        target_spacing_z = 1.0,                        # resample all to 1mm Z
    )

    # Diagnostic: print voxel counts
    for name in dataset_names:
        metal_count = int(metal_masks[name].sum())
        print(f"  {name}: aorta_vol={aorta_volumes[name].shape} "
              f"metal_voxels={metal_count:,}")

    # ================================================================== #
    # STAGE 2 — EADTV ENHANCEMENT  (BUG-M2 FIX)                         #
    # ================================================================== #
    # BEFORE (wrong): enhance_all_datasets(pre_volumes, weight=cfg.EADTV_WEIGHT)
    #   → pre_volumes doesn't exist; metal_masks not passed; stent channel TV'd
    # AFTER (correct): aorta_volumes + metal_masks; stent channel NOT enhanced
    # ================================================================== #
    print("\n[2/12] EADTV Enhancement (soft tissue only — stent protected)")
    enhanced_aorta, enhancement_results = enhance_all_datasets(
        aorta_volumes  = aorta_volumes,
        metal_masks    = metal_masks,
        weight         = cfg.EADTV_WEIGHT_SOFT,   # 0.05  [Stage 1 fix]
        max_iter       = cfg.EADTV_MAX_ITER,       # 20    [Stage 1 fix]
    )
    # Stent channel is NEVER TV-denoised — pass through unchanged
    enhanced_stent = stent_volumes

    # Enhancement visualisations (aorta channel only)
    for name in dataset_names:
        z    = enhanced_aorta[name].shape[0] // 2
        path = os.path.join(cfg.OUTPUT_DIR, "enhancement", f"{name}_z{z}.png")
        plot_enhancement_comparison(
            name,
            aorta_volumes[name][z],
            enhanced_aorta[name][z],
            save_path=path, z=z,
        )

    # ================================================================== #
    # STAGE 3 — REGISTRATION  (BUG-M3 FIX)                              #
    # ================================================================== #
    # BEFORE (wrong): register_all_to_baseline(pre_volumes, raw_volumes, ...)
    #   → pre_volumes doesn't exist; no aorta_masks for metric masking
    #   → 10% sampling missed stent; metric not restricted to aorta region
    # AFTER (correct): passes aorta_volumes, stent_volumes, raw_volumes,
    #   aorta_masks; sampling=30%; metric restricted to aorta ROI
    # ================================================================== #

    # We need aorta masks before registration for metric masking
    # Run a quick HU-threshold mask generation first (fast, no morphology needed)
    print("\n[3/12] Generating preliminary aorta masks for registration masking")
    from mask_generation import generate_aorta_mask
    prelim_aorta_masks = {}
    for name in dataset_names:
        prelim_aorta_masks[name] = generate_aorta_mask(
            raw_volumes[name],
            hu_min       = cfg.HU_MIN_AORTA,
            hu_max       = cfg.HU_MAX_AORTA,
            min_area     = cfg.AORTA_MIN_AREA,
            max_area     = cfg.AORTA_MAX_AREA,
            z_close_radius = 2,   # light closing for registration mask only
        )

    if run_registration:
        print("  Inter-scan Rigid Registration (T2, T3 → T1 baseline)")
        reg_aorta, reg_stent, reg_raw, transforms = register_all_to_baseline(
            aorta_volumes  = enhanced_aorta,      # [BUG-M3 FIX] was pre_volumes
            stent_volumes  = enhanced_stent,       # [BUG-M3 FIX] NEW
            raw_volumes    = raw_volumes,
            aorta_masks    = prelim_aorta_masks,   # [BUG-M3 FIX] for metric masking
            spacings       = spacings,
            origins        = origins,
            baseline       = cfg.BASELINE_DATASET,
            do_deformable  = cfg.REG_DEFORMABLE,
            verbose        = False,
        )
    else:
        print("  Skipping registration")
        reg_aorta  = enhanced_aorta
        reg_stent  = enhanced_stent
        reg_raw    = raw_volumes

    # ================================================================== #
    # STAGE 4 — MASK GENERATION  (BUG-M4 FIX)                           #
    # ================================================================== #
    # BEFORE (wrong): stent_dilate_iters=cfg.STENT_DILATE_AORTA_ITERS (=8)
    #   → 8mm dilation included bone/calcification as false-positive stent
    # AFTER (correct): dilate_iters=3 + closing_radius=1 + z_close_radius=3
    # ================================================================== #
    print("\n[4/12] Mask Generation (aorta + stent — corrected parameters)")
    aorta_masks, stent_masks = generate_all_masks(
        raw_volumes,
        hu_min_aorta         = cfg.HU_MIN_AORTA,
        hu_max_aorta         = cfg.HU_MAX_AORTA,
        hu_min_stent         = cfg.HU_MIN_STENT,
        stent_dilate_iters   = 3,               # [BUG-M4 FIX] was 8
        stent_closing_radius = 1,               # [BUG-M4 FIX] NEW — bridges strut gaps
        aorta_min_area       = cfg.AORTA_MIN_AREA,
        aorta_max_area       = cfg.AORTA_MAX_AREA,
        stent_min_cc_voxels  = cfg.STENT_MIN_CC_VOXELS,
        aorta_z_close_radius = 3,               # [BUG-M4 FIX] was default 2 → smoother
    )

    # Bounding-box crops per slice (used by dataset + inference)
    crops_per_dataset = {}
    for name in dataset_names:
        crops_per_dataset[name] = get_bbox_2d_per_slice(
            aorta_masks[name],
            padding=20,    # [Stage 4 fix] was 8 — stent struts extend to wall
        )

    # ================================================================== #
    # STAGE 5 — MODEL BUILDING  (BUG-M5 FIX)                            #
    # ================================================================== #
    # BEFORE (wrong): build_all_models() → all 4 models with in_ch=1
    # AFTER (correct): build_segmentation_models() →
    #   aorta_models (in_ch=1) + stent_models (in_ch=2) + separate losses
    # ================================================================== #
    print("\n[5/12] Building Models")
    model_bundle   = build_segmentation_models(base=cfg.BASE_CHANNELS)
    aorta_models   = model_bundle["aorta_models"]   # {name: nn.Module, in_ch=1}
    stent_models   = model_bundle["stent_models"]   # {name: nn.Module, in_ch=2}

    # ================================================================== #
    # STAGE 6 — TRAINING  (BUG-M6, BUG-M7 FIX)                         #
    # ================================================================== #
    # BEFORE (wrong):
    #   - StentDataset2D (1 channel) used for stent model
    #   - train_model called without is_stent_model or val_loader
    #   - smooth=1e-6 caused exploding gradients (fixed in training.py)
    #   - pos_weight stayed on CPU → crashed on GPU (fixed in training.py)
    # AFTER (correct):
    #   - StentDataset2D for aorta, StentDataset2D_DualChannel for stent
    #   - 80/20 train/val split to detect overfitting
    #   - is_stent_model=True uses dice_weight=0.8
    # ================================================================== #
    print("\n[6/12] Dataset Construction & Training")

    train_tf = get_train_transform(cfg.IMAGE_SIZE)  # IMAGE_SIZE now (512,512)

    # ── Aorta dataset (single channel) ────────────────────────────────
    aorta_ds = StentDataset2D(
        volumes      = reg_aorta,
        stent_masks  = aorta_masks,   # GT = aorta masks for aorta model
        crops        = crops_per_dataset,
        transform    = train_tf,
        image_size   = cfg.IMAGE_SIZE,
        allow_empty  = True,
    )
    aorta_val_n   = max(1, int(0.20 * len(aorta_ds)))
    aorta_trn_n   = len(aorta_ds) - aorta_val_n
    aorta_trn_ds, aorta_val_ds = random_split(aorta_ds, [aorta_trn_n, aorta_val_n])

    # ── Stent dataset (dual channel)  ──────────────────────────────────
    stent_ds = StentDataset2D_DualChannel(
        stent_volumes = reg_stent,
        metal_masks   = metal_masks,
        stent_gt      = stent_masks,
        crops         = crops_per_dataset,
        transform     = train_tf,
        image_size    = cfg.IMAGE_SIZE,
        allow_empty   = False,   # skip slices with no stent — important!
    )
    stent_val_n   = max(1, int(0.20 * len(stent_ds)))
    stent_trn_n   = len(stent_ds) - stent_val_n
    stent_trn_ds, stent_val_ds = random_split(stent_ds, [stent_trn_n, stent_val_n])

    def _make_loader(ds, shuffle=True):
        return DataLoader(
            ds,
            batch_size  = batch_size,
            shuffle     = shuffle,
            num_workers = 0,
            pin_memory  = (device.type == "cuda"),
        )

    print(f"  Aorta dataset : {len(aorta_trn_ds)} train / {aorta_val_n} val slices")
    print(f"  Stent dataset : {len(stent_trn_ds)} train / {stent_val_n} val slices")

    trained_aorta: dict = {}
    trained_stent: dict = {}

    # Train / load aorta models
    print("\n  --- Aorta Models ---")
    for mname, model in aorta_models.items():
        ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"{mname}_aorta.pth")
        if skip_training and os.path.isfile(ckpt):
            print(f"  Loading {ckpt}")
            state = torch.load(ckpt, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
        else:
            print(f"  Training {mname} (aorta) …")
            model = train_model(
                model            = model,
                train_loader     = _make_loader(aorta_trn_ds),
                val_loader       = _make_loader(aorta_val_ds, shuffle=False),
                device           = device,
                epochs           = epochs,
                lr               = cfg.LEARNING_RATE,
                checkpoint_path  = ckpt,
                warmup_epochs    = 5,
                is_stent_model   = False,   # dice_weight=0.5 for aorta
            )
        trained_aorta[mname] = model

    # Train / load stent models
    print("\n  --- Stent Models ---")
    for mname, model in stent_models.items():
        ckpt = os.path.join(cfg.CHECKPOINT_DIR, f"{mname}_stent.pth")
        if skip_training and os.path.isfile(ckpt):
            print(f"  Loading {ckpt}")
            state = torch.load(ckpt, map_location="cpu")
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            else:
                model.load_state_dict(state)
        else:
            print(f"  Training {mname} (stent) …")
            # Estimate pos_weight from dataset: negative/positive ratio
            total_stent_vox = sum(int(stent_masks[n].sum()) for n in dataset_names)
            total_vox       = sum(int(np.prod(stent_masks[n].shape)) for n in dataset_names)
            pos_w = (total_vox - total_stent_vox) / max(total_stent_vox, 1)
            pos_w = float(np.clip(pos_w, 1.0, 200.0))  # cap to avoid instability
            print(f"    pos_weight={pos_w:.1f} (stent voxel ratio={total_stent_vox/total_vox:.4f})")

            model = train_model(
                model            = model,
                train_loader     = _make_loader(stent_trn_ds),
                val_loader       = _make_loader(stent_val_ds, shuffle=False),
                device           = device,
                epochs           = epochs,
                lr               = cfg.LEARNING_RATE,
                pos_weight       = pos_w,
                checkpoint_path  = ckpt,
                warmup_epochs    = 5,
                is_stent_model   = True,   # dice_weight=0.8 for extreme imbalance
            )
        trained_stent[mname] = model

    # ================================================================== #
    # STAGE 7 — INFERENCE  (BUG-M8, BUG-M9 FIX)                        #
    # ================================================================== #
    # BEFORE (wrong):
    #   - predict_volume(model, work_pre[name], ...) — wrong channel for stent
    #   - metal_prior not passed → stent model crashed with shape error
    #   - postprocess_mask (aorta radii) applied to stent → merged struts
    # AFTER (correct):
    #   - aorta inference: reg_aorta channel, no metal_prior
    #   - stent inference: reg_stent channel, metal_prior passed
    #   - separate postprocessing functions per structure
    # ================================================================== #
    print("\n[7/12] Inference & Post-processing")

    aorta_preds_by_model: Dict[str, Dict[str, np.ndarray]] = {}
    stent_preds_by_model: Dict[str, Dict[str, np.ndarray]] = {}

    # Aorta inference
    for mname, model in trained_aorta.items():
        aorta_preds_by_model[mname] = {}
        for name in dataset_names:
            prob = predict_volume(
                model       = model,
                volume      = reg_aorta[name],     # aorta-windowed channel
                crops       = crops_per_dataset[name],
                device      = device,
                image_size  = cfg.IMAGE_SIZE,
                metal_prior = None,                # aorta model: 1 channel
            )
            binary = apply_shape_filter(prob, min_voxels=cfg.CC_MIN_VOXELS_AORTA)
            final  = postprocess_aorta_mask(binary, spacing=spacings[name])
            aorta_preds_by_model[mname][name] = final
            print(f"  Aorta {mname} {name}: voxels={final.sum():,}")

    # Stent inference
    for mname, model in trained_stent.items():
        stent_preds_by_model[mname] = {}
        for name in dataset_names:
            prob = predict_volume(
                model       = model,
                volume      = reg_stent[name],     # stent-windowed channel
                crops       = crops_per_dataset[name],
                device      = device,
                image_size  = cfg.IMAGE_SIZE,
                metal_prior = metal_masks[name].astype(np.float32),  # 2nd channel
            )
            binary = apply_shape_filter(prob, min_voxels=cfg.CC_MIN_VOXELS_STENT)
            final  = postprocess_stent_mask(binary, spacing=spacings[name])
            stent_preds_by_model[mname][name] = final
            pct = 100.0 * final.sum() / max(aorta_masks[name].sum(), 1)
            print(f"  Stent {mname} {name}: voxels={final.sum():,} ({pct:.1f}% of aorta)")

    # ================================================================== #
    # STAGE 8 — DISPLACEMENT  (BUG-M11 FIX)                             #
    # ================================================================== #
    # BEFORE (wrong): compute_all_displacements called without spacing check
    #   → different slice thicknesses across datasets → wrong mm values
    # AFTER (correct): validate_spacing_consistency first (raises if wrong)
    # ================================================================== #
    print("\n[8/12] Centroid Tracking & Displacement")

    # Spacing guard — will raise ValueError if resampling was skipped
    try:
        validate_spacing_consistency(spacings, tolerance_mm=0.15)
    except ValueError as e:
        print(f"  WARNING: {e}")
        print("  Displacement values may be inaccurate — re-run with target_spacing_z=1.0")

    pairs = [
        (dataset_names[i], dataset_names[j])
        for i in range(len(dataset_names))
        for j in range(i + 1, len(dataset_names))
    ]

    # Use stent predictions for displacement (primary clinical metric)
    centroids_by_model, displacements_by_model, disp_stats_by_model = \
        compute_all_displacements(
            preds_by_model = stent_preds_by_model,
            spacings       = spacings,
            origins        = origins,
            pairs          = pairs,
        )

    # Per-slice centroid tracking (reveals focal migration)
    print("  Per-slice centroid tracking …")
    slice_centroids: Dict[str, Dict[str, dict]] = {}
    best_stent_model = next(iter(stent_preds_by_model))
    for name in dataset_names:
        sc = centroid_per_slice(
            stent_preds_by_model[best_stent_model][name],
            spacings[name],
            origins[name],
        )
        valid = sum(1 for v in sc.values() if v is not None)
        print(f"  {name}: {valid} slices with stent centroid")
        slice_centroids[name] = sc

    # ================================================================== #
    # STAGE 9 — EVALUATION  (uses corrected evaluate_all_structures)     #
    # ================================================================== #
    print("\n[9/12] Segmentation Evaluation")

    seg_results = evaluate_all_structures(
        aorta_preds  = aorta_preds_by_model,
        stent_preds  = stent_preds_by_model,
        aorta_masks  = aorta_masks,
        stent_masks  = stent_masks,
        spacings     = spacings,
        include_hausdorff = cfg.INCLUDE_HAUSDORFF,
    )

    print("\n  AORTA metrics:")
    for mname, by_ds in seg_results["aorta"].items():
        for ds_name, m in by_ds.items():
            hd = f" HD95={m.get('HD95', float('nan')):.2f}" if "HD95" in m else ""
            print(f"    {mname} {ds_name}: DSC={m['DSC']:.4f} IoU={m['IoU']:.4f}{hd}")

    print("\n  STENT metrics:")
    for mname, by_ds in seg_results["stent"].items():
        for ds_name, m in by_ds.items():
            hd = f" HD95={m.get('HD95', float('nan')):.2f}" if "HD95" in m else ""
            dsc_str = f"{m['DSC']:.4f}" if not np.isnan(m['DSC']) else "NaN (no stent detected)"
            print(f"    {mname} {ds_name}: DSC={dsc_str} IoU={m['IoU']:.4f}{hd}")

    print("\n  Cross-dataset DSC (stent spatial overlap — migration indicator):")
    cross_dsc_results = {}
    for mname in stent_preds_by_model:
        cross_dsc_results[mname] = cross_dataset_dsc(stent_preds_by_model[mname])
        for (a, b), v in cross_dsc_results[mname].items():
            print(f"    {mname} {a}↔{b}: cross-DSC={v:.4f}")

    # ================================================================== #
    # STAGE 10 — 2D VISUALISATION  (BUG-M10 FIX)                        #
    # ================================================================== #
    # BEFORE (wrong): voxel_coords_all[name] = {"X": xs, "Y": ys, "Z": zs}
    #   plot_displacement_histograms expected tuple, not dict → blank plots
    # AFTER (correct): store as tuple (xs, ys, zs)
    # ================================================================== #
    print("\n[10/12] 2-D Visualisation")

    # Segmentation overlays — show stent predictions
    for mname in trained_stent:
        for name in dataset_names:
            # Find a representative slice that actually has stent voxels
            stent_pred = stent_preds_by_model[mname][name]
            slices_with_stent = [z for z in range(stent_pred.shape[0])
                                 if stent_pred[z].sum() > 0]
            z = slices_with_stent[len(slices_with_stent)//2] if slices_with_stent \
                else reg_stent[name].shape[0] // 2

            path = os.path.join(
                cfg.OUTPUT_DIR, "segmentation",
                f"{name}_{mname}_stent_z{z}.png"
            )
            plot_segmentation_overlay(
                name,
                reg_aorta[name][z],          # use aorta channel as background CT
                (stent_masks[name][z] > 0).astype(float),
                stent_pred[z],
                mname,
                save_path=path, z=z,
            )

    # Metrics comparison
    # plot_metrics_comparison expects {model: {dataset: {DSC, IoU}}}
    mc_path = os.path.join(cfg.OUTPUT_DIR, "histograms", "stent_metrics.png")
    plot_metrics_comparison(seg_results["stent"], save_path=mc_path)

    # Displacement histograms — BUG-M10 FIX: store as tuple not dict
    voxel_coords_all = {}
    for name in dataset_names:
        xs, ys, zs = voxel_coords_physical(
            stent_preds_by_model[best_stent_model][name],
            spacings[name],
            origins[name],
        )
        voxel_coords_all[name] = (xs, ys, zs)   # [BUG-M10 FIX] was {"X":xs,...}

    hist_path = os.path.join(cfg.OUTPUT_DIR, "histograms", "stent_positions.png")
    plot_displacement_histograms(voxel_coords_all, save_path=hist_path)

    # Longitudinal centroid displacement plot
    for mname in trained_stent:
        lp_path = os.path.join(
            cfg.OUTPUT_DIR, "histograms", f"longitudinal_stent_{mname}.png"
        )
        plot_longitudinal_displacement(
            centroids_by_model[mname], dataset_names, save_path=lp_path
        )

    # ================================================================== #
    # STAGE 11 — 3D VISUALISATION                                        #
    # ================================================================== #
    # spacing is (X,Y,Z) from SimpleITK — _build_image_data in Stage 9
    # was fixed to handle this correctly. No change needed here.
    # Using aorta_masks (HU-threshold) for 3D — more complete than model preds
    # Using stent_preds for 3D — model output after postprocess_stent_mask
    # ================================================================== #
    print("\n[11/12] 3-D Visualisation")

    best_stent_m = next(iter(stent_preds_by_model))

    for name in dataset_names:
        base_3d = os.path.join(cfg.OUTPUT_DIR, "3d", name)
        os.makedirs(base_3d, exist_ok=True)

        sp = spacings[name]   # (X_spacing, Y_spacing, Z_spacing) — correct for viz

        # Diagnostic: print what will be rendered
        print(f"  {name}: aorta_vox={aorta_masks[name].sum():,} "
              f"stent_vox={stent_preds_by_model[best_stent_m][name].sum():,} "
              f"spacing={sp}")

        # Single-view renders
        for view in cfg.VIZ_VIEWS:
            path = os.path.join(base_3d, f"{view}.png")
            plot_aorta_and_stent_3d(
                aorta_mask          = aorta_masks[name],
                stent_mask          = stent_preds_by_model[best_stent_m][name],
                spacing             = sp,
                original_centroid   = centroids_by_model[best_stent_m].get(name),
                displacement_vector = None,
                show_displaced      = False,
                view                = view,
                save_screenshot     = path,
                off_screen          = cfg.VIZ_OFFSCREEN,
                window_size         = cfg.VIZ_WINDOW_SIZE,
            )

        # Composite 2×2 all-views
        composite_path = os.path.join(base_3d, "composite_4views.png")
        plot_aorta_and_stent_3d_composite(
            aorta_mask        = aorta_masks[name],
            stent_mask        = stent_preds_by_model[best_stent_m][name],
            spacing           = sp,
            original_centroid = centroids_by_model[best_stent_m].get(name),
            show_displaced    = False,
            save_screenshot   = composite_path,
            off_screen        = cfg.VIZ_OFFSCREEN,
            window_size       = (2400, 1800),
            title             = name,
        )

    # Displacement 3D renders (T1→T2, T1→T3, T2→T3)
    for (ds_a, ds_b), disp_vec in displacements_by_model[best_stent_m].items():
        if disp_vec is None:
            continue
        base_d = os.path.join(cfg.OUTPUT_DIR, "3d", f"displacement_{ds_a}_to_{ds_b}")
        os.makedirs(base_d, exist_ok=True)
        for view in ["isometric", "sagittal", "coronal"]:
            path = os.path.join(base_d, f"{view}.png")
            plot_aorta_and_stent_3d(
                aorta_mask          = aorta_masks[ds_a],
                stent_mask          = stent_preds_by_model[best_stent_m][ds_a],
                spacing             = spacings[ds_a],
                original_centroid   = centroids_by_model[best_stent_m].get(ds_a),
                displacement_vector = disp_vec,
                show_displaced      = True,
                view                = view,
                save_screenshot     = path,
                off_screen          = cfg.VIZ_OFFSCREEN,
                window_size         = cfg.VIZ_WINDOW_SIZE,
            )

    # ================================================================== #
    # STAGE 12 — REPORT                                                  #
    # ================================================================== #
    print("\n[12/12] Reports (CSV + PDF)")
    first_model = next(iter(cross_dsc_results))
    export_csv(seg_results["stent"], disp_stats_by_model, cfg.OUTPUT_DIR)
    generate_pdf_report(
        seg_results         = seg_results["stent"],
        disp_stats          = disp_stats_by_model,
        enhancement_results = enhancement_results,
        cross_dsc           = cross_dsc_results[first_model],
        output_dir          = cfg.OUTPUT_DIR,
        pdf_path            = cfg.REPORT_PDF,
    )

    # Results JSON
    def _safe(v):
        if isinstance(v, np.ndarray): return v.tolist()
        if isinstance(v, float):      return v if not np.isnan(v) else None
        return v

    results = {
        "enhancement":          {k: {mk: _safe(mv) for mk, mv in m.items()}
                                  for k, m in enhancement_results.items()},
        "segmentation_aorta":   {m: {d: {k: _safe(v) for k, v in s.items()}
                                      for d, s in by_ds.items()}
                                  for m, by_ds in seg_results["aorta"].items()},
        "segmentation_stent":   {m: {d: {k: _safe(v) for k, v in s.items()}
                                      for d, s in by_ds.items()}
                                  for m, by_ds in seg_results["stent"].items()},
        "displacement_stats":   {m: {f"{a}→{b}": {k: _safe(v) for k, v in st.items()}
                                      for (a, b), st in pair_stats.items()}
                                  for m, pair_stats in disp_stats_by_model.items()},
        "cross_dataset_dsc":    {m: {f"{a}↔{b}": float(v)
                                      for (a, b), v in cdsc.items()}
                                  for m, cdsc in cross_dsc_results.items()},
    }

    with open(cfg.RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete!   Results → {cfg.OUTPUT_DIR}")
    print(f"{'=' * 60}\n")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Longitudinal Aortic Stent Displacement Analysis"
    )
    parser.add_argument("--epochs",       type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch_size",   type=int, default=cfg.BATCH_SIZE)
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--no_registration", action="store_true")
    parser.add_argument("--data_root",    type=str, default=None)
    args = parser.parse_args()

    if args.data_root:
        cfg.DATA_ROOT = args.data_root
        cfg.DATASETS  = {k: os.path.join(args.data_root, k) for k in cfg.DATASETS}

    run_pipeline(
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        skip_training    = args.skip_training,
        run_registration = not args.no_registration,
    )