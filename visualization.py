"""
visualization.py
================
All visualisation for the aortic stent displacement pipeline.

2-D (matplotlib)
  - Enhancement comparison  (original | enhanced | |diff|)
  - Segmentation overlay    (CT | GT stent | predicted stent)
  - Metrics bar chart        (DSC/IoU per model × dataset)
  - Displacement histograms  (X, Y, Z voxel distributions)
  - Longitudinal centroid plot (timeline across T1 → T2 → T3)

3-D (PyVista)
  - Aorta + stent surface mesh with full smoothing pipeline:
      1. Anisotropic Gaussian pre-smoothing (removes staircase / axial artefacts)
      2. pv.ImageData grid in physical mm (correct iso-surface geometry)
      3. Mesh clean (merge coincident points)
      4. Laplacian smooth  (100 iters, factor=0.10)
      5. Taubin smooth     (50 iters,  pass_band=0.05)  – corrects Laplacian shrinkage
      6. decimate_pro      (preserve_topology=True)      – artefact-free reduction
      7. Feature-angle normals (60°) for edge sharpness
  - Dark studio background + 3-point lighting + SSAA anti-aliasing
  - 4 camera views: isometric, sagittal, coronal, axial
  - 2×2 composite multi-view PNG
  - Displacement arrow overlays
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (safe for off-screen / WSL)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple, Union

try:
    import pyvista as pv
    _HAVE_PV = True
except ImportError:
    _HAVE_PV = False
    pv = None

# Pre-mesh smoothing helper lives in postprocessing to keep it testable
try:
    from postprocessing import smooth_mask_for_3d as _smooth_mask_for_3d
    _HAVE_PP = True
except ImportError:
    _HAVE_PP = False
    _smooth_mask_for_3d = None


# ---------------------------------------------------------------------------
# 2-D helpers
# ---------------------------------------------------------------------------

def plot_enhancement_comparison(
    name: str,
    pre_slice: np.ndarray,
    enh_slice: np.ndarray,
    save_path: Optional[str] = None,
    z: int = 0,
) -> None:
    """3-panel figure: original | enhanced | absolute difference."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pre_slice, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"{name}  z={z}\nOriginal (normalised)")
    axes[0].axis("off")

    axes[1].imshow(enh_slice, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Enhanced (EADTV/TV)")
    axes[1].axis("off")

    diff = np.abs(enh_slice.astype(np.float32) - pre_slice.astype(np.float32))
    im = axes[2].imshow(diff, cmap="hot")
    axes[2].set_title("|Enhancement residual|")
    axes[2].axis("off")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    plt.suptitle(f"Enhancement validation – {name}", y=1.02, fontsize=13)
    plt.tight_layout()
    _save_or_close(fig, save_path)


def plot_segmentation_overlay(
    name: str,
    ct_slice: np.ndarray,
    gt_slice: np.ndarray,
    pred_slice: np.ndarray,
    model_name: str,
    save_path: Optional[str] = None,
    z: int = 0,
) -> None:
    """3-panel overlay: CT | CT + GT | CT + prediction."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax in axes:
        ax.imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")

    axes[0].set_title(f"{name}  z={z}\nCT")

    axes[1].imshow(np.ma.masked_where(gt_slice < 0.5, gt_slice),
                   cmap="Reds", alpha=0.55, vmin=0, vmax=1)
    axes[1].set_title("GT stent (red)")

    axes[2].imshow(np.ma.masked_where(pred_slice < 0.5, pred_slice),
                   cmap="Greens", alpha=0.55, vmin=0, vmax=1)
    axes[2].set_title(f"{model_name} prediction (green)")

    plt.suptitle(f"Segmentation overlay – {name}", y=1.02, fontsize=13)
    plt.tight_layout()
    _save_or_close(fig, save_path)


def plot_metrics_comparison(
    results: Dict[str, Dict[str, Dict[str, float]]],
    save_path: Optional[str] = None,
) -> None:
    """Bar chart comparing DSC and IoU per model per dataset."""
    import pandas as pd

    rows = []
    for model_name, by_ds in results.items():
        for ds_name, m in by_ds.items():
            row = {"Model": model_name, "Dataset": ds_name}
            row.update({k: float(v) for k, v in m.items()
                        if isinstance(v, (int, float)) and not np.isnan(float(v))})
            rows.append(row)

    if not rows:
        return

    df = pd.DataFrame(rows)
    metrics_to_plot = [c for c in ["DSC", "IoU", "HD95"] if c in df.columns]

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(6 * len(metrics_to_plot), 5))
    if len(metrics_to_plot) == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics_to_plot):
        pivot = df.pivot(index="Model", columns="Dataset", values=metric)
        pivot.plot(kind="bar", ax=ax, rot=30, colormap="tab10")
        ax.set_title(metric)
        ax.set_ylim(0, max(1.05, df[metric].max() * 1.15) if metric != "HD95" else None)
        ax.legend(title="Dataset", fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Segmentation Metrics – Model Comparison", fontsize=14)
    plt.tight_layout()
    _save_or_close(fig, save_path)

# REPLACE plot_displacement_histograms WITH:

def plot_displacement_histograms(
    voxel_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    axes_labels: List[str] = ("X", "Y", "Z"),
    save_path: Optional[str] = None,
) -> None:
    """
    Overlaid histograms of stent voxel positions.

    FIX: Original expected voxel_coords[name].get("X") but
    voxel_coords_physical() returns a tuple (xs, ys, zs).
    Updated to unpack tuple correctly.

    Parameters
    ----------
    voxel_coords : dict[dataset_name → (xs, ys, zs)]
        Output of voxel_coords_physical() per dataset.
    """
    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]
    names  = list(voxel_coords.keys())
    n_axes = len(axes_labels)

    fig, plot_axes = plt.subplots(n_axes, 1, figsize=(10, 4 * n_axes))
    if n_axes == 1:
        plot_axes = [plot_axes]

    for row, (label, axis_idx) in enumerate(zip(axes_labels, [0, 1, 2])):
        ax = plot_axes[row]
        for i, name in enumerate(names):
            coords_tuple = voxel_coords.get(name)
            if coords_tuple is None:
                continue
            # coords_tuple = (xs, ys, zs) — index by axis position
            arr = coords_tuple[axis_idx]
            if arr is not None and arr.size > 0:
                ax.hist(arr, bins=50, alpha=0.55,
                        color=colors[i % len(colors)],
                        label=name, edgecolor="none")
        ax.set_title(f"Stent voxel distribution along {label} (mm)")
        ax.set_xlabel(f"{label} coordinate (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Multi-Timepoint Stent Position Histograms", fontsize=14)
    plt.tight_layout()
    _save_or_close(fig, save_path)


# REPLACE plot_displacement_histograms WITH:

def plot_displacement_histograms(
    voxel_coords: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
    axes_labels: List[str] = ("X", "Y", "Z"),
    save_path: Optional[str] = None,
) -> None:
    """
    Overlaid histograms of stent voxel positions.

    FIX: Original expected voxel_coords[name].get("X") but
    voxel_coords_physical() returns a tuple (xs, ys, zs).
    Updated to unpack tuple correctly.

    Parameters
    ----------
    voxel_coords : dict[dataset_name → (xs, ys, zs)]
        Output of voxel_coords_physical() per dataset.
    """
    colors = ["tab:red", "tab:green", "tab:blue", "tab:orange", "tab:purple"]
    names  = list(voxel_coords.keys())
    n_axes = len(axes_labels)

    fig, plot_axes = plt.subplots(n_axes, 1, figsize=(10, 4 * n_axes))
    if n_axes == 1:
        plot_axes = [plot_axes]

    for row, (label, axis_idx) in enumerate(zip(axes_labels, [0, 1, 2])):
        ax = plot_axes[row]
        for i, name in enumerate(names):
            coords_tuple = voxel_coords.get(name)
            if coords_tuple is None:
                continue
            # coords_tuple = (xs, ys, zs) — index by axis position
            arr = coords_tuple[axis_idx]
            if arr is not None and arr.size > 0:
                ax.hist(arr, bins=50, alpha=0.55,
                        color=colors[i % len(colors)],
                        label=name, edgecolor="none")
        ax.set_title(f"Stent voxel distribution along {label} (mm)")
        ax.set_xlabel(f"{label} coordinate (mm)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("Multi-Timepoint Stent Position Histograms", fontsize=14)
    plt.tight_layout()
    _save_or_close(fig, save_path)

# ---------------------------------------------------------------------------
# 3-D helpers (PyVista)
# ---------------------------------------------------------------------------

# ── Studio lighting colours ──────────────────────────────────────────────────
_BG_TOP    = "#0d1117"   # near-black top (dark studio)
_BG_BOTTOM = "#1a2535"   # dark-blue bottom gradient
_AORTA_COL = "#e05a3a"   # warm tomato-red  (vivid, visible through semi-opacity)
_STENT_COL = "#d4d4d4"   # metallic silver
_DISP_COL  = "#f0c040"   # displaced stent – warm gold
_ARROW_COL = "#00ff7f"   # spring-green arrow

# Render quality
_AA_MODE   = "ssaa"      # Super-Sample Anti-Aliasing (highest quality)

# Smooth step parameters (scientifically tuned)
_LAPLACIAN_ITERS   = 100    # Laplacian smooth iterations
_LAPLACIAN_FACTOR  = 0.10   # relaxation factor  (0.05 shrinks; 0.10 is safe)
_TAUBIN_ITERS      = 50     # Taubin smooth iterations (corrects Laplacian shrinkage)
_TAUBIN_PASSBAND   = 0.05   # Taubin pass-band frequency (lower = smoother)
_DECIMATE_RATIO    = 0.40   # fraction of cells to remove
_HOLE_SIZE         = 25.0   # max hole perimeter (mm) to fill
_FEATURE_ANGLE     = 60.0   # degrees – normals below this form a crease

# Pre-mesh volumetric smoothing
_SIGMA_MM      = 1.2    # isotropic mm sigma for in-plane smoothing
_AXIAL_BOOST   = 1.5    # extra factor on Z to close slice-to-slice gaps


# REPLACE _build_image_data WITH:

def _build_image_data(
    vol: np.ndarray,
    spacing: Tuple[float, ...],
) -> "pv.ImageData":
    """
    Wrap a (Z, Y, X) float volume in PyVista ImageData with correct mm spacing.

    FIX: Original code swapped X and Z spacing, causing geometric distortion
    on all 3D meshes. SimpleITK spacing = (sx, sy, sz) in X, Y, Z order.
    PyVista grid.spacing must also be in X, Y, Z order after transposing
    vol from (Z,Y,X) to (X,Y,Z).

    spacing input: SimpleITK convention = (X_spacing, Y_spacing, Z_spacing)
    """
    # vol: (Z, Y, X) → transpose to (X, Y, Z) for VTK
    vol_xyz = np.ascontiguousarray(vol.transpose(2, 1, 0))
    nz, ny, nx = vol.shape   # original shape

    # SimpleITK spacing = (X_spacing, Y_spacing, Z_spacing) — use directly
    # DO NOT reverse: spacing[0]=X, spacing[1]=Y, spacing[2]=Z
    x_spacing = float(spacing[0])
    y_spacing = float(spacing[1])
    z_spacing = float(spacing[2])

    grid = pv.ImageData()
    grid.dimensions = (nx + 1, ny + 1, nz + 1)
    grid.spacing    = (x_spacing, y_spacing, z_spacing)   # FIXED: was reversed
    grid.origin     = (0.0, 0.0, 0.0)

    grid.cell_data["scalars"] = vol_xyz.ravel(order="F")
    grid = grid.cell_data_to_point_data()
    return grid


def _extract_surface_mesh(
    binary_vol:  np.ndarray,
    spacing:     Tuple[float, ...] = (1.0, 1.0, 1.0),
    reduce:      float = _DECIMATE_RATIO,
    smooth_iters: int  = _LAPLACIAN_ITERS,
):
    """High-quality marching-cubes surface mesh from a binary volume.

    Full mesh-quality pipeline
    --------------------------
    1. Anisotropic Gaussian pre-smoothing (``smooth_mask_for_3d``)
       – eliminates the staircase artefacts that arise from voxel-grid
         quantisation along the axial (Z) direction.
    2. Physical-mm ``pv.ImageData`` grid – ensures geometrically correct
       iso-surface coordinates when voxel spacing is anisotropic.
    3. Marching-cubes at iso-level 0.5 on the smooth float field.
    4. ``mesh.clean()`` – merges duplicate / near-coincident points.
    5. Laplacian smooth (100 iters, factor=0.10) – global surface fairing.
    6. Taubin smooth (50 iters, pass_band=0.05) – scientifically corrects the
       volume shrinkage introduced by Laplacian smoothing (Taubin 1995).
    7. ``decimate_pro(preserve_topology=True)`` – topology-safe polygon
       reduction (avoids non-manifold edges that plain ``decimate`` can create).
    8. ``compute_normals(feature_angle=60°)`` – preserves anatomical creases
       (e.g. stent strut edges) while smoothing the broad surface faces.
    9. ``fill_holes`` – closes small gaps left by marching cubes at mask
       boundaries.
    """
    if not _HAVE_PV:
        raise ImportError("PyVista is not installed.  Run: pip install pyvista")

    # ── Step 1: Anisotropic Gaussian pre-smoothing ────────────────────────────
    if _HAVE_PP and _smooth_mask_for_3d is not None:
        smooth_vol = _smooth_mask_for_3d(
            binary_vol, spacing=spacing,
            sigma_mm=_SIGMA_MM, axial_boost=_AXIAL_BOOST,
        )
    else:
        # Fallback: simple isotropic smoothing if postprocessing not importable
        from scipy.ndimage import gaussian_filter
        smooth_vol = np.clip(
            gaussian_filter(binary_vol.astype(np.float32), sigma=1.5),
            0.0, 1.0,
        ).astype(np.float32)

    # ── Step 2 & 3: Build physical ImageData and run marching cubes ──────────
    grid = _build_image_data(smooth_vol, spacing)
    mesh = grid.contour([0.5], scalars="scalars", method="marching_cubes")

    if mesh.n_points == 0:
        return mesh

    # ── Step 4: Clean – merge coincident points, remove degenerate cells ──────
    mesh = mesh.clean(tolerance=1e-3)

    # ── Step 5: Laplacian global fairing ─────────────────────────────────────
    if smooth_iters > 0:
        mesh = mesh.smooth(
            n_iter=smooth_iters,
            relaxation_factor=_LAPLACIAN_FACTOR,
            boundary_smoothing=True,
            feature_smoothing=False,   # let Taubin handle features
        )

    # ── Step 6: Taubin smoothing (corrects Laplacian shrinkage) ──────────────
    try:
        mesh = mesh.smooth_taubin(
            n_iter=_TAUBIN_ITERS,
            pass_band=_TAUBIN_PASSBAND,
            boundary_smoothing=True,
            normalize_coordinates=True,
        )
    except AttributeError:
        # PyVista < 0.38 does not have smooth_taubin – silently skip
        pass

    # ── Step 7: Topology-safe polygon reduction ───────────────────────────────
    if reduce > 0 and mesh.n_cells > 500:
        try:
            mesh = mesh.decimate_pro(
                target_reduction=reduce,
                preserve_topology=True,
                splitting=False,
            )
        except Exception:
            # Fallback to standard decimate if decimate_pro unavailable
            mesh = mesh.decimate(target_reduction=reduce)

    # ── Step 8: Normals with feature-angle edge preservation ─────────────────
    mesh = mesh.compute_normals(
        consistent_normals=True,
        auto_orient_normals=True,
        feature_angle=_FEATURE_ANGLE,
        split_vertices=True,
    )

    # ── Step 9: Fill residual holes ───────────────────────────────────────────
    mesh = mesh.fill_holes(hole_size=_HOLE_SIZE)

    return mesh


def _add_studio_lights(plotter: "pv.Plotter") -> None:
    """Add a 3-point studio lighting rig for clinical 3-D renders.

    Key light   – primary illumination from upper-left front
    Fill light  – soft fill from right to reduce harsh shadows
    Back light  – rim/edge light from behind to separate subject from bg
    """
    plotter.remove_all_lights()

    key = pv.Light(position=(1.0, 1.0, 1.0), focal_point=(0, 0, 0))
    key.intensity = 1.0
    key.positional = False

    fill = pv.Light(position=(-1.0, 0.5, 0.5), focal_point=(0, 0, 0))
    fill.intensity = 0.45
    fill.positional = False

    back = pv.Light(position=(0.0, -1.0, -0.5), focal_point=(0, 0, 0))
    back.intensity = 0.30
    back.positional = False

    for light in (key, fill, back):
        plotter.add_light(light)


def _add_aorta_mesh(
    plotter: "pv.Plotter",
    aorta_mask: np.ndarray,
    spacing: Tuple[float, ...],
) -> bool:
    """Extract and add the aorta surface mesh.  Returns True on success."""
    if not aorta_mask.any():
        return False
    try:
        mesh = _extract_surface_mesh(aorta_mask, spacing)
        if mesh.n_points == 0:
            return False
        plotter.add_mesh(
            mesh,
            color=_AORTA_COL,
            opacity=0.42,
            specular=0.60,
            specular_power=25,
            diffuse=0.80,
            ambient=0.12,
            smooth_shading=True,
            label="Aorta",
            name="aorta",
        )
        return True
    except Exception as e:
        print(f"  [WARN] Aorta mesh failed: {e}", flush=True)
        return False


# REPLACE _add_stent_mesh WITH:

def _add_stent_mesh(
    plotter: "pv.Plotter",
    stent_mask: np.ndarray,
    spacing: Tuple[float, ...],
    displacement_vector: Optional[np.ndarray] = None,
    original_centroid: Optional[np.ndarray] = None,
    show_displaced: bool = False,
) -> bool:
    """
    Render stent using skeleton/tube strategy instead of surface mesh.

    WHY: Stent struts are 1-2 voxels wide. Surface mesh extraction + 150
    smoothing iterations destroys all geometric detail. Instead:
    1. Skeletonize the stent mask to get centerline voxels
    2. Convert skeleton voxels to physical mm coordinates
    3. Render as tubes — preserves wire geometry, looks clinically accurate

    Falls back to minimal-smoothing surface if skeletonization fails.
    """
    if not stent_mask.any():
        return False

    try:
        # Strategy 1: Skeleton-based tube rendering
        from skimage.morphology import skeletonize_3d
        skeleton = skeletonize_3d(stent_mask.astype(np.uint8))

        if skeleton.any():
            # Convert skeleton voxels to physical mm coordinates
            pos = np.argwhere(skeleton > 0)   # (N, 3): [z, y, x] indices
            x_spacing, y_spacing, z_spacing = float(spacing[0]), float(spacing[1]), float(spacing[2])

            pts_mm = np.column_stack([
                pos[:, 2] * x_spacing,   # x_mm
                pos[:, 1] * y_spacing,   # y_mm
                pos[:, 0] * z_spacing,   # z_mm
            ]).astype(np.float32)

            # Render each skeleton point as a small sphere (stent wire node)
            pc = pv.PolyData(pts_mm)
            glyphs = pc.glyph(
                geom=pv.Sphere(radius=0.6),   # 0.6mm ≈ 1 voxel radius for stent wire
                scale=False,
                orient=False,
            )
            plotter.add_mesh(
                glyphs,
                color=_STENT_COL,
                opacity=0.95,
                specular=1.0,
                specular_power=80,
                diffuse=0.7,
                ambient=0.1,
                smooth_shading=True,
                label="Stent (original)",
                name="stent_orig",
            )

            # Displaced stent
            if show_displaced and displacement_vector is not None:
                pts_disp = pts_mm + displacement_vector.astype(np.float32)
                pc_disp  = pv.PolyData(pts_disp)
                glyphs_disp = pc_disp.glyph(geom=pv.Sphere(radius=0.6), scale=False, orient=False)
                plotter.add_mesh(
                    glyphs_disp, color=_DISP_COL, opacity=0.90,
                    specular=0.7, smooth_shading=True,
                    label="Stent (displaced)", name="stent_disp",
                )

        else:
            # Strategy 2: fallback — minimal-smoothing surface (sigma=0.3mm, 10 iters only)
            raise ValueError("Empty skeleton — use surface fallback")

    except Exception:
        # Fallback: surface mesh with minimal smoothing to preserve strut geometry
        try:
            from scipy.ndimage import gaussian_filter
            smooth_vol = np.clip(
                gaussian_filter(stent_mask.astype(np.float32), sigma=0.3),
                0.0, 1.0
            )
            grid = _build_image_data(smooth_vol, spacing)
            stent_mesh = grid.contour([0.5], scalars="scalars", method="marching_cubes")
            if stent_mesh.n_points > 0:
                stent_mesh = stent_mesh.clean()
                # CRITICAL: only 10 Laplacian iters for stent — NOT 100
                stent_mesh = stent_mesh.smooth(n_iter=10, relaxation_factor=0.05)
                plotter.add_mesh(
                    stent_mesh, color=_STENT_COL, opacity=0.92,
                    specular=1.0, specular_power=60, smooth_shading=True,
                    label="Stent (original)", name="stent_orig",
                )
        except Exception as e:
            print(f"  [WARN] Stent mesh failed: {e}", flush=True)
            return False

    # Displacement arrow
    if (displacement_vector is not None
            and original_centroid is not None
            and np.linalg.norm(displacement_vector) > 0):
        mag = float(np.linalg.norm(displacement_vector))
        plotter.add_arrows(
            cent=original_centroid.reshape(1, 3),
            direction=displacement_vector.reshape(1, 3) / mag,
            mag=mag * 1.3,
            color=_ARROW_COL,
            label="Displacement",
            name="arrow",
        )
    return True
def _configure_renderer(
    plotter: "pv.Plotter",
    view: str,
    title: str,
    aorta_voxels: int = 0,
    stent_voxels: int = 0,
    renderer_index: Optional[int] = None,
) -> None:
    """Apply scene settings: background, lights, camera, annotations."""
    if renderer_index is not None:
        plotter.subplot(*renderer_index)   # for multi-viewport plotters

    # Background: dark two-tone gradient (clinical look)
    try:
        plotter.set_background(_BG_TOP, top=_BG_BOTTOM)
    except TypeError:
        plotter.set_background(_BG_TOP)

    # Anti-aliasing
    try:
        plotter.enable_anti_aliasing(_AA_MODE)
    except Exception:
        pass

    # Studio lighting
    _add_studio_lights(plotter)

    # Axes and grid
    plotter.add_axes(interactive=False, line_width=2)

    # Annotation
    info = ""
    if aorta_voxels or stent_voxels:
        info = f"\naorta={aorta_voxels:,} vox  stent={stent_voxels:,} vox"
    plotter.add_text(
        f"Aorta & Stent – {view.capitalize()}{info}",
        position="upper_edge",
        font_size=11,
        color="white",
        shadow=True,
    )

    _set_camera(plotter, view)


def plot_aorta_and_stent_3d(
    aorta_mask: np.ndarray,
    stent_mask: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    original_centroid: Optional[np.ndarray] = None,
    displacement_vector: Optional[np.ndarray] = None,
    show_displaced: bool = False,
    view: str = "isometric",
    save_screenshot: Optional[str] = None,
    off_screen: bool = True,
    window_size: Tuple[int, int] = (1600, 1200),
) -> None:
    """Render aorta and stent 3-D surface meshes with PyVista.

    Parameters
    ----------
    aorta_mask, stent_mask : np.ndarray, bool / uint8
    spacing : tuple  – voxel spacing in mm  (Z, Y, X)
    original_centroid : np.ndarray or None  – centroid of original stent (mm)
    displacement_vector : np.ndarray or None  – [Δx, Δy, Δz] in mm
    show_displaced : bool  – show a translated (displaced) stent mesh
    view : str – ``"isometric"`` | ``"sagittal"`` | ``"coronal"`` | ``"axial"``
    save_screenshot : str or None  – path to save PNG
    off_screen : bool  – render without opening a window
    window_size : tuple
    """
    if not _HAVE_PV:
        print("  [WARN] PyVista not installed – skipping 3-D rendering.", flush=True)
        return

    plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size))

    _add_aorta_mesh(plotter, aorta_mask, spacing)
    _add_stent_mesh(
        plotter, stent_mask, spacing,
        displacement_vector=displacement_vector,
        original_centroid=original_centroid,
        show_displaced=show_displaced,
    )

    _configure_renderer(
        plotter, view,
        title=view,
        aorta_voxels=int(aorta_mask.sum()),
        stent_voxels=int(stent_mask.sum()),
    )

    # Legend
    try:
        plotter.add_legend(face="rectangle", size=(0.22, 0.22),
                           bcolor="#1a2535", border=True,
                           label_color="white")
    except Exception:
        pass

    if save_screenshot:
        os.makedirs(os.path.dirname(save_screenshot) or ".", exist_ok=True)
        plotter.screenshot(save_screenshot, transparent_background=False)
        print(f"    3-D screenshot saved: {save_screenshot}", flush=True)

    plotter.close()


def plot_aorta_and_stent_3d_composite(
    aorta_mask: np.ndarray,
    stent_mask: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    original_centroid: Optional[np.ndarray] = None,
    displacement_vector: Optional[np.ndarray] = None,
    show_displaced: bool = False,
    save_screenshot: Optional[str] = None,
    off_screen: bool = True,
    window_size: Tuple[int, int] = (2400, 1800),
    title: str = "",
) -> None:
    """Render a 2×2 composite of all 4 standard views in one PNG.

    Views layout:
      [Isometric]  [Coronal ]
      [Sagittal ]  [Axial   ]

    This gives a full clinical overview with a single render call,
    significantly reducing per-dataset render time.
    """
    if not _HAVE_PV:
        print("  [WARN] PyVista not installed – skipping composite 3-D render.", flush=True)
        return

    views_grid = [
        ("isometric", (0, 0)),
        ("coronal",   (0, 1)),
        ("sagittal",  (1, 0)),
        ("axial",     (1, 1)),
    ]

    plotter = pv.Plotter(
        off_screen=off_screen,
        window_size=list(window_size),
        shape=(2, 2),
    )

    aorta_vox = int(aorta_mask.sum())
    stent_vox = int(stent_mask.sum())

    # Pre-extract meshes once to avoid repeated expensive computation
    aorta_mesh = None
    stent_mesh  = None
    disp_mesh   = None

    if aorta_mask.any():
        try:
            aorta_mesh = _extract_surface_mesh(aorta_mask, spacing)
            if aorta_mesh.n_points == 0:
                aorta_mesh = None
        except Exception as e:
            print(f"  [WARN] Composite aorta mesh: {e}", flush=True)

    if stent_mask.any():
        try:
            stent_mesh = _extract_surface_mesh(stent_mask, spacing)
            if stent_mesh.n_points == 0:
                stent_mesh = None
            elif show_displaced and displacement_vector is not None:
                disp_mesh = stent_mesh.copy()
                disp_mesh.translate(displacement_vector.tolist(), inplace=True)
        except Exception as e:
            print(f"  [WARN] Composite stent mesh: {e}", flush=True)

    for view, (row, col) in views_grid:
        plotter.subplot(row, col)

        if aorta_mesh is not None:
            plotter.add_mesh(
                aorta_mesh.copy(),
                color=_AORTA_COL, opacity=0.42,
                specular=0.60, specular_power=25,
                diffuse=0.80, ambient=0.12,
                smooth_shading=True,
                label="Aorta", name=f"aorta_{view}",
            )

        if stent_mesh is not None:
            plotter.add_mesh(
                stent_mesh.copy(),
                color=_STENT_COL, opacity=0.92,
                specular=1.00, specular_power=60,
                diffuse=0.70, ambient=0.08,
                smooth_shading=True,
                label="Stent", name=f"stent_{view}",
            )

        if disp_mesh is not None:
            plotter.add_mesh(
                disp_mesh.copy(),
                color=_DISP_COL, opacity=0.88,
                specular=0.70, specular_power=30,
                diffuse=0.80, ambient=0.10,
                smooth_shading=True,
                label="Stent (displaced)", name=f"stent_disp_{view}",
            )

        if (displacement_vector is not None
                and original_centroid is not None
                and np.linalg.norm(displacement_vector) > 0):
            mag = float(np.linalg.norm(displacement_vector))
            plotter.add_arrows(
                cent=original_centroid.reshape(1, 3),
                direction=displacement_vector.reshape(1, 3) / mag,
                mag=mag * 1.3,
                color=_ARROW_COL,
                name=f"arrow_{view}",
            )

        # Per-viewport scene settings
        try:
            plotter.set_background(_BG_TOP, top=_BG_BOTTOM)
        except TypeError:
            plotter.set_background(_BG_TOP)
        try:
            plotter.enable_anti_aliasing(_AA_MODE)
        except Exception:
            pass
        _add_studio_lights(plotter)
        plotter.add_axes(interactive=False, line_width=2)
        plotter.add_text(
            view.capitalize(),
            position="upper_edge", font_size=10,
            color="white", shadow=True,
        )
        _set_camera(plotter, view)

    # Global title via scalar bar or text in first subplot
    plotter.subplot(0, 0)
    if title:
        plotter.add_text(
            title, position="lower_left", font_size=9,
            color="#aaaaaa", shadow=False,
        )

    if save_screenshot:
        os.makedirs(os.path.dirname(save_screenshot) or ".", exist_ok=True)
        plotter.screenshot(save_screenshot, transparent_background=False)
        print(f"    3-D composite screenshot saved: {save_screenshot}", flush=True)

    plotter.close()


def _set_camera(plotter, view: str) -> None:
    """Set standard camera positions."""
    if view == "isometric":
        plotter.view_isometric()
    elif view == "sagittal":
        plotter.view_yz()
    elif view == "coronal":
        plotter.view_xz()
    elif view == "axial":
        plotter.view_xy()
    else:
        plotter.view_isometric()



# ---------------------------------------------------------------------------
# 3-D Lumen visualisation (True / False lumen for aortic dissection)
# ---------------------------------------------------------------------------

_TRUE_LUMEN_COL  = "#4488ff"   # blue  – True lumen
_FALSE_LUMEN_COL = "#ff4422"   # red   – False lumen


def plot_lumen_3d(
    aorta_mask: np.ndarray,
    true_lumen_mask: np.ndarray,
    false_lumen_mask: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    view: str = "isometric",
    save_screenshot: Optional[str] = None,
    off_screen: bool = True,
    window_size: Tuple[int, int] = (1600, 1200),
    title: str = "",
) -> None:
    """Render aorta wall (semi-transparent) + true lumen (blue) + false lumen (red).

    Parameters
    ----------
    aorta_mask       : binary aorta mask, uint8
    true_lumen_mask  : binary true lumen mask, uint8
    false_lumen_mask : binary false lumen mask, uint8
    spacing          : voxel spacing (sz, sy, sx) in mm
    view             : "isometric" | "sagittal" | "coronal" | "axial"
    save_screenshot  : path to save PNG, or None
    off_screen       : render without a window
    window_size      : (W, H) pixels
    title            : optional text annotation
    """
    if not _HAVE_PV:
        print("  [WARN] PyVista not installed – skipping lumen 3-D render.", flush=True)
        return

    plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size))

    # Aorta wall (very transparent)
    if aorta_mask.any():
        try:
            am = _extract_surface_mesh(aorta_mask, spacing)
            if am.n_points > 0:
                plotter.add_mesh(am, color=_AORTA_COL, opacity=0.15,
                                 specular=0.3, diffuse=0.5, ambient=0.1,
                                 smooth_shading=True, name="aorta_wall")
        except Exception as e:
            print(f"  [WARN] Lumen aorta mesh: {e}", flush=True)

    # True lumen (blue, opaque)
    if true_lumen_mask.any():
        try:
            tm = _extract_surface_mesh(true_lumen_mask, spacing)
            if tm.n_points > 0:
                plotter.add_mesh(tm, color=_TRUE_LUMEN_COL, opacity=0.88,
                                 specular=0.7, specular_power=40,
                                 diffuse=0.8, ambient=0.1,
                                 smooth_shading=True,
                                 label="True lumen", name="true_lumen")
        except Exception as e:
            print(f"  [WARN] True lumen mesh: {e}", flush=True)

    # False lumen (red, slightly transparent)
    if false_lumen_mask.any():
        try:
            fm = _extract_surface_mesh(false_lumen_mask, spacing)
            if fm.n_points > 0:
                plotter.add_mesh(fm, color=_FALSE_LUMEN_COL, opacity=0.75,
                                 specular=0.6, specular_power=30,
                                 diffuse=0.8, ambient=0.1,
                                 smooth_shading=True,
                                 label="False lumen", name="false_lumen")
        except Exception as e:
            print(f"  [WARN] False lumen mesh: {e}", flush=True)

    # Scene
    try:
        plotter.set_background(_BG_TOP, top=_BG_BOTTOM)
    except TypeError:
        plotter.set_background(_BG_TOP)
    try:
        plotter.enable_anti_aliasing(_AA_MODE)
    except Exception:
        pass
    _add_studio_lights(plotter)
    plotter.add_axes(interactive=False, line_width=2)
    ann = title or f"True/False Lumen – {view.capitalize()}"
    plotter.add_text(ann, position="upper_edge", font_size=11,
                     color="white", shadow=True)
    _set_camera(plotter, view)

    try:
        plotter.add_legend(face="rectangle", size=(0.26, 0.20),
                           bcolor="#1a2535", border=True, label_color="white")
    except Exception:
        pass

    if save_screenshot:
        os.makedirs(os.path.dirname(save_screenshot) or ".", exist_ok=True)
        plotter.screenshot(save_screenshot, transparent_background=False)
        print(f"    Lumen 3-D screenshot: {save_screenshot}", flush=True)
    plotter.close()


def plot_lumen_3d_composite(
    aorta_mask: np.ndarray,
    true_lumen_mask: np.ndarray,
    false_lumen_mask: np.ndarray,
    spacing: Tuple[float, ...] = (1.0, 1.0, 1.0),
    save_screenshot: Optional[str] = None,
    off_screen: bool = True,
    window_size: Tuple[int, int] = (2400, 1800),
    title: str = "",
) -> None:
    """2×2 composite of isometric / coronal / sagittal / axial lumen views."""
    if not _HAVE_PV:
        print("  [WARN] PyVista not installed – skipping composite lumen render.", flush=True)
        return

    views_grid = [
        ("isometric", (0, 0)),
        ("coronal",   (0, 1)),
        ("sagittal",  (1, 0)),
        ("axial",     (1, 1)),
    ]

    plotter = pv.Plotter(off_screen=off_screen, window_size=list(window_size), shape=(2, 2))

    # Pre-extract meshes once
    aorta_m = true_m = false_m = None
    if aorta_mask.any():
        try:
            aorta_m = _extract_surface_mesh(aorta_mask, spacing)
            if aorta_m.n_points == 0:
                aorta_m = None
        except Exception:
            pass
    if true_lumen_mask.any():
        try:
            true_m = _extract_surface_mesh(true_lumen_mask, spacing)
            if true_m.n_points == 0:
                true_m = None
        except Exception:
            pass
    if false_lumen_mask.any():
        try:
            false_m = _extract_surface_mesh(false_lumen_mask, spacing)
            if false_m.n_points == 0:
                false_m = None
        except Exception:
            pass

    for view, (row, col) in views_grid:
        plotter.subplot(row, col)
        if aorta_m is not None:
            plotter.add_mesh(aorta_m.copy(), color=_AORTA_COL, opacity=0.15,
                             smooth_shading=True, name=f"aw_{view}")
        if true_m is not None:
            plotter.add_mesh(true_m.copy(), color=_TRUE_LUMEN_COL, opacity=0.88,
                             specular=0.7, smooth_shading=True, name=f"tl_{view}")
        if false_m is not None:
            plotter.add_mesh(false_m.copy(), color=_FALSE_LUMEN_COL, opacity=0.75,
                             specular=0.6, smooth_shading=True, name=f"fl_{view}")

        try:
            plotter.set_background(_BG_TOP, top=_BG_BOTTOM)
        except TypeError:
            plotter.set_background(_BG_TOP)
        try:
            plotter.enable_anti_aliasing(_AA_MODE)
        except Exception:
            pass
        _add_studio_lights(plotter)
        plotter.add_axes(interactive=False, line_width=2)
        plotter.add_text(view.capitalize(), position="upper_edge",
                         font_size=10, color="white", shadow=True)
        _set_camera(plotter, view)

    if save_screenshot:
        os.makedirs(os.path.dirname(save_screenshot) or ".", exist_ok=True)
        plotter.screenshot(save_screenshot, transparent_background=False)
        print(f"    Lumen composite saved: {save_screenshot}", flush=True)
    plotter.close()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _save_or_close(fig, save_path: Optional[str]) -> None:
    """Save figure to file or just close it."""
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"    Plot saved: {save_path}", flush=True)
    plt.close(fig)
