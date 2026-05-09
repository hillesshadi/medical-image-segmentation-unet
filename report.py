"""
report.py
=========
Clinical report generation – CSV + PDF.

Exports
-------
- ``displacement_report.csv``  – one row per model per dataset pair;
  columns: model, pair, dx_mm, dy_mm, dz_mm, magnitude_mm, DSC, IoU, HD95
- ``clinical_report.pdf``      – A4 PDF with title, displacement table,
  metrics table, and embedded key visualisations (first 4 PNGs found in
  the output sub-directories).

Dependencies
------------
  pandas, fpdf2  (``pip install fpdf2``)
"""

import os
import datetime
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Any

try:
    from fpdf import FPDF
    _HAVE_FPDF = True
except ImportError:
    _HAVE_FPDF = False
    FPDF = object   # dummy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CSV export
# ---------------------------------------------------------------------------

# REPLACE _flatten_results WITH:

def _flatten_results(
    seg_results: Dict,
    disp_stats:  Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge segmentation metrics and displacement stats into flat DataFrames.

    FIX 1: NaN check now works for numpy.float64 (was only checking float).
    FIX 2: Displacement NaN values written as empty string, not "nan" text.
    """
    rows = []
    for model_name, by_ds in seg_results.items():
        for ds_name, metrics in by_ds.items():
            row = {"model": model_name, "dataset": ds_name}
            for k, v in metrics.items():
                try:
                    fv = float(v)
                    row[k] = round(fv, 4) if not np.isnan(fv) else float("nan")
                except (TypeError, ValueError):
                    row[k] = float("nan")
            rows.append(row)
    seg_df = pd.DataFrame(rows)

    disp_rows = []
    for model_name, pair_stats in disp_stats.items():
        for (a, b), st in pair_stats.items():
            disp_row = {"model": model_name, "pair": f"{a} → {b}"}
            for k, v in st.items():
                try:
                    fv = float(v)
                    # Write NaN as None so pandas outputs empty cell in CSV
                    # (not the text "nan" which confuses Excel/clinical readers)
                    disp_row[k] = round(fv, 3) if not np.isnan(fv) else None
                except (TypeError, ValueError):
                    disp_row[k] = None
            disp_rows.append(disp_row)
    disp_df = pd.DataFrame(disp_rows)

    return seg_df, disp_df

def export_csv(
    seg_results:  Dict,
    disp_stats:   Dict,
    output_dir:   str,
) -> Tuple[str, str]:
    """Save segmentation metrics and displacement tables as CSV files.

    Returns
    -------
    (seg_csv_path, disp_csv_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    seg_df, disp_df = _flatten_results(seg_results, disp_stats)

    seg_path  = os.path.join(output_dir, "segmentation_metrics.csv")
    disp_path = os.path.join(output_dir, "displacement_report.csv")

    seg_df.to_csv(seg_path,  index=False)
    disp_df.to_csv(disp_path, index=False)
    print(f"  CSV saved: {seg_path}", flush=True)
    print(f"  CSV saved: {disp_path}", flush=True)
    return seg_path, disp_path


# ---------------------------------------------------------------------------
# PDF report (FPDF2)
# ---------------------------------------------------------------------------

class _PDF(FPDF if _HAVE_FPDF else object):
    """Custom FPDF subclass with header and footer."""

    def header(self):
        self.set_font("Helvetica", "B", 11)
        self.set_fill_color(30, 60, 120)
        self.set_text_color(255, 255, 255)
        self.cell(0, 10, "Longitudinal Aortic Stent Displacement Analysis Report",
                  new_x="LMARGIN", new_y="NEXT", align="C", fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.date.today()}",
                  align="C")


def _add_table(pdf, df: pd.DataFrame, title: str) -> None:
    """Render a pandas DataFrame as a table in the PDF."""
    if not _HAVE_FPDF:
        return

    pdf.set_font("Helvetica", "B", 11)
    pdf.set_fill_color(200, 220, 255)
    pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT", fill=True)
    pdf.ln(2)

    col_w = max(20, min(38, 190 // max(len(df.columns), 1)))
    # Header row
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_fill_color(220, 230, 255)
    for col in df.columns:
        pdf.cell(col_w, 7, str(col)[:14], border=1, align="C", fill=True)
    pdf.ln()

    # Data rows
    pdf.set_font("Helvetica", "", 7)
    for i, row in df.iterrows():
        fill = (i % 2 == 0)
        pdf.set_fill_color(245, 248, 255) if fill else pdf.set_fill_color(255, 255, 255)
        for val in row:
            cell_str = str(val)[:14]
            pdf.cell(col_w, 6, cell_str, border=1, align="C", fill=True)
        pdf.ln()
    pdf.ln(4)


# REPLACE _find_images WITH:

def _find_images(output_dir: str, max_images: int = 8) -> list:
    """
    Collect PNG paths from known sub-folders, including nested subfolders.

    FIX: Original only searched one level deep. 3D composite images are at
    results/3d/{dataset_name}/composite_4views.png — one level deeper.
    Now searches recursively up to 2 levels deep.

    Priority order: composite 4-view renders first, then other images.
    """
    found = []

    # Priority 1: composite 4-view renders (most informative for PDF)
    d3d = os.path.join(output_dir, "3d")
    if os.path.isdir(d3d):
        for dataset_folder in sorted(os.listdir(d3d)):
            composite = os.path.join(d3d, dataset_folder, "composite_4views.png")
            if os.path.isfile(composite):
                found.append(composite)
                if len(found) >= max_images:
                    return found

    # Priority 2: other output images (enhancement, segmentation, histograms)
    for sub in ["enhancement", "segmentation", "histograms"]:
        folder = os.path.join(output_dir, sub)
        if not os.path.isdir(folder):
            continue
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(".png"):
                found.append(os.path.join(folder, f))
                if len(found) >= max_images:
                    return found

    # Priority 3: any remaining 3D single-view renders
    if os.path.isdir(d3d):
        for dataset_folder in sorted(os.listdir(d3d)):
            subfolder = os.path.join(d3d, dataset_folder)
            if not os.path.isdir(subfolder):
                continue
            for f in sorted(os.listdir(subfolder)):
                if f.lower().endswith(".png") and "composite" not in f:
                    found.append(os.path.join(subfolder, f))
                    if len(found) >= max_images:
                        return found
    return found


def generate_pdf_report(
    seg_results:   Dict,
    disp_stats:    Dict,
    enhancement_results: Dict,
    cross_dsc:     Dict,
    output_dir:    str,
    pdf_path:      Optional[str] = None,
) -> str:
    """Generate a single-file clinical PDF report.

    Parameters
    ----------
    seg_results   : model × dataset × metric dict
    disp_stats    : model × pair × stat dict
    enhancement_results : dataset × {"MSE", "PSNR", "SSIM"} dict
    cross_dsc     : (ds_a, ds_b) → float  from evaluation.cross_dataset_dsc
    output_dir    : directory where result images live
    pdf_path      : optional override for PDF file path

    Returns
    -------
    str – path to the generated PDF.
    """
    if not _HAVE_FPDF:
        print("  [WARN] fpdf2 not installed – skipping PDF report.", flush=True)
        return ""

    if pdf_path is None:
        pdf_path = os.path.join(output_dir, "clinical_report.pdf")

    os.makedirs(output_dir, exist_ok=True)
    pdf = _PDF()
    pdf.set_auto_page_break(auto=True, margin=20)

    # ---- Page 1: Summary ----
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "Executive Summary", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 6,
        "This report summarises the results of an automated longitudinal "
        "aortic stent displacement analysis performed on three CT time-points "
        "(T1 = baseline, T2, T3) of the same patient. "
        "The pipeline includes: DICOM ingestion, HU windowing, EADTV denoising, "
        "inter-scan rigid registration, dual-stage segmentation (aorta + stent) "
        "using four deep-learning architectures (UNet, ResUNet, AttUNet, nnUNet), "
        "morphological post-processing, centroid/centerline tracking, and "
        "quantitative evaluation (DSC, IoU, HD95)."
    )

    # Enhancement metrics
    pdf.ln(4)
    enh_df = pd.DataFrame([
        {"Dataset": n, **{k: round(float(v), 4) for k, v in m.items()}}
        for n, m in enhancement_results.items()
    ])
    _add_table(pdf, enh_df, "Enhancement Validation (EADTV): MSE / PSNR / SSIM")

    # Cross-dataset DSC
    if cross_dsc:
        cross_rows = [{"Pair": f"{a} → {b}", "DSC": round(float(v), 4)}
                      for (a, b), v in cross_dsc.items()]
        _add_table(pdf, pd.DataFrame(cross_rows),
                   "Cross-Dataset DSC (stent migration)")

    # ---- Page 2: Segmentation Metrics ----
    pdf.add_page()
    seg_df, disp_df = _flatten_results(seg_results, disp_stats)
    _add_table(pdf, seg_df,  "Segmentation Metrics per Model × Dataset")

    # ---- Page 3: Displacement ----
    pdf.add_page()
    _add_table(pdf, disp_df, "Stent Centroid Displacement (mm)")

    # ---- Page 4+: Key visualisations ----
    images = _find_images(output_dir, max_images=8)
    if images:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Key Visualisations", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        x_positions = [10, 110]
        img_w, img_h = 90, 70
        col_idx = 0    # tracks column position independently of image index
        y_cursor = pdf.get_y()   # tracks Y position independently of row counter

        for img_path in images:
            x = x_positions[col_idx]
            y = y_cursor

            # Check if this image fits on current page — if not, new page
            if y + img_h + 12 > pdf.h - 25:
                pdf.add_page()
                y_cursor  = pdf.get_y()
                col_idx   = 0           # FIX: reset column on new page
                x         = x_positions[0]
                y         = y_cursor

            try:
                pdf.image(img_path, x=x, y=y, w=img_w, h=img_h)
                pdf.set_xy(x, y + img_h + 1)
                pdf.set_font("Helvetica", "I", 6)
                pdf.cell(img_w, 4, os.path.basename(img_path)[:40], align="C")
            except Exception as e:
                print(f"  [WARN] Could not embed {img_path}: {e}", flush=True)
            
            # Advance col/row
            col_idx += 1
            if col_idx >= 2:
                col_idx   = 0
                y_cursor += img_h + 12   # move down one row        

    pdf.output(pdf_path)
    print(f"  PDF report saved: {pdf_path}", flush=True)
    return pdf_path
