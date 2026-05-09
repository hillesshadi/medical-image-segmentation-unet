# Medical Image Segmentation using U-Net for Aortic and Stent Analysis

Deep learning-based medical image segmentation framework for automated analysis of aortic CT angiography (CTA) scans using U-Net architectures and image processing techniques.

## Project Overview

This project focuses on the segmentation and analysis of:

- Aorta structures from CT angiography scans
- Endovascular stent regions
- Stent localization inside the aorta
- Detection of stent start and end points
- Quantitative analysis of segmented regions

The framework is designed for cardiovascular medical imaging research and AI-assisted vascular analysis.

---

## Features

- CT/DICOM image preprocessing
- Aorta segmentation using U-Net
- Stent segmentation and localization
- Centerline extraction
- Startpoint and endpoint detection of stents
- Model training and inference pipeline
- Segmentation evaluation metrics
- Visualization and reporting tools

---

## Repository Structure

```text
medical-image-segmentation-unet/
│
├── centerline/         # Centerline extraction and analysis
├── dataset/            # Dataset loading and preprocessing
├── evaluation/         # Segmentation evaluation metrics
├── inference/          # Model inference and prediction
├── training/           # U-Net training pipeline
├── visualization/      # Visualization and plotting utilities
├── report/             # Generated analysis reports
├── main.py             # Main execution script
└── README.md
```

---

## Technologies

- Python
- PyTorch
- OpenCV
- NumPy
- SimpleITK
- scikit-image
- Matplotlib

---

## Research Objectives

The main objectives of this project include:

- Automated segmentation of aortic structures
- Accurate localization of vascular stents
- Detection of stent boundaries inside CTA scans
- AI-assisted cardiovascular image analysis
- Supporting medical imaging research workflows

---

## Dataset

The current experimental dataset includes CT angiography scans from 3 patients for research and validation purposes.

> Note: All medical data used in this project should be anonymized to preserve patient privacy.

---

## Applications

- Medical image analysis
- Cardiovascular imaging research
- AI-assisted healthcare systems
- Endovascular stent assessment
- Deep learning for medical imaging

---

## Disclaimer

This repository is intended for research and educational purposes only and is not designed for clinical use.
