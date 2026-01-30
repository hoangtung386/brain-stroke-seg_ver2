
# Universal Symmetry-Aware Medical Segmentation (SymFormer)

> **New**: Integrated **Mask Alignment Fix** and **Enhanced Preprocessing** (ICIP 2015) for Stroke Segmentation.

## 🌟 Introduction
This project implements **SymFormer**, a symmetry-aware transformer network for medical image segmentation (CT/MRI). It leverages the inherent symmetry of the human body (brain, abdomen) to detect anomalies like stroke lesions or tumors.

## 🚀 Key Features
- **Symmetry-Aware Bottleneck**: Explicitly models left-right hemisphere symmetry.
- **Alignment Network**: Autoregmatically aligns input images to a canonical symmetric pose.
- **Mask Alignment (Critial Fix)**: Ensures ground truth masks are aligned exactly like the input images during training.
- **Enhanced Preprocessing**: Advanced contrast enhancement for low-contrast stroke lesions.
- **Multi-Dataset Support**: CPAISD (Stroke CT), BraTS (Brain Tumor MRI), RSNA (Abdominal CT).

---

## 🛠️ Usage Guide

### 1. Environment Setup
```bash
conda activate brain_seg_stroke
pip install -r requirements.txt
```

### 2. Training with Flexible GPU Config
You can specify exactly which GPUs to use:

- **Single GPU (cuda:0)**:
  ```bash
  python train.py --devices "0" --dataset cpaisd
  ```

- **Single GPU (cuda:1)** (if cuda:0 is busy):
  ```bash
  python train.py --devices "1" --dataset cpaisd
  ```

- **Multi-GPU (DataParallel)**:
  ```bash
  python train.py --devices "0,1" --dataset cpaisd
  ```

### 3. Using Enhanced Dataset (CPAISD)
For CPAISD (Stroke), we have a new **Enhanced Pipeline** (Windowing + CLAHE + Detail Boost) to improve visibility of dark lesions.

- **Recommended Mode (3-Channel Input)**:
  Uses 3 separate channels (Stroke Window, Context, Enhanced) for maximum information.
  ```bash
  python train.py --dataset cpaisd_enhanced --devices "0"
  ```

- **Visualization**:
  Check `README_ENHANCEMENT.md` or run:
  ```bash
  python visualize_enhancement.py
  ```

---

## 🐛 Critical Fixes Verification

### ✅ Mask Alignment
**Problem**: The model contains an `AlignmentNetwork` that rotates/shifts the input image to be symmetric. Previously, the ground truth mask was *not* transformed, leading to a mismatch (Image is rotated, Mask is upright).

**Solution**: 
- We updated `train.py` and `evaluate.py`.
- The alignment parameters predicted by the model are extracted.
- These parameters are applied to the Mask using `Nearest Neighbor` interpolation (to preserve class labels).
- **Verification**: This applies to **ALL** datasets (CPAISD, BraTS, RSNA) because the logic is embedded in the main training loop and `SymFormer` architecture.

### ✅ Evaluation Consistency
The `evaluate.py` script has been updated to use the exact same preprocessing and mask alignment logic as `train.py`. This ensures your validation metrics are accurate.

---

## 📂 Project Structure

- `models/`: SymFormer architecture, AlignmentNetwork, etc.
- `datasets/`: Data loaders for CPAISD, BraTS, RSNA.
- `preprocessing/`: Enhancement logic (`enhancement.py`).
- `configs/`: Configuration files (`config.py`).
- `train.py`: Main training script (supports Multi-GPU & Mask Alignment).
- `evaluate.py`: Evaluation and visualization script.

---
**Note**: Always ensure your dataset paths are correct in `configs/config.py`.
