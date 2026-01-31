# Universal Symmetry-Aware Medical Segmentation (SymFormer)

> **New**: Integrated **Mask Alignment Fix** and **Enhanced Preprocessing** (ICIP 2015) for Stroke Segmentation.

## Introduction

This project implements **SymFormer**, a symmetry-aware transformer network for medical image segmentation (CT/MRI). It leverages the inherent symmetry of the human body (brain, abdomen) to detect anomalies like stroke lesions or tumors.

## Key Features

- **Symmetry-Aware Bottleneck**: Explicitly models left-right hemisphere symmetry.
- **Alignment Network**: Automatically aligns input images to a canonical symmetric pose.
- **Mask Alignment (Critical Fix)**: Ensures ground truth masks are aligned exactly like the input images during training.
- **Enhanced Preprocessing**: Advanced contrast enhancement for low-contrast stroke lesions.
- **Multi-Dataset Support**: CPAISD (Stroke CT), BraTS (Brain Tumor MRI), RSNA (Abdominal CT).

---

## Quick Start

### 1. Environment Setup

```bash
conda activate brain_seg_stroke
pip install -r requirements.txt
```

### 2. Dataset Paths

Ensure your dataset paths are correct in `configs/config.py`:

```python
DATA_PATHS = {
    'cpaisd': 'dataset_APIS/dataset',
    'cpaisd_enhanced': 'dataset_APIS/dataset',
    'brats': 'Dataset_BraTs',
    'rsna': 'datasets/RSNA'
}
```

---

## Training Commands

### CPAISD Dataset (Stroke CT - Standard)

Standard single-channel input with HU windowing:

```powershell
# Single GPU (cuda:0)
python train.py --devices "0" --dataset cpaisd

# Single GPU (cuda:1)
python train.py --devices "1" --dataset cpaisd

# Multi-GPU (DataParallel)
python train.py --devices "0,1" --dataset cpaisd
```

### CPAISD Enhanced Dataset (Stroke CT - Multi-Channel)

3-channel enhanced input (Stroke Window + Context + Enhanced) for maximum lesion visibility:

```powershell
# Single GPU (cuda:0)
python train.py --dataset cpaisd_enhanced --devices "0"

# Single GPU (cuda:1)
python train.py --dataset cpaisd_enhanced --devices "1"

# Multi-GPU (DataParallel)
python train.py --dataset cpaisd_enhanced --devices "0,1"
```

### BraTS Dataset (Brain Tumor MRI)

```powershell
# Single GPU (cuda:0)
python train.py --devices "0" --dataset brats

# Single GPU (cuda:1)
python train.py --devices "1" --dataset brats

# Multi-GPU
python train.py --devices "0,1" --dataset brats
```

### RSNA Dataset (Abdominal Trauma CT)

```powershell
# Single GPU (cuda:0)
python train.py --devices "0" --dataset rsna

# Single GPU (cuda:1)
python train.py --devices "1" --dataset rsna

# Multi-GPU
python train.py --devices "0,1" --dataset rsna
```

---

## Evaluation Commands

**Important**: Use the same `--dataset` argument for evaluation as you used for training. Otherwise, the model architecture will not match (e.g., 1-channel vs 3-channel).

### CPAISD Evaluation (Standard)

```powershell
# Basic evaluation
python evaluate.py --checkpoint checkpoints/best_model.pth --dataset cpaisd

# With GPU selection
python evaluate.py --checkpoint checkpoints/best_model.pth --dataset cpaisd --device cuda:1

# Save to custom folder
python evaluate.py --checkpoint checkpoints/best_model.pth --dataset cpaisd --output-dir ./results_cpaisd

# Visualize only 10 samples
python evaluate.py --checkpoint checkpoints/best_model.pth --dataset cpaisd --num-samples 10
```

### CPAISD Enhanced Evaluation (Multi-Channel)

**Critical**: Must use `--dataset cpaisd_enhanced` if model was trained with enhanced dataset:

```powershell
# Basic evaluation (3-channel model)
python evaluate.py --checkpoint checkpoints/best_model_enhanced.pth --dataset cpaisd_enhanced

# With GPU selection
python evaluate.py --checkpoint checkpoints/best_model_enhanced.pth --dataset cpaisd_enhanced --device cuda:1

# Save to custom folder
python evaluate.py --checkpoint checkpoints/best_model_enhanced.pth --dataset cpaisd_enhanced --output-dir ./results_enhanced

# Visualize only 10 samples
python evaluate.py --checkpoint checkpoints/best_model_enhanced.pth --dataset cpaisd_enhanced --num-samples 10
```

### BraTS Evaluation (Brain Tumor MRI)

```powershell
# Basic evaluation
python evaluate.py --checkpoint checkpoints/best_model_brats.pth --dataset brats

# Visualize only 10 samples
python evaluate.py --checkpoint checkpoints/best_model_brats.pth --dataset brats --num-samples 10
```

### RSNA Evaluation (Abdominal Trauma CT)

```powershell
# Basic evaluation
python evaluate.py --checkpoint checkpoints/best_model_rsna.pth --dataset rsna

# Visualize only 10 samples
python evaluate.py --checkpoint checkpoints/best_model_rsna.pth --dataset rsna --num-samples 10
```

---

## Evaluation Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--checkpoint` | Yes | - | Path to model checkpoint (.pth file) |
| `--dataset` | No | `cpaisd` | Dataset name (cpaisd, cpaisd_enhanced, brats, rsna) |
| `--output-dir` | No | `./output` | Directory to save results |
| `--batch-size` | No | `1` | Batch size for evaluation |
| `--num-samples` | No | `-1` | Number of samples to visualize (-1 = all) |
| `--device` | No | `cuda` if available | Device (cuda, cuda:0, cpu) |

---

## Critical Fixes

### Mask Alignment Fix

**Problem**: The model contains an `AlignmentNetwork` that rotates/shifts the input image to be symmetric. Previously, the ground truth mask was *not* transformed, leading to a mismatch (Image is rotated, Mask is upright).

**Solution**:
- Updated `train.py` and `evaluate.py`.
- The alignment parameters predicted by the model are extracted.
- These parameters are applied to the Mask using `Nearest Neighbor` interpolation (to preserve class labels).
- **Applies to ALL datasets** (CPAISD, CPAISD_Enhanced, BraTS, RSNA).

### Double Normalization Fix

**Problem**: Double normalization was destroying data distribution:
- HU windowing → [0, 1]
- Then Z-Score → [-1.85, +1.12]

**Solution**:
- Removed Z-Score normalization after windowing.
- Set `MEAN = None` and `STD = None` in config.
- Images now stay in [0, 1] range after preprocessing.

### Dataset Channel Mismatch Fix

**Problem**: Training with `cpaisd_enhanced` (3-channel) but evaluating with `cpaisd` (1-channel) caused model weights mismatch.

**Solution**:
- Updated `evaluate.py` to support `cpaisd_enhanced` dataset.
- Dynamically set `NUM_CHANNELS` based on dataset type.
- Always use matching `--dataset` argument for training and evaluation.

---

## Project Structure

```
Universal_Symmetry-Aware_Med_Seg/
├── models/
│   ├── symformer.py          # Main SymFormer architecture
│   └── losses.py             # Custom loss functions
├── datasets/
│   ├── cpaisd.py             # CPAISD dataset loader
│   ├── cpaisd_enhanced.py    # Enhanced CPAISD with 3-channel input
│   ├── brats.py              # BraTS dataset loader
│   ├── rsna.py               # RSNA dataset loader
│   └── factory.py            # Dataset factory
├── preprocessing/
│   └── enhancement.py        # Image enhancement pipeline
├── configs/
│   └── config.py             # Configuration files
├── train.py                  # Training script
├── evaluate.py               # Evaluation script
└── checkpoints/              # Saved model checkpoints
```

---

## Checkpoint Management

Checkpoints are saved in `checkpoints/` directory with the following naming convention:
- `symformer_best_{dataset}.pth` - Best model based on validation Dice
- `symformer_last.pth` - Last epoch checkpoint

**Note**: Always match the checkpoint with the correct dataset configuration when evaluating.
