# Preprocessing Module

This directory contains scripts for data analysis and preparation for different datasets.

## Structure

```
preprocessing/
├── cpaisd/                 # Scripts for Stroke Dataset (CT)
│   ├── check_npz.py        # Check NPZ file integrity
│   └── explore.py          # Visual exploration of data
├── brats/                  # Scripts for BraTS Dataset (MRI)
│   ├── analyze.py          # Analyze 3D NIfTI stats
│   └── prepare.py          # (Optional) Convert 3D NIfTI to 2D NPZ
```

## Usage

### BraTS Analysis
Check if your BraTS dataset is recognized correctly:
```bash
python preprocessing/brats/analyze.py
```

### CPAISD Exploration
Explore the structure of the Stroke dataset:
```bash
python preprocessing/cpaisd/explore.py
```
