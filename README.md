# Brain Stroke Segmentation with SymFormer

A state-of-the-art framework for brain stroke lesion segmentation from CT scans, featuring **SymFormer** (Symmetry-Aware Hybrid Transformer).

## 🌟 Key Features

### Model Architecture: SymFormer
- **Symmetry-Aware Bottleneck**: Specifically designed to catch asymmetrical stroke features by comparing left and right brain hemispheres.
- **Mamba-2 Integration**: Utilizes the Mamba-2 State Space Model for efficient long-range dependency modeling.
- **KAN Decoder**: Implements Kolmogorov-Arnold Networks (KAN) in the decoder for improved boundary precision.
- **Clinical Conditioning**: Incorporates patient metadata (NIHSS, Age, Time from onset) to guide segmentation.

### Project Structure
```
.
├── configs/            # Configuration files (hyperparameters, paths)
├── dataset_APIS/       # Dataset directory (CPAISD)
├── models/             # Neural Network architectures
│   ├── layers/         # SOTA blocks (Mamba, KAN, Attention)
│   ├── symformer.py    # Main SymFormer model
│   └── components.py   # Encoder/Decoder blocks
├── scripts/            # Utility scripts (check_npz.py, explore_dataset.py)
├── utils/              # Helper functions (metrics, logging)
├── train.py            # Main training script
└── evaluate.py         # Evaluation script
```

## 💾 Dataset
The project uses the **CPAISD** dataset. 
- **Download Link**: [Zenodo Record 10892316](https://zenodo.org/records/10892316)
- **Format**: Pre-processed `.npz` files (image/mask) or raw DICOM.

## 🚀 Usage

### 1. Installation
```powershell
pip install -r requirements.txt
# Update WandB (Required for v1 keys)
pip install --upgrade wandb
```

### 2. Training
Run the training script with specific GPU selection:

```powershell
# Run on GPU 0
python train.py --devices 0

# Run on GPU 1
python train.py --devices 1

# Run on multiple GPUs (0 and 1)
python train.py --devices 0,1

# Run on CPU
# (Automatically falls back if no CUDA devices found)
python train.py
```

### 3. WandB Integration
The project is configured to log training metrics to Weights & Biases.
If you encounter login issues, set your API key manually before training:

```powershell
$env:WANDB_API_KEY = "your_wandb_api_key_here"
python train.py
```
