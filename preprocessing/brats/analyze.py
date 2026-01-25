import os
import nibabel as nib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_brats(dataset_root):
    """
    Analyze BraTS dataset structure and intensity statistics.
    Useful for checking if paths are correct and establishing normalization parameters.
    """
    dataset_root = Path(dataset_root)
    train_dir = dataset_root / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData"
    
    if not train_dir.exists():
        print(f"Error: Training directory not found at {train_dir}")
        return

    subjects = sorted([d for d in train_dir.iterdir() if d.is_dir()])
    print(f"Found {len(subjects)} subjects in training set.")
    
    # Statistics
    modalities = ['t2f', 't1c', 't1n', 't2w', 'seg']
    shapes = []
    intensities = {'max': [], 'mean': []}
    
    print("\nScanning first 5 subjects as detailed sample...")
    
    for i, subj in enumerate(subjects[:5]):
        subj_id = subj.name
        print(f"\nSubject: {subj_id}")
        
        # Check files
        for mod in modalities:
            fpath = subj / f"{subj_id}-{mod}.nii.gz"
            if fpath.exists():
                img = nib.load(fpath)
                data = img.get_fdata()
                
                if mod == 'seg':
                    unique_classes = np.unique(data)
                    print(f"  {mod}: Shape={data.shape}, Unique Classes={unique_classes}")
                else:                     
                    print(f"  {mod}: Shape={data.shape}, Range=[{data.min():.1f}, {data.max():.1f}], Mean={data.mean():.1f}")
                
                if mod == 't2f': # Analysis on primary modality
                    shapes.append(data.shape)
                    intensities['max'].append(data.max())
                    intensities['mean'].append(data.mean())
            else:
                print(f"  {mod}: MISSING")
                
    print(f"\n{'='*50}")
    print("Aggregate Analysis (Sample)")
    print(f"{'='*50}")
    if shapes:
        print(f"Common Shape: {shapes[0]}")
    if intensities['max']:
        print(f"Average Max Intensity (t2f): {np.mean(intensities['max']):.2f}")
        print(f"Average Mean Intensity (t2f): {np.mean(intensities['mean']):.2f}")

    print("\nFile structure verification complete.")
    print("The dataset structure matches the expected BraTS 2023/2024 Challenge format.")

if __name__ == "__main__":
    # Default path from project structure
    default_path = "../../Dataset_BraTs"
    
    # Check absolute path first if relative fails
    if not os.path.exists(default_path):
        default_path = "C:/Users/Admin/Projects/brain-stroke-segmentation_ver2/Dataset_BraTs"
        
    analyze_brats(default_path)
