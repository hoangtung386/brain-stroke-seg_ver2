import os
import nibabel as nib
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_brats_slices(input_root, output_root, modality='t2f'):
    """
    Optional: Convert BraTS 3D NIfTI volumes to 2D NPZ slices.
    This mimics the CPAISD format and speeds up training by avoiding NIfTI reads during training.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)
    
    splits = ['ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData']
    
    for split_name in splits:
        split_in = input_root / split_name
        split_out_name = 'train' if 'Training' in split_name else 'val'
        split_out = output_root / split_out_name
        
        if not split_in.exists():
            continue
            
        print(f"Processing {split_name} -> {split_out}")
        os.makedirs(split_out, exist_ok=True)
        
        subjects = sorted([d for d in split_in.iterdir() if d.is_dir()])
        
        for subj in tqdm(subjects):
            subj_id = subj.name
            img_path = subj / f"{subj_id}-{modality}.nii.gz"
            seg_path = subj / f"{subj_id}-seg.nii.gz"
            
            if not img_path.exists() or not seg_path.exists():
                continue
                
            # Load Volumes
            img = nib.load(img_path).get_fdata().astype(np.float32)
            seg = nib.load(seg_path).get_fdata().astype(np.uint8)
            
            # Normalize Volume
            if img.max() > 0:
                img = (img - img.min()) / (img.max() - img.min())
            
            # Save Slices
            # Skip margins
            depth = img.shape[2]
            margin = 15
            
            subj_out = split_out / subj_id
            os.makedirs(subj_out, exist_ok=True)
            
            for i in range(margin, depth - margin):
                img_slice = img[..., i]
                seg_slice = seg[..., i]
                
                # Check for content (optional) or save all
                if img_slice.max() > 0:
                    np.savez_compressed(subj_out / f"slice_{i:03d}.npz", 
                                        image=img_slice, 
                                        mask=seg_slice)
                                        
    print("Done! Slices extracted.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../../Dataset_BraTs")
    parser.add_argument("--output", default="../../dataset_brats_processed")
    args = parser.parse_args()
    
    # Not running by default, just providing the tool
    print("This script converts 3D BraTS to 2D NPZ slices.")
    print("Usage: python prepare.py --root ../Dataset_BraTs --output ./data_processed")
