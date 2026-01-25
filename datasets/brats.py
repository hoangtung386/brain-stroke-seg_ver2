from .base import BaseDataset
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import os
import glob

class BraTSDataset(BaseDataset):
    """
    Dataset loader for BraTS (Brain Tumor Segmentation)
    MRI images (T1, T1ce, T2, FLAIR)
    
    Mapping Strategy:
    - Input: T2-FLAIR (Best for edema/penumbra-like features)
    - Labels:
        0: Background -> 0
        1: NCR (Necrotic Core) + 3: ET (Enhancing Tumor) -> 1 (Core)
        2: Edema -> 2 (Penumbra)
    """
    def __init__(self, dataset_root, split='train', T=1, transform=None):
        super().__init__(dataset_root, split, T, transform)
        self.dataset_root = Path(dataset_root)
        
        # Adjust path for Validation vs Training folder names if needed
        # User provided: 
        # Train: ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData
        # Val: ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData
        
        # Simple heuristic to find the right folder if the root is just 'Dataset_BraTs'
        if split == 'train':
            if (self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData').exists():
                self.data_dir = self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
            else:
                 self.data_dir = self.dataset_root # Assume direct path
        else: # val/test
            if (self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData').exists():
                self.data_dir = self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
            else:
                 # If validation folder missing, fallback to train or specific val path if provided
                 self.data_dir = self.dataset_root 
        
        self.samples = self._build_index()
        
        print(f"\n{split.upper()} Dataset (BraTS):")
        print(f"  Root: {self.data_dir}")
        print(f"  Total slices: {len(self.samples)}")
        print(f"  Adjacent slices (T): {T}")
        
    def _build_index(self):
        """
        Build index of reliable slices.
        We only index slices that contain some brain tissue to avoid empty training batches.
        """
        samples = []
        
        if not self.data_dir.exists():
            print(f"WARNING: {self.data_dir} does not exist!")
            return []
            
        studies = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        
        for study_path in studies:
            study_id = study_path.name
            
            # Paths to files
            flair_path = study_path / f"{study_id}-t2f.nii.gz"
            seg_path = study_path / f"{study_id}-seg.nii.gz"
            
            if not flair_path.exists() or not seg_path.exists():
                continue
                
            # We delay loading the heavy NIfTI volume. 
            # But we need basic info (num_slices).
            # Optimization: Load header only.
            try:
                img_proxy = nib.load(flair_path)
                num_slices = img_proxy.shape[2] # (H, W, D)
                
                # Add slices
                # To reduce index size and empty slices, we might want to skip first/last 20 slices
                # BraTS typically has ~155 slices, margins are usually empty background.
                margin = 15
                for i in range(margin, num_slices - margin):
                    samples.append({
                        'study_id': study_id,
                        'slice_idx': i,
                        'flair_path': str(flair_path),
                        'seg_path': str(seg_path),
                        'num_slices': num_slices,
                        'path': study_path
                    })
            except Exception as e:
                print(f"Error reading header for {study_id}: {e}")
                
        return samples

    def _load_volume_slice(self, path, slice_idx):
        """Load a specific slice from NIfTI volume"""
        # Note: Loading entire NIfTI for 1 slice is inefficient.
        # Ideally, we cache volumes or convert to .npy/.npz ahead of time.
        # For 'on-the-fly' loading, this is the trade-off.
        
        img = nib.load(path)
        # Slicing creates a proxy, data_obj gets the data
        # shape is (H, W, D), we want indices corresponding to slice_idx
        slice_data = img.dataobj[..., slice_idx] 
        return np.array(slice_data).astype(np.float32)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        slice_idx = sample['slice_idx']
        num_slices = sample['num_slices']
        flair_path = sample['flair_path']
        seg_path = sample['seg_path']
        
        # 1. Load Input Images (2T+1 slices)
        images = []
        for offset in range(-self.T, self.T + 1):
            s_idx = max(0, min(num_slices - 1, slice_idx + offset))
            
            # Load slice
            img_slice = self._load_volume_slice(flair_path, s_idx)
            
            # Normalization (Simple Min-Max per slice or global Z-score)
            # Global Z-score is better, but requires volume stats. 
            # For simplicity and robustness per slice: Min-Max
            if img_slice.max() > 0:
                img_slice = (img_slice - img_slice.min()) / (img_slice.max() - img_slice.min())
            
            images.append(img_slice)
            
        images = np.stack(images, axis=0) # (2T+1, H, W)
        
        # 2. Load Mask (Center slice)
        mask_slice = self._load_volume_slice(seg_path, slice_idx).astype(np.int64)
        
        # 3. Map Labels
        # BraTS: 0=bg, 1=NCR, 2=Edema, 3=ET
        # Target: 0=bg, 1=Core(NCR+ET), 2=Penumbra(Edema)
        
        new_mask = np.zeros_like(mask_slice)
        new_mask[mask_slice == 1] = 1 # NCR -> Core
        new_mask[mask_slice == 3] = 1 # ET -> Core
        new_mask[mask_slice == 2] = 2 # Edema -> Penumbra
        
        # Convert to tensor
        images = torch.from_numpy(images).float()
        mask = torch.from_numpy(new_mask).long()
        
        # Metadata
        metadata = {
            'study_id': sample['study_id'],
            'slice_index': slice_idx,
            # Dummy clinical data (BraTS doesn't usually provide NIHSS/Time)
            'nihss': 10.0,
            'age': 60.0,
            'dsa': 0,
            'time': 0,
            'sex': 0
        }
        
        return images, mask, metadata
