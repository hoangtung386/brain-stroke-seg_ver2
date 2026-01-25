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
        if split == 'train':
            if (self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData').exists():
                self.data_dir = self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
            else:
                 self.data_dir = self.dataset_root # Assume direct path
        else: # val/test
            if (self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData').exists():
                self.data_dir = self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
            else:
                 self.data_dir = self.dataset_root 
        
        self.samples = self._build_index()
        
        # Simple cache for volumes to avoid repetitive IO
        # Stores: path -> volume array (normalized)
        self.volume_cache = {} 
        self.max_cache_size = 2 # Keep 2 volumes in memory (current + next usually)
        
        print(f"\n{split.upper()} Dataset (BraTS):")
        print(f"  Root: {self.data_dir}")
        print(f"  Total slices: {len(self.samples)}")
        print(f"  Adjacent slices (T): {T}")
        print(f"  Normalization: Z-Score (Per-Volume)")
        
    def _build_index(self):
        """
        Build index of reliable slices.
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
            
            # Check existence
            if not flair_path.exists():
                continue
            
            # For Training, we MUST have labels
            # For Validation/Test, it's okay to miss regular labels (we returned dummy)
            if self.split == 'train' and not seg_path.exists():
                continue
                
            try:
                # Optimized header read
                img_proxy = nib.load(flair_path)
                num_slices = img_proxy.shape[2] 
                
                margin = 20 # Avoid empty edges
                for i in range(margin, num_slices - margin):
                    samples.append({
                        'study_id': study_id,
                        'slice_idx': i,
                        'flair_path': str(flair_path),
                        'seg_path': str(seg_path),
                        'num_slices': num_slices
                    })
            except Exception as e:
                print(f"Error reading header for {study_id}: {e}")
                
        return samples

    def _load_volume_cached(self, path):
        """
        Load and normalize entire volume with caching.
        Performs Z-Score normalization per volume.
        """
        if path in self.volume_cache:
            return self.volume_cache[path]
            
        # Manage cache size
        if len(self.volume_cache) >= self.max_cache_size:
            # Remove a random key (or oldest if ordered, python dicts are ordered by insertion now)
            # Efficient FIFO removal
            first_key = next(iter(self.volume_cache))
            del self.volume_cache[first_key]
            
        # Load Volume
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        
        # Z-Score Normalization
        brain_mask = data > 0
        if brain_mask.any():
            brain_voxels = data[brain_mask]
            mean = brain_voxels.mean()
            std = brain_voxels.std()
            
            # Z-Score
            data_norm = (data - mean) / (std + 1e-8)
            
            # Robust scaling (Percentile clipping)
            # Clip between 0.5% and 99.5% to remove outliers
            p05 = np.percentile(brain_voxels, 0.5)
            # Reconstruct approximately from z-score
            # or just clip the z-score itself.
            # Clipping Z-score to [-3, 3] is also common.
            # Let's clip to [p05, p995] of original values transformed
            
            # Simplified robust Min-Max on Z-Scored data
            # Typically range is approx [-3, 3]
            data_norm = np.clip(data_norm, -3.0, 3.0)
            
            # Scale to [0, 1] for model consistency
            data_norm = (data_norm - data_norm.min()) / (data_norm.max() - data_norm.min() + 1e-8)
            
            data = data_norm
        else:
            # Empty volume (unlikely)
            data = data / (data.max() + 1e-8)
            
        self.volume_cache[path] = data
        return data

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        slice_idx = sample['slice_idx']
        num_slices = sample['num_slices']
        flair_path = sample['flair_path']
        seg_path = sample['seg_path']
        
        # 1. Load Normalized Volume (Cached)
        flair_volume = self._load_volume_cached(flair_path)
        
        # 2. Extract Input Slices (2T+1)
        images = []
        for offset in range(-self.T, self.T + 1):
            s_idx = max(0, min(num_slices - 1, slice_idx + offset))
            images.append(flair_volume[..., s_idx])
            
        images = np.stack(images, axis=0) # (2T+1, H, W)
        
        # 3. Load Mask (Center slice only)
        # Handle missing segregation (for Validation/Test sets)
        if os.path.exists(seg_path):
            img_seg = nib.load(seg_path)
            mask_slice = img_seg.dataobj[..., slice_idx].astype(np.int64)
            
            # 4. Map Labels
            # Return native BraTS labels: 0=bg, 1=NCR, 2=Edema, 3=ET
            # We do NOT map to Core/Penumbra anymore.
            new_mask = mask_slice # Keep original classes
        else:
            # Dummy mask for validation/inference
            new_mask = np.zeros(images.shape[1:], dtype=np.int64) 

        
        # Convert to tensor
        images = torch.from_numpy(images).float()
        mask = torch.from_numpy(new_mask).long()
        
        # Metadata
        metadata = {
            'study_id': sample['study_id'],
            'slice_index': slice_idx,
            'nihss': 10.0,
            'age': 60.0,
            'dsa': 0,
            'time': 0,
            'sex': 0
        }
        
        return images, mask, metadata
