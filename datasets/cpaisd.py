"""
CPAISD Dataset Loader - Proper preprocessing for Stroke Segmentation
Based on CPAISD paper analysis.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import pydicom
from pathlib import Path
from .base import BaseDataset

class CPAISDDataset(BaseDataset):
    """
    Dataset loader cho CPAISD (Dataset APIS)
    
    Theo paper (Section 3.2):
    - NPZ files đã được chuẩn bị sẵn
    - Mask có 3 classes: 0=background, 1=core, 2=penumbra
    - Images cần chuẩn hóa thêm
    """
    
    def __init__(self, dataset_root, split='train', T=1, 
                 use_hu_window=True, transform=None, config=None):
        super().__init__(dataset_root, split, T, transform)
        self.config = config
        self.use_hu_window = use_hu_window
        
        # Brain window parameters (standard for stroke CT)
        # Use config values if available, otherwise use defaults
        if config and hasattr(config, 'WINDOW_CENTER'):
            self.window_center = config.WINDOW_CENTER
            self.window_width = config.WINDOW_WIDTH
        else:
            self.window_center = 40  # HU
            self.window_width = 80   # HU
        
        # Filtering parameters
        self.skip_empty_slices = getattr(config, 'SKIP_EMPTY_SLICES', False) if config else False
        self.negative_sample_ratio = getattr(config, 'NEGATIVE_SAMPLE_RATIO', 0.2) if config else 0.2
        
        # Build sample index
        self.samples, self.filter_stats = self._build_index()
        
        print(f"\n{split.upper()} Dataset (CPAISD):")
        print(f"  Root: {self.dataset_root}")
        if self.skip_empty_slices:
            print(f"  ⚠️  Empty slice filtering: ENABLED")
            print(f"  Total slices (before filter): {self.filter_stats['total']}")
            print(f"  Empty slices: {self.filter_stats['empty']} (dropped: {self.filter_stats['dropped_empty']})")
            print(f"  Non-empty slices: {self.filter_stats['non_empty']}")
            print(f"  Final dataset size: {len(self.samples)}")
        else:
            print(f"  Total slices: {len(self.samples)}")
        print(f"  HU windowing: {use_hu_window}")
        print(f"  Adjacent slices (T): {T}")
    
    def _build_index(self):
        """Build index of all slices with optional empty slice filtering"""
        import random
        random.seed(42)  # For reproducible negative sampling
        
        samples = []
        filter_stats = {
            'total': 0,
            'empty': 0,
            'non_empty': 0,
            'dropped_empty': 0
        }
        
        # Support both 'dataset_APIS/dataset/train' and direct 'train' if root is correct
        split_path = Path(self.dataset_root) / self.split
        
        if not split_path.exists():
            print(f"WARNING: {split_path} does not exist!")
            return [], filter_stats

        for study_dir in sorted(split_path.iterdir()):
            if not study_dir.is_dir():
                continue
            
            # Filter distinct directories only (Fix for NotADirectoryError)
            slice_dirs = sorted([d for d in study_dir.iterdir() if d.is_dir()])
            num_slices = len(slice_dirs)
            
            for idx, slice_dir in enumerate(slice_dirs):
                # Check files exist
                if not (slice_dir / "image.npz").exists():
                    continue
                if not (slice_dir / "mask.npz").exists():
                    continue
                
                filter_stats['total'] += 1
                
                # Check if slice is empty (optional filtering)
                is_empty = False
                if self.skip_empty_slices:
                    mask_file = slice_dir / "mask.npz"
                    try:
                        mask_data = np.load(mask_file)
                        mask_key = list(mask_data.keys())[0]
                        mask = mask_data[mask_key]
                        is_empty = (mask.sum() == 0)
                        
                        if is_empty:
                            filter_stats['empty'] += 1
                            # Apply negative sampling: keep only X% of empty slices
                            if random.random() > self.negative_sample_ratio:
                                filter_stats['dropped_empty'] += 1
                                continue  # Skip this slice
                        else:
                            filter_stats['non_empty'] += 1
                    except Exception as e:
                        # If error loading mask, include the slice anyway
                        print(f"Warning: Error checking mask for {slice_dir}: {e}")
                
                samples.append({
                    'study': study_dir.name,
                    'slice_idx': idx,
                    'slice_path': slice_dir,
                    'num_slices': num_slices,
                    'all_slices': slice_dirs,
                    'is_empty': is_empty
                })
        
        return samples, filter_stats
    
    def __len__(self):
        return len(self.samples)
    
    def _load_slice(self, slice_path, use_raw_dicom=False):
        """
        Load a single slice
        """
        if use_raw_dicom and (slice_path / "raw.dcm").exists():
            # Load from DICOM for custom HU processing
            dcm = pydicom.dcmread(slice_path / "raw.dcm")
            pixels = dcm.pixel_array.astype(np.float32)
            
            # Convert to HU
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            hu = pixels * slope + intercept
            
            if self.use_hu_window:
                # Apply brain window
                lower = self.window_center - (self.window_width / 2)
                upper = self.window_center + (self.window_width / 2)
                hu = np.clip(hu, lower, upper)
                hu = (hu - lower) / (upper - lower)  # [0, 1]
            else:
                # Simple normalization to roughly [0, 1]
                hu = (hu + 1024) / (3072 + 1024)  # CT range [-1024, 3072]
                hu = np.clip(hu, 0, 1)
            
            image = hu
        else:
            # Load from NPZ (pre-extracted)
            image_npz = np.load(slice_path / "image.npz")
            image = image_npz['image'].astype(np.float32)
            
            # Check if likely in HU range or already normalized
            # If range is big (e.g. -100 to 100), assume HU
            if image.max() > 1.0 or image.min() < 0.0:
                # Need normalization
                if self.use_hu_window:
                    # Assume this is HU data
                    lower = self.window_center - (self.window_width / 2)
                    upper = self.window_center + (self.window_width / 2)
                    image = np.clip(image, lower, upper)
                    image = (image - lower) / (upper - lower)
                else:
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Apply Z-Score Normalization (Global for both DICOM and NPZ)
        if self.config and hasattr(self.config, 'MEAN') and hasattr(self.config, 'STD'):
            mean = self.config.MEAN[0] if isinstance(self.config.MEAN, list) else self.config.MEAN
            std = self.config.STD[0] if isinstance(self.config.STD, list) else self.config.STD
            image = (image - mean) / (std + 1e-8)
        
        # Load mask
        mask_npz = np.load(slice_path / "mask.npz")
        mask = mask_npz['mask'].astype(np.int64)
        
        return image, mask
    
    def _load_metadata(self, slice_path):
        """Load clinical metadata for the slice's study"""
        study_path = slice_path.parent
        meta_path = study_path / "metadata.json"
        
        if meta_path.exists():
            import json
            with open(meta_path, 'r') as f:
                meta = json.load(f)
        else:
            # Check dataset root metadata for global info or defaults
            meta = {}
            
        # Parse into standard format
        parsed_meta = {
            'nihss': self._safe_float(meta.get('nihss', 0)),
            'age': self._safe_float(meta.get('age', 60)),
            'sex': 0 if meta.get('sex', 'M') == 'M' else 1,
            'time': self._parse_time(meta.get('time', '0')),
            'dsa': 1 if meta.get('dsa', False) else 0
        }
        return parsed_meta

    def _safe_float(self, value, default=0.0):
        try:
            return float(value)
        except (ValueError, TypeError):
            return default

    def _parse_time(self, time_str):
        if isinstance(time_str, (int, float)): return float(time_str)
        try:
            if '-' in str(time_str):
                parts = str(time_str).split('-')
                return (float(parts[0]) + float(parts[1])) / 2
            return float(time_str)
        except:
            return 0.0

    def __getitem__(self, idx):
        sample = self.samples[idx]
        center_idx = sample['slice_idx']
        all_slices = sample['all_slices']
        num_slices = sample['num_slices']
        
        # Collect 2T+1 adjacent slices
        images = []
        
        for offset in range(-self.T, self.T + 1):
            slice_idx = center_idx + offset
            # Clamp to valid range
            slice_idx = max(0, min(num_slices - 1, slice_idx))
            
            image, _ = self._load_slice(all_slices[slice_idx])
            images.append(image)
        
        # Load center mask and metadata
        center_slice_path = all_slices[center_idx]
        _, mask = self._load_slice(center_slice_path)
        metadata = self._load_metadata(center_slice_path)
        
        # Stack images: (2T+1, H, W)
        images = np.stack(images, axis=0)
        
        # Convert to tensors
        images = torch.from_numpy(images).float()
        mask = torch.from_numpy(mask).long()
        
        # Apply transforms
        if self.transform:
             pass
        
        return images, mask, metadata
