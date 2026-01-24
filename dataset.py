"""
CPAISD Dataset Loader - Proper preprocessing for Stroke Segmentation
Based on CPAISD paper analysis.
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pydicom
from pathlib import Path
from torchvision import transforms

class CPAISDDataset(Dataset):
    """
    Dataset loader cho CPAISD
    
    Theo paper (Section 3.2):
    - NPZ files đã được chuẩn bị sẵn
    - Mask có 3 classes: 0=background, 1=core, 2=penumbra
    - Images cần chuẩn hóa thêm
    """
    
    def __init__(self, dataset_root, split='train', T=1, 
                 use_hu_window=True, transform=None):
        """
        Args:
            dataset_root: Path to dataset folder
            split: 'train', 'val', or 'test'
            T: Number of adjacent slices
            use_hu_window: Apply brain window to HU values
            transform: Additional transforms
        """
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.T = T
        self.use_hu_window = use_hu_window
        self.transform = transform
        
        # Brain window parameters (standard for stroke CT)
        self.window_center = 40  # HU
        self.window_width = 80   # HU
        
        # Build sample index
        self.samples = self._build_index()
        
        print(f"\n{split.upper()} Dataset:")
        print(f"  Root: {self.dataset_root}")
        print(f"  Total slices: {len(self.samples)}")
        print(f"  HU windowing: {use_hu_window}")
        print(f"  Adjacent slices (T): {T}")
    
    def _build_index(self):
        """Build index of all slices"""
        samples = []
        split_path = self.dataset_root / self.split
        
        if not split_path.exists():
            print(f"WARNING: {split_path} does not exist!")
            return []

        for study_dir in sorted(split_path.iterdir()):
            if not study_dir.is_dir():
                continue
            
            slice_dirs = sorted(study_dir.iterdir())
            num_slices = len(slice_dirs)
            
            for idx, slice_dir in enumerate(slice_dirs):
                # Check files exist
                if not (slice_dir / "image.npz").exists():
                    continue
                if not (slice_dir / "mask.npz").exists():
                    continue
                
                samples.append({
                    'study': study_dir.name,
                    'slice_idx': idx,
                    'slice_path': slice_dir,
                    'num_slices': num_slices,
                    'all_slices': slice_dirs
                })
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def _load_slice(self, slice_path, use_raw_dicom=False):
        """
        Load a single slice
        
        Options:
        1. Use NPZ (faster, already extracted)
        2. Use raw DICOM (slower, but can apply custom HU window)
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
                    # Simple min-max
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Load mask
        mask_npz = np.load(slice_path / "mask.npz")
        mask = mask_npz['mask'].astype(np.int64)
        
        return image, mask
    
    def _load_metadata(self, slice_path):
        """Load clinical metadata for the slice's study"""
        # Metadata is likely at study level, but we check slice folder first or parent
        # Based on check_data.py, there is a metadata.json at dataset root, 
        # but patient specific metadata might be in study folder.
        # Assuming minimal metadata is available or using defaults.
        
        # NOTE: Real implementation should load from patient-specific JSON.
        # For now, we simulate or look for it.
        
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


def create_dataloaders(config):
    """
    Create train and test dataloaders for CPAISD
    """
    # Assuming dataset is in 'dataset_APIS/dataset' relative to project root
    # OR config.BASE_PATH defines it.
    # The exploration script found it at 'dataset_APIS/dataset'
    
    # Try to resolve dataset path
    possible_paths = [
        # os.path.join(config.BASE_PATH, 'dataset'),  # Default from config
        "dataset_APIS/dataset",
        "dataset",
        "C:/Users/Admin/Projects/brain-stroke-segmentation_ver2/dataset_APIS/dataset"
    ]
    
    dataset_root = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isdir(p):
            dataset_root = p
            break
            
    if dataset_root is None:
        raise FileNotFoundError(f"Could not find dataset directory. Checked: {possible_paths}")

    train_dataset = CPAISDDataset(
        dataset_root=dataset_root,
        split='train',
        T=config.T,
        use_hu_window=config.USE_HU_WINDOW
    )
    
    # Use 'test' or 'val' for validation/testing
    # If explicit 'val' folder exists, use it.
    val_split = 'val' if (Path(dataset_root) / 'val').exists() else 'test'
    
    val_dataset = CPAISDDataset(
        dataset_root=dataset_root,
        split=val_split,
        T=config.T,
        use_hu_window=config.USE_HU_WINDOW
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader
