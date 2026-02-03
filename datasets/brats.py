from .base import BaseDataset
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
import os
import glob

class BraTSDataset(BaseDataset):
    """
    Dataset loader cho BraTS
    
    Preprocessing Strategy:
    1. Z-Score normalization per volume (brain voxels only)
    2. Robust clipping (percentile-based)
    3. Min-Max scaling to [0, 1]
    4. Modality-specific handling
    
    Label Mapping (BraTS Native):
    - 0: Background
    - 1: Necrotic Core (NCR)
    - 2: Edema
    - 3: Enhancing Tumor (ET)
    
    Grouping strategies:
    - Whole Tumor (WT): Classes 1, 2, 3
    - Tumor Core (TC): Classes 1, 3
    - Enhancing Tumor (ET): Class 3
    """
    
    def __init__(self, dataset_root, split='train', T=1, transform=None,
                 modality='t2f', label_mode='native', use_cache=True, config=None):
        super().__init__(dataset_root, split, T, transform)
        
        self.dataset_root = Path(dataset_root)
        self.modality = modality  # 't2f', 't1c', 't1n', 't2w'
        self.config = config
        self.label_mode = label_mode  # Use the label_mode as passed
            
        self.use_cache = use_cache
        
        # Determine data directory
        if split == 'train':
            if (self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData').exists():
                self.data_dir = self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData'
            else:
                self.data_dir = self.dataset_root
        else:
            if (self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData').exists():
                self.data_dir = self.dataset_root / 'ASNR-MICCAI-BraTS2023-GLI-Challenge-ValidationData'
            else:
                self.data_dir = self.dataset_root
        
        # Cache volumes
        self.volume_cache = {} if use_cache else None
        self.max_cache_size = 3
        
        # Filtering parameters
        self.skip_empty_slices = getattr(config, 'SKIP_EMPTY_SLICES', False) if config else False
        self.negative_sample_ratio = getattr(config, 'NEGATIVE_SAMPLE_RATIO', 0.2) if config else 0.2
        
        # Build sample index (Improved with filtering)
        self.samples, self.filter_stats = self._build_index()
        
        # Load normalization stats from Config OR JSON
        self.norm_stats = self._load_norm_stats()
        
        print(f"\n{split.upper()} Dataset (BraTS - Improved):")
        print(f"  Root: {self.data_dir}")
        print(f"  Modality: {self.modality}")
        print(f"  Label mode: {self.label_mode}")
        if self.skip_empty_slices:
            print(f"  ⚠️  Empty slice filtering: ENABLED")
            print(f"  Total slices (before filter): {self.filter_stats['total']}")
            print(f"  Empty slices: {self.filter_stats['empty']} (dropped: {self.filter_stats['dropped_empty']})")
            print(f"  Non-empty slices: {self.filter_stats['non_empty']}")
            print(f"  Final dataset size: {len(self.samples)}")
        else:
            print(f"  Total slices: {len(self.samples)}")
        print(f"  Adjacent slices (T): {T}")
        print(f"  Caching: {'Enabled' if use_cache else 'Disabled'}")
        if self.norm_stats:
             print(f"  Normalization: Global Stats ({self.modality})")
        else:
             print(f"  Normalization: Per-Volume Z-Score")

    def _load_norm_stats(self):
        """Load normalization stats from Config (priority) or JSON file"""
        # 1. Try Config
        if self.config and hasattr(self.config, 'GLOBAL_STATS'):
            stats = self.config.GLOBAL_STATS.get(self.modality)
            if stats:
                # Rename keys to match expected format if needed, or just use as is
                # Config has: {'mean': ..., 'std': ..., 'min': ..., 'max': ...}
                # Code expects keys like 'global_mean', 'global_std' if using old logic
                # Let's standardize to the Config keys directly in _normalize_volume
                # Or map them here
                
                return {
                    'global_mean': stats['mean'],
                    'global_std': stats['std']
                }
        
        # 2. Try JSON File (Fallback)
        config_path = self.dataset_root / 'brats_normalization_config.json'
        
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            if self.modality in config.get('modalities', {}):
                print(f"  ✅ Loaded normalization stats from JSON for {self.modality}")
                return config['modalities'][self.modality]
        
        print(f"  ⚠️  No normalization stats found, using per-volume Z-score")
        return None
    
    def _build_index(self):
        """Xây dựng index các slices có nội dung với filtering"""
        import nibabel as nib
        import random
        random.seed(42)
        
        samples = []
        filter_stats = {
            'total': 0,
            'empty': 0,
            'non_empty': 0,
            'dropped_empty': 0
        }
        
        if not self.data_dir.exists():
            print(f"WARNING: {self.data_dir} does not exist!")
            return [], filter_stats
        
        studies = sorted([d for d in self.data_dir.iterdir() if d.is_dir()])
        print(f"  Scanning {len(studies)} volumes for valid slices...")
        
        for study_path in studies:
            study_id = study_path.name
            
            # Đường dẫn files
            img_path = study_path / f"{study_id}-{self.modality}.nii.gz"
            seg_path = study_path / f"{study_id}-seg.nii.gz"
            
            # Kiểm tra tồn tại
            if not img_path.exists():
                continue
            
            # Training set PHẢI có label
            if self.split == 'train' and not seg_path.exists():
                continue
            
            try:
                # Đọc header để lấy số slices
                img_proxy = nib.load(img_path)
                num_slices = img_proxy.shape[2]
                
                # Load label volume if filtering is enabled
                seg_data = None
                if self.skip_empty_slices and seg_path.exists():
                    seg_data = nib.load(seg_path).get_fdata().astype(np.int8)
                
                # Bỏ qua slices rìa (thường là background)
                margin = 20
                start_slice = margin
                end_slice = num_slices - margin
                
                if start_slice >= end_slice: # Volume quá mỏng
                    start_slice, end_slice = 0, num_slices
                
                for i in range(start_slice, end_slice):
                    filter_stats['total'] += 1
                    
                    is_empty = False
                    if self.skip_empty_slices and seg_data is not None:
                        # Check slice i content
                        # Assuming Native: 0=BG, others=Tumor
                        # If label_mode changes, 'non-empty' means having meaningful labels
                        has_tumor = np.any(seg_data[..., i] > 0)
                        if not has_tumor:
                            is_empty = True
                            filter_stats['empty'] += 1
                            # Negative sampling
                            if random.random() > self.negative_sample_ratio:
                                filter_stats['dropped_empty'] += 1
                                continue
                        else:
                            filter_stats['non_empty'] += 1
                    
                    samples.append({
                        'study_id': study_id,
                        'slice_idx': i,
                        'img_path': str(img_path),
                        'seg_path': str(seg_path) if seg_path.exists() else None,
                        'num_slices': num_slices
                    })
            except Exception as e:
                print(f"Error reading {study_id}: {e}")
        
        return samples, filter_stats
    
    def _normalize_volume(self, volume):
        """
        Normalization chuẩn cho MRI:
        1. Z-Score trên brain voxels
        2. Robust clipping (percentile)
        3. Scale về [0, 1]
        """
        # Brain mask (non-zero voxels)
        brain_mask = volume > 0
        
        if not brain_mask.any():
            return volume  # Empty volume
        
        brain_voxels = volume[brain_mask]
        
        # 1. Z-Score normalization
        if self.norm_stats:
            # Sử dụng global stats nếu có
            mean = self.norm_stats['global_mean']
            std = self.norm_stats['global_std']
        else:
            # Per-volume Z-score
            mean = brain_voxels.mean()
            std = brain_voxels.std()
        
        volume_norm = (volume - mean) / (std + 1e-8)
        
        # 2. Robust clipping (3-sigma rule)
        volume_norm = np.clip(volume_norm, -3.0, 3.0)
        
        # 3. Scale to [0, 1]
        # Sau khi clip [-3, 3], shift và scale
        volume_norm = (volume_norm + 3.0) / 6.0  # [0, 1]
        
        # Đảm bảo background = 0
        volume_norm[~brain_mask] = 0.0
        
        return volume_norm
    
    def _load_volume_cached(self, path):
        """Load và normalize volume với caching"""
        if self.volume_cache is not None and path in self.volume_cache:
            return self.volume_cache[path]
            
        # Manage cache size
        if self.volume_cache is not None and len(self.volume_cache) >= self.max_cache_size:
            first_key = next(iter(self.volume_cache))
            del self.volume_cache[first_key]
            
        # Load Volume
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        
        # Determine Normalization Mode
        # Only 'brats' config has GLOBAL_STATS, safe check
        # This line assumes 'self.config' exists, which is not defined in the provided snippet.
        # For now, I'll assume it's meant to be 'self.norm_stats' or a similar config object.
        # Given the context of _normalize_volume, it seems self.norm_stats is the intended source for global stats.
        # I will adapt the logic to use self.norm_stats if it's available for global normalization.
        
        # Re-implementing the logic based on the original _normalize_volume and the requested change.
        # The requested change seems to introduce a 'self.config' object which is not present.
        # I will use the existing 'self.norm_stats' for global normalization if available,
        # and fall back to per-volume if not, aligning with the original _normalize_volume's intent.
        
        brain_mask = data > 0
        if not brain_mask.any():
            volume_norm = data # Empty volume, return as is
        else:
            brain_voxels = data[brain_mask]
            
            if self.norm_stats: # Global normalization mode
                mean = self.norm_stats['global_mean']
                std = self.norm_stats['global_std']
                
                # Z-Score
                volume_norm = (data - mean) / (std + 1e-8)
                
                # Robust clipping (3-sigma rule, or configurable if self.config was available)
                clip_min, clip_max = -3.0, 3.0 # Default from original _normalize_volume
                volume_norm = np.clip(volume_norm, clip_min, clip_max)
                
                # Scale to [0, 1]
                volume_norm = (volume_norm - clip_min) / (clip_max - clip_min + 1e-8)
                
            else: # Per-volume Z-score (Fallback or Default)
                mean = brain_voxels.mean()
                std = brain_voxels.std()
                
                # Z-Score
                volume_norm = (data - mean) / (std + 1e-8)
                
                # Robust scaling (Percentile clipping)
                # Clip to [-3, 3] and scale to [0, 1]
                volume_norm = np.clip(volume_norm, -3.0, 3.0)
                volume_norm = (volume_norm - (-3.0)) / (3.0 - (-3.0) + 1e-8)
            
            # Ensure background = 0
            volume_norm[~brain_mask] = 0.0
        
        # Cache
        if self.volume_cache is not None:
            self.volume_cache[path] = volume_norm
        
        return volume_norm
    
    def _map_labels(self, seg_slice):
        """
        Map labels theo mode
        
        native: 0, 1, 2, 3 (giữ nguyên BraTS)
        binary: 0 (background), 1 (tumor - any class)
        wt_tc_et: 0 (bg), 1 (WT), 2 (TC), 3 (ET) - multi-class
        """
        if self.label_mode == 'native':
            return seg_slice
        
        elif self.label_mode == 'binary':
            # Background vs Any Tumor
            return (seg_slice > 0).astype(np.int64)
        
        elif self.label_mode == 'wt_tc_et':
            # 3-class segmentation
            # WT = 1|2|3, TC = 1|3, ET = 3
            # Map: 0->0, 1->2(TC), 2->1(WT), 3->3(ET)
            mapped = np.zeros_like(seg_slice, dtype=np.int64)
            mapped[seg_slice == 2] = 1  # Edema only -> WT
            mapped[seg_slice == 1] = 2  # NCR -> TC
            mapped[seg_slice == 3] = 3  # ET -> ET
            return mapped
            
        elif self.label_mode == 'stroke_compatible':
            # Map to 3 classes (0, 1, 2) for config.NUM_CLASSES = 3
            # 0: Background
            # 1: Core (Necrotic + Non-Enhancing + Enhancing) = BraTS 1 + 3
            # 2: Edema = BraTS 2
            mapped = np.zeros_like(seg_slice, dtype=np.int64)
            mapped[seg_slice == 2] = 2  # Edema -> 2
            mapped[seg_slice == 1] = 1  # NCR -> 1
            mapped[seg_slice == 3] = 1  # ET -> 1 (Merge with Core)
            return mapped
        
        else:
            raise ValueError(f"Unknown label_mode: {self.label_mode}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        slice_idx = sample['slice_idx']
        num_slices = sample['num_slices']
        img_path = sample['img_path']
        seg_path = sample['seg_path']
        
        # 1. Load normalized volume
        volume = self._load_volume_cached(img_path)
        
        # 2. Extract 2T+1 adjacent slices
        images = []
        for offset in range(-self.T, self.T + 1):
            s_idx = max(0, min(num_slices - 1, slice_idx + offset))
            images.append(volume[..., s_idx])
        
        images = np.stack(images, axis=0)  # (2T+1, H, W)
        
        # 3. Load mask (center slice only)
        if seg_path and os.path.exists(seg_path):
            seg_img = nib.load(seg_path)
            mask_slice = seg_img.dataobj[..., slice_idx].astype(np.int64)
            mask_slice = self._map_labels(mask_slice)
        else:
            # Dummy mask for validation/test
            mask_slice = np.zeros(images.shape[1:], dtype=np.int64)
        
        # 4. Convert to tensors
        images = torch.from_numpy(images).float()
        mask = torch.from_numpy(mask_slice).long()
        
        # 5. Metadata (dummy cho BraTS)
        metadata = {
            'study_id': sample['study_id'],
            'slice_index': slice_idx,
            'nihss': 0.0,  # Không có clinical data
            'age': 50.0,
            'dsa': 0,
            'time': 0,
            'sex': 0
        }
        
        return images, mask, metadata
    
    def get_num_classes(self):
        """Trả về số classes theo label_mode"""
        if self.label_mode == 'native':
            return 4
        elif self.label_mode == 'binary':
            return 2
        elif self.label_mode == 'wt_tc_et':
            return 4
        else:
            return 4


# ==============================================================================
# Validation Script
# ==============================================================================

def validate_brats_preprocessing():
    """Script kiểm tra preprocessing"""
    import matplotlib.pyplot as plt
    
    print("="*60)
    print("KIỂM TRA PREPROCESSING BraTS")
    print("="*60)
    
    # Khởi tạo dataset
    dataset = BraTSDataset(
        dataset_root='Dataset_BraTs',
        split='train',
        T=1,
        modality='t2f',
        label_mode='native',
        use_cache=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    # Lấy 1 sample
    images, mask, metadata = dataset[0]
    
    print(f"\nSample shapes:")
    print(f"  Images: {images.shape}")  # (2T+1, H, W)
    print(f"  Mask: {mask.shape}")      # (H, W)
    
    print(f"\nImage statistics:")
    print(f"  Min: {images.min():.4f}")
    print(f"  Max: {images.max():.4f}")
    print(f"  Mean: {images.mean():.4f}")
    print(f"  Std: {images.std():.4f}")
    
    print(f"\nMask classes:")
    unique, counts = torch.unique(mask, return_counts=True)
    for cls, count in zip(unique, counts):
        pct = 100 * count / mask.numel()
        print(f"  Class {cls}: {count} pixels ({pct:.2f}%)")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    center_idx = images.shape[0] // 2
    
    axes[0].imshow(images[center_idx].numpy(), cmap='gray')
    axes[0].set_title('Normalized Image')
    axes[0].axis('off')
    
    axes[1].imshow(mask.numpy(), cmap='tab10', vmin=0, vmax=3)
    axes[1].set_title('Segmentation Mask')
    axes[1].axis('off')
    
    axes[2].imshow(images[center_idx].numpy(), cmap='gray')
    axes[2].imshow(mask.numpy(), cmap='tab10', alpha=0.5, vmin=0, vmax=3)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('brats_preprocessing_check.png', dpi=150)
    print(f"\n✅ Đã lưu visualization: brats_preprocessing_check.png")
    plt.close()
    
    print("\n" + "="*60)
    print("✅ KIỂM TRA HOÀN TẤT")
    print("="*60)


if __name__ == "__main__":
    validate_brats_preprocessing()
    