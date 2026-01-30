
from .cpaisd import CPAISDDataset
from preprocessing.enhancement import EnhancementPipeline
import numpy as np
import torch
import pydicom

class EnhancedCPAISDDataset(CPAISDDataset):
    """
    Enhanced version of CPAISD Dataset with advanced preprocessing pipeline
    specifically tuned for ischemic stroke detection (ICIP 2015).
    """
    def __init__(self, dataset_root, split='train', T=1, 
                 use_enhancement=True, transform=None, config=None):
        super().__init__(dataset_root, split, T, use_hu_window=True, transform=transform, config=config)
        self.use_enhancement = use_enhancement
        
        # Initialize enhancement pipeline
        # Tamed parameters as per user feedback:
        self.enhancer = EnhancementPipeline(
            stroke_center=40,    # Standard stroke center
            stroke_width=40,     # Standard stroke width
            clahe_clip_limit=1.5, # Reduced from 2.0
            boost_strength=0.5,   # Reduced from 1.5
            alpha=0.3             # Blend: 30% Enhanced, 70% Base
        )
        
        self.multichannel = getattr(config, 'MULTICHANNEL', False)
        
        print(f"  ✨ Enhanced Preprocessing: {'ENABLED' if use_enhancement else 'DISABLED'}")
        if use_enhancement:
            print(f"     - Window: {self.enhancer.stroke_center} +/- {self.enhancer.stroke_width/2}")
            print(f"     - CLAHE: Clip={self.enhancer.clahe_clip_limit}")
            print(f"     - Detail Boost: {self.enhancer.boost_strength}")
            print(f"     - Alpha Blending: {self.enhancer.alpha*100}%")
            if self.multichannel:
                print("     - Output: 3-Channel Strategy (Stroke, Context, Enhanced)")
            else:
                print("     - Output: Single Channel (Blended)")

    def _load_slice(self, slice_path, use_raw_dicom=False):
        """
        Override to apply enhancement pipeline
        """
        # 1. Load Raw Data (DICOM or NPZ)
        if use_raw_dicom and (slice_path / "raw.dcm").exists():
            dcm = pydicom.dcmread(slice_path / "raw.dcm")
            pixels = dcm.pixel_array.astype(np.float32)
            intercept = float(dcm.RescaleIntercept)
            slope = float(dcm.RescaleSlope)
            image_hu = pixels * slope + intercept
        else:
            # Load from NPZ
            image_npz = np.load(slice_path / "image.npz")
            image_raw = image_npz['image'].astype(np.float32)
            
            # Heuristic to check if already normalized or HU
            if image_raw.max() <= 1.0 and image_raw.min() >= 0.0:
                # Approx reverse to HU: 40 + (val - 0.5) * 80
                image_hu = 40 + (image_raw - 0.5) * 80
            else:
                image_hu = image_raw

        # 2. Apply Enhancement or Standard Windowing
        if self.use_enhancement:
            if self.multichannel:
                # Returns (3, H, W) where channels are (Stroke, Context, Enhanced)
                # Note: This changes the output dimension from (1, H, W) to (3, H, W)
                image = self.enhancer.get_3_channels(image_hu)
                # Shape: (3, H, W)
                # When stacked for T slices, it will be (T, 3, H, W) -> flattened to (3*T, H, W)
            else:
                # Blended Single Channel
                image = self.enhancer.process_blended(image_hu)
                # Shape: (H, W) or (1, H, W) depending on impl. Usually just (H, W)
        else:
            # Fallback to standard windowing
            center, width = 40, 80
            lower = center - (width / 2)
            upper = center + (width / 2)
            image = np.clip(image_hu, lower, upper)
            image = (image - lower) / (upper - lower)

        # 3. Normalization (Z-Score)
        if self.config and hasattr(self.config, 'MEAN') and hasattr(self.config, 'STD'):
             mean = self.config.MEAN[0] if isinstance(self.config.MEAN, list) else self.config.MEAN
             std = self.config.STD[0] if isinstance(self.config.STD, list) else self.config.STD
             image = (image - mean) / (std + 1e-8)

        # 4. Load Mask
        mask_npz = np.load(slice_path / "mask.npz")
        mask = mask_npz['mask'].astype(np.int64)

        return image, mask

    def __getitem__(self, idx):
        # Override getitem to handle multi-channel stacking if needed
        # Or rely on parent if shape is consistent
        # If multichannel, parent __getitem__ stacks images (2T+1, H, W)
        # But here image is (3, H, W). Stack will be (2T+1, 3, H, W).
        # We need to reshape to ( (2T+1)*3, H, W ) for the model.
        
        images, mask, metadata = super().__getitem__(idx)
        
        if self.multichannel and self.use_enhancement:
            # images shape: (2T+1, 3, H, W) (from numpy stack of (3, H, W))
            # Reshape to (C_total, H, W)
            # C_total = (2T+1) * 3
            if isinstance(images, torch.Tensor):
                B, C, H, W = images.shape
                images = images.view(B*C, H, W)
            elif isinstance(images, np.ndarray):
                B, C, H, W = images.shape
                images = images.reshape(B*C, H, W)
                
        return images, mask, metadata

def create_enhanced_dataloaders(config, use_enhancement=True):
    """
    Helper to create dataloaders with enhancement enabled
    """
    from torch.utils.data import DataLoader
    from .factory import custom_collate_fn
    
    # Same logic as factory.create_dataloaders but forces EnhancedCPAISDDataset
    dataset_root = None
    if hasattr(config, 'DATA_PATHS') and 'cpaisd' in config.DATA_PATHS:
        dataset_root = config.DATA_PATHS['cpaisd']
    elif hasattr(config, 'BASE_PATH'):
        dataset_root = os.path.join(config.BASE_PATH, 'cpaisd')
    
    # Fallback
    if dataset_root is None or not os.path.exists(dataset_root):
         dataset_root = "dataset_APIS/dataset" # Default relative path

    train_dataset = EnhancedCPAISDDataset(
        dataset_root=dataset_root,
        split='train',
        T=config.T,
        use_enhancement=use_enhancement,
        transform=None,
        config=config
    )
    
    val_dataset = EnhancedCPAISDDataset(
        dataset_root=dataset_root,
        split='val' if os.path.exists(os.path.join(dataset_root, 'val')) else 'test',
        T=config.T,
        use_enhancement=use_enhancement, 
        transform=None,
        config=config
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, val_loader
