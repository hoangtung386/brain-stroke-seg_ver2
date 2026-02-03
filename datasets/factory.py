import os
import torch
from torch.utils.data import DataLoader
from .cpaisd import CPAISDDataset
from .brats import BraTSDataset
from .rsna import RSNADataset

def custom_collate_fn(batch):
    """
    Custom collate function to properly batch metadata dicts
    
    Args:
        batch: List of (images, masks, metadata) tuples
    
    Returns:
        images: (B, 2T+1, H, W)
        masks: (B, H, W)
        metadata: Dict[str, Tensor] where each Tensor is (B,)
    """
    # Handle both 2-tuple and 3-tuple formats
    if len(batch[0]) == 3:
        images, masks, metadatas = zip(*batch)
        
        # Stack images and masks
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        
        # Convert list of metadata dicts to dict of tensors
        metadata_dict = {}
        if metadatas and metadatas[0]:  # Check if metadata exists
            keys = metadatas[0].keys()
            for key in keys:
                values = [m[key] for m in metadatas]
                # Handle both numeric and string values
                if isinstance(values[0], str):
                    metadata_dict[key] = values  # Keep as list for strings
                else:
                    metadata_dict[key] = torch.tensor(values)
        
        return images, masks, metadata_dict
    else:
        # Fallback for datasets without metadata
        images, masks = zip(*batch)
        images = torch.stack(images, dim=0)
        masks = torch.stack(masks, dim=0)
        return images, masks

def get_dataset_class(name):
    name = name.lower()
    if name == 'cpaisd' or name == 'apis':
        return CPAISDDataset
    elif name == 'cpaisd_enhanced':
        from .cpaisd_enhanced import EnhancedCPAISDDataset
        return EnhancedCPAISDDataset
    elif name == 'brats': 
        return BraTSDataset
    elif name == 'rsna':
        return RSNADataset
    else:
        raise ValueError(f"Unknown dataset: {name}")

def create_dataloaders(config):
    """
    Create train and val dataloaders based on config
    """
    dataset_name = getattr(config, 'DATASET_NAME', 'cpaisd') 
    
    # Resolve dataset root
    # Priority: Config path -> Local defaults
    dataset_root = None
    
    if hasattr(config, 'DATA_PATHS') and dataset_name in config.DATA_PATHS:
        dataset_root = config.DATA_PATHS[dataset_name]
    elif hasattr(config, 'BASE_PATH'):
        dataset_root = os.path.join(config.BASE_PATH, dataset_name)
    
    # Fallback search for CPAISD (legacy support)
    if dataset_name == 'cpaisd' and (dataset_root is None or not os.path.exists(dataset_root)):
        possible_paths = [
            "dataset_APIS/dataset",
            "dataset",
            "C:/Users/Admin/Projects/brain-stroke-segmentation_ver2/dataset_APIS/dataset"
        ]
        for p in possible_paths:
            if os.path.exists(p) and os.path.isdir(p):
                dataset_root = p
                break
    
    # VALIDATE DATASET PATH
    if dataset_root is None or not os.path.exists(dataset_root):
        raise FileNotFoundError(
            f"❌ Dataset root not found for '{dataset_name}'!\n"
            f"   Expected: {dataset_root}\n"
            f"   Please check configs/config.py DATA_PATHS"
        )
    
    print(f"✓ Dataset root: {dataset_root}")
        # raise FileNotFoundError(f"Dataset root not found for {dataset_name}")

    DatasetClass = get_dataset_class(dataset_name)
    
    # Train Dataset
    # BraTS-specific: Add label_mode
    if dataset_name.lower() == 'brats':
        train_dataset = DatasetClass(
            dataset_root=dataset_root,
            split='train',
            T=config.T,
            transform=None,
            label_mode='native',  # Explicit: Use 4 classes
            config=config
        )
    else:
        train_dataset = DatasetClass(
            dataset_root=dataset_root,
            split='train',
            T=config.T,
            transform=None,
            config=config # Pass config object
        )
    
    # Val Dataset
    # Try different split names
    val_split = 'val'
    if dataset_root and not os.path.exists(os.path.join(dataset_root, 'val')):
        val_split = 'test'
    
    if dataset_name.lower() == 'brats':
        val_dataset = DatasetClass(
            dataset_root=dataset_root,
            split=val_split,
            T=config.T,
            transform=None,
            label_mode='native',  # Explicit: Use 4 classes
            config=config
        )
    else:
        val_dataset = DatasetClass(
            dataset_root=dataset_root,
            split=val_split,
            T=config.T,
            transform=None,
            config=config # Pass config object
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
