import os
from torch.utils.data import DataLoader
from .cpaisd import CPAISDDataset
from .brats import BraTSDataset
from .rsna import RSNADataset

def get_dataset_class(name):
    name = name.lower()
    if name == 'cpaisd' or name == 'apis':
        return CPAISDDataset
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
