"""
Optimized to prevent NaN and OOM issues
"""
import os

class Config:
    # Basic settings
    SEED = 42
    
    # Data paths
    KEY = None
    DATASET_NAME = None
    
    DATA_PATHS = {
        'cpaisd': 'dataset_APIS/dataset',
        'brats': 'Dataset_BraTs',
        'rsna': 'datasets/RSNA'
    }
    
    BASE_PATH = './data' 
    IMAGE_DIR = os.path.join(BASE_PATH, 'images')
    MASK_DIR = os.path.join(BASE_PATH, 'masks')
    OUTPUT_DIR = './outputs'
    CHECKPOINT_DIR = './checkpoints'
    
    # Data split
    TRAIN_SPLIT = 0.8
    
    # Model parameters
    NUM_CHANNELS = 1
    NUM_CLASSES = 3  # 0=background, 1=core, 2=penumbra
    INIT_FEATURES = 32
    IMAGE_SIZE = (512, 512)
    
    # HU Windowing (Brain Stroke Optimized)
    USE_HU_WINDOW = True
    # Stroke Window: High contrast for intraparenchymal differentiation
    WINDOW_CENTER = 40 
    WINDOW_WIDTH = 40
    
    # Dataset filtering for class balance
    SKIP_EMPTY_SLICES = True        # Filter out slices with no stroke annotation
    NEGATIVE_SAMPLE_RATIO = 0.05    # Reducing from 0.2 to 0.05 to drastically reduce background dominance
    
    # Loss Weights
    FP_PENALTY_WEIGHT = 0.3         # New: Penalty for predicting stroke in background regions
    
    # Batch size (optimized for RTX 3090 24GB VRAM)
    BATCH_SIZE = 5  # Increased from 5 (CPAISD) , 20 (BraTS) for better GPU utilization
    NUM_EPOCHS = 150
    LEARNING_RATE = 1e-4  # Increased from 1e-5 for better convergence
    
    # DataLoader parameters (optimized for multi-core CPU)
    NUM_WORKERS = 8  # Increased from 4 to maximize I/O throughput
    CACHE_RATE = 0
    PIN_MEMORY = True
    PERSISTENT_WORKERS = True
    
    # Model architecture
    T = 1                       # Number of adjacent slices
    NUM_PARTITIONS_H = 4
    NUM_PARTITIONS_W = 4
    GLOBAL_IMPACT = 0.3
    LOCAL_IMPACT = 0.7
    
    # Transformer Parameters
    TRANSFORMER_NUM_HEADS = 4
    TRANSFORMER_NUM_LAYERS = 2
    TRANSFORMER_EMBED_DIM = 1024  # Should match bottleneck channels
    
    # Normalization (Optimized for Stroke Window WC=40, WW=40)
    MEAN = [0.6234]    # Computed from 500 samples
    STD = [0.3369]     # Computed from 500 samples
    
    WEIGHT_DECAY = 1e-4
    
    # Training stability
    GRAD_CLIP_NORM = 1.0
    USE_AMP = False
    DEBUG_MODE = False
    DETECT_ANOMALY = False
    
    # ⭐ SOTA Components
    USE_MAMBA = True        # Use Mamba-2 bottleneck
    USE_KAN = True          # Use Efficient-KAN decoder heads
    USE_CONDITIONING = True # Use Clinical Conditioning
    
    # Component-specific settings
    MAMBA_DEPTH = 4
    KAN_DEGREE = 3
    
    # SymFormer specific
    KMAX_NUM_HEADS = 8
    KMAX_NUM_LAYERS = 2
    SYMMETRY_WEIGHT = 0.05
    CLUSTER_WEIGHT = 0.1
    
    # Loss weights
    DICE_WEIGHT = 0.7
    CE_WEIGHT = 0.3
    FOCAL_WEIGHT = 1.0
    ALIGNMENT_WEIGHT = 0.05
    PERCEPTUAL_WEIGHT = 0.1   
    
    # W&B settings
    USE_WANDB = True
    WANDB_PROJECT = "OmniSym-dataset-"
    WANDB_ENTITY = None
    WANDB_MODE = "online"
    
    # Scheduler parameters
    SCHEDULER_T0 = 10
    SCHEDULER_T_MULT = 2
    SCHEDULER_ETA_MIN = 1e-6
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 30
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary"""
        return {
            'seed': cls.SEED,
            'train_split': cls.TRAIN_SPLIT,
            'batch_size': cls.BATCH_SIZE,
            'num_epochs': cls.NUM_EPOCHS,
            'learning_rate': cls.LEARNING_RATE,
            'image_size': cls.IMAGE_SIZE,
            'init_features': cls.INIT_FEATURES,
            'num_channels': cls.NUM_CHANNELS,
            'num_classes': cls.NUM_CLASSES,
            'T': cls.T,
            'global_impact': cls.GLOBAL_IMPACT,
            'local_impact': cls.LOCAL_IMPACT,
            'dice_weight': cls.DICE_WEIGHT,
            'ce_weight': cls.CE_WEIGHT,
            'alignment_weight': cls.ALIGNMENT_WEIGHT,
            'grad_clip_norm': cls.GRAD_CLIP_NORM,
            'use_amp': cls.USE_AMP,
            'use_mamba': cls.USE_MAMBA,
            'use_kan': cls.USE_KAN,
            'use_conditioning': cls.USE_CONDITIONING,
        }
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)
        os.makedirs(cls.CHECKPOINT_DIR, exist_ok=True)
        print(f"Directories created: {cls.OUTPUT_DIR}, {cls.CHECKPOINT_DIR}")
    
    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "="*60)
        print("CURRENT CONFIGURATION")
        print("="*60)
        print(f"Batch Size:        {cls.BATCH_SIZE}")
        print(f"Learning Rate:     {cls.LEARNING_RATE}")
        print(f"Epochs:            {cls.NUM_EPOCHS}")
        print(f"Image Size:        {cls.IMAGE_SIZE}")
        print(f"Gradient Clip:     {cls.GRAD_CLIP_NORM}")
        print(f"Alignment Weight:  {cls.ALIGNMENT_WEIGHT}")
        print(f"Use AMP:           {cls.USE_AMP}")
        print(f"Debug Mode:        {cls.DEBUG_MODE}")
        print(f"SOTA Components:   Mamba={cls.USE_MAMBA}, KAN={cls.USE_KAN}, Cond={cls.USE_CONDITIONING}")
        print("="*60 + "\n")



# ============================================================================
# DATASET-SPECIFIC CONFIGURATIONS
# ============================================================================

class CPAISDConfig(Config):
    """Configuration for Stroke (CT)"""
    DATASET_NAME = 'cpaisd'
    NUM_CLASSES = 3      # 0=bg, 1=core, 2=penumbra
    IMAGE_SIZE = (512, 512) # CT size
    USE_HU_WINDOW = True # CT specific
    
    # Weights optimized for Stroke
    DICE_WEIGHT = 0.7
    CE_WEIGHT = 0.3
    
class BraTSConfig(Config):
    """Configuration for Brain Tumor (MRI)"""
    DATASET_NAME = 'brats'
    NUM_CLASSES = 4      
    IMAGE_SIZE = (240, 240) # BraTS Native Resolution
    
    USE_HU_WINDOW = False # MRI does not use HU
    
    # Normalization Strategy
    # Options: 'global' (uses dataset-wide stats) or 'per_volume' (standard)
    NORMALIZATION_MODE = 'global' 
    
    # Global Statistics from brats_normalization_config.json
    GLOBAL_STATS = {
        't2f': {'mean': 941.0089, 'std': 351.0287, 'min': 0.0, 'max': 4050.0},
        't1c': {'mean': 2099.6694, 'std': 760.0310, 'min': 0.0, 'max': 20289.6},
        't1n': {'mean': 803.4387, 'std': 174.5820, 'min': 0.0, 'max': 4191.0},
        't2w': {'mean': 673.8523, 'std': 322.7553, 'min': 0.0, 'max': 3841.8}
    }
    
    # Clipping and Scaling
    CLIP_RANGE = [-3.0, 3.0]
    TARGET_RANGE = [0.0, 1.0]

    # MRI usually needs normalization per volume (handled in loader)
    MEAN = [0.0] # Not used if loader does internal norm
    STD = [1.0]
    
    # Class weights might differ
    # Core is often smaller than Edema
    DICE_WEIGHT = 0.5
    FOCAL_WEIGHT = 0.5 

class RSNAConfig(Config):
    """Configuration for Abdominal Trauma (CT)"""
    DATASET_NAME = 'rsna'
    NUM_CLASSES = 2 # Placeholder: 0=bg, 1=injury
    USE_HU_WINDOW = True # CT
    
class CPAISDEnhancedConfig(CPAISDConfig):
    """
    Configuration for Enhanced Stroke Segmentation (3-Channel Strategy)
    Channel 1: Original Stroke Window (Anchor)
    Channel 2: Context Window
    Channel 3: Enhanced (Detail Boost)
    """
    DATASET_NAME = 'cpaisd_enhanced'
    NUM_CHANNELS = 3 # Multi-channel input
    MULTICHANNEL = True
    
    # Adjust weights if needed for multi-channel
    # Maybe slightly higher alignment weight?
    ALIGNMENT_WEIGHT = 0.05

def get_config(dataset_name):
    """Factory to get config by name"""
    if dataset_name == 'cpaisd':
        return CPAISDConfig
    elif dataset_name == 'cpaisd_enhanced':
        return CPAISDEnhancedConfig
    elif dataset_name == 'brats':
        return BraTSConfig
    elif dataset_name == 'rsna':
        return RSNAConfig
    else:
        return Config # Default

