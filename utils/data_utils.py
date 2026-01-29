import torch
from tqdm import tqdm
import numpy as np

def compute_class_weights(dataset, num_classes=2, num_samples=None):
    """
    Compute inverse frequency weights for class imbalance handling.
    
    Args:
        dataset: PyTorch dataset
        num_classes: Number of classes
        num_samples: Number of samples to use for calculation (None for all)
        
    Returns:
        torch.Tensor: Class weights of shape (num_classes,)
    """
    print(f"Computing class weights from dataset ({'all' if num_samples is None else num_samples} samples)...")
    
    # Initialize counts
    pixel_counts = torch.zeros(num_classes, dtype=torch.long)
    
    # Determine samples to iterate
    if num_samples is None or num_samples >= len(dataset):
        indices = range(len(dataset))
    else:
        # Randomly sample indices
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    for idx in tqdm(indices, desc="Scanning dataset for class weights"):
        try:
            _, mask, _ = dataset[idx]
            
            # Mask should be (H, W) or (1, H, W) with integer class labels
            if mask.ndim == 3:
                mask = mask.squeeze(0)
                
            # Count pixels per class
            unique, counts = torch.unique(mask, return_counts=True)
            
            for u, c in zip(unique, counts):
                if u < num_classes:
                    pixel_counts[u.long()] += c.long()
                    
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            continue
            
    # Avoid division by zero
    pixel_counts = pixel_counts + 1e-8
    
    # Calculate weights: Total / (num_classes * count_c)
    # This balances the contribution of each class
    total_pixels = pixel_counts.sum()
    weights = total_pixels / (num_classes * pixel_counts)
    
    # --- CRITICAL FIX FOR STROKE OVERPREDICTION ---
    # Boost minority classes significantly to penalize missing them
    # But relies on FP penalty in loss to stop overprediction
    if num_classes >= 3:
        print("  ⚡ Boosting weights for Stroke classes (Core: 10x, Penumbra: 8x)")
        weights[1] *= 10.0  # Core
        weights[2] *= 8.0   # Penumbra
    # ----------------------------------------------
    
    # Normalize weights so they sum to num_classes (optional but good for stability)
    weights = weights / weights.mean()
    
    print(f"Class pixel counts: {pixel_counts.tolist()}")
    print(f"Computed Class Weights: {weights.tolist()}")
    
    return weights.float()
