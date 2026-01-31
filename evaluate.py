#!/usr/bin/env python3
"""
Evaluation script for Brain Stroke Segmentation
Generates metrics and visualizations using the trained model
"""
import os
import sys
import torch
import argparse
from pathlib import Path

from configs.config import get_config
from datasets import create_dataloaders
from datasets.cpaisd_enhanced import create_enhanced_dataloaders
from models.symformer import SymFormer


def main():
    parser = argparse.ArgumentParser(description='Evaluate Brain Stroke Segmentation Model')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='./output', help='Directory to save results')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size for evaluation')
    parser.add_argument('--num-samples', type=int, default=-1, help='Number of visualization samples (-1 for all validation samples)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--dataset', type=str, default='cpaisd', help='Dataset name (cpaisd, cpaisd_enhanced, brats, rsna)')
    
    args = parser.parse_args()
    
    # Load appropriate config for dataset
    ConfigClass = get_config(args.dataset)
    
    print(f"Evaluation Settings:")
    print(f"  Dataset: {ConfigClass.DATASET_NAME}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Output Dir: {args.output_dir}")
    print(f"  Device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Create dataloaders with enhanced support
    print("\nLoading dataset...")
    
    # Override batch size for evaluation
    original_batch_size = ConfigClass.BATCH_SIZE
    ConfigClass.BATCH_SIZE = args.batch_size
    
    # Check if using enhanced dataset
    if args.dataset == 'cpaisd_enhanced':
        print("  Using EnhancedCPAISDDataset (3-channel)...")
        _, val_loader = create_enhanced_dataloaders(ConfigClass, use_enhancement=True)
    else:
        _, val_loader = create_dataloaders(ConfigClass)
    
    print(f"Validation set size: {len(val_loader.dataset)}")
    
    # Initialize model with correct number of input channels
    print("\nInitializing model...")
    in_channels = getattr(ConfigClass, 'NUM_CHANNELS', 1)
    print(f"  Input channels: {in_channels}")
    
    model = SymFormer(
        in_channels=in_channels,
        num_classes=ConfigClass.NUM_CLASSES,
        T=ConfigClass.T,
        input_size=ConfigClass.IMAGE_SIZE
    )
    
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return 1
        
    print(f"Loading checkpoint from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle both full checkpoint dict and just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'best_val_dice' in checkpoint:
            print(f"Best validation Dice: {checkpoint['best_val_dice']:.4f}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dictionary")
    
    # Initialize evaluator
    from evaluation.evaluator import Evaluator
    evaluator = Evaluator(
        model=model,
        val_loader=val_loader,
        device=device,
        config=ConfigClass,
        num_samples=args.num_samples,  # Pass num_samples parameter
        output_dir=args.output_dir
    )
    
    # Run evaluation
    evaluator.run()
    
    print(f"\nEvaluation complete! Results saved to evaluation_results/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
