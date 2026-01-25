"""
Training script for SymFormer
Replace old LCNN with this new architecture
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from models.losses import SymFormerLoss
import argparse
import os
import wandb
import csv


# ============================================================================
# INTEGRATION with Existing Code
# ============================================================================

class SymFormerTrainer:
    """
    Drop-in replacement for existing Trainer
    Just change model initialization
    """
    def __init__(self, model, train_loader, val_loader, config, device, multi_gpu=False):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.multi_gpu = multi_gpu
        
        # Compute class weights (from existing utils)
        # Note: If running on DDP, this should be done only on main process or cached
        # For DataParallel, it is fine.
        from utils.data_utils import compute_class_weights
        print("Computing/Loading class weights...")
        class_weights = compute_class_weights(
            train_loader.dataset,
            num_classes=config.NUM_CLASSES,
            num_samples=500
        ).to(device)
        
        # SymFormer Loss
        self.criterion = SymFormerLoss(
            num_classes=config.NUM_CLASSES,
            class_weights=class_weights
        )
        
        # Optimizer (unchanged)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=1e-4
        )
        
        # Scheduler (unchanged)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Metrics
        from monai.metrics import DiceMetric
        self.dice_metric = DiceMetric(include_background=False, reduction='mean')
        
        self.best_dice = 0.0
        self.history = []
        self.loss_accumulators = {}
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_loss = 0
        # total_seg/cluster/sym are now handled dynamically via self.loss_accumulators
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        for batch in pbar:
            # Unpack batch (now includes metadata)
            if len(batch) == 3:
                images, masks, metadata = batch
            else:
                images, masks = batch
                metadata = None

            images = images.to(self.device)
            masks = masks.to(self.device)
            
            # Convert metadata dict to tensors if present
            if metadata:
                 # Ensure metadata is properly formatted for batching if it's a list of dicts
                 # DataLoader usually returns a dict of lists/tensors if collate is default
                 pass
            
            self.optimizer.zero_grad()
            
            # Forward pass
            output, cluster_outputs, asymmetry_map = self.model(
                images, metadata=metadata, return_alignment=False
            )
            
            # Compute loss
            loss, loss_dict = self.criterion(
                output, masks, cluster_outputs, asymmetry_map
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping (IMPORTANT!)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.GRAD_CLIP_NORM
            )
            
            self.optimizer.step()
            
            # Accumulate metrics
            total_loss += loss.item()
            
            # Dynamic accumulation of all loss components
            for k, v in loss_dict.items():
                if k not in self.loss_accumulators:
                    self.loss_accumulators[k] = 0.0
                self.loss_accumulators[k] += v if isinstance(v, (int, float)) else v.item()
            
            # Update pbar
            postfix = {'loss': f'{loss.item():.4f}'}
            # Add other major losses to pbar if they exist
            if 'main' in loss_dict:
                postfix['main'] = f"{loss_dict['main']:.4f}"
            pbar.set_postfix(postfix)
            
            # W&B Logging (Iterative)
            if self.config.USE_WANDB:
                log_dict = {'batch/train_loss': loss.item(), 'batch/grad_norm': grad_norm, 'epoch': epoch}
                # Add all loss components
                for k, v in loss_dict.items():
                    log_dict[f'batch/{k}'] = v if isinstance(v, (int, float)) else v.item()
                wandb.log(log_dict)
        
        # Calculate averages
        avg_loss = total_loss / len(self.train_loader)
        avg_metrics = {k: v / len(self.train_loader) for k, v in self.loss_accumulators.items()}
        
        return avg_loss, avg_metrics
    
    def validate(self, epoch):
        """Validate"""
        self.model.eval()
        self.dice_metric.reset()
        
        total_val_loss = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            for batch in pbar:
                if len(batch) == 3:
                    images, masks, metadata = batch
                else:
                    images, masks = batch
                    metadata = None
                    
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                # Forward
                output, cluster_outputs, asymmetry_map = self.model(
                    images, metadata=metadata, return_alignment=False
                )
                
                # Loss
                loss, _ = self.criterion(
                    output, masks, cluster_outputs, asymmetry_map
                )
                
                total_val_loss += loss.item()
                
                # Dice metric
                if masks.ndim == 3:
                    masks_metric = masks.unsqueeze(1)
                else:
                    masks_metric = masks
                
                self.dice_metric(y_pred=output, y=masks_metric)
                
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
        
        # Average metrics
        val_dice = self.dice_metric.aggregate().item()
        val_loss = total_val_loss / len(self.val_loader)
        
        print(f"\nValidation: Loss={val_loss:.4f}, Dice={val_dice:.4f}")
        
        return val_dice, val_loss
    
    def train(self, num_epochs):
        """Main training loop"""
        print(f"\nStarting SymFormer Training")
        print(f"Epochs: {num_epochs}")
        print(f"Device: {self.device}")
        print("="*60)
        
        for epoch in range(1, num_epochs + 1):
            # Reset accumulators for each epoch
            self.loss_accumulators = {}
            
            # Train
            train_loss, train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_dice, val_loss = self.validate(epoch)
            
            # Scheduler step
            self.scheduler.step()
            
            # Save best model
            if val_dice > self.best_dice:
                self.best_dice = val_dice
                
                # Handle DataParallel saving
                if self.multi_gpu:
                     model_state = self.model.module.state_dict()
                else:
                     model_state = self.model.state_dict()

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice
                }, 'checkpoints/symformer_best.pth')
                print(f"✓ Best model saved! Dice: {val_dice:.4f}")
            
            # Log
            history_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_dice': val_dice
            }
            # Add dynamic metrics to history
            history_dict.update(train_metrics)
            self.history.append(history_dict)
            
            # CSV Logging
            log_file = os.path.join(self.config.OUTPUT_DIR, f"training_{self.config.DATASET_NAME}_log.csv")
            file_exists = os.path.isfile(log_file)
            
            # If resetting at epoch 1 (optional, depends if we want to append or overwrite)
            # Assuming we want to overwrite if starting from epoch 1, or append if resuming
            # For simplicity, if epoch == 1, we overwrite
            mode = 'w' if epoch == 1 else 'a'
            
            with open(log_file, mode, newline='') as f:
                writer = csv.DictWriter(f, fieldnames=history_dict.keys())
                if epoch == 1:
                    writer.writeheader()
                    print(f"Created log file: {log_file}")
                
                # Verify header match for subsequent appends (if keys change, DictWriter might error or ignore)
                # But for now assuming consistent keys
                writer.writerow(history_dict)

            
            print(f"Epoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            # Print dynamic metrics
            for k, v in train_metrics.items():
                print(f"    - {k}: {v:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  Best Dice: {self.best_dice:.4f}")
            print("="*60)
            
            # W&B Logging (Epoch)
            if self.config.USE_WANDB:
                log_dict = {
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'val/loss': val_loss,
                    'val/dice': val_dice,
                    'val/best_dice': self.best_dice,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                }
                # Add dynamic train metrics
                for k, v in train_metrics.items():
                    log_dict[f'train/{k}'] = v
                    
                wandb.log(log_dict)
        
        print(f"\nTraining Complete! Best Dice: {self.best_dice:.4f}")


# ============================================================================
# MAIN SCRIPT - Replace train.py
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train SymFormer")
    parser.add_argument('--devices', type=str, default=None, 
                        help='Comma-separated GPU IDs (e.g. "0" or "0,1"). Default: CPU')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (cpaisd, brats, rsna)')
    return parser.parse_args()

def main():
    """
    Usage:
    python train.py                     # CPU
    python train.py --devices 0         # GPU 0
    python train.py --devices 1         # GPU 1
    python train.py --devices 0,1       # Multi-GPU
    """
    args = parse_args()
    
    # Handle Device Selection
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    else:
        # Default to CPU if no devices specified
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    
    from configs.config import Config, get_config
    
    # 1. Determine Dataset Name first (default to cpaisd if not specified)
    dataset_name = args.dataset if args.dataset else 'cpaisd'
    
    # 2. Get Specific Configuration
    Config = get_config(dataset_name)
    print(f"Loaded Configuration for: {Config.DATASET_NAME}")
    
    from datasets import create_dataloaders
    
    # Setup directories
    Config.create_directories()
    
    # Override dataset from args and update WANDB Project
    if args.dataset:
        Config.DATASET_NAME = args.dataset
        # Dynamically update WANDB project name
        Config.WANDB_PROJECT = f"{Config.WANDB_PROJECT}{args.dataset}"
        print(f"Overriding dataset to: {args.dataset}")
        print(f"W&B Project set to: {Config.WANDB_PROJECT}")
    
    # Initialize W&B
    if Config.USE_WANDB:
        wandb.init(
            project=Config.WANDB_PROJECT,
            config=Config.to_dict(),
            name=f"SymFormerV2_{Config.DATASET_NAME}_{'cpu' if not args.devices else 'gpu'+args.devices}",
            mode=Config.WANDB_MODE
        )
    
    # Configure Devices (Support Multi-GPU)
    # Check if we have visible devices consistent with args
    if torch.cuda.is_available() and args.devices is not None:
        
        # NOTE: After setting CUDA_VISIBLE_DEVICES, the visible GPUs are re-indexed to 0, 1, ...
        # So 'device_ids' for DataParallel should always start from 0 up to count.
        
        device_count = torch.cuda.device_count()
        device_ids = list(range(device_count)) # [0, 1, ...] relative to visible devices
        device = torch.device('cuda:0')
        multi_gpu = device_count > 1
        
        print(f"\nCUDA Enabled.")
        print(f"Requested GPUs: {args.devices}")
        print(f"Visible Device Count: {device_count}")
        print(f"Using Devices: {device_ids}")
    else:
        device = torch.device('cpu')
        device_ids = []
        multi_gpu = False
        print("\nUsing CPU (No devices specified or CUDA unavailable).")
    
    # Data
    train_loader, val_loader = create_dataloaders(Config)
    
    # Model (NEW!)
    from models.symformer import SymFormer
    model = SymFormer(
        in_channels=Config.NUM_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        T=Config.T,
        input_size=Config.IMAGE_SIZE
    )
    
    print(f"\nSymFormer Architecture:")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    
    # Apply DataParallel
    if multi_gpu:
        print(f"  Enabling DataParallel on devices: {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    
    model = model.to(device)
    
    # Trainer
    trainer = SymFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=Config,
        device=device,
        multi_gpu=multi_gpu
    )
    
    # Train
    trainer.train(num_epochs=Config.NUM_EPOCHS)


if __name__ == "__main__":
    main()
