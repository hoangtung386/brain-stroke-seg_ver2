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
        
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        total_seg = 0
        total_cluster = 0
        total_sym = 0
        
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
            total_seg += loss_dict['seg_loss']
            total_cluster += loss_dict['cluster_loss']
            total_sym += loss_dict['symmetry_loss']
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'grad': f'{grad_norm:.2f}'
            })
            
            # W&B Logging (Iterative)
            if self.config.USE_WANDB:
                wandb.log({
                    'batch/train_loss': loss.item(),
                    'batch/seg_loss': loss_dict['seg_loss'],
                    'batch/cluster_loss': loss_dict['cluster_loss'],
                    'batch/symmetry_loss': loss_dict['symmetry_loss'],
                    'batch/grad_norm': grad_norm,
                    'epoch': epoch
                })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_seg = total_seg / len(self.train_loader)
        avg_cluster = total_cluster / len(self.train_loader)
        avg_sym = total_sym / len(self.train_loader)
        
        return avg_loss, avg_seg, avg_cluster, avg_sym
    
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
            # Train
            train_loss, seg_loss, cluster_loss, sym_loss = self.train_epoch(epoch)
            
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
            self.history.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'seg_loss': seg_loss,
                'cluster_loss': cluster_loss,
                'symmetry_loss': sym_loss,
                'val_loss': val_loss,
                'val_dice': val_dice
            })
            
            print(f"Epoch {epoch}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"    - Seg: {seg_loss:.4f}")
            print(f"    - Cluster: {cluster_loss:.4f}")
            print(f"    - Symmetry: {sym_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Dice: {val_dice:.4f}")
            print(f"  Best Dice: {self.best_dice:.4f}")
            print("="*60)
            
            # W&B Logging (Epoch)
            if self.config.USE_WANDB:
                wandb.log({
                    'epoch': epoch,
                    'train/loss': train_loss,
                    'train/seg_loss': seg_loss,
                    'train/cluster_loss': cluster_loss,
                    'train/symmetry_loss': sym_loss,
                    'val/loss': val_loss,
                    'val/dice': val_dice,
                    'val/best_dice': self.best_dice,
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        print(f"\nTraining Complete! Best Dice: {self.best_dice:.4f}")


# ============================================================================
# MAIN SCRIPT - Replace train.py
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train SymFormer")
    parser.add_argument('--devices', type=str, default='0', 
                        help='Comma-separated GPU IDs (e.g. "0" or "0,1" or "0,1,2")')
    return parser.parse_args()

def main():
    """
    Usage:
    python train.py --devices 0
    python train.py --devices 0,1
    """
    args = parse_args()
    
    # Set visible devices BEFORE verifying availability
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    
    from configs.config import Config
    from dataset import create_dataloaders
    
    # Setup directories
    Config.create_directories()
    
    # Initialize W&B
    if Config.USE_WANDB:
        wandb.init(
            project=Config.WANDB_PROJECT,
            config=Config.to_dict(),
            name=f"SymFormerV2_gpu{args.devices}",
            mode=Config.WANDB_MODE
        )
    
    # Configure Devices (Support Multi-GPU)
    if torch.cuda.is_available():
        
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
        print("CUDA not available. Using CPU.")
    
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
