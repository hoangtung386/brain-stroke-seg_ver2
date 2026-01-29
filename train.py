"""
Training script for SymFormer
Replace old LCNN with this new architecture
"""
import os
import sys
import argparse

# ============================================================================
# EARLY CONFIGURATION (Before importing torch)
# ============================================================================
def early_device_setup():
    """Parse --devices arg and set CUDA_VISIBLE_DEVICES before torch import"""
    # Create a partial parser just for devices to avoid conflicts
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--devices', type=str, default=None)
    
    # Parse known args only (ignore other args like --dataset for now)
    args, _ = parser.parse_known_args()
    
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        print(f"🎯 [Early Setup] Setting CUDA_VISIBLE_DEVICES={args.devices}")
        print(f"   (Physical GPU {args.devices} will be mapped to cuda:0)")
    else:
        # Don't set empty string as it might hide all GPUs in some envs
        # Only set if explicitly requested empty (e.g. for CPU testing)
        if args.devices == "":
             os.environ["CUDA_VISIBLE_DEVICES"] = ""
             print("🎯 [Early Setup] Using CPU (no GPUs specified)")

early_device_setup()

# ============================================================================
# STANDARD IMPORTS (Now safe)
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from models.losses import SymFormerLoss
import wandb
import csv
from monai.networks.utils import one_hot


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
            class_weights=class_weights,
            fp_penalty_weight=getattr(config, 'FP_PENALTY_WEIGHT', 0.0)
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
            
            # Move metadata to device if present
            if metadata:
                metadata = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in metadata.items()
                }
            
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
                
                # Move metadata to device if present
                if metadata:
                    metadata = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in metadata.items()
                    }
                
                # Forward
                output, cluster_outputs, asymmetry_map = self.model(
                    images, metadata=metadata, return_alignment=False
                )
                
                # Loss
                loss, _ = self.criterion(
                    output, masks, cluster_outputs, asymmetry_map
                )
                
                total_val_loss += loss.item()
                
                # Dice metric - CRITICAL FIX
                # DiceMetric expects:
                # - y_pred: (B, C, H, W) with softmax applied OR one-hot
                # - y: (B, 1, H, W) for class indices OR (B, C, H, W) for one-hot
                
                # CRITICAL FIX: Ensure HARD DICE calculation (Standard for validation)
                # 0. Prepare masks
                if masks.ndim == 3:
                     masks_metric = masks.unsqueeze(1)
                else:
                     masks_metric = masks

                # 1. Get discrete predictions (Argmax)
                y_pred_idx = torch.argmax(output, dim=1, keepdim=True)
                y_pred_onehot = one_hot(y_pred_idx, num_classes=self.config.NUM_CLASSES)
                
                # 2. Get one-hot targets
                y_target_onehot = one_hot(masks_metric, num_classes=self.config.NUM_CLASSES)
                
                # 3. Update metric
                self.dice_metric(y_pred=y_pred_onehot, y=y_target_onehot)
                
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

                # Construct dynamic filename
                dataset_suffix = self.config.DATASET_NAME if self.config.DATASET_NAME else "unknown"
                best_model_path = os.path.join(self.config.CHECKPOINT_DIR, f'symformer_best_{dataset_suffix}.pth')
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model_state,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_dice': self.best_dice,
                    'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
                }, best_model_path)
                print(f"✓ Best model saved to {best_model_path}! Dice: {val_dice:.4f}")
            
            # Save Last Model (Every Epoch)
            dataset_suffix = self.config.DATASET_NAME if self.config.DATASET_NAME else "unknown"
            last_model_path = os.path.join(self.config.CHECKPOINT_DIR, f'symformer_{dataset_suffix}.pth')
            
            # Get state dict again if not already got
            if self.multi_gpu:
                 model_state = self.model.module.state_dict()
            else:
                 model_state = self.model.state_dict()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model_state,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'val_dice': val_dice,
                 'config': self.config.to_dict() if hasattr(self.config, 'to_dict') else str(self.config)
            }, last_model_path)
            # print(f"  (Last model saved to {last_model_path})")
            
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
    args = parse_args()
    
    # 2. IMPORT CONFIG & DATASETS
    # (Torch is already imported globally)
    from configs.config import get_config, Config
    from datasets import create_dataloaders
    
    # 3. GET CORRECT CONFIG (Initial setup)
    dataset_name = args.dataset if args.dataset else 'cpaisd'
    ConfigClass = get_config(dataset_name) # Get base config for the dataset
    
    # 4. SETUP CONFIG (Dynamic updates)
    # ConfigClass is now an instance of the specific dataset config.
    # We can directly modify its attributes.
    
    # Dynamic Class Configuration
    if dataset_name.lower() == 'brats':
        print("\n🔧 Configuring for BraTS Dataset (Native 4 classes)")
        ConfigClass.NUM_CLASSES = 4
    elif dataset_name.lower() == 'rsna':
         # RSNA typically 6 classes for subtype or 2 for binary
         # For now, keep default or configurable
         pass
         
    # 5. SETUP DEVICE VARIABLE (This comment is now correctly placed before device setup)
    
    print(f"\n{'='*60}")
    print(f"📦 DATASET: {ConfigClass.DATASET_NAME}")
    # 4. SETUP DIRECTORIES
    ConfigClass.create_directories()
    
    # 5. UPDATE WANDB PROJECT NAME
    wandb_project = f"{ConfigClass.WANDB_PROJECT}{dataset_name}"
    
    # 6. INIT WANDB
    if ConfigClass.USE_WANDB:
        wandb.init(
            project=wandb_project,
            config=ConfigClass.to_dict(),
            name=f"SymFormer_{dataset_name}_{'cpu' if not args.devices else 'gpu'+args.devices.replace(',','_')}",
            mode=ConfigClass.WANDB_MODE
        )
    
    # 7. DEVICE SETUP AFTER ENV VAR SET
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        
        if device_count == 0:
            print("⚠️ CUDA available but no visible devices!")
            device = torch.device('cpu')
            device_ids = []
            multi_gpu = False
        else:
            # After setting CUDA_VISIBLE_DEVICES:
            # - Physical GPU [1] → cuda:0 in PyTorch
            # - Physical GPU [0,1] → cuda:0, cuda:1
            device_ids = list(range(device_count))
            device = torch.device('cuda:0')
            multi_gpu = device_count > 1
            
            print(f"\n🚀 GPU Configuration:")
            print(f"   Requested: --devices {args.devices}")
            print(f"   Visible Count: {device_count}")
            print(f"   Device IDs: {device_ids}")
            print(f"   Primary Device: {device}")
            print(f"   Multi-GPU: {multi_gpu}")
            
            # VALIDATE
            for i in device_ids:
                print(f"   ✓ cuda:{i} = {torch.cuda.get_device_name(i)}")
    else:
        device = torch.device('cpu')
        device_ids = []
        multi_gpu = False
        print("\n💻 Using CPU")
    
    # 8. CREATE DATALOADERS
    print(f"\n📂 Loading dataset from: {ConfigClass.DATA_PATHS.get(dataset_name, 'NOT FOUND')}")
    train_loader, val_loader = create_dataloaders(ConfigClass)
    
    print(f"✓ Train samples: {len(train_loader.dataset)}")
    print(f"✓ Val samples: {len(val_loader.dataset)}")
    
    # 9. CREATE MODEL
    from models.symformer import SymFormer
    model = SymFormer(
        in_channels=ConfigClass.NUM_CHANNELS,
        num_classes=ConfigClass.NUM_CLASSES,
        T=ConfigClass.T,
        input_size=ConfigClass.IMAGE_SIZE
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model: SymFormer")
    print(f"   Parameters: {total_params/1e6:.2f}M")
    print(f"   Classes: {ConfigClass.NUM_CLASSES}")
    print(f"   Input Size: {ConfigClass.IMAGE_SIZE}")
    
    # 10. APPLY DATAPARALLEL
    if multi_gpu:
        print(f"\n⚡ Enabling DataParallel on {device_ids}")
        model = nn.DataParallel(model, device_ids=device_ids)
    
    model = model.to(device)
    
    # 11. CREATE TRAINER
    trainer = SymFormerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=ConfigClass,  # ← PASS ConfigClass
        device=device,
        multi_gpu=multi_gpu
    )
    
    # 12. TRAIN
    print(f"\n{'='*60}")
    print(f"🏋️ STARTING TRAINING")
    print(f"{'='*60}")
    trainer.train(num_epochs=ConfigClass.NUM_EPOCHS)

if __name__ == "__main__":
    main()
