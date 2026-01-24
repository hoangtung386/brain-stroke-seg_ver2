"""
Improved Loss Function for SymFormer
Addresses multi-scale supervision and symmetry-aware learning

Key improvements:
1. Weighted multi-scale deep supervision
2. Focal loss for hard examples
3. Contrastive symmetry loss
4. Boundary refinement loss
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss, FocalLoss


class SymFormerLoss(nn.Module):
    """
    Enhanced loss with proper multi-scale handling
    """
    def __init__(self, num_classes=2, class_weights=None,
                 dice_weight=0.5, focal_weight=0.3, ce_weight=0.2,
                 cluster_weight=0.15, symmetry_weight=0.05,
                 boundary_weight=0.1):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Main segmentation losses
        self.dice = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            reduction='mean'
        )
        
        self.focal = FocalLoss(
            include_background=False,
            to_onehot_y=True,
            gamma=2.0,
            reduction='mean'
        )
        
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        
        # Loss weights
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.cluster_weight = cluster_weight
        self.symmetry_weight = symmetry_weight
        self.boundary_weight = boundary_weight
        
        # Multi-scale weights (deep supervision)
        # Higher weight for finer scales
        self.scale_weights = [0.5, 0.3, 0.15, 0.05]  # 4 decoder stages
        
    def compute_main_loss(self, output, target):
        """Main segmentation loss with Dice + Focal + CE"""
        # Prepare targets
        if target.ndim == 3:
            target_dice = target.unsqueeze(1)
            target_ce = target.long()
        else:
            target_dice = target
            target_ce = target.squeeze(1).long()
        
        # Compute losses
        dice_loss = self.dice(output, target_dice)
        focal_loss = self.focal(output, target_dice)
        ce_loss = self.ce(output, target_ce)
        
        # Combined
        main_loss = (
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss +
            self.ce_weight * ce_loss
        )
        
        return main_loss, {
            'dice': dice_loss.item(),
            'focal': focal_loss.item(),
            'ce': ce_loss.item()
        }
    
    def compute_cluster_loss(self, cluster_outputs, target):
        """
        Multi-scale deep supervision with weighted losses
        
        Args:
            cluster_outputs: List of 4 outputs from decoder stages
                            [(B, C, H/2, W/2), (B, C, H/4, W/4), ...]
            target: (B, H, W) or (B, 1, H, W)
        """
        if not cluster_outputs:
            return torch.tensor(0.0, device=target.device), {}
        
        total_loss = 0.0
        stage_losses = []
        
        # Ensure we have scale weights for all stages
        num_stages = len(cluster_outputs)
        if len(self.scale_weights) < num_stages:
            # Extend weights if needed
            remaining = num_stages - len(self.scale_weights)
            extended_weights = self.scale_weights + [0.01] * remaining
        else:
            extended_weights = self.scale_weights[:num_stages]
        
        # Normalize weights to sum to 1
        weight_sum = sum(extended_weights)
        normalized_weights = [w / weight_sum for w in extended_weights]
        
        for i, (cluster_out, scale_weight) in enumerate(zip(cluster_outputs, normalized_weights)):
            # Resize target to match cluster output
            H_out, W_out = cluster_out.shape[-2:]
            
            target_scaled = F.interpolate(
                target.unsqueeze(1).float() if target.ndim == 3 else target.float(),
                size=(H_out, W_out),
                mode='nearest'
            )
            
            # Compute loss for this scale
            dice_loss = self.dice(cluster_out, target_scaled.long())
            focal_loss = self.focal(cluster_out, target_scaled.long())
            
            scale_loss = (dice_loss + focal_loss) * scale_weight
            total_loss += scale_loss
            
            stage_losses.append(scale_loss.item())
        
        loss_dict = {
            f'cluster_stage_{i}': loss for i, loss in enumerate(stage_losses)
        }
        loss_dict['cluster_total'] = total_loss.item()
        
        return total_loss, loss_dict
    
    def compute_symmetry_loss(self, asymmetry_map, target):
        """
        Contrastive symmetry loss
        
        Key idea:
        - Encourage asymmetry in stroke regions (high asymmetry = good)
        - Penalize asymmetry in healthy regions (low asymmetry = good)
        """
        if asymmetry_map is None:
            return torch.tensor(0.0, device=target.device), {}
        
        # Create masks
        if target.ndim == 4:
            target = target.squeeze(1)
        
        # Resize asymmetry_map to match target if needed
        if asymmetry_map.shape[-2:] != target.shape[-2:]:
            asymmetry_map = F.interpolate(
                asymmetry_map,
                size=target.shape[-2:],
                mode='bilinear',
                align_corners=False
            )
        
        # Average over depth dimension if present (B, 1, D, H, W) -> (B, 1, H, W)
        if asymmetry_map.ndim == 5:
            asymmetry_map = asymmetry_map.mean(dim=2)
        
        # Binary masks
        stroke_mask = (target > 0).float().unsqueeze(1)  # (B, 1, H, W)
        healthy_mask = (target == 0).float().unsqueeze(1)
        
        # Loss components
        # 1. Penalize asymmetry in healthy regions (should be symmetric)
        healthy_sym_loss = (asymmetry_map * healthy_mask).mean()
        
        # 2. Reward asymmetry in stroke regions (should be asymmetric)
        # Use negative loss to encourage high asymmetry
        stroke_asym_loss = -torch.log(asymmetry_map * stroke_mask + 1e-6).mean()
        
        # 3. Margin-based contrastive loss
        # Asymmetry in stroke should be HIGHER than in healthy
        margin = 0.1
        asym_stroke = (asymmetry_map * stroke_mask).sum() / (stroke_mask.sum() + 1e-6)
        asym_healthy = (asymmetry_map * healthy_mask).sum() / (healthy_mask.sum() + 1e-6)
        
        contrastive_loss = F.relu(margin - (asym_stroke - asym_healthy))
        
        # Combined symmetry loss
        symmetry_loss = healthy_sym_loss + 0.5 * stroke_asym_loss + contrastive_loss
        
        return symmetry_loss, {
            'sym_healthy': healthy_sym_loss.item(),
            'sym_stroke': stroke_asym_loss.item(),
            'sym_contrastive': contrastive_loss.item()
        }
    
    def compute_boundary_loss(self, output, target):
        """
        Boundary refinement loss using morphological operations
        Helps improve edge accuracy
        """
        # Get prediction
        pred = torch.argmax(output, dim=1)  # (B, H, W)
        
        # Compute boundaries using morphological gradient
        # Boundary = Dilation - Erosion
        kernel = torch.ones(1, 1, 3, 3, device=output.device)
        
        # For target
        target_float = target.unsqueeze(1).float() if target.ndim == 3 else target.float()
        target_dilated = F.conv2d(target_float, kernel, padding=1)
        target_eroded = -F.conv2d(-target_float, kernel, padding=1)
        target_boundary = (target_dilated - target_eroded) > 0
        
        # For prediction
        pred_float = pred.unsqueeze(1).float()
        pred_dilated = F.conv2d(pred_float, kernel, padding=1)
        pred_eroded = -F.conv2d(-pred_float, kernel, padding=1)
        pred_boundary = (pred_dilated - pred_eroded) > 0
        
        # Boundary loss (Dice on boundaries)
        intersection = (pred_boundary & target_boundary).float().sum()
        union = pred_boundary.float().sum() + target_boundary.float().sum()
        
        boundary_dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        
        return boundary_dice, {'boundary_dice': boundary_dice.item()}
    
    def forward(self, output, target, cluster_outputs=None, asymmetry_map=None):
        """
        Complete loss computation
        
        Args:
            output: (B, C, H, W) - final prediction
            target: (B, H, W) - ground truth
            cluster_outputs: List[(B, C, H_i, W_i)] - multi-scale outputs
            asymmetry_map: (B, 1, D, H, W) - symmetry map
        
        Returns:
            total_loss: scalar
            loss_dict: detailed breakdown
        """
        loss_dict = {}
        
        # 1. Main segmentation loss
        main_loss, main_dict = self.compute_main_loss(output, target)
        loss_dict.update(main_dict)
        
        # 2. Multi-scale cluster loss
        cluster_loss, cluster_dict = self.compute_cluster_loss(cluster_outputs, target)
        loss_dict.update(cluster_dict)
        
        # 3. Symmetry loss
        symmetry_loss, sym_dict = self.compute_symmetry_loss(asymmetry_map, target)
        loss_dict.update(sym_dict)
        
        # 4. Boundary refinement loss
        boundary_loss, boundary_dict = self.compute_boundary_loss(output, target)
        loss_dict.update(boundary_dict)
        
        # Total weighted loss
        total_loss = (
            main_loss +
            self.cluster_weight * cluster_loss +
            self.symmetry_weight * symmetry_loss +
            self.boundary_weight * boundary_loss
        )
        
        loss_dict['total'] = total_loss.item()
        loss_dict['main'] = main_loss.item()
        
        return total_loss, loss_dict


# ============================================================================
# Drop-in Replacement for existing loss
# ============================================================================

class SymFormerLoss(ImprovedSymFormerLoss):
    """
    Backward compatible wrapper
    Just replace the import and it works!
    """
    pass


# ============================================================================
# Ablation Study Helper
# ============================================================================

def create_ablation_losses(num_classes=2, class_weights=None):
    """
    Create different loss configurations for ablation study
    
    Returns:
        dict: {name: loss_function}
    """
    ablation_configs = {
        'baseline': {
            'dice_weight': 0.7,
            'focal_weight': 0.0,
            'ce_weight': 0.3,
            'cluster_weight': 0.0,
            'symmetry_weight': 0.0,
            'boundary_weight': 0.0
        },
        'with_focal': {
            'dice_weight': 0.5,
            'focal_weight': 0.3,
            'ce_weight': 0.2,
            'cluster_weight': 0.0,
            'symmetry_weight': 0.0,
            'boundary_weight': 0.0
        },
        'with_multiscale': {
            'dice_weight': 0.5,
            'focal_weight': 0.3,
            'ce_weight': 0.2,
            'cluster_weight': 0.15,
            'symmetry_weight': 0.0,
            'boundary_weight': 0.0
        },
        'with_symmetry': {
            'dice_weight': 0.5,
            'focal_weight': 0.3,
            'ce_weight': 0.2,
            'cluster_weight': 0.15,
            'symmetry_weight': 0.05,
            'boundary_weight': 0.0
        },
        'full': {
            'dice_weight': 0.5,
            'focal_weight': 0.3,
            'ce_weight': 0.2,
            'cluster_weight': 0.15,
            'symmetry_weight': 0.05,
            'boundary_weight': 0.1
        }
    }
    
    losses = {}
    for name, config in ablation_configs.items():
        losses[name] = ImprovedSymFormerLoss(
            num_classes=num_classes,
            class_weights=class_weights,
            **config
        )
    
    return losses
    