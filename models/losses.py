import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.losses import DiceLoss

class SymFormerLoss(nn.Module):
    """
    Combined loss for SymFormer
    Components:
    1. Segmentation Loss (Dice + CE)
    2. Clustering Loss (if applicable)
    3. Symmetry Loss (if applicable)
    """
    def __init__(self, num_classes, class_weights=None, 
                 dice_weight=0.7, ce_weight=0.3, 
                 cluster_weight=0.1, symmetry_weight=0.05):
        super().__init__()
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.cluster_weight = cluster_weight
        self.symmetry_weight = symmetry_weight
        
        # Main Seg Losses
        self.ce_loss = nn.CrossEntropyLoss(weight=class_weights)
        self.dice_loss = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            reduction='mean'
        )
        
    def forward(self, output, target, cluster_outputs=None, asymmetry_map=None):
        """
        Args:
            output: (B, C, H, W)
            target: (B, H, W)
            cluster_outputs: Optional auxiliary outputs
            asymmetry_map: Optional symmetry map
        """
        loss_dict = {}
        
        # 1. Segmentation Loss
        ce = self.ce_loss(output, target)
        dice = self.dice_loss(output, target.unsqueeze(1))
        seg_loss = self.ce_weight * ce + self.dice_weight * dice
        
        loss_dict['seg_loss'] = seg_loss.item()
        loss_dict['ce_loss'] = ce.item()
        loss_dict['dice_loss'] = dice.item()
        
        total_loss = seg_loss
        
        # 2. Cluster Loss (Placeholder)
        if cluster_outputs is not None:
            # Assuming cluster_outputs matches some target or consistency
            # For now, 0
            cluster_loss = torch.tensor(0.0, device=output.device)
            loss_dict['cluster_loss'] = cluster_loss.item()
            total_loss += self.cluster_weight * cluster_loss
        else:
            loss_dict['cluster_loss'] = 0.0
            
        # 3. Symmetry Loss (Placeholder)
        if asymmetry_map is not None:
             # Encourage sparsity or consistency
             sym_loss = torch.mean(torch.abs(asymmetry_map))
             loss_dict['symmetry_loss'] = sym_loss.item()
             total_loss += self.symmetry_weight * sym_loss
        else:
             loss_dict['symmetry_loss'] = 0.0
             
        return total_loss, loss_dict
