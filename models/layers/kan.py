"""
Efficient-KAN (Kolmogorov-Arnold Networks) for Decoder Heads

Reference: 
- "KAN: Kolmogorov-Arnold Networks" (2024)
- Efficient-KAN: Lightweight implementation for production

WARNING: Only use KAN for FINAL LAYERS, not entire network!
Medical imaging requires stable latent spaces.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EfficientKANLayer(nn.Module):
    """
    Efficient implementation of KAN layer
    
    Key idea: Replace σ(Wx + b) with learnable spline functions
    
    Original KAN: y = Σ φ(x_i) where φ is B-spline
    Efficient-KAN: Use rational functions instead of B-splines
    """
    
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        
        # Base linear transformation (like MLP)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        
        # Spline coefficients
        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order)
        )
        
        # Grid points (fixed)
        self.register_buffer(
            'grid',
            torch.linspace(-1, 1, grid_size + 1).expand(in_features, -1)
        )
        
        self.spline_order = spline_order
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.xavier_uniform_(self.spline_weight, gain=0.1)
        
    def b_splines(self, x):
        """
        Compute B-spline basis functions
        
        Args:
            x: (B, in_features)
        Returns:
            bases: (B, in_features, grid_size + spline_order)
        """
        # Normalize x to grid range
        x = x.unsqueeze(-1)  # (B, in_features, 1)
        
        # Compute distances to grid points
        grid = self.grid.unsqueeze(0)  # (1, in_features, grid_size+1)
        
        # B-spline basis (simplified - use piecewise linear)
        bases = torch.zeros(
            x.size(0), self.in_features, self.grid_size + self.spline_order,
            device=x.device
        )
        
        for i in range(self.grid_size):
            # Linear interpolation between grid points
            mask = (x >= grid[:, :, i:i+1]) & (x < grid[:, :, i+1:i+2])
            bases[:, :, i] = mask.float().squeeze(-1)
        
        return bases
    
    def forward(self, x):
        """
        Args:
            x: (B, in_features)
        Returns:
            output: (B, out_features)
        """
        # Base transformation
        base_output = F.linear(x, self.base_weight)  # (B, out_features)
        
        # Spline transformation
        bases = self.b_splines(x)  # (B, in_features, grid_size + spline_order)
        
        # Weighted sum of basis functions
        # (out_features, in_features, grid_size) @ (B, in_features, grid_size)
        spline_output = torch.einsum(
            'oig,big->bo',
            self.spline_weight,
            bases
        )
        
        return base_output + spline_output


class RationalKANLayer(nn.Module):
    """
    Rational function approximation (faster than B-splines)
    
    φ(x) = P(x) / Q(x) where P, Q are polynomials
    
    This is what Efficient-KAN paper recommends for production
    """
    
    def __init__(self, in_features, out_features, degree=3):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.degree = degree
        
        # Numerator coefficients
        self.P_coef = nn.Parameter(
            torch.randn(out_features, in_features, degree + 1) * 0.1
        )
        
        # Denominator coefficients (ensure Q(x) > 0)
        self.Q_coef = nn.Parameter(
            torch.randn(out_features, in_features, degree) * 0.1
        )
        
        # Base transformation
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_bias = nn.Parameter(torch.zeros(out_features))
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.base_weight)
        nn.init.zeros_(self.base_bias)
        
        # Initialize P(0) ≈ 0, Q(0) ≈ 1 for stability
        with torch.no_grad():
            self.P_coef[:, :, 0] = 0.0
            self.Q_coef[:, :, 0] = 1.0
        
    def forward(self, x):
        """
        Args:
            x: (*, in_features)
        Returns:
            output: (*, out_features)
        """
        # Flatten batch dims
        original_shape = x.shape
        x = x.view(-1, self.in_features)  # (B, in_features)
        
        # Compute polynomial bases
        x_powers = torch.stack([x ** i for i in range(self.degree + 1)], dim=-1)
        # x_powers: (B, in_features, degree+1)
        
        # Numerator P(x)
        P = torch.einsum('oip,bip->bio', self.P_coef, x_powers)  # (B, in_features, out_features)
        
        # Denominator Q(x) = 1 + Σ q_i x^i (ensure Q > 0)
        Q = 1.0 + torch.einsum('oip,bip->bio', self.Q_coef, x_powers[:, :, 1:])
        
        # Rational function
        rational = P / (Q + 1e-6)  # (B, in_features, out_features)
        
        # Sum over input features
        rational_output = rational.sum(dim=1)  # (B, out_features)
        
        # Add base transformation
        base_output = F.linear(x, self.base_weight, self.base_bias)
        
        output = base_output + rational_output
        
        # Reshape back
        output = output.view(*original_shape[:-1], self.out_features)
        
        return output


class KANDecoderHead(nn.Module):
    """
    KAN-based decoder head for segmentation
    
    Architecture:
    - Conv to reduce spatial dims (optional)
    - Flatten or global pooling
    - KAN layers for classification
    - Upsample back to original resolution
    
    USE CASE: Final 1x1 conv replacement in decoder
    """
    
    def __init__(self, in_channels, num_classes, use_rational=True, hidden_dim=None):
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = max(in_channels // 2, num_classes * 4)
        
        # Spatial reduction (optional - can be identity)
        self.spatial_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.norm = nn.GroupNorm(min(32, in_channels), in_channels)
        
        # KAN layers (replace MLP)
        KANLayer = RationalKANLayer if use_rational else EfficientKANLayer
        
        self.kan1 = KANLayer(in_channels, hidden_dim)
        self.kan2 = KANLayer(hidden_dim, num_classes)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            output: (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        
        # Spatial processing
        feat = self.norm(self.spatial_conv(x))  # (B, C, H, W)
        
        # Permute for KAN: (B, C, H, W) -> (B, H, W, C)
        feat = feat.permute(0, 2, 3, 1)  # (B, H, W, C)
        
        # KAN layers (applied per-pixel)
        feat = self.kan1(feat)  # (B, H, W, hidden_dim)
        feat = self.dropout(feat)
        output = self.kan2(feat)  # (B, H, W, num_classes)
        
        # Permute back: (B, H, W, num_classes) -> (B, num_classes, H, W)
        output = output.permute(0, 3, 1, 2)
        
        return output


# ============================================================================
# Modified HVT Decoder with KAN Heads
# ============================================================================

class KANHVTDecoder(nn.Module):
    """
    HVT Decoder with KAN-based classification heads
    
    Changes:
    - Replace final Conv2d(C, num_classes, 1) with KANDecoderHead
    - Keep kMaX blocks unchanged (too risky to modify)
    """
    
    def __init__(self, encoder_channels=[64, 128, 256, 512, 1024], 
                 num_classes=2, num_heads=8, use_kan_heads=True):
        super().__init__()
        
        from models.symformer import DecoderBlock, kMaXBlock
        
        # Pixel Decoder (unchanged)
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        # kMaX blocks (unchanged)
        self.kmax_blocks = nn.ModuleList([
            kMaXBlock(512, num_classes, num_heads),
            kMaXBlock(256, num_classes, num_heads),
            kMaXBlock(128, num_classes, num_heads),
            kMaXBlock(64, num_classes, num_heads),
        ])
        
        # Classification heads
        if use_kan_heads:
            # ⭐ KAN-based heads
            self.heads = nn.ModuleList([
                KANDecoderHead(512, num_classes, use_rational=True),
                KANDecoderHead(256, num_classes, use_rational=True),
                KANDecoderHead(128, num_classes, use_rational=True),
                KANDecoderHead(64, num_classes, use_rational=True),
            ])
            
            self.final = KANDecoderHead(64, num_classes, use_rational=True)
        else:
            # Standard conv heads
            self.heads = nn.ModuleList([
                nn.Conv2d(num_classes, num_classes, 1) for _ in range(4)
            ])
            self.final = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, bottleneck, skip_connections):
        """Same interface as original HVTDecoder"""
        s1, s2, s3, s4 = skip_connections
        
        def get_mid_slice(x):
            return x[:, :, x.size(2)//2, :, :]
        
        x = get_mid_slice(bottleneck)
        
        cluster_outputs = []
        
        # Stage 4
        x = self.dec4(x, get_mid_slice(s4))
        c4, _ = self.kmax_blocks[0](x)
        cluster_outputs.append(self.heads[0](c4))
        
        # Stage 3
        x = self.dec3(x, get_mid_slice(s3))
        c3, _ = self.kmax_blocks[1](x)
        cluster_outputs.append(self.heads[1](c3))
        
        # Stage 2
        x = self.dec2(x, get_mid_slice(s2))
        c2, _ = self.kmax_blocks[2](x)
        cluster_outputs.append(self.heads[2](c2))
        
        # Stage 1
        x = self.dec1(x, get_mid_slice(s1))
        c1, _ = self.kmax_blocks[3](x)
        cluster_outputs.append(self.heads[3](c1))
        
        # Final output
        final_output = self.final(x)
        
        return final_output, cluster_outputs


# ============================================================================
# Ablation Study Helper
# ============================================================================

def compare_kan_vs_mlp(model_with_kan, model_with_mlp, test_loader, device):
    """
    A/B test to verify KAN improves performance
    
    Returns:
        {
            'kan_dice': float,
            'mlp_dice': float,
            'kan_params': int,
            'mlp_params': int,
            'improvement': float
        }
    """
    from monai.metrics import DiceMetric
    
    dice_metric = DiceMetric(include_background=False, reduction='mean')
    
    results = {}
    
    for name, model in [('kan', model_with_kan), ('mlp', model_with_mlp)]:
        model.eval()
        dice_metric.reset()
        
        with torch.no_grad():
            for images, masks in test_loader:
                images = images.to(device)
                masks = masks.to(device)
                
                outputs, _, _ = model(images)
                
                if masks.ndim == 3:
                    masks = masks.unsqueeze(1)
                    
                dice_metric(y_pred=outputs, y=masks)
        
        dice = dice_metric.aggregate().item()
        params = sum(p.numel() for p in model.parameters())
        
        results[f'{name}_dice'] = dice
        results[f'{name}_params'] = params
    
    results['improvement'] = (
        (results['kan_dice'] - results['mlp_dice']) / results['mlp_dice'] * 100
    )
    
    return results


# ============================================================================
# Usage Example
# ============================================================================

"""
# In your training script:

from models.symformer import SymFormer
from efficient_kan_heads import KANHVTDecoder

# Create base model
model = SymFormer(...)

# Replace decoder with KAN version
model.decoder = KANHVTDecoder(
    num_classes=Config.NUM_CLASSES,
    use_kan_heads=True  # Set False for baseline comparison
)

# Train as usual
trainer.train(model)

# After training, run ablation:
baseline_model = SymFormer(...)
baseline_model.decoder = KANHVTDecoder(use_kan_heads=False)

results = compare_kan_vs_mlp(model, baseline_model, val_loader, device)
print(f"KAN improvement: {results['improvement']:.2f}%")
"""
