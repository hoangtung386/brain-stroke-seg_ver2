"""
Mamba-2 Bottleneck for SymFormer
Replaces SymmetryAwareBottleneck with State Space Models

Reference: Mamba-2 (2024) - "Transformers are SSMs"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


try:
    from mamba_ssm import Mamba2
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    print("Warning: mamba-ssm not installed. Using fallback implementation.")


class MambaBottleneck(nn.Module):
    """
    Mamba-2 based bottleneck for brain symmetry modeling
    
    Advantages over Transformer bottleneck:
    1. Linear O(N) complexity
    2. Better at modeling sequential dependencies (slices)
    3. State space formulation naturally handles symmetry
    """
    
    def __init__(self, channels=1024, depth=4, d_state=128, d_conv=4, expand=2):
        super().__init__()
        
        self.channels = channels
        
        if MAMBA_AVAILABLE:
            # Use official Mamba-2 implementation
            self.mamba_layers = nn.ModuleList([
                Mamba2(
                    d_model=channels,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand
                )
                for _ in range(depth)
            ])
        else:
            # Fallback: Simplified SSM
            self.mamba_layers = nn.ModuleList([
                SimplifiedSSM(channels, d_state)
                for _ in range(depth)
            ])
        
        # Normalization layers
        self.norms = nn.ModuleList([
            nn.LayerNorm(channels) for _ in range(depth)
        ])
        
        # Symmetry-aware processing
        self.symmetry_gate = SymmetryGate(channels)
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            output: (B, C, D, H, W)
            asymmetry_map: (B, 1, D, H, W)
        """
        B, C, D, H, W = x.shape
        
        # Split hemispheres
        mid_w = W // 2
        x_left = x[..., :mid_w]  # (B, C, D, H, W/2)
        x_right = x[..., mid_w:]
        
        # Flip right for symmetry
        x_right_flipped = torch.flip(x_right, dims=[-1])
        
        # Process each hemisphere with Mamba
        # Reshape for sequence modeling: (B, C, D, H, W/2) -> (B, D*H*W/2, C)
        left_seq = rearrange(x_left, 'b c d h w -> b (d h w) c')
        right_seq = rearrange(x_right_flipped, 'b c d h w -> b (d h w) c')
        
        # Apply Mamba layers
        for mamba, norm in zip(self.mamba_layers, self.norms):
            # Left hemisphere
            left_seq = left_seq + mamba(norm(left_seq))
            # Right hemisphere
            right_seq = right_seq + mamba(norm(right_seq))
        
        # Reshape back
        left_out = rearrange(left_seq, 'b (d h w) c -> b c d h w', d=D, h=H, w=mid_w)
        right_out = rearrange(right_seq, 'b (d h w) c -> b c d h w', d=D, h=H, w=mid_w)
        
        # Unflip right
        right_out = torch.flip(right_out, dims=[-1])
        
        # Compute asymmetry map
        asymmetry_map = torch.abs(
            x_left - torch.flip(x_right, dims=[-1])
        ).mean(dim=1, keepdim=True)
        
        # Apply symmetry gate
        left_gated, right_gated = self.symmetry_gate(left_out, right_out, asymmetry_map)
        
        # Concatenate
        output = torch.cat([left_gated, right_gated], dim=-1)
        
        return output, asymmetry_map


class SymmetryGate(nn.Module):
    """
    Gating mechanism to balance normal symmetry vs pathological asymmetry
    """
    
    def __init__(self, channels):
        super().__init__()
        
        self.gate = nn.Sequential(
            nn.Conv3d(channels * 2 + 1, 64, 1),  # +1 for asymmetry map
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Conv3d(64, 2, 1),  # 2 gates: left_weight, right_weight
            nn.Sigmoid()
        )
        
    def forward(self, left_feat, right_feat, asymmetry_map):
        """
        Args:
            left_feat: (B, C, D, H, W/2)
            right_feat: (B, C, D, H, W/2)
            asymmetry_map: (B, 1, D, H, W)
        """
        # Resize asymmetry map
        asym_left = asymmetry_map[..., :asymmetry_map.size(-1)//2]
        asym_right = asymmetry_map[..., asymmetry_map.size(-1)//2:]
        
        # Concatenate features
        combined = torch.cat([left_feat, right_feat, asym_left], dim=1)
        
        # Compute gates
        gates = self.gate(combined)  # (B, 2, D, H, W/2)
        left_gate = gates[:, 0:1, ...]
        right_gate = gates[:, 1:2, ...]
        
        # Apply gates
        left_out = left_feat * left_gate
        right_out = right_feat * right_gate
        
        return left_out, right_out


class SimplifiedSSM(nn.Module):
    """
    Simplified State Space Model (fallback when mamba-ssm not available)
    
    Based on S4 (Structured State Space Sequence Model)
    """
    
    def __init__(self, d_model, d_state=64):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # SSM parameters
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Projections
        self.in_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            output: (B, L, d_model)
        """
        B, L, D = x.shape
        
        # Project input
        u = self.in_proj(x)  # (B, L, d_model)
        
        # Discretize SSM (Euler method)
        dt = 0.001  # Time step
        A_bar = torch.eye(self.d_state, device=x.device) + dt * self.A
        B_bar = dt * self.B
        
        # Apply SSM
        h = torch.zeros(B, self.d_state, device=x.device)
        outputs = []
        
        for t in range(L):
            h = h @ A_bar.T + u[:, t, :] @ B_bar.T  # (B, d_state)
            y = h @ self.C.T + u[:, t, :] * self.D  # (B, d_model)
            outputs.append(y)
        
        output = torch.stack(outputs, dim=1)  # (B, L, d_model)
        output = self.out_proj(output)
        
        return output


# ============================================================================
# Mamba-Enhanced SymFormer
# ============================================================================

class MambaSymFormer(nn.Module):
    """
    SymFormer with Mamba-2 bottleneck
    
    Changes:
    1. Replace SymmetryAwareBottleneck with MambaBottleneck
    2. Keep rest of architecture unchanged
    """
    
    def __init__(self, in_channels=1, num_classes=2, T=1, input_size=(512, 512)):
        super().__init__()
        
        self.T = T
        
        from models.components import EncoderBlock3D, AlignmentNetwork
        
        self.alignment_net = AlignmentNetwork(input_size)
        
        # Encoder (unchanged)
        self.enc1 = EncoderBlock3D(1, 64)
        self.enc2 = EncoderBlock3D(64, 128)
        self.enc3 = EncoderBlock3D(128, 256)
        self.enc4 = EncoderBlock3D(256, 512)
        
        # Bottleneck preprocessing
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True)
        )
        
        # ⭐ NEW: Mamba-2 Bottleneck
        self.bottleneck = MambaBottleneck(
            channels=1024,
            depth=4,  # Number of Mamba layers
            d_state=128,  # State dimension
            d_conv=4,  # Conv kernel size in Mamba
            expand=2  # Expansion factor
        )
        
        # Decoder (unchanged)
        from models.symformer import HVTDecoder
        self.decoder = HVTDecoder(num_classes=num_classes)
        
    def forward(self, x, return_alignment=False):
        """Standard forward pass"""
        # Alignment
        B, num_slices, H, W = x.shape
        x_flat = x.view(B * num_slices, 1, H, W)
        params = self.alignment_net(x_flat)
        params = torch.tanh(params) * 0.2
        aligned_flat, _ = self.alignment_net.apply_transform(x_flat, params)
        x_aligned = aligned_flat.view(B, num_slices, H, W).unsqueeze(1)
        
        # Encoding
        s1, x = self.enc1(x_aligned)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        
        # Bottleneck with Mamba
        x = self.bottleneck_conv(x)
        x, asymmetry_map = self.bottleneck(x)  # ⭐ Mamba magic here
        
        # Decoder
        output, cluster_outputs = self.decoder(x, [s1, s2, s3, s4])
        
        if return_alignment:
            aligned_slices = [aligned_flat[i::num_slices] for i in range(num_slices)]
            alignment_params = [params.view(B, num_slices, 3)[:, i] for i in range(num_slices)]
            return output, aligned_slices, alignment_params, cluster_outputs, asymmetry_map
        
        return output, cluster_outputs, asymmetry_map


# ============================================================================
# Installation Guide
# ============================================================================

"""
To use Mamba-2, install:

```bash
pip install mamba-ssm
# Or from source:
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
```

If installation fails, the code will use SimplifiedSSM fallback.

Mamba-2 requires:
- CUDA 11.8+
- PyTorch 2.0+
- Triton (for optimized kernels)
"""
