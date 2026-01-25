"""
SymFormer: Symmetry-Aware Hybrid Transformer for Stroke Segmentation

Key Innovations:
1. Symmetry-Aware Bottleneck (NO Transformer here!)
2. HVT Decoder with k-Means Mask Transformer
3. Cross-Hemisphere Attention for stroke detection
4. Multi-scale cluster assignment loss

Based on:
- HResFormer (Hybrid Local-Global fusion)
- kMaXU (k-Means Mask Transformer)
- Vision Transformer Survey (HVT in decoder)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from configs.config import Config

# ⭐ SOTA Components
try:
    from models.layers.mamba import MambaBottleneck
    from models.layers.conditioning import ClinicalConditionEncoder, ConditionalCrossAttention
    from models.layers.kan import KANDecoderHead
    SOTA_AVAILABLE = True
except ImportError:
    print("Warning: SOTA components not found in models/layers/")
    SOTA_AVAILABLE = False


# ============================================================================
# 1. SYMMETRY-AWARE BOTTLENECK (Replaces Transformer Bottleneck)
# ============================================================================

class SymmetryAwareBottleneck(nn.Module):
    """
    Bottleneck that explicitly models brain hemisphere symmetry
    NO Transformer - pure CNN with symmetry constraints
    """
    def __init__(self, in_channels=1024, num_heads=8):
        super().__init__()
        
        # Dual-branch processing
        self.left_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.right_branch = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 3, padding=1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )
        
        # Cross-hemisphere attention (lightweight)
        self.cross_attn = CrossHemisphereAttention(in_channels, num_heads)
        
        # Fusion
        self.fusion = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, 1),
            nn.GroupNorm(32, in_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, D, H, W)
        Returns:
            fused: (B, C, D, H, W)
            asymmetry_map: (B, 1, D, H, W) - for loss
        """
        B, C, D, H, W = x.shape
        
        # Split left/right hemispheres with equal widths
        mid_w = W // 2
        x_left = x[..., :mid_w]
        x_right = x[..., mid_w:mid_w*2]  # Ensure same width as left
        
        # Process each hemisphere
        f_left = self.left_branch(x_left)
        f_right = self.right_branch(x_right)
        
        # Cross-hemisphere attention
        f_left_attended, f_right_attended, asym_map = self.cross_attn(f_left, f_right)
        
        # Concatenate and fuse
        f_concat = torch.cat([f_left_attended, f_right_attended], dim=-1)
        
        # If original W was odd, pad to restore original width
        if f_concat.shape[-1] < W:
            pad_width = W - f_concat.shape[-1]
            f_concat = F.pad(f_concat, (0, pad_width), mode='constant', value=0)
        
        fused = self.fusion(f_concat)
        
        return fused, asym_map


class CrossHemisphereAttention(nn.Module):
    """
    Attention between left and right brain hemispheres
    Detects asymmetry (potential stroke indicator)
    """
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.q_left = nn.Conv3d(channels, channels, 1)
        self.k_right = nn.Conv3d(channels, channels, 1)
        self.v_right = nn.Conv3d(channels, channels, 1)
        
        self.q_right = nn.Conv3d(channels, channels, 1)
        self.k_left = nn.Conv3d(channels, channels, 1)
        self.v_left = nn.Conv3d(channels, channels, 1)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, left, right):
        """
        Args:
            left: (B, C, D, H, W_half)
            right: (B, C, D, H, W_half)
        """
        # Flip right hemisphere for symmetry
        right_flipped = torch.flip(right, dims=[-1])
        
        # Left queries attend to right (flipped) keys
        q_l = self.q_left(left)  # (B, C, D, H, W_half)
        k_r = self.k_right(right_flipped)
        v_r = self.v_right(right_flipped)
        
        # Get actual spatial dimensions from tensor shape
        B, C, D, H, W = q_l.shape
        spatial_size = D * H * W  # Use actual dimensions, not assumed
        
        # Reshape for multi-head attention
        q_l = q_l.view(B, self.num_heads, self.head_dim, spatial_size).transpose(-2, -1)
        k_r = k_r.view(B, self.num_heads, self.head_dim, spatial_size)
        v_r = v_r.view(B, self.num_heads, self.head_dim, spatial_size).transpose(-2, -1)
        
        # Attention
        attn_l = torch.softmax(q_l @ k_r * self.scale, dim=-1)
        out_l = (attn_l @ v_r).transpose(-2, -1).reshape(B, C, D, H, W)
        
        # Same for right
        q_r = self.q_right(right_flipped)
        k_l = self.k_left(left)
        v_l = self.v_left(left)
        
        # Use same spatial_size (already computed from left)
        q_r = q_r.view(B, self.num_heads, self.head_dim, spatial_size).transpose(-2, -1)
        k_l = k_l.view(B, self.num_heads, self.head_dim, spatial_size)
        v_l = v_l.view(B, self.num_heads, self.head_dim, spatial_size).transpose(-2, -1)
        
        attn_r = torch.softmax(q_r @ k_l * self.scale, dim=-1)
        out_r = (attn_r @ v_l).transpose(-2, -1).reshape(B, C, D, H, W)
        
        # Compute asymmetry map (for loss)
        asym_map = torch.abs(left - right_flipped).mean(dim=1, keepdim=True)
        
        # Unflip right output
        out_r = torch.flip(out_r, dims=[-1])
        
        return out_l, out_r, asym_map


# ============================================================================
# 2. k-MEANS MASK TRANSFORMER DECODER (From kMaXU paper)
# ============================================================================

class kMaXBlock(nn.Module):
    """
    k-Means Mask Transformer Block for Decoder
    Learns K cluster centers (K = num_classes) and assigns pixels
    """
    def __init__(self, pixel_channels, num_clusters, num_heads=8, num_layers=2):
        super().__init__()
        self.num_clusters = num_clusters
        
        # Cluster centers (learnable)
        self.cluster_centers = nn.Parameter(torch.randn(1, num_clusters, pixel_channels))
        nn.init.trunc_normal_(self.cluster_centers, std=0.02)
        
        # k-Means Cross-Attention layers
        self.cross_attn_layers = nn.ModuleList([
            kMeansCrossAttention(pixel_channels, num_heads)
            for _ in range(num_layers)
        ])
        
        # Self-Attention on cluster centers
        self.self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(pixel_channels, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # FFN
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(pixel_channels, pixel_channels * 4),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(pixel_channels * 4, pixel_channels)
            )
            for _ in range(num_layers)
        ])
        
        self.norms = nn.ModuleList([
            nn.LayerNorm(pixel_channels) for _ in range(num_layers * 3)
        ])
        
    def forward(self, pixel_features):
        """
        Args:
            pixel_features: (B, C, H, W) from pixel decoder
        Returns:
            cluster_assignments: (B, K, H, W)
            updated_centers: (B, K, C)
        """
        B, C, H, W = pixel_features.shape
        
        # Initialize centers
        centers = self.cluster_centers.expand(B, -1, -1)  # (B, K, C)
        
        # Flatten pixel features
        pixels = pixel_features.view(B, C, H*W).permute(0, 2, 1)  # (B, N, C)
        
        for i in range(len(self.cross_attn_layers)):
            # 1. k-Means Cross-Attention
            centers_updated = self.norms[i*3](centers)
            centers = centers + self.cross_attn_layers[i](centers_updated, pixels)
            
            # 2. Self-Attention on centers
            centers_sa = self.norms[i*3+1](centers)
            centers_sa, _ = self.self_attn_layers[i](centers_sa, centers_sa, centers_sa)
            centers = centers + centers_sa
            
            # 3. FFN
            centers_ffn = self.norms[i*3+2](centers)
            centers = centers + self.ffn_layers[i](centers_ffn)
        
        # Final cluster assignment
        assignments = self.compute_assignments(centers, pixels)  # (B, K, N)
        assignments = assignments.view(B, self.num_clusters, H, W)
        
        return assignments, centers
    
    def compute_assignments(self, centers, pixels):
        """
        Cluster-wise argmax (from kMaX paper)
        
        Args:
            centers: (B, K, C)
            pixels: (B, N, C)
        Returns:
            assignments: (B, K, N)
        """
        # Similarity: (B, K, C) @ (B, C, N) = (B, K, N)
        similarity = torch.bmm(centers, pixels.transpose(1, 2))
        
        # Cluster-wise argmax
        assignments = F.softmax(similarity, dim=1)  # Normalize over clusters
        
        return assignments


class kMeansCrossAttention(nn.Module):
    """
    Cross-attention: queries from cluster centers, keys/values from pixels
    """
    def __init__(self, channels, num_heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)
        
    def forward(self, centers, pixels):
        """
        Args:
            centers: (B, K, C) - queries
            pixels: (B, N, C) - keys/values
        """
        out, _ = self.attn(centers, pixels, pixels)
        return out


# ============================================================================
# 3. HVT DECODER with Multi-Scale kMaX
# ============================================================================

class HVTDecoder(nn.Module):
    """
    Hybrid Vision Transformer Decoder
    Combines pixel decoder (CNN) with kMaX blocks at multiple scales
    """
    def __init__(self, encoder_channels=[64, 128, 256, 512, 1024], 
                 num_classes=2, num_heads=8):
        super().__init__()
        
        # Pixel Decoder (cascaded upsampling)
        self.dec4 = DecoderBlock(1024, 512)
        self.dec3 = DecoderBlock(512, 256)
        self.dec2 = DecoderBlock(256, 128)
        self.dec1 = DecoderBlock(128, 64)
        
        # kMaX blocks at each scale
        self.kmax_blocks = nn.ModuleList([
            kMaXBlock(512, num_classes, num_heads),
            kMaXBlock(256, num_classes, num_heads),
            kMaXBlock(128, num_classes, num_heads),
            kMaXBlock(64, num_classes, num_heads),
        ])
        
        # Final fusion heads
        self.heads = nn.ModuleList([
            nn.Conv2d(num_classes, num_classes, 1) for _ in range(4)
        ])
        
        self.final = nn.Conv2d(64, num_classes, 1)
        
    def forward(self, bottleneck, skip_connections):
        """
        Args:
            bottleneck: (B, 1024, D, H/32, W/32)
            skip_connections: List[(B, C, D, H, W)]
        Returns:
            final_output: (B, num_classes, H, W)
            cluster_outputs: List[(B, num_classes, H_i, W_i)]
        """
        s1, s2, s3, s4 = skip_connections
        
        # Extract middle slice from 3D features
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
        
        # Final pixel output
        final_output = self.final(x)
        
        return final_output, cluster_outputs


class DecoderBlock(nn.Module):
    """Standard 2D decoder block with skip connection"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2)
        self.conv1 = nn.Conv2d(out_channels * 2, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


# ============================================================================
# 4. SYMFORMER - Complete Architecture
# ============================================================================

class SymFormer(nn.Module):
    """
    Complete SymFormer Architecture
    
    Pipeline:
    1. Alignment Network (unchanged)
    2. 3D CNN Encoder
    3. Symmetry-Aware Bottleneck (NO Transformer!)
    4. HVT Decoder with k-Means clustering
    5. Multi-scale fusion
    """
    def __init__(self, in_channels=1, num_classes=2, T=1, input_size=(512, 512)):
        super().__init__()
        
        self.T = T
        
        # Encoder (from original SEAN)
        from models.components import EncoderBlock3D, AlignmentNetwork
        
        self.alignment_net = AlignmentNetwork(input_size)
        
        self.enc1 = EncoderBlock3D(1, 64)
        self.enc2 = EncoderBlock3D(64, 128)
        self.enc3 = EncoderBlock3D(128, 256)
        self.enc4 = EncoderBlock3D(256, 512)
        
        # Symmetry-Aware Bottleneck (NEW!)
        self.bottleneck_conv = nn.Sequential(
            nn.Conv3d(512, 1024, 3, padding=1),
            nn.GroupNorm(32, 1024),
            nn.ReLU(inplace=True)
        )
        if Config.USE_MAMBA and SOTA_AVAILABLE:
            # ⭐ Mamba-2 Bottleneck
            self.bottleneck = MambaBottleneck(
                channels=1024,
                depth=Config.MAMBA_DEPTH
            )
        else:
            # Original Symmetry-Aware Bottleneck
            self.bottleneck = SymmetryAwareBottleneck(1024)
            
        # ⭐ Clinical Conditioning
        self.use_conditioning = Config.USE_CONDITIONING and SOTA_AVAILABLE
        if self.use_conditioning:
            self.condition_encoder = ClinicalConditionEncoder(embed_dim=256)
            self.bottleneck_condition = ConditionalCrossAttention(1024, 256)
        
        # HVT Decoder with k-Means (NEW!)
        # Pass KAN usage flag to decoder if supported, else wrap it
        self.decoder = HVTDecoder(num_classes=num_classes)
        
        if Config.USE_KAN and SOTA_AVAILABLE:
             # Replace standard heads with KAN heads
             self.decoder.heads = nn.ModuleList([
                KANDecoderHead(512, num_classes, use_rational=True),
                KANDecoderHead(256, num_classes, use_rational=True),
                KANDecoderHead(128, num_classes, use_rational=True),
                KANDecoderHead(64, num_classes, use_rational=True),
            ])
             self.decoder.final = KANDecoderHead(64, num_classes, use_rational=True)
        
    def forward(self, x, metadata=None, return_alignment=False):
        """
        Args:
            x: (B, 2T+1, H, W)
            metadata: Dict of clinical features (optional)
        Returns:
            output: (B, num_classes, H, W)
            cluster_outputs: List for multi-scale loss
            asymmetry_map: For symmetry loss
        """
        # 0. Encode clinical condition
        condition_vector = None
        if self.use_conditioning and metadata is not None:
            # Check if metadata has required keys for conditioning
            required_keys = ['nihss', 'age', 'sex', 'time', 'dsa']
            if all(k in metadata for k in required_keys):
                # Check if values are meaningful (not all zeros - dummy data)
                has_real_data = any(
                    metadata[k].sum().item() != 0 if isinstance(metadata[k], torch.Tensor) 
                    else sum(metadata[k]) != 0 
                    for k in required_keys
                )
                if has_real_data:
                    condition_vector = self.condition_encoder(metadata)
                # else: silently skip conditioning for dummy metadata
            
        # 1. Alignment (unchanged)
        B, num_slices, H, W = x.shape
        x_flat = x.view(B * num_slices, 1, H, W)
        params = self.alignment_net(x_flat)
        params = torch.tanh(params) * 0.2
        aligned_flat, _ = self.alignment_net.apply_transform(x_flat, params)
        x_aligned = aligned_flat.view(B, num_slices, H, W).unsqueeze(1)
        
        # 2. 3D Encoding
        s1, x = self.enc1(x_aligned)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        
        # 3. Symmetry-Aware Bottleneck (Mamba or Standard)
        x = self.bottleneck_conv(x)
        x, asymmetry_map = self.bottleneck(x)
        
        # Apply conditioning to bottleneck if enabled
        if condition_vector is not None:
             # Extract middle slice and apply conditioning
            mid_slice = x[:, :, x.size(2)//2, :, :]
            mid_slice_conditioned = self.bottleneck_condition(mid_slice, condition_vector)
            x[:, :, x.size(2)//2, :, :] = mid_slice_conditioned
        
        # 4. HVT Decoder
        output, cluster_outputs = self.decoder(x, [s1, s2, s3, s4])
        
        if return_alignment:
            aligned_slices = [aligned_flat[i::num_slices] for i in range(num_slices)]
            alignment_params = [params.view(B, num_slices, 3)[:, i] for i in range(num_slices)]
            return output, aligned_slices, alignment_params, cluster_outputs, asymmetry_map
        
        return output, cluster_outputs, asymmetry_map



