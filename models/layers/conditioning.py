"""
Conditional Encoder for SymFormer
Integrates clinical metadata (NIHSS, age, sex, time) into segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalConditionEncoder(nn.Module):
    """
    Encodes clinical metadata into conditioning vectors
    
    Input features (from CPAISD metadata):
    - NIHSS score (0-42, stroke severity)
    - Age (years)
    - Sex (M/F → 0/1)
    - Time from onset (hours, can be range)
    - DSA flag (cerebral angiography performed)
    """
    
    def __init__(self, embed_dim=256):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Continuous features
        self.nihss_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim // 4)
        )
        
        self.age_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim // 4)
        )
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, embed_dim // 4)
        )
        
        # Categorical features
        self.sex_embed = nn.Embedding(2, embed_dim // 8)  # M/F
        self.dsa_embed = nn.Embedding(2, embed_dim // 8)  # Yes/No
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
    def forward(self, metadata_dict):
        """
        Args:
            metadata_dict: {
                'nihss': (B,) or (B, 1),
                'age': (B,) or (B, 1),
                'sex': (B,) - 0=M, 1=F,
                'time': (B,) or (B, 1),
                'dsa': (B,) - 0=No, 1=Yes
            }
        Returns:
            condition_vector: (B, embed_dim)
        """
        # Normalize continuous features
        nihss = metadata_dict['nihss'].float().unsqueeze(-1) / 42.0  # Normalize to [0,1]
        age = metadata_dict['age'].float().unsqueeze(-1) / 100.0
        time = metadata_dict['time'].float().unsqueeze(-1) / 24.0  # Assume max 24h
        
        # Embed
        nihss_feat = self.nihss_embed(nihss)  # (B, embed_dim/4)
        age_feat = self.age_embed(age)
        time_feat = self.time_embed(time)
        sex_feat = self.sex_embed(metadata_dict['sex'].long())  # (B, embed_dim/8)
        dsa_feat = self.dsa_embed(metadata_dict['dsa'].long())
        
        # Concatenate
        combined = torch.cat([
            nihss_feat, age_feat, time_feat, sex_feat, dsa_feat
        ], dim=-1)  # (B, embed_dim)
        
        # Fusion
        condition_vector = self.fusion(combined)
        
        return condition_vector


class ConditionalCrossAttention(nn.Module):
    """
    Cross-attention that conditions image features on clinical metadata
    
    Used in bottleneck and decoder stages
    """
    
    def __init__(self, feature_dim, condition_dim, num_heads=8):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query from image features
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        
        # Key/Value from condition vector
        self.kv_proj = nn.Linear(condition_dim, feature_dim * 2)
        
        # Output projection
        self.out_proj = nn.Linear(feature_dim, feature_dim)
        
        self.norm = nn.LayerNorm(feature_dim)
        
    def forward(self, image_features, condition_vector):
        """
        Args:
            image_features: (B, C, H, W) or (B, N, C)
            condition_vector: (B, condition_dim)
        Returns:
            conditioned_features: Same shape as image_features
        """
        # Handle 4D input (B, C, H, W)
        is_4d = image_features.dim() == 4
        if is_4d:
            B, C, H, W = image_features.shape
            image_features = image_features.view(B, C, H*W).transpose(1, 2)  # (B, N, C)
        
        B, N, C = image_features.shape
        
        # Queries from image
        q = self.q_proj(image_features)  # (B, N, C)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, N, head_dim)
        
        # Keys/Values from condition
        kv = self.kv_proj(condition_vector).view(B, 1, 2, self.num_heads, self.head_dim)
        k, v = kv.unbind(2)  # Each: (B, 1, heads, head_dim)
        k = k.transpose(1, 2)  # (B, heads, 1, head_dim)
        v = v.transpose(1, 2)
        
        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, N, 1)
        attn = F.softmax(attn, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        out = self.out_proj(out)
        
        # Residual + Norm
        out = self.norm(out + image_features)
        
        # Reshape back to 4D if needed
        if is_4d:
            out = out.transpose(1, 2).view(B, C, H, W)
        
        return out


class ConditionedSymFormer(nn.Module):
    """
    SymFormer with Clinical Conditioning
    
    Modifications:
    1. Add ClinicalConditionEncoder
    2. Inject conditioning via Cross-Attention in bottleneck
    3. Optional: Condition decoder stages
    """
    
    def __init__(self, base_symformer, condition_dim=256, use_decoder_conditioning=True):
        super().__init__()
        
        self.base_model = base_symformer
        
        # Clinical encoder
        self.condition_encoder = ClinicalConditionEncoder(embed_dim=condition_dim)
        
        # Conditioning modules
        self.bottleneck_condition = ConditionalCrossAttention(
            feature_dim=1024,  # Bottleneck channels
            condition_dim=condition_dim,
            num_heads=8
        )
        
        self.use_decoder_conditioning = use_decoder_conditioning
        if use_decoder_conditioning:
            # Add conditioning to each decoder stage
            self.dec4_condition = ConditionalCrossAttention(512, condition_dim, 8)
            self.dec3_condition = ConditionalCrossAttention(256, condition_dim, 8)
            self.dec2_condition = ConditionalCrossAttention(128, condition_dim, 4)
            self.dec1_condition = ConditionalCrossAttention(64, condition_dim, 4)
    
    def forward(self, x, metadata_dict, return_alignment=False):
        """
        Args:
            x: (B, 2T+1, H, W) - Input slices
            metadata_dict: Clinical metadata
            return_alignment: Return alignment info
        """
        # 1. Encode clinical condition
        condition_vector = self.condition_encoder(metadata_dict)  # (B, condition_dim)
        
        # 2. Run encoder (unchanged)
        B, num_slices, H, W = x.shape
        x_flat = x.view(B * num_slices, 1, H, W)
        params = self.base_model.alignment_net(x_flat)
        params = torch.tanh(params) * 0.2
        aligned_flat, _ = self.base_model.alignment_net.apply_transform(x_flat, params)
        x_aligned = aligned_flat.view(B, num_slices, H, W).unsqueeze(1)
        
        s1, x = self.base_model.enc1(x_aligned)
        s2, x = self.base_model.enc2(x)
        s3, x = self.base_model.enc3(x)
        s4, x = self.base_model.enc4(x)
        
        # 3. Bottleneck with conditioning
        x = self.base_model.bottleneck_conv(x)
        x, asymmetry_map = self.base_model.bottleneck(x)
        
        # Apply clinical conditioning to bottleneck
        # Extract middle slice and apply conditioning
        mid_slice = x[:, :, x.size(2)//2, :, :]  # (B, 1024, H, W)
        mid_slice_conditioned = self.bottleneck_condition(mid_slice, condition_vector)
        
        # Put back
        x[:, :, x.size(2)//2, :, :] = mid_slice_conditioned
        
        # 4. Decoder (with optional conditioning)
        def get_mid_slice(tensor):
            return tensor[:, :, tensor.size(2)//2, :, :]
        
        skip_connections = [s1, s2, s3, s4]
        
        # Access decoder blocks
        decoder = self.base_model.decoder
        
        # Stage 4
        feat = get_mid_slice(x)
        feat = decoder.dec4(feat, get_mid_slice(s4))
        if self.use_decoder_conditioning:
            feat = self.dec4_condition(feat, condition_vector)
        c4, _ = decoder.kmax_blocks[0](feat)
        cluster_outputs = [decoder.heads[0](c4)]
        
        # Stage 3
        feat = decoder.dec3(feat, get_mid_slice(s3))
        if self.use_decoder_conditioning:
            feat = self.dec3_condition(feat, condition_vector)
        c3, _ = decoder.kmax_blocks[1](feat)
        cluster_outputs.append(decoder.heads[1](c3))
        
        # Stage 2
        feat = decoder.dec2(feat, get_mid_slice(s2))
        if self.use_decoder_conditioning:
            feat = self.dec2_condition(feat, condition_vector)
        c2, _ = decoder.kmax_blocks[2](feat)
        cluster_outputs.append(decoder.heads[2](c2))
        
        # Stage 1
        feat = decoder.dec1(feat, get_mid_slice(s1))
        if self.use_decoder_conditioning:
            feat = self.dec1_condition(feat, condition_vector)
        c1, _ = decoder.kmax_blocks[3](feat)
        cluster_outputs.append(decoder.heads[3](c1))
        
        # Final output
        output = decoder.final(feat)
        
        if return_alignment:
            aligned_slices = [aligned_flat[i::num_slices] for i in range(num_slices)]
            alignment_params = [params.view(B, num_slices, 3)[:, i] for i in range(num_slices)]
            return output, aligned_slices, alignment_params, cluster_outputs, asymmetry_map
        
        return output, cluster_outputs, asymmetry_map


# ============================================================================
# Usage Example
# ============================================================================

def create_conditioned_model(base_config):
    """
    Factory function to create conditioned SymFormer
    """
    from models.symformer import SymFormer
    
    # Create base model
    base_model = SymFormer(
        in_channels=base_config.NUM_CHANNELS,
        num_classes=base_config.NUM_CLASSES,
        T=base_config.T,
        input_size=base_config.IMAGE_SIZE
    )
    
    # Wrap with conditioning
    conditioned_model = ConditionedSymFormer(
        base_symformer=base_model,
        condition_dim=256,
        use_decoder_conditioning=True  # Set False for ablation study
    )
    
    return conditioned_model


# ============================================================================
# Dataset Modification (Add metadata loading)
# ============================================================================

class CPAISDWithMetadata:
    """
    Wrapper to add metadata loading to existing dataset
    """
    
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        images, masks = self.base_dataset[idx]
        
        # Load metadata from sample
        sample = self.base_dataset.samples[idx]
        metadata_path = sample['slice_path'].parent / 'metadata.json'
        
        import json
        with open(metadata_path, 'r') as f:
            meta = json.load(f)
        
        # Parse metadata
        metadata = {
            'nihss': float(meta.get('nihss', 0)),  # Default 0 if unknown
            'age': float(meta.get('age', 60)),
            'sex': 0 if meta.get('sex', 'M') == 'M' else 1,
            'time': self._parse_time(meta.get('time', '0')),
            'dsa': 1 if meta.get('dsa', False) else 0
        }
        
        return images, masks, metadata
    
    def _parse_time(self, time_str):
        """Parse time field (can be number or range)"""
        if isinstance(time_str, (int, float)):
            return float(time_str)
        
        # Handle range "X-Y"
        if '-' in str(time_str):
            parts = str(time_str).split('-')
            return (float(parts[0]) + float(parts[1])) / 2
        
        try:
            return float(time_str)
        except:
            return 0.0


# ============================================================================
# Modified Training Loop
# ============================================================================

def train_step_with_metadata(model, batch, criterion, device):
    """
    Training step for conditioned model
    """
    images, masks, metadata_dict = batch
    
    # Move to device
    images = images.to(device)
    masks = masks.to(device)
    
    # Convert metadata dict to tensors
    meta_tensors = {
        k: torch.tensor([m[k] for m in metadata_dict]).to(device)
        for k in metadata_dict[0].keys()
    }
    
    # Forward
    output, cluster_outputs, asymmetry_map = model(
        images, 
        meta_tensors,
        return_alignment=False
    )
    
    # Loss
    loss, loss_dict = criterion(output, masks, cluster_outputs, asymmetry_map)
    
    return loss, loss_dict
