"""
Occlusion-aware cross-modal attention fusion module.
Corresponds to paper Methods: "The two representations were fused with a 
cross-attention block in which appearance features served as queries and 
motion features as keys and values."
"""
import torch
import torch.nn as nn


class OcclusionAwareFusion(nn.Module):
    """
    Occlusion-aware cross-modal attention fusion.
    Paper design: appearance as query, motion as key/value.
    Detection confidence < 0.5 suppresses motion features in attention.
    """
    def __init__(self, appearance_dim=768, motion_dim=256, num_heads=8, 
                 dropout=0.1, ff_dim=1024):
        super().__init__()
        self.appearance_dim = appearance_dim
        self.motion_dim = motion_dim
        self.num_heads = num_heads

        # Project appearance to motion dimension for cross-attention
        self.q_proj = nn.Linear(appearance_dim, motion_dim)
        self.k_proj = nn.Linear(motion_dim, motion_dim)
        self.v_proj = nn.Linear(motion_dim, motion_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=motion_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # LayerNorm + FFN
        self.norm1 = nn.LayerNorm(motion_dim)
        self.norm2 = nn.LayerNorm(motion_dim)

        self.ffn = nn.Sequential(
            nn.Linear(motion_dim, ff_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, motion_dim),
            nn.Dropout(dropout)
        )

        # Residual connection from appearance
        self.appearance_residual = nn.Linear(appearance_dim, motion_dim)

    def forward(self, appearance_feat, motion_feat, occlusion_mask=None):
        """
        Args:
            appearance_feat: [B, 768] appearance features
            motion_feat: [B, T, 256] motion feature sequence (T = clip_len)
            occlusion_mask: [B, T] occlusion mask (1=valid, 0=occluded)
        Returns:
            [B, 256] fused representation, [B, 1, T] attention weights
        """
        B = appearance_feat.shape[0]

        # Appearance as query [B, 1, 256]
        q = self.q_proj(appearance_feat).unsqueeze(1)

        # Motion as key/value
        k = self.k_proj(motion_feat)  # [B, T, 256]
        v = self.v_proj(motion_feat)  # [B, T, 256]

        # Build key padding mask: occluded positions are ignored
        key_padding_mask = None
        if occlusion_mask is not None:
            key_padding_mask = (occlusion_mask == 0)  # [B, T], bool

        # Cross-modal attention
        attn_out, attn_weights = self.cross_attn(
            query=q,
            key=k,
            value=v,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True
        )

        attn_out = attn_out.squeeze(1)  # [B, 256]

        # Residual + LayerNorm
        residual = self.appearance_residual(appearance_feat)  # [B, 256]
        fused = self.norm1(attn_out + residual)

        # FFN
        out = self.norm2(fused + self.ffn(fused))

        return out, attn_weights
