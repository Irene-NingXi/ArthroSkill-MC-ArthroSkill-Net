"""
ArthroSkill-MC full model
Corresponds to paper Fig. 2a: appearance branch + motion branch + 
occlusion-aware fusion + multi-task head.
"""
import torch
import torch.nn as nn
from .appearance_branch import AppearanceBranch
from .motion_branch import MotionBranch
from .fusion import OcclusionAwareFusion
from .heads import MultiTaskHead


class ArthroSkillMC(nn.Module):
    """
    ArthroSkill-MC dual-branch network.
    Input: {
        'appearance': [B, 3, 16, 480, 640] video clip,
        'motion': [B, 16, 6] kinematic feature sequence,
        'detection_conf': [B, 16] detection confidence (optional)
    }
    Output: {
        'cls_logits': [B, 3] classification logits,
        'dim_scores': [B, 6] GRS dimension regression,
        'total_score': [B, 1] GRS total regression,
        'attn_weights': [B, 1, 16] attention weights (for interpretability)
    }
    """
    def __init__(self, config):
        super().__init__()
        cfg_model = config["model"]
        cfg_train = config["training"]
        num_classes = config["data"]["num_classes"]

        # Appearance branch: Video Swin -> 768-D
        self.appearance_branch = AppearanceBranch(
            output_dim=cfg_model["appearance"]["output_dim"],
            pretrained=cfg_model["appearance"].get("pretrained", True)
        )

        # Motion branch: YOLO+DeepSORT features -> 1D-CNN+BiLSTM -> 256-D
        self.motion_branch = MotionBranch(
            input_dim=6,
            output_dim=cfg_model["motion"]["output_dim"],
            conf_threshold=cfg_model["motion"]["conf_threshold"],
            cnn_channels=cfg_model["motion"]["cnn_channels"],
            cnn_kernels=cfg_model["motion"]["cnn_kernels"],
            lstm_hidden=cfg_model["motion"]["lstm_hidden"],
            lstm_layers=cfg_model["motion"]["lstm_layers"]
        )

        # Fusion: occlusion-aware cross-modal attention
        self.fusion = OcclusionAwareFusion(
            appearance_dim=cfg_model["appearance"]["output_dim"],
            motion_dim=cfg_model["motion"]["output_dim"],
            num_heads=cfg_model["fusion"]["num_heads"],
            dropout=cfg_model["fusion"]["dropout"]
        )

        # Multi-task head: classification + regression
        self.head = MultiTaskHead(
            input_dim=cfg_model["motion"]["output_dim"],
            num_classes=num_classes,
            num_dimensions=config["data"]["grs_dimensions"],
            hidden_dim=cfg_model["heads"]["classification"]["hidden_dim"],
            dropout=cfg_model["heads"]["classification"]["dropout"],
            lambda_reg=cfg_train["lambda_reg"]
        )

    def forward(self, appearance, motion, detection_conf=None):
        """
        Args:
            appearance: [B, 3, T, H, W]
            motion: [B, T, 6]
            detection_conf: [B, T] (optional)
        Returns:
            dict with cls_logits, dim_scores, total_score, attn_weights
        """
        # Appearance branch
        app_feat = self.appearance_branch(appearance)  # [B, 768]

        # Motion branch
        mot_feat, occ_mask = self.motion_branch(motion, detection_conf)  # [B, 256], [B, T]

        # Re-extract motion sequence features (post-CNN, pre-LSTM) for fusion
        mot_seq = motion.permute(0, 2, 1)  # [B, 6, T]
        for layer in self.motion_branch.encoder.cnn:
            mot_seq = layer(mot_seq)
        mot_seq = mot_seq.permute(0, 2, 1)  # [B, T, 256]

        # Fusion
        fused_feat, attn_weights = self.fusion(app_feat, mot_seq, occ_mask)

        # Prediction heads
        cls_logits, dim_scores, total_score = self.head(fused_feat)

        return {
            "cls_logits": cls_logits,
            "dim_scores": dim_scores,
            "total_score": total_score,
            "attn_weights": attn_weights,
            "occlusion_mask": occ_mask
        }

    def compute_loss(self, outputs, labels_cls, labels_reg):
        """
        Multi-task loss: L = L_cls + lambda * L_reg (smooth L1)
        labels_reg: [B, 7] -> first 6 dims are GRS dimensions, last is total score
        """
        ce_loss = nn.functional.cross_entropy(outputs["cls_logits"], labels_cls)

        dim_targets = labels_reg[:, :6]
        total_target = labels_reg[:, 6:7]

        dim_loss = nn.functional.smooth_l1_loss(outputs["dim_scores"], dim_targets)
        total_loss = nn.functional.smooth_l1_loss(outputs["total_score"], total_target)
        reg_loss = dim_loss + total_loss

        total = ce_loss + self.head.lambda_reg * reg_loss

        return {
            "total": total,
            "cls": ce_loss,
            "reg": reg_loss,
            "dim": dim_loss,
            "total_score": total_loss
        }
