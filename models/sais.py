"""SAIS-style appearance-only baseline used in the paper comparison.

The original SAIS implementation is not redistributed here.  This module keeps
its reproducible interface while using the same 16-frame clips and optimisation
budget as ArthroSkill-MC.  A temporal transformer aggregates frame features and
predicts the three skill classes plus the continuous GRS total.
"""
import torch
import torch.nn as nn
from .appearance_branch import AppearanceBranch


class SAISBaseline(nn.Module):
    """Appearance-only temporal baseline with configurable frame stride."""
    def __init__(self, config, stride=1):
        super().__init__()
        self.stride = max(1, int(stride))
        model_cfg = config["model"]
        data_cfg = config["data"]
        dim = int(model_cfg["appearance"].get("output_dim", 768))
        self.appearance = AppearanceBranch(
            output_dim=dim,
            pretrained=model_cfg["appearance"].get("pretrained", True),
        )
        heads = model_cfg.get("heads", {})
        cls_cfg = heads.get("classification", {})
        reg_cfg = heads.get("regression", {})
        hidden = int(cls_cfg.get("hidden_dim", 512))
        dropout = float(cls_cfg.get("dropout", 0.3))
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim, nhead=8, dim_feedforward=hidden,
                dropout=dropout, batch_first=True, norm_first=True
            ), num_layers=2
        )
        self.cls = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, data_cfg["num_classes"]))
        self.reg = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU(),
                                 nn.Dropout(float(reg_cfg.get("dropout", dropout))), nn.Linear(hidden, 1), nn.Sigmoid())
        self.lambda_reg = float(config["training"].get("lambda_reg", 1.0))

    def forward(self, appearance, motion=None, detection_conf=None):
        # Keep the original clip layout and apply SAIS frame stride before the
        # appearance encoder.  The encoder already performs temporal pooling.
        x = appearance[:, :, ::self.stride]
        feat = self.appearance(x).unsqueeze(1)
        feat = self.temporal(feat).squeeze(1)
        return {"cls_logits": self.cls(feat), "total_score": self.reg(feat),
                "dim_scores": self.reg(feat).expand(-1, 6),
                "attn_weights": None, "occlusion_mask": None}

    def compute_loss(self, outputs, labels_cls, labels_reg):
        cls_loss = nn.functional.cross_entropy(outputs["cls_logits"], labels_cls)
        reg_loss = nn.functional.smooth_l1_loss(outputs["total_score"], labels_reg[:, 6:7])
        return {"total": cls_loss + self.lambda_reg * reg_loss, "cls": cls_loss, "reg": reg_loss}
