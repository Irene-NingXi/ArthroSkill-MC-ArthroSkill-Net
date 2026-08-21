"""
Multi-task prediction head.
Corresponds to paper Methods: "The fused representation fed two parallel heads: 
a classification head... and a regression head"
"""
import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Classification head: 3-level skill classification.
    Paper: novice / intermediate / expert, trained with cross-entropy loss.
    """
    def __init__(self, input_dim=256, hidden_dim=512, num_classes=3, dropout=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [B, 256] fused features
        Returns:
            logits: [B, 3] unnormalised class scores
        """
        return self.mlp(x)


class RegressionHead(nn.Module):
    """
    Regression head: 6 GRS dimensions + total score.
    Paper: "regressing each GRS dimension independently... predicting 
    the six GRS dimensions and the overall GRS score"
    Loss: smooth L1 loss.
    """
    def __init__(self, input_dim=256, hidden_dim=512, num_dimensions=6, dropout=0.3):
        super().__init__()
        self.num_dimensions = num_dimensions

        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )

        # Separate regressors for each dimension
        self.dim_regressors = nn.ModuleList([
            nn.Linear(hidden_dim // 2, 1) for _ in range(num_dimensions)
        ])

        # Total score regressor
        self.total_regressor = nn.Linear(hidden_dim // 2, 1)

    def forward(self, x):
        """
        Args:
            x: [B, 256] fused features
        Returns:
            dim_scores: [B, 6] 6 GRS dimension predictions (0-1 normalised)
            total_score: [B, 1] GRS total prediction (0-1 normalised)
        """
        feat = self.shared(x)

        dim_scores = []
        for regressor in self.dim_regressors:
            dim_scores.append(regressor(feat))
        dim_scores = torch.cat(dim_scores, dim=-1)

        total_score = self.total_regressor(feat)

        # Sigmoid to 0-1 (assuming labels are normalised)
        dim_scores = torch.sigmoid(dim_scores)
        total_score = torch.sigmoid(total_score)

        return dim_scores, total_score


class MultiTaskHead(nn.Module):
    """
    Multi-task head: classification + regression.
    Joint loss: L = L_cls + lambda * L_reg, lambda=1.0
    """
    def __init__(self, input_dim=256, num_classes=3, num_dimensions=6, 
                 hidden_dim=512, dropout=0.3, lambda_reg=1.0):
        super().__init__()
        self.lambda_reg = lambda_reg
        self.classifier = ClassificationHead(input_dim, hidden_dim, num_classes, dropout)
        self.regressor = RegressionHead(input_dim, hidden_dim, num_dimensions, dropout)

    def forward(self, x):
        """
        Args:
            x: [B, 256] fused features
        Returns:
            cls_logits: [B, 3]
            dim_scores: [B, 6]
            total_score: [B, 1]
        """
        cls_logits = self.classifier(x)
        dim_scores, total_score = self.regressor(x)
        return cls_logits, dim_scores, total_score
