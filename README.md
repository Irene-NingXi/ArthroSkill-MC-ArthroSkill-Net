# ArthroSkill-MC

**Multi-centre, cross-joint interpretable skill assessment in arthroscopic surgery videos via occlusion-aware multimodal fusion**

This repository contains the PyTorch reimplementation of the paper. The code is implemented in standard PyTorch (without hard dependency on MMAction2 specific APIs), with detailed comments mapping each module to the corresponding section in the paper.

---

## Project Structure

```
ArthroSkill-MC/
├── configs/
│   └── config.yaml          # Centralised hyperparameters
├── data/
│   ├── __init__.py
│   └── dataset.py           # Custom Dataset for 16-frame clips
├── models/
│   ├── __init__.py
│   ├── appearance_branch.py # Video Swin Transformer (Swin-T) -> 768-D
│   ├── motion_branch.py     # YOLOv8+DeepSORT, 6 kinematic features, 1D-CNN+BiLSTM -> 256-D
│   ├── fusion.py            # Occlusion-aware cross-modal attention (Q=appearance, K/V=motion)
│   ├── heads.py             # Classification (3-class) + Regression (6-D GRS + total)
│   ├── sais.py              # SAIS baseline (appearance-only, for comparison)
│   └── arthroskill.py       # Full model assembly + multi-task loss
├── scripts/
│   ├── train_sais.py        # SAIS training with grid search
│   ├── evaluate_sais.py     # SAIS evaluation with calibration analysis
│   └── train_loso.py        # Leave-one-site-out cross-validation
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py     # Video clipping, brightness/blur filtering, per-video normalisation
│   └── metrics.py           # Accuracy, Pearson r, MAE, RMSE, ICC(A,1), confusion matrix, calibration
├── train.py                 # Training script (AdamW + Cosine Annealing)
├── evaluate.py              # Evaluation script (all metrics from the paper)
├── preprocess.py            # Data preprocessing entry point
├── requirements.txt         # Dependencies
└── README.md                # This file
```

---

## Installation

```bash
# 1. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# or venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt
```

---

## Quick Start

### 1. Data Preprocessing

Place your arthroscopic videos (`.mp4`/`.avi`/`.mov`) in `./videos`:

```bash
python preprocess.py --input ./videos --output ./processed
```

Preprocessing pipeline (corresponding to the paper Methods):
- Decode at 25 fps, resize to 640 x 480
- Filter out-of-body and turbid frames using brightness and Laplacian blur thresholds
- Per-video intensity normalisation
- Split into non-overlapping 16-frame clips, saved as `.pt`

### 2. Prepare Labels

Each `.pt` file in `./processed/train/` and `./processed/val/` should contain:

```python
{
    "frames": tensor [3, 16, 480, 640],      # video clip
    "motion": tensor [16, 6],                # pre-computed kinematic features (optional, placeholder acceptable during training)
    "label_cls": int,                        # 0=novice, 1=intermediate, 2=expert
    "label_reg": tensor [7]                  # [6 normalised GRS dimensions, normalised total score], range 0-1
}
```

> **Note**: YOLOv8+DeepSORT feature extraction should be completed during preprocessing, or run separately to generate `motion` and `detection_conf`.

### 3. Training

```bash
python train.py --config configs/config.yaml --data ./processed --output ./outputs
```

Training configuration (corresponding to the paper):
- Optimiser: AdamW, lr=1e-4, weight_decay=0.05
- Batch size: 4
- Scheduler: Cosine Annealing + 5-epoch Warmup
- Loss: L = L_cls + lambda * L_reg, lambda=1.0
- Data augmentation: random crop, horizontal flip, colour jitter

### 4. Evaluation

```bash
python evaluate.py --config configs/config.yaml --data ./processed --checkpoint ./outputs/best_model.pth --output ./outputs
```

Evaluation outputs (corresponding to the paper Results):
- Classification accuracy
- Pearson correlation coefficient r
- MAE / RMSE
- ICC(A,1) (two-way random-effects, absolute agreement, single rater)
- Confusion matrix
- Adjacent-grade vs cross-grade error rate
- Expected Calibration Error (ECE)

### 5. SAIS Baseline (Comparison)

```bash
python scripts/train_sais.py --config configs/config.yaml --data ./processed --output ./outputs/sais
python scripts/evaluate_sais.py --config configs/config.yaml --data ./processed --checkpoint ./outputs/sais/sais_best_model.pth --output ./outputs/sais
```

### 6. Leave-One-Site-Out (LOSO)

```bash
python scripts/train_loso.py --config configs/config.yaml --data ./processed --output ./outputs/loso
```

---

## Paper-to-Code Mapping

| Paper Module | Code File | Notes |
|-------------|-----------|-------|
| Appearance branch (Video Swin-T) | `models/appearance_branch.py` | Uses timm Swin as approximation; for deployment, replace with MMAction2 official Video Swin and load Kinetics-400 -> HeiChole pretrained weights |
| Motion branch (YOLOv8 + DeepSORT) | `models/motion_branch.py` | `KinematicFeatureExtractor` implements 6 kinematic features; `MotionEncoder` implements 1D-CNN (k7,5,3) + BiLSTM (128) |
| Occlusion-aware fusion | `models/fusion.py` | Cross-attention: Q=appearance, K/V=motion; `key_padding_mask` implements confidence<0.5 suppression |
| Multi-task head | `models/heads.py` | Classification head (CE loss) + Regression head (Smooth L1) |
| SAIS baseline | `models/sais.py` | Reproduced from Kiyasseh et al., Nat. Biomed. Eng. 2023; grid search over lr and stride |
| Training protocol | `train.py` | AdamW + Cosine Annealing; three-stage pretraining requires loading corresponding pretrained weights in config |
| LOSO evaluation | `scripts/train_loso.py` | 3-fold leave-one-site-out cross-validation |
| Evaluation | `evaluate.py` + `utils/metrics.py` | All metrics from the paper: accuracy, Pearson r, MAE, RMSE, ICC(A,1), confusion matrix, calibration analysis |

---

## Key Hyperparameters

All hyperparameters are centralised in `configs/config.yaml`:

```yaml
data:
  clip_len: 16
  img_size: [480, 640]
  brightness_threshold: 30
  blur_threshold: 100

model:
  appearance:
    output_dim: 768
  motion:
    conf_threshold: 0.5
    cnn_channels: [64, 128, 256]
    cnn_kernels: [7, 5, 3]
    lstm_hidden: 128
    output_dim: 256
  fusion:
    num_heads: 8

training:
  batch_size: 4
  learning_rate: 0.0001
  num_epochs: 100
  lambda_reg: 1.0
```

---

## Notes

1. **GPU requirement**: Video Swin Transformer training requires substantial VRAM; at least 12 GB is recommended (e.g. RTX 3090). If VRAM is insufficient, reduce batch_size or use a smaller backbone.
2. **Pretrained weights**: The paper uses three-stage transfer: Kinetics-400 -> HeiChole -> arthroscopy data. The skeleton code has reserved interfaces; please download corresponding pretrained weights and load them in `appearance_branch.py`.
3. **Motion features**: `motion` and `detection_conf` should be generated during preprocessing via YOLOv8 + DeepSORT. The current version provides feature computation logic; a full detection-tracking pipeline needs to be added.
4. **Data privacy**: Surgical videos involve patient privacy. Please ensure ethics approval has been obtained and videos are de-identified before processing.

---

## Citation

```bibtex
@article{xi2026arthroskill,
  title={Multi-centre, cross-joint interpretable skill assessment in arthroscopic surgery videos via occlusion-aware multimodal fusion},
  author={Xi, Ning and others},
  journal={In preparation},
  year={2026}
}
```

---

## Author

- Ning Xi (Universiti Kebangsaan Malaysia)

---

## License

MIT License
