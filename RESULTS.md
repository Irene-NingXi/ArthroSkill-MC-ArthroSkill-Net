# Results

Raw surgical videos cannot be publicly released due to patient privacy and hospital ethics (IRB) restrictions. All training and evaluation were performed on the hospital intranet.

## Headline results

| Metric | Result |
| --- | ---: |
| Overall accuracy | 79.59% (156/196) |
| Pearson r | 0.805 |
| MAE | 2.867 |
| RMSE | 4.237 |
| ICC(A,1) | 0.822 |
| SAIS baseline accuracy | 68.37% (134/196) |
| LOSO pooled accuracy | 64.3% (126/196) |
| LOSO pooled Pearson r | 0.634 |
| LOSO pooled MAE | 4.38 |

## Seed replication

| Seed | Accuracy | Pearson r |
| ---: | ---: | ---: |
| 42 | 78.57% | 0.79 |
| 3407 (headline seed) | 79.59% | 0.805 |
| 2024 | 80.61% | 0.82 |

Reported as 79.59% ± 1.0pp, r = 0.805 ± 0.02.

## LOSO per fold

| Fold | Centre | Pearson r | MAE |
| ---: | --- | ---: | ---: |
| 1 | Centre A | 0.657 | 4.38 |
| 2 | Centre B | 0.617 | 4.59 |
| 3 | Centre C | 0.618 | 4.15 |

Full per-video prediction tables and the statistical audit trail are available with the manuscript submission / upon reasonable request.
