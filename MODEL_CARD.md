# V4 Model Notes

| Field | Value |
|---|---|
| Package | `tornet-detection==1.0.0` |
| Checkpoint | Epoch 23 |
| Parameters | 2,970,049 |
| Input | `[4, 2, 6, 120, 240]` |
| Variables | `DBZ`, `KDP`, `RHOHV`, `VEL`, `WIDTH`, `ZDR` |
| Temporal layer | Bidirectional ConvGRU |
| Spatial pooling | Highest-scoring 5% of cells |
| Decision threshold | `0.9532915353775024` |

## Data split

| Split | Frames | Positive | Negative |
|---|---:|---:|---:|
| Official train | 686,564 | 22,430 | 664,134 |
| Official test | 125,868 | 3,909 | 121,959 |

Validation uses 20% of the official training split. Event groups do not cross between training and validation. Split seed: `20260913`. Training seed: `20260928`.

## Results

| Metric | Validation | Test |
|---|---:|---:|
| PR-AUC | 0.6103 | 0.5233 |
| Macro-year PR-AUC | 0.6123 | — |
| F1 | 0.5861 | 0.5273 |
| ROC-AUC | 0.9464 | 0.9252 |
| Precision | — | 0.5434 |
| Recall | — | 0.5122 |

At the selected threshold:

| True positive | False negative | False positive | True negative |
|---:|---:|---:|---:|
| 2,002 | 1,907 | 1,682 | 120,277 |

Frame scores are uncalibrated sigmoid outputs.

### Test results by year

| Year | PR-AUC | F1 | ROC-AUC |
|---:|---:|---:|---:|
| 2013 | 0.7683 | 0.6952 | 0.9598 |
| 2014 | 0.6959 | 0.6431 | 0.9214 |
| 2015 | 0.5926 | 0.5747 | 0.9252 |
| 2016 | 0.4128 | 0.4410 | 0.8959 |
| 2017 | 0.6206 | 0.5903 | 0.9362 |
| 2018 | 0.4958 | 0.4945 | 0.9214 |
| 2019 | 0.4994 | 0.4988 | 0.9289 |
| 2020 | 0.3180 | 0.4000 | 0.9290 |
| 2021 | 0.3603 | 0.3770 | 0.9199 |
| 2022 | 0.4602 | 0.4840 | 0.9176 |

## Test-set use

The test split was run after V1 and after V4 selection. V5 was rejected on validation results and was not run on the test split.

## Limitations

- Bidirectional inference requires all four frames.
- Test recall at the selected threshold is `0.5122`.
- Year-level test PR-AUC ranges from `0.3180` to `0.7683`.
- Frame scores are not calibrated probabilities.

## Data and weights

TorNet data and the V4 checkpoint are not included. See [Veillette et al.](https://arxiv.org/abs/2401.16437) and [`DATA_AND_MODEL_LICENSE.md`](DATA_AND_MODEL_LICENSE.md).
