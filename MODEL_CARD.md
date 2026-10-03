# V4 Model Notes

## What it is

V4 is the final model from this project. It assigns a tornado score to each of the four frames in a TorNet file.

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

The bidirectional ConvGRU sees the complete four-frame sequence. It cannot make a causal, live prediction.

## Data split

| Split | Frames | Positive | Negative |
|---|---:|---:|---:|
| Official train | 686,564 | 22,430 | 664,134 |
| Official test | 125,868 | 3,909 | 121,959 |

The validation split came only from the official training data. Event groups do not cross between training and validation. The validation fraction was 20%, with split seed `20260913` and training seed `20260928`.

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

The sigmoid scores are not calibrated probabilities. A score of `0.95` does not mean a 95% chance of a tornado.

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

The large year-to-year spread matters. The aggregate score does not show uniform performance across storms, years, or radar sites.

## Test-set use

The test split was run after V1 and again after the V4 model and threshold had been selected from validation data. V5 performed worse on validation and was not tested. The project stops with the V4 result.

## Reasonable uses

- Reproducing the reported TorNet experiment
- Comparing temporal or sweep-aware radar models
- Inspecting frame-level likelihood maps
- Testing the packaged inference code with compatible TorNet files

Do not use it to issue warnings or make safety decisions. The model uses future frames, misses roughly half of the positive test frames, and has not been tested in live radar operations.

## Data and weights

TorNet is described in [Veillette et al., *A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data*](https://arxiv.org/abs/2401.16437). No TorNet data are included here.

The V4 checkpoint is not publicly distributed. The repository's MIT license applies to this project's code, not to TorNet or the private checkpoint. See [`DATA_AND_MODEL_LICENSE.md`](DATA_AND_MODEL_LICENSE.md).
