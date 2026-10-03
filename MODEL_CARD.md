# Model Card: Bidirectional Spatiotemporal Tornado Detector V4

## Summary

V4 is an offline research model for frame-level detection of tornadic signatures in four-frame TorNet polarimetric weather-radar sequences. It preserves explicit time, elevation-sweep, radar-variable, azimuth, and range axes.

The model is not a real-time warning system, has not been operationally validated, and must not replace expert meteorological judgment or public warning authorities.

## Model details

| Field | Value |
|---|---|
| Package version | `1.0.0` |
| Model class | `SpatiotemporalTornadoDetector` |
| Checkpoint | V4 epoch 23 |
| Parameters | 2,970,049 |
| Input shape | `[time=4, sweep=2, variable=6, azimuth=120, range=240]` |
| Variables | `DBZ`, `KDP`, `RHOHV`, `VEL`, `WIDTH`, `ZDR` |
| Temporal model | Bidirectional ConvGRU |
| Spatial pooling | Top-5% log-mean-exp |
| Frozen threshold | `0.9532915353775024` |
| Score semantics | Uncalibrated sigmoid score |
| Intended mode | Offline, complete-sequence inference |

Each sweep is encoded independently with shared residual weights. Sweep features are fused using both sweeps and their signed and absolute differences. Forward and reverse ConvGRU streams process the full sequence before the likelihood head produces frame-level spatial maps and scores.

Because the reverse stream uses future frames, a score for an early frame depends on later frames in the same sequence. V4 is not causal.

## Data

The project uses the TorNet benchmark: full-resolution polarimetric WSR-88D radar observations sampled from reported storm events across 2013–2022. TorNet includes tornadic examples and challenging nontornadic warning and null cases.

Canonical index totals used by this project:

| Split | Frames | Positive frames | Negative frames |
|---|---:|---:|---:|
| Official train | 686,564 | 22,430 | 664,134 |
| Official test | 125,868 | 3,909 | 121,959 |

Only the official training split was divided for development. Event groups were kept disjoint between model-training and validation subsets. The validation fraction was 20% with split seed `20260913`; the training seed was `20260928`.

TorNet data are not included in this repository or model bundle. The dataset and upstream TorNet software retain their own provenance and licensing terms. Users must obtain TorNet from its official distribution and comply with those terms. Public redistribution of this project's trained weights remains pending an explicit rights decision.

## Evaluation

The final checkpoint and operating threshold were frozen using validation data. V4 was then evaluated once on the official test split.

| Metric | Validation | Official test |
|---|---:|---:|
| PR-AUC | 0.6103 | 0.5233 |
| Macro-year PR-AUC | 0.6123 | — |
| F1 | 0.5861 | 0.5273 |
| ROC-AUC | 0.9464 | 0.9252 |
| Precision | — | 0.5434 |
| Recall | — | 0.5122 |

Official-test confusion counts at the frozen threshold:

| True positive | False negative | False positive | True negative |
|---:|---:|---:|---:|
| 2,002 | 1,907 | 1,682 | 120,277 |

The raw sigmoid outputs are **not calibrated probabilities**. The high numerical threshold must not be interpreted as 95.3% confidence.

### Official-test performance by year

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

Performance varies substantially across years. In particular, 2020 and 2021 remain difficult. Aggregate metrics must not be treated as evidence of uniform robustness across seasons, regions, radar sites, or storm regimes.

## Evaluation disclosure

The official test split was evaluated after the causal spatiotemporal V1 milestone and again after V4 was frozen. No V5 result was evaluated on the official test. The official test is closed to further architecture selection, threshold tuning, or repeated experimentation.

V5 added an auxiliary sequence-category task and was rejected on validation evidence. Validation PR-AUC fell to `0.5779056354733761`, validation F1 fell to `0.5604611555089742`, and the learned frame-task weight grew to `1,982,829.25`, producing numerically unhealthy million-scale training losses.

## Intended uses

Suitable uses:

- reproducible research on TorNet;
- offline comparison of radar-sequence modeling approaches;
- analysis of learned spatial likelihood maps;
- educational demonstrations of sweep-aware and temporal deep learning;
- controlled experiments that preserve the official test boundary.

Unsuitable uses:

- issuing or cancelling tornado warnings;
- autonomous public-safety decisions;
- real-time detection, because the model requires future frames;
- interpreting scores as calibrated probabilities;
- deployment on other radar products, geometries, variables, or sequence lengths without new validation;
- claims of operational performance based only on TorNet metrics.

## Risks and limitations

False negatives can miss tornadic signatures and create severe safety consequences if the model is misused operationally. False positives can contribute to alarm fatigue and unnecessary action. The frozen threshold reflects one validation objective and does not encode the asymmetric costs of real warning decisions.

The dataset is highly imbalanced. Performance may be affected by temporal drift, event-selection policy, radar-site characteristics, missing or range-folded measurements, geographic coverage, and changes in observing systems. The model does not ingest the complete information available to operational meteorologists.

## Reproducibility and artifacts

The release bundle binds together the `1.0.0` wheel, epoch-23 checkpoint, canonical model configuration, training normalization, frozen threshold, and preserved validation and official-test metrics. Checksums must be verified before use. Mixing artifacts from different experiments or versions is unsupported.

The canonical experiment path is:

```text
00 → 01 → 02 → 03 → 16 → 27 → 28 → 29 → 30
```

## Citation

TorNet was introduced by Mark S. Veillette, James M. Kurdzo, Phillip M. Stepanian, John Y. N. Cho, Siddharth Samsi, and Joseph McDonald in *A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data* ([arXiv:2401.16437](https://arxiv.org/abs/2401.16437)).
