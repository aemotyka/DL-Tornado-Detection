# DL Tornado Detection

Research pipeline and packaged inference interface for detecting tornadic radar signatures in the [TorNet](https://github.com/mit-ll/tornet) benchmark dataset.

The final model is an offline, bidirectional spatiotemporal detector. It processes each four-frame TorNet sequence with its physical axes intact:

- time: 4 frames;
- elevation sweep: 2 angles;
- radar variable: `DBZ`, `KDP`, `RHOHV`, `VEL`, `WIDTH`, and `ZDR`;
- azimuth: 120 bins;
- range: 240 bins.

This is research software. It is not a real-time tornado-warning system and has not been operationally validated.

## Final model

For every time step, a shared residual encoder processes each elevation sweep independently. The two encoded sweeps are fused using the lower sweep, upper sweep, signed difference, and absolute difference. A bidirectional ConvGRU then processes the complete four-frame feature sequence. A spatial likelihood head and sparse top-5% log-mean-exp pooling produce one score per frame.

Bidirectional processing uses past and future frames. The model therefore requires a complete four-frame sequence and must not be presented as causal or real-time.

## Results

| Model | Validation PR-AUC | Test PR-AUC | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| DBZ+VEL 2D baseline | — | 0.3473 | 0.4032 | 0.8794 |
| Causal spatiotemporal V1 | 0.5670 | 0.4940 | 0.4949 | 0.9150 |
| Bidirectional V4 | **0.6103** | **0.5233** | **0.5273** | **0.9252** |

V4 improved official-test PR-AUC by approximately 50.7% relative to the original DBZ+VEL 2D baseline. The frozen V4 operating threshold is `0.9532915353775024`. This threshold is applied to an **uncalibrated sigmoid model score**, not a calibrated tornado probability.

The official test set was evaluated during the V1 milestone and once more for the frozen V4 model. It is closed to further model selection or threshold tuning.

## Installation

Python 3.11 or 3.12 is recommended.

From a source checkout:

```bash
python -m pip install -e ".[ml]"
```

From the release wheel:

```bash
python -m pip install "tornet_detection-1.0.0-py3-none-any.whl[ml]"
```

Development and tests:

```bash
python -m pip install -e ".[dev,ml]"
python -m compileall -q src tests
python -m pytest
```

## Model bundle

Inference requires the versioned release bundle, not a bare checkpoint. Obtain `tornet-detection-v1.0.0` from the project release location after it is published and keep its files together:

```text
tornet-detection-v1.0.0/
├── best_model.pt
├── model_config.json
├── normalization.json
├── validation_metrics.json
├── official_test_metrics.json
├── MODEL_CARD.md
├── README.md
├── SHA256SUMS
└── tornet_detection-1.0.0-py3-none-any.whl
```

`model_config.json` binds the architecture, input ordering, checkpoint epoch, and frozen decision threshold. `normalization.json` supplies the exact training-set statistics. The wheel supplies the implementation. `best_model.pt` supplies the epoch-23 V4 weights. Do not mix files across versions.

Google Drive paths used by the research notebooks are not part of the runtime interface.

## Inference

The input must be one TorNet-compatible NetCDF file containing a complete four-frame sequence.

Python:

```python
from tornado_detection.inference import TornadoDetector

detector = TornadoDetector.from_bundle(
    "/path/to/tornet-detection-v1.0.0"
)
result = detector.predict_file("/path/to/example.nc")

print(result.frame_scores)
print(result.frame_predictions)
```

CLI:

```bash
tornet-detect \
  --bundle /path/to/tornet-detection-v1.0.0 \
  --input /path/to/example.nc \
  --output prediction.json
```

Both interfaces return four frame scores and four binary decisions. Required bundle files are validated before inference, and checkpoint loading uses strict state-dictionary validation.

## Data isolation and reproduction

The official TorNet train/test assignment remains intact. Model development divides only the official training split into event-group-disjoint training and validation subsets using a fixed 20% validation fraction and seed `20260913`. Training uses seed `20260928`. The final checkpoint was selected on validation PR-AUC and macro-year validation PR-AUC; the decision threshold was also selected on validation data.

Canonical notebook path:

```text
00 → 01 → 02 → 03 → 16 → 27 → 28 → 29 → 30
```

See [`notebooks/README.md`](notebooks/README.md) for the status of every notebook. Historical experiments remain in the repository for provenance but are not part of the canonical reproduction path.

## Limitations

- The model requires future temporal context and cannot issue causal warnings.
- Scores are not calibrated probabilities.
- Positive examples are rare, and false negatives remain consequential.
- Performance varies materially by year; 2020 and 2021 are particularly difficult.
- The model was evaluated on TorNet, not as part of an operational forecasting or warning workflow.
- Predictions must not replace meteorologists, warning authorities, or established severe-weather procedures.

See [`MODEL_CARD.md`](MODEL_CARD.md) for detailed metrics, intended uses, and risk information.

## Dataset and citation

This project uses TorNet, introduced by Mark S. Veillette, James M. Kurdzo, Phillip M. Stepanian, John Y. N. Cho, Siddharth Samsi, and Joseph McDonald in *A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data* ([arXiv:2401.16437](https://arxiv.org/abs/2401.16437)).

TorNet data and upstream software are external works governed by their own distribution and licensing terms. No TorNet data are included in this repository. This repository does not yet grant a code license, and the trained checkpoint will not be publicly redistributed until its redistribution status is confirmed.
