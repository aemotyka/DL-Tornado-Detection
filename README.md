# DL Tornado Detection

PyTorch experiments for frame-level tornado detection on the [TorNet](https://github.com/mit-ll/tornet) radar dataset.

## Results

| Model | Validation PR-AUC | Test PR-AUC | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| DBZ+VEL 2D baseline | — | 0.3473 | 0.4032 | 0.8794 |
| Causal spatiotemporal V1 | 0.5670 | 0.4940 | 0.4949 | 0.9150 |
| Bidirectional V4 | **0.6103** | **0.5233** | **0.5273** | **0.9252** |

## Model

| Field | Value |
|---|---|
| Input shape | `[time=4, sweep=2, variable=6, azimuth=120, range=240]` |
| Variables | `DBZ`, `KDP`, `RHOHV`, `VEL`, `WIDTH`, `ZDR` |
| Spatial encoder | Shared residual encoder per sweep |
| Sweep fusion | Lower sweep, upper sweep, signed difference, absolute difference |
| Temporal layer | Bidirectional ConvGRU over all four frames |
| Spatial output | One `15 × 30` likelihood map per frame |
| Frame score | Top-5% spatial pooling |

## Installation

Python 3.11 or 3.12:

```bash
python -m pip install -e ".[ml]"
```

For development:

```bash
python -m pip install -e ".[dev,ml]"
python -m pytest
```

## Inference

Inference needs three files from the same run:

```text
best_model.pt
model_config.json
normalization.json
```

Python:

```python
from tornado_detection.inference import TornadoDetector

detector = TornadoDetector.from_bundle("/path/to/model_bundle")
result = detector.predict_file("/path/to/sample.nc")

print(result.frame_scores)
print(result.frame_predictions)
```

Command line:

```bash
tornet-detect \
  --bundle /path/to/model_bundle \
  --input /path/to/sample.nc \
  --output prediction.json
```

The output contains four sigmoid scores and four thresholded predictions. The scores are not calibrated probabilities. The threshold is `0.9532915353775024`.

## Experiment split

TorNet's official train/test split was kept intact. The official training data was divided into event-group-disjoint training and validation sets using a 20% validation fraction and seed `20260913`. Training used seed `20260928`.

## Reproducing the final run

The shortest path through the notebooks is:

```text
00 → 01 → 02 → 03 → 16 → 27 → 28 → 29 → 30
```

See [`notebooks/README.md`](notebooks/README.md) for the notebook index.

## Data, code, and weights

TorNet was created by Mark S. Veillette, James M. Kurdzo, Phillip M. Stepanian, John Y. N. Cho, Siddharth Samsi, and Joseph McDonald. See [*A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data*](https://arxiv.org/abs/2401.16437).

TorNet data and the V4 checkpoint are not included. Project code is MIT-licensed. See [`DATA_AND_MODEL_LICENSE.md`](DATA_AND_MODEL_LICENSE.md).
