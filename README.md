# DL Tornado Detection

PyTorch experiments for frame-level tornado detection on the [TorNet](https://github.com/mit-ll/tornet) radar dataset.

The final experiment uses four radar frames, two elevation sweeps, and six radar variables. It reached a test PR-AUC of `0.5233` and F1 of `0.5273`. That is a meaningful improvement over the 2D baseline, but it is not close to an operational warning model.

## Results

| Model | Validation PR-AUC | Test PR-AUC | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| DBZ+VEL 2D baseline | — | 0.3473 | 0.4032 | 0.8794 |
| Causal spatiotemporal V1 | 0.5670 | 0.4940 | 0.4949 | 0.9150 |
| Bidirectional V4 | **0.6103** | **0.5233** | **0.5273** | **0.9252** |

V4 raised test PR-AUC by `0.1760` over the 2D baseline. At the selected threshold it found 2,002 of 3,909 positive test frames and produced 1,682 false positives.

## Model

Each TorNet file contains four frames with shape:

```text
[time=4, sweep=2, variable=6, azimuth=120, range=240]
```

The variables are `DBZ`, `KDP`, `RHOHV`, `VEL`, `WIDTH`, and `ZDR`.

The model:

1. encodes each elevation sweep with shared residual blocks;
2. combines the two sweeps and their differences;
3. runs the sequence forward and backward through a ConvGRU;
4. produces a spatial likelihood map for each frame; and
5. pools the highest-scoring 5% of spatial cells into a frame score.

Because the ConvGRU runs in both directions, each score can use later frames from the same file. The model is for offline experiments, not live detection.

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

The checkpoint is not published in this repository. The code and synthetic inference tests still show the complete loading and prediction path.

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

The output contains four scores and four thresholded predictions. The scores are not calibrated probabilities. V4 uses the validation-selected threshold `0.9532915353775024`.

## Experiment split

TorNet's official train/test split was kept intact. The official training data was divided into event-group-disjoint training and validation sets using a 20% validation fraction and seed `20260913`. Training used seed `20260928`.

The official test split was run after V1 and once after V4. V5 was rejected on validation results and was not run on the test split. No further test-set tuning is planned.

## Reproducing the final run

The shortest path through the notebooks is:

```text
00 → 01 → 02 → 03 → 16 → 27 → 28 → 29 → 30
```

[`notebooks/README.md`](notebooks/README.md) explains what each notebook does and which experiments were superseded.

## Limits

- Test F1 is `0.5273`; this is an experiment, not a warning product.
- The model uses future frames and cannot run causally.
- It finds only about half of the positive test frames at the selected threshold.
- Results vary considerably by year, especially in 2020 and 2021.
- The scores are not calibrated probabilities.
- It was tested on TorNet, not in an operational weather workflow.

## Data, code, and weights

TorNet was created by Mark S. Veillette, James M. Kurdzo, Phillip M. Stepanian, John Y. N. Cho, Siddharth Samsi, and Joseph McDonald. See [*A Benchmark Dataset for Tornado Detection and Prediction using Full-Resolution Polarimetric Weather Radar Data*](https://arxiv.org/abs/2401.16437).

This repository contains no TorNet data. Download the data from the [official TorNet repository](https://github.com/mit-ll/tornet) and follow its terms and attribution requirements.

The code in this repository is MIT-licensed. That license does not cover TorNet data or the trained V4 checkpoint. The checkpoint is kept as a private project artifact rather than distributed publicly. See [`DATA_AND_MODEL_LICENSE.md`](DATA_AND_MODEL_LICENSE.md).
