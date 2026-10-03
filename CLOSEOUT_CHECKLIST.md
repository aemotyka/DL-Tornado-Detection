# Tornado Detection Project Closeout Checklist

This document is the authoritative checklist for closing, packaging, and releasing the rebuilt TorNet tornado-detection project. No additional model experimentation is required for the initial release.

## Frozen final-model decision

- [x] Select V4 as the final model.
- [x] Freeze checkpoint epoch `23`.
- [x] Freeze classification threshold `0.9532915353775024`.
- [x] Record final architecture:
  - Six radar variables: `DBZ`, `KDP`, `RHOHV`, `VEL`, `WIDTH`, `ZDR`
  - Two elevation sweeps
  - Four temporal frames
  - Bidirectional ConvGRU temporal processing
  - Sparse spatial pooling with `pooling_topk_fraction=0.05`
  - Offline, full-sequence inference
- [x] Record validation metrics:
  - PR-AUC: `0.6103114794623021`
  - Macro-year PR-AUC: `0.6122810708211335`
  - F1: `0.5860555826426134`
  - ROC-AUC: `0.9464051829712374`
- [x] Record official-test metrics:
  - PR-AUC: `0.5233280973770362`
  - F1: `0.5273278019228237`
  - ROC-AUC: `0.9251858020023173`
  - Precision: `0.5434310532030402`
  - Recall: `0.5121514453824507`
  - True positives: `2,002`
  - False negatives: `1,907`
  - False positives: `1,682`
  - True negatives: `120,277`
- [x] Permanently close the official test set after the V4 evaluation.
- [ ] Verify the preserved V4 checkpoint, normalization artifact, validation metrics, official-test metrics, and diagnostic artifacts still exist in Drive.

## Rejected V5 experiment

- [x] Reject V5 as a final-model candidate.
- [x] Do not evaluate V5 on the official test.
- [x] Preserve notebook 31 as a documented failed experiment.
- [ ] Document the V5 failure in the notebook index and model-development history:
  - Validation PR-AUC fell to `0.5779056354733761`.
  - Validation F1 fell to `0.5604611555089742`.
  - Learned frame-task weight exploded to `1,982,829.25`.
  - Training loss reached millions.
  - The learned multitask weighting was numerically unhealthy.
- [x] Remove V5-only auxiliary-category code from the final `1.0.0` package surface.

## Current packaging gaps

- [ ] Add a root `README.md`.
- [ ] Add `MODEL_CARD.md`.
- [ ] Make and document a license decision.
- [ ] Add continuous integration.
- [x] Add a supported inference API.
- [x] Add a command-line inference entry point.
- [x] Add a canonical machine-readable model configuration.
- [ ] Add a release artifact manifest.
- [ ] Add SHA-256 checksums for release artifacts.
- [ ] Document the relationship between the wheel, checkpoint, normalization, configuration, and frozen threshold.
- [ ] Add a notebook index identifying authoritative and superseded notebooks.
- [ ] Add a clean-install, end-to-end inference smoke test.
- [x] Remove the obsolete root-level original final-project notebook and PDF.

## Pass 1 — Freeze the final package

### Package cleanup

- [x] Remove the optional V5 sequence-category head and its tests from the release package.
- [x] Retain the successful V4 features:
  - Bidirectional temporal processing
  - Sparse top-k spatial pooling
  - Six-variable, two-sweep, four-frame input contract
  - Explicit finite, range-folded, and coordinate inputs
- [x] Bump the package version to `1.0.0`.
- [x] Confirm the final model has `2,970,049` parameters.
- [ ] Confirm existing V4 checkpoint keys load strictly into the final model.
- [ ] Confirm all unit tests pass.

### Canonical model configuration

- [x] Create `artifacts/v1/model_config.json`.
- [x] Record these constructor arguments:

```python
SpatiotemporalTornadoDetector(
    variable_count=6,
    sweep_count=2,
    coordinate_count=5,
    encoder_widths=(32, 64, 128),
    temporal_channels=128,
    pooling_temperature=1.0,
    pooling_topk_fraction=0.05,
    bidirectional_temporal=True,
)
```

- [x] Record variable order and channel order.
- [x] Record coordinate names.
- [x] Record sequence shape `[4, 2, 6, 120, 240]`.
- [x] Record normalization provenance.
- [x] Record validation fraction and split seed.
- [x] Record training seed.
- [x] Record checkpoint epoch `23`.
- [x] Record frozen threshold `0.9532915353775024`.
- [x] Record package version `1.0.0`.
- [x] Record that the model is bidirectional and offline-only.

### Inference interface

- [x] Add `src/tornado_detection/inference.py`.
- [x] Add `src/tornado_detection/cli.py`.
- [x] Add `tests/test_inference.py`.
- [x] Implement the supported Python API:

```python
detector = TornadoDetector.from_bundle(bundle_directory)
result = detector.predict_file(netcdf_path)
```

- [x] Return four frame scores and four frozen-threshold predictions.
- [x] Include the threshold, variables, sequence shape, and bidirectional designation in the result.
- [x] Do not call raw sigmoid outputs calibrated probabilities in documentation or output schemas.
- [x] Add the `tornet-detect` console entry point to `pyproject.toml`.
- [x] Support this command:

```bash
tornet-detect \
  --bundle /path/to/tornet-detection-v1.0.0 \
  --input /path/to/example.nc \
  --output prediction.json
```

- [x] Validate required bundle files before inference.
- [x] Load the checkpoint with strict state-dictionary validation.
- [x] Apply the exact stored normalization and channel order.
- [x] Default to CPU when CUDA is unavailable.
- [x] Add deterministic synthetic inference tests.

## Pass 2 — Build the release bundle

- [ ] Create a staging directory named `tornet-detection-v1.0.0/`.
- [ ] Include exactly these deliverables:

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

- [ ] Copy the frozen V4 `best_model.pt`, not a V5 checkpoint.
- [ ] Copy the exact all-six-variable normalization artifact used by V4.
- [ ] Copy the preserved V4 validation metrics.
- [ ] Copy the preserved V4 official-test metrics.
- [ ] Build the `1.0.0` wheel.
- [ ] Generate and verify `SHA256SUMS` after bundle contents are final.
- [ ] Create a compressed release archive.
- [ ] Decide whether the trained checkpoint may be redistributed.
- [ ] If the checkpoint is not committed, publish it through a GitHub Release or another durable location and document its checksum.
- [ ] Do not make notebook-specific Google Drive paths part of the runtime contract.

## Pass 3 — Documentation and project presentation

### Root README

- [ ] Create `README.md`.
- [ ] Explain the tornado-detection problem and TorNet dataset.
- [ ] Explain the explicit time, sweep, variable, azimuth, and range axes.
- [ ] Explain the final bidirectional architecture.
- [ ] Explain train, validation, and official-test isolation.
- [ ] Include the final results table:

| Model | Validation PR-AUC | Test PR-AUC | Test F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|
| DBZ+VEL 2D baseline | — | 0.3473 | 0.4032 | 0.8794 |
| Causal spatiotemporal V1 | 0.5670 | 0.4940 | 0.4949 | 0.9150 |
| Bidirectional V4 | **0.6103** | **0.5233** | **0.5273** | **0.9252** |

- [ ] Include installation instructions.
- [ ] Include model-bundle acquisition instructions.
- [ ] Include Python inference and CLI examples.
- [ ] Include the canonical reproduction path.
- [ ] Include limitations and intended-use warnings.
- [ ] State that the `0.953` threshold operates on an uncalibrated model score.
- [ ] State that V4 improved official-test PR-AUC by approximately `50.7%` over the original 2D baseline.

### Model card

- [ ] Create `MODEL_CARD.md`.
- [ ] State that the model is an offline research detector.
- [ ] State that bidirectional processing uses future frames.
- [ ] State that the model is not a real-time warning system.
- [ ] State that the model is not operationally validated.
- [ ] Document year-to-year performance variation.
- [ ] Note that 2020 and 2021 remain difficult years.
- [ ] State that model scores are not calibrated probabilities.
- [ ] Document false-negative and false-positive risks.
- [ ] Document dataset scope, provenance, and licensing.
- [ ] Disclose that the official test was evaluated after V1 and again after V4.
- [ ] Document suitable and unsuitable uses.

### Notebook index

- [ ] Create `notebooks/README.md`.
- [ ] Classify notebooks `00–03` as data foundation.
- [ ] Classify notebooks `04–19` as early baselines.
- [ ] Classify notebooks `20–27` as spatiotemporal development.
- [ ] Mark notebook `28` as final-model training.
- [ ] Mark notebook `29` as final official evaluation.
- [ ] Mark notebook `30` as final validation diagnostics.
- [ ] Mark notebook `31` as a rejected experiment.
- [ ] Document the canonical closeout path:

```text
00 → 01 → 02 → 03 → 16 → 27 → 28 → 29 → 30
```

- [ ] Leave historical notebooks in place to preserve experimental provenance.
- [ ] Explicitly mark noncanonical notebooks as historical or superseded.

### Attribution and licensing

- [ ] Choose a code license, such as MIT or Apache-2.0, only after deciding how the code may be reused.
- [ ] Confirm the TorNet dataset license and required attribution.
- [ ] Confirm whether trained weights may be redistributed.
- [ ] Add the TorNet citation.
- [ ] Add attribution for any retained course-derived material.
- [ ] Keep code licensing distinct from dataset and model-weight licensing.

## Pass 4 — CI, verification, and release

### Continuous integration

- [ ] Add `.github/workflows/test.yml`.
- [ ] Test Python 3.11 and 3.12.
- [ ] Run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev,ml]"
python -m compileall -q src tests
python -m pytest
python -m build --wheel
```

- [ ] Add `build` to the appropriate development dependency group.
- [ ] Confirm CI passes from a clean clone.

### Release verification

- [ ] Install the built wheel into a clean environment.
- [ ] Load a synthetic model bundle.
- [ ] Run a synthetic NetCDF prediction.
- [ ] Verify four finite frame scores and four binary predictions.
- [ ] Verify the frozen threshold is loaded from configuration.
- [ ] Verify strict checkpoint loading.
- [ ] Verify bundle checksums.
- [ ] Verify README metrics exactly match preserved JSON artifacts.
- [ ] Verify the official test is not executed by any automated workflow.
- [ ] Verify no Google Drive path is required for normal inference.

### Release

- [ ] Review the final diff for secrets, machine-specific paths, and generated junk.
- [ ] Create the final release archive.
- [ ] Write concise release notes.
- [ ] Create the `v1.0.0` tag manually.
- [ ] Publish the GitHub release manually.
- [ ] Attach the wheel and, if permitted, the model bundle.
- [ ] Verify release downloads and checksums.

## Definition of done

- [ ] A clean machine can install `tornet_detection-1.0.0-py3-none-any.whl`.
- [ ] A clean machine can run:

```bash
tornet-detect \
  --bundle tornet-detection-v1.0.0 \
  --input sample.nc \
  --output prediction.json
```

- [ ] The command produces four finite scores and four predictions.
- [ ] No notebook must be opened or edited for inference.
- [ ] No machine-specific or Google Drive path is required for inference.
- [ ] All unit and inference tests pass in CI.
- [ ] The V4 checkpoint loads exactly and strictly.
- [ ] The release bundle is complete and checksum-verified.
- [ ] README claims match preserved metrics.
- [ ] V5 is clearly documented as rejected.
- [ ] The official test is never evaluated again.
- [ ] Licensing and redistribution decisions are documented.

## Implementation order

- [ ] **Pass 1:** Final package cleanup, `1.0.0` configuration, inference API, CLI, and inference tests.
- [ ] **Pass 2:** Self-contained release bundle, metrics, checkpoint, normalization, and checksums.
- [ ] **Pass 3:** README, model card, notebook index, attribution, and licensing decision.
- [ ] **Pass 4:** CI, clean-install verification, release archive, and manual `v1.0.0` release.

