# Project Closeout

## Final model

- [x] Use bidirectional V4 as the final model.
- [x] Freeze checkpoint epoch 23.
- [x] Freeze threshold `0.9532915353775024`.
- [x] Keep six variables: `DBZ`, `KDP`, `RHOHV`, `VEL`, `WIDTH`, and `ZDR`.
- [x] Keep two sweeps and four frames per file.
- [x] Keep top-5% spatial pooling.
- [x] Confirm 2,970,049 parameters.
- [x] Load the real V4 checkpoint strictly with package `1.0.0`.
- [x] Stop model development after V5 failed validation.
- [x] Do not run V5 on the official test split.

Final validation results:

| PR-AUC | Macro-year PR-AUC | F1 | ROC-AUC |
|---:|---:|---:|---:|
| 0.6103 | 0.6123 | 0.5861 | 0.9464 |

Final test results:

| PR-AUC | F1 | ROC-AUC | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.5233 | 0.5273 | 0.9252 | 0.5434 | 0.5122 |

| TP | FN | FP | TN |
|---:|---:|---:|---:|
| 2,002 | 1,907 | 1,682 | 120,277 |

## Package

- [x] Remove the V5-only category head.
- [x] Set package version `1.0.0`.
- [x] Add `artifacts/v1/model_config.json`.
- [x] Record variable order, channel order, coordinates, shape, seeds, epoch, and threshold.
- [x] Add `TornadoDetector.from_bundle()`.
- [x] Add `TornadoDetector.predict_file()`.
- [x] Add the `tornet-detect` command.
- [x] Validate bundle files and normalization metadata.
- [x] Load checkpoints with `strict=True`.
- [x] Add synthetic end-to-end inference tests.
- [x] Confirm all 42 tests pass locally.
- [x] Build and load the `1.0.0` wheel in Colab.
- [x] Load the real epoch-23 checkpoint on CPU.

## Documentation

- [x] Add the root README.
- [x] Add V4 model notes and yearly results.
- [x] Add an index covering notebooks 00–31.
- [x] Mark V5 as rejected.
- [x] Record the shortest notebook path:

```text
00 → 01 → 02 → 03 → 16 → 27 → 28 → 29 → 30
```

- [x] Remove the obsolete root notebook and PDF.
- [x] Cite the TorNet paper and official repository.

## Licensing and distribution

- [x] License this project's code under MIT.
- [x] Keep TorNet data out of the repository and release files.
- [x] State that this project's MIT license does not cover TorNet.
- [x] Record that the Zenodo dataset entry has no named standard license in its Rights field.
- [x] Keep the V4 checkpoint private.
- [x] Keep the private checkpoint and research artifacts in Drive.
- [x] Remove the old course-submission files; no third-party course material is distributed.

## Automation

- [x] Add a GitHub Actions workflow.
- [x] Test Python 3.11 and 3.12.
- [x] Compile the package, run tests, and build the wheel in CI.
- [x] Confirm the first GitHub Actions run passes.

- [x] Keep TorNet, Drive access, training, and official-test evaluation out of CI.

## Private archive bundle

- [ ] Create `tornet-detection-v1.0.0-private.tar.gz` for private archival storage.
- [ ] Include:

```text
tornet-detection-v1.0.0/
├── best_model.pt
├── model_config.json
├── normalization.json
├── validation_metrics.json
├── official_test_metrics.json
├── MODEL_CARD.md
├── README.md
├── DATA_AND_MODEL_LICENSE.md
├── SHA256SUMS
└── tornet_detection-1.0.0-py3-none-any.whl
```

- [ ] Copy the epoch-23 V4 checkpoint.
- [ ] Copy the six-variable normalization used by V4.
- [ ] Copy the saved validation and test metrics.
- [ ] Build a fresh wheel from the final source.
- [ ] Generate and verify SHA-256 checksums.
- [ ] Store the archive privately in Drive.

## Public release

- [ ] Review the repository for secrets, machine-specific paths, and generated files.
- [ ] Install the final wheel in a clean environment.
- [ ] Run a synthetic NetCDF prediction through the installed command.
- [ ] Confirm four finite scores and four binary predictions.
- [ ] Confirm normal inference needs no notebook or Drive path.
- [ ] Confirm README metrics match the saved JSON metrics.
- [ ] Write short release notes.
- [ ] Create the `v1.0.0` tag manually.
- [ ] Publish the GitHub release manually.
- [ ] Attach the wheel; do not attach TorNet data or the V4 checkpoint.

## Done when

- [ ] CI passes.
- [ ] The private archive is complete and checksum-verified.
- [ ] The wheel installs on a clean machine.
- [ ] The CLI completes a synthetic prediction from the installed wheel.
- [ ] The public release contains code and documentation but no TorNet data or V4 weights.
