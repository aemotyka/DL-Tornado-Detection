# Notebook Index

The canonical closeout path is:

```text
00 → 01 → 02 → 03 → 16 → 27 → 28 → 29 → 30
```

Only that path is required to understand the final dataset preparation, six-variable normalization, bidirectional model, final training run, official evaluation, and validation diagnostics. Other notebooks are retained as experimental provenance.

## Data foundation

| Notebook | Status | Purpose |
|---|---|---|
| `00_dataset_audit.ipynb` | Canonical | Audit source archives and dataset structure. |
| `01_build_year_manifests.ipynb` | Canonical | Build raw yearly manifests. |
| `02_build_modeling_manifests.ipynb` | Canonical | Build corrected modeling manifests where required. |
| `03_ml_data_smoke.ipynb` | Canonical | Load the canonical index and verify real NetCDF tensor I/O. |

## Early baselines and infrastructure

Notebooks 04–19 establish I/O performance, normalization, training mechanics, and 2D baselines. They are historical or supporting work except for notebook 16, whose normalization artifact is used by V4.

| Notebook | Status | Purpose |
|---|---|---|
| `04_2013_io_benchmark.ipynb` | Historical | Initial 2013 archive I/O benchmark. |
| `05_tiny_overfit.ipynb` | Historical | Early small-sample overfit check. |
| `06_2013_train_normalization.ipynb` | Superseded | Two-variable 2013 normalization. |
| `07_2013_baseline.ipynb` | Superseded | Initial 2013 baseline. |
| `08_2013_baseline_v2.ipynb` | Superseded | Revised 2013 baseline. |
| `09_2013_official_test.ipynb` | Historical | Early 2013 official-test evaluation. |
| `10_file_tensor_io_benchmark.ipynb` | Historical | File-level tensor I/O benchmark. |
| `11_file_grouped_loader_benchmark.ipynb` | Historical | File-grouped loader benchmark. |
| `12_all_year_normalization.ipynb` | Superseded | Earlier all-year normalization. |
| `13_all_year_baseline_v1.ipynb` | Superseded | Initial all-year baseline. |
| `14_all_year_baseline_v2.ipynb` | Superseded | Revised all-year baseline. |
| `15_all_year_official_test.ipynb` | Historical | Earlier all-year official-test evaluation. |
| `16_all6_normalization.ipynb` | Canonical supporting artifact | Produce the six-variable normalization used by V4. |
| `17_all6_loader_benchmark.ipynb` | Historical | Six-variable loader benchmark. |
| `18_all6_baseline_v1.ipynb` | Superseded | Six-variable 2D baseline training. |
| `19_all6_official_test.ipynb` | Historical baseline | Official evaluation of the six-variable 2D baseline. |

## Spatiotemporal development

Notebooks 20–27 develop the explicit temporal and sweep-aware architecture. Notebook 27 is canonical because it fixes the practical batch size for final V4 training; the earlier notebooks are historical development evidence.

| Notebook | Status | Purpose |
|---|---|---|
| `20_spatiotemporal_tiny_overfit.ipynb` | Historical | Verify the first spatiotemporal model can overfit a tiny sample. |
| `21_spatiotemporal_throughput.ipynb` | Historical | Benchmark initial temporal-model throughput. |
| `22_spatiotemporal_batch_scaling.ipynb` | Historical | Measure batch-size scaling. |
| `23_spatiotemporal_all_year_v1.ipynb` | Superseded | Train causal spatiotemporal V1. |
| `24_spatiotemporal_official_test.ipynb` | Historical | Official-test evaluation of V1. |
| `25_spatiotemporal_all_year_v2.ipynb` | Superseded | Train the V2 refinement. |
| `26_spatiotemporal_all_year_v3.ipynb` | Superseded | Train the V3 refinement. |
| `27_bidirectional_throughput.ipynb` | Canonical | Benchmark the final bidirectional architecture and select batch size 8. |

## Final and post-final experiments

| Notebook | Status | Purpose |
|---|---|---|
| `28_bidirectional_all_year_v4.ipynb` | **Final training** | Train V4 and freeze epoch 23 using validation evidence. |
| `29_bidirectional_v4_official_test.ipynb` | **Final official evaluation** | Evaluate frozen V4 once on the official test split. |
| `30_v4_validation_diagnostics.ipynb` | **Final diagnostics** | Analyze V4 validation errors and preserve prediction-level diagnostics. |
| `31_auxiliary_category_all_year_v5.ipynb` | **Rejected experiment** | Test auxiliary category supervision; retain as failure evidence only. |

V5 is not part of the release model. Its validation PR-AUC fell to `0.5779056354733761`, its validation F1 fell to `0.5604611555089742`, and its learned frame-task weight grew to `1,982,829.25`, producing numerically unhealthy million-scale losses. It was not evaluated on the official test.

Historical and superseded notebooks remain in place so the development record is inspectable. Their checkpoints, thresholds, paths, and model definitions are not part of the supported `1.0.0` runtime contract.
