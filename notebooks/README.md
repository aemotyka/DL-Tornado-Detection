# Notebooks

The shortest path to the final result is:

```text
00 → 01 → 02 → 03 → 16 → 27 → 28 → 29 → 30
```

The other notebooks show earlier baselines, performance tests, and model versions that were later replaced.

## Data setup

| Notebook | Use |
|---|---|
| `00_dataset_audit.ipynb` | Check the source archives and dataset layout. |
| `01_build_year_manifests.ipynb` | Build yearly manifests. |
| `02_build_modeling_manifests.ipynb` | Build corrected manifests for the affected years. |
| `03_ml_data_smoke.ipynb` | Check the combined index and load a real NetCDF file. |

## Early baselines and loading work

| Notebook | Status | Use |
|---|---|---|
| `04_2013_io_benchmark.ipynb` | Earlier work | Benchmark 2013 archive reads. |
| `05_tiny_overfit.ipynb` | Earlier work | Check that the first model can fit a tiny sample. |
| `06_2013_train_normalization.ipynb` | Replaced | Compute the first two-variable normalization. |
| `07_2013_baseline.ipynb` | Replaced | Train the first 2013 baseline. |
| `08_2013_baseline_v2.ipynb` | Replaced | Revise the 2013 baseline. |
| `09_2013_official_test.ipynb` | Earlier result | Evaluate the 2013 baseline. |
| `10_file_tensor_io_benchmark.ipynb` | Earlier work | Benchmark file-level tensor reads. |
| `11_file_grouped_loader_benchmark.ipynb` | Earlier work | Benchmark grouped loading. |
| `12_all_year_normalization.ipynb` | Replaced | Compute the earlier all-year normalization. |
| `13_all_year_baseline_v1.ipynb` | Replaced | Train the first all-year baseline. |
| `14_all_year_baseline_v2.ipynb` | Replaced | Revise the all-year baseline. |
| `15_all_year_official_test.ipynb` | Earlier result | Evaluate the earlier all-year model. |
| `16_all6_normalization.ipynb` | Final path | Compute the six-variable normalization used by V4. |
| `17_all6_loader_benchmark.ipynb` | Earlier work | Benchmark the six-variable loader. |
| `18_all6_baseline_v1.ipynb` | Replaced | Train the six-variable 2D baseline. |
| `19_all6_official_test.ipynb` | Baseline result | Evaluate the six-variable 2D baseline. |

## Temporal models

| Notebook | Status | Use |
|---|---|---|
| `20_spatiotemporal_tiny_overfit.ipynb` | Earlier work | Check the first temporal model on a tiny sample. |
| `21_spatiotemporal_throughput.ipynb` | Earlier work | Benchmark temporal-model throughput. |
| `22_spatiotemporal_batch_scaling.ipynb` | Earlier work | Compare batch sizes. |
| `23_spatiotemporal_all_year_v1.ipynb` | Replaced | Train causal V1. |
| `24_spatiotemporal_official_test.ipynb` | Earlier result | Evaluate V1 on the test split. |
| `25_spatiotemporal_all_year_v2.ipynb` | Replaced | Train V2. |
| `26_spatiotemporal_all_year_v3.ipynb` | Replaced | Train V3. |
| `27_bidirectional_throughput.ipynb` | Final path | Benchmark the bidirectional model and select batch size 8. |

## Final run and last experiment

| Notebook | Status | Use |
|---|---|---|
| `28_bidirectional_all_year_v4.ipynb` | Final | Train V4 and select epoch 23. |
| `29_bidirectional_v4_official_test.ipynb` | Final | Evaluate V4 on the official test split. |
| `30_v4_validation_diagnostics.ipynb` | Final | Analyze validation errors and save prediction-level results. |
| `31_auxiliary_category_all_year_v5.ipynb` | Rejected | Test auxiliary category supervision. |

V5 validation PR-AUC fell to `0.5779056354733761` and F1 fell to `0.5604611555089742`. Its learned frame-task weight grew to `1,982,829.25`, and training losses reached the millions. It was not run on the test split.
