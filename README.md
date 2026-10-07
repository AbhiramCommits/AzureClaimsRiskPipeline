# AzureClaimsRiskPipeline

An end-to-end claims-severity risk modeling pipeline built on an Azure-shaped stack. It generates tens of millions of synthetic property/equipment loss records with a real frequency–severity process, stores them as partitioned Parquet in a local Azure Data Lake Storage Gen2 emulator layout, engineers features with PySpark in a Databricks-compatible module, trains gradient-boosted and PyTorch tabular severity models tracked in MLflow with an Azure-ML-style registry, reports calibration and quantile loss with a policy-year holdout backtest, and serves scores through a containerized FastAPI app shaped like an Azure ML managed online endpoint, wired up in GitHub Actions CI.

## Architecture

```
[ Synthetic loss generator (PySpark, local[*]) ]
                     |
                     v
[ ADLS Gen2 partitioned Parquet lake (local emulator / abfss:// in azure mode) ]
                     |
                     v
[ Databricks-compatible PySpark feature engineering, policy-year splits + leakage guards ]
                     |
                     v
[ MLflow tracking (file/sqlite local, Azure ML tracking URI in azure mode) + model registry ]
                     |
                     v
[ Evaluation: calibration, pinball/quantile loss, policy-year holdout backtest, figures ]
                     |
                     v
[ FastAPI scoring service in Docker (Azure ML managed-online-endpoint contract) + GitHub Actions CI ]
```

### File-by-file

| Path | Purpose |
|---|---|
| `src/claims_risk/config.py` | `Settings` dataclass loaded from env (`storage_mode`, `lake_root`, `n_rows`, `mlflow_tracking_uri`, `azure_*`). |
| `src/claims_risk/storage.py` | `LakeClient` abstraction: `LocalLakeClient` (filesystem ADLS layout) and `AzureLakeClient` (DataLakeServiceClient + `abfss://` + `DefaultAzureCredential`); `get_lake_client`. |
| `src/claims_risk/generate.py` | PySpark synthetic loss generator (frequency ~ Poisson, severity ~ Gamma with log-link mean + heavy tail + zero-inflation). |
| `src/claims_risk/profile.py` | PySpark data profiler → `results/data_profile.json`. |
| `src/claims_risk/features.py` | PySpark feature engineering, policy-year splits, leakage assertions → `results/feature_report.json`. |
| `src/claims_risk/data.py` | Memory-efficient pyarrow parquet loader + categorical/target matrix builders shared by training and evaluation. |
| `src/claims_risk/baselines.py` | Global-mean and Tweedie-GLM baselines (isolated process to avoid the macOS OpenMP conflict). |
| `src/claims_risk/train_gbm.py` | LightGBM Tweedie severity model + quantile trio (0.5/0.9/0.99) with early stopping, feature-importance plot, MLflow logging. |
| `src/claims_risk/models/tabular.py` | PyTorch `TabularSeverityNet` (categorical embeddings + BatchNorm MLP + log-link head) and `GammaNLLLoss`. |
| `src/claims_risk/train_torch.py` | PyTorch training loop (AdamW, cosine LR, early stopping, seed control) logged to MLflow. |
| `src/claims_risk/registry.py` | Registers the winning run as `claims-severity` and sets the `champion` alias. |
| `src/claims_risk/export_model.py` | Exports the boosters + category map to `results/model_artifacts/` for self-contained serving. |
| `src/claims_risk/metrics.py` | MAE, RMSE, Gamma deviance, pinball loss, normalized Gini, calibration slope/intercept (numpy only). |
| `src/claims_risk/evaluate.py` | Scores validation + holdout, writes `metrics.json`, `backtest_by_year.json`, four figures, and `model_comparison.md`. |
| `src/claims_risk/serve.py` | FastAPI service implementing the Azure ML `{"input_data": {...}}` contract with `/`, `/health`, `/score`. |
| `src/claims_risk/batch_score.py` | Batch-scores the holdout split to the lake and reports throughput. |
| `azureml/score.py` | `init()`/`run()` entry script for a managed online endpoint. |
| `azureml/register_model.py` | `azure-ai-ml` / `azureml-core` registration path guarded by env vars. |
| `azureml/endpoint.yml`, `azureml/deployment.yml` | Managed online endpoint + deployment definitions. |
| `databricks/job_severity_features.json` | Databricks job definition for the PySpark feature module. |
| `Dockerfile`, `requirements-serve.txt` | Slim CPU serving image (non-root, HEALTHCHECK). |
| `.github/workflows/ci.yml` | CI: Python 3.11 + Java, `pip install`, ruff, `pytest --cov`, `docker build`. |
| `tests/` | Metrics, pinball, storage, leakage, small end-to-end, and FastAPI TestClient tests. |

## How to run

```bash
make setup        # pip install -r requirements.txt && pip install -e .
make generate     # python -m claims_risk.generate && python -m claims_risk.profile
make features     # python -m claims_risk.features
make train        # baselines -> train_gbm -> registry -> train_torch
make evaluate     # python -m claims_risk.evaluate
make test         # pytest -q --cov=src/claims_risk
make serve        # uvicorn claims_risk.serve:app --host 0.0.0.0 --port 8000
make all          # setup generate features train evaluate test
```

Docker:

```bash
docker build -t claims-risk-service:latest .
docker run -p 8000:8000 claims-risk-service:latest
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/score -H 'Content-Type: application/json' \
  -d '{"input_data":{"columns":["log_tiv","tiv_per_building_age","building_age_winsor","inspection_score_imputed","prior_claim_count","deductible","report_lag_days","state_freq","occ_freq","state_mean_loss","state_loss_prob","tiv_x_sprinkler","sprinkler_flag"],"data":[[12.5,50000,15,85,0,1000,5,120,45,5000,0.12,12.5,1]]}}'
```

Flip to a live Azure backend (code paths implemented, not exercised in CI):

```bash
export STORAGE_MODE=azure
export AZURE_STORAGE_ACCOUNT=<account>       # -> abfss://<container>@<account>.dfs.core.windows.net/...
export AZURE_CONTAINER=<container>
export AZURE_WORKSPACE=<workspace> AZURE_SUBSCRIPTION=<sub> AZURE_RESOURCE_GROUP=<rg>
export MLFLOW_TRACKING_URI=azureml://<region>.api.azureml.ms/mlflow/v1.0/subscriptions/<sub>/resourceGroups/<rg>/providers/Microsoft.MachineLearningServices/workspaces/<ws>
```

## Results

Every number below is taken from `results/*.json`, produced on this machine. Data volume target was 30,000,000 rows; this volume completed locally and is what is reported.

### Data and features

| Metric | Value |
|---|---|
| Rows generated | 30,000,000 |
| Generation wall-clock | 76.79 s |
| Raw lake size on disk | 676,363,005 bytes (~645 MiB) |
| Raw partition count | 10 `policy_year` partitions (× 5 `peril`), 4,800 files |
| Profiling wall-clock | 60.37 s |
| Mean loss | 7,084.64 |
| Median / p90 / p99 / max loss | 0.0 / 12,139.63 / 28,424,972.93 / 28,424,972.93 |
| Zero-loss share | 0.7972 |
| Gini of loss concentration | 0.7059 |
| Feature count | 27 |
| Rows: train / valid / holdout | 21,001,550 / 2,998,218 / 6,000,232 |
| Spark feature wall-clock | 120.21 s |
| Leakage assertions passed | true (train ≤ 2021, valid = 2022, holdout ≥ 2023; policy IDs disjoint) |

### Validation (2022) — model selection

| Model | RMSE | MAE | Best iter | Train wall-clock |
|---|---|---|---|---|
| global mean | 85,341.28 | 11,561.98 | – | – |
| Tweedie GLM | 84,832.76 | 10,175.24 | – | 3.42 s |
| LightGBM (Tweedie) | 84,800.34 | 10,569.92 | 87 | 127.30 s |
| PyTorch TabularSeverityNet | – | – | – | 405.96 s (best val loss 9.2805, 3,000,000-row subsample) |

Model selection is by validation RMSE, so LightGBM is registered as the winner (`claims-severity` version 3, alias `champion`).

### Holdout (2023–2024) — policy-year backtest

| Model | MAE | RMSE | Gamma deviance | Pinball 0.5 | Pinball 0.9 | Pinball 0.99 | Norm. Gini | Calib slope | Calib intercept |
|---|---|---|---|---|---|---|---|---|---|
| global mean | 11,661.7 | 80,945.3 | 35.9062 | 5,830.8 | 5,816.7 | 5,813.5 | 0.0000 | 0.4975 | 3,534.3 |
| Tweedie GLM | 10,294.1 | 80,466.7 | 34.8987 | 5,147.1 | 5,574.9 | 5,671.1 | 0.1236 | 1.3284 | -900.4 |
| LightGBM | 10,698.5 | 80,388.1 | 34.7896 | 3,534.3 | 4,827.4 | 2,985.1 | 0.1181 | 1.0951 | -363.6 |

LightGBM improvement over baselines (holdout):

| Metric | vs global mean | vs Tweedie GLM |
|---|---|---|
| MAE | 8.26% | -3.93% |
| RMSE | 0.69% | 0.10% |
| Gamma deviance | 3.11% | 0.31% |

Top-decile lift (LightGBM, holdout): **4.0822** (decile mean actual loss ranges 645.67 → 28,855.41).

Actual-to-expected ratio by policy year (holdout):

| Year | Rows | global mean A/E | Tweedie GLM A/E | LightGBM A/E |
|---|---|---|---|---|
| 2023 | 2,997,901 | 0.9973 | 1.1810 | 1.0439 |
| 2024 | 3,002,331 | 0.9927 | 1.1756 | 1.0391 |

The GLM wins on holdout MAE and normalized Gini because the synthetic severity is generated with a log-link mean, which a log-link GLM recovers well; LightGBM wins on RMSE, Gamma deviance, and pinball loss over the GLM, and on all metrics over the mean.

### Serving and CI

| Metric | Value |
|---|---|
| Batch scoring | 6,000,232 rows in 12.60 s (476,067 rows/s) |
| Docker image size | 868 MB |
| `/score` latency (300 local requests) | p50 1.80 ms, p95 3.43 ms, p99 5.20 ms, 476 req/s |
| Sample `/score` response | `results/api_sample.json` |
| Tests | 11 passed |
| Test coverage | 34% (`--cov=src/claims_risk`) |

## Limitations

- **Synthetic data.** Losses are generated from a simulated frequency–severity process, not a real insurer book. The heavy tail is injected as random noise, so it is largely irreducible and dominates RMSE for every model.
- **Local Spark, not a cluster.** Everything runs in `local[*]`; the Databricks job JSON is provided but not executed on a real cluster.
- **Azure paths implemented, not exercised.** `AzureLakeClient`, the `abfss://` builder, `azureml/score.py`, and `azureml/register_model.py` are real code guarded by env vars, but CI and the reported numbers use the local filesystem emulator and a local MLflow tracking store (sqlite). No Azure subscription was used.
- **macOS OpenMP note.** scikit-learn and LightGBM cannot share a process here, so the GLM baseline runs in its own process (`claims_risk.baselines`) and metrics avoid scikit-learn; the GLM is fit on a 3,000,000-row subsample with a 99.9th-percentile-clipped target. The PyTorch model trains on a 3,000,000-row subsample.
