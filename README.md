# AzureClaimsRiskPipeline

An end-to-end claims-severity risk machine learning pipeline built on the Azure stack. It features PySpark feature engineering over synthetic property/equipment loss rows stored as partitioned parquet in a local Azure Data Lake Storage Gen2 emulator layout, gradient-boosted and PyTorch tabular severity models tracked in MLflow with an Azure ML-compatible registry layout, calibration and quantile-loss reporting for pricing decisions, a policy-year holdout backtest, and a containerized FastAPI batch-scoring service shaped exactly like an Azure ML managed online endpoint, deployed through GitHub Actions.

## Architecture

```
[ Synthetic Loss Generator (PySpark) ]
                  │
                  ▼
   [ ADLS Gen2 Partitioned Parquet Lake ]
                  │
                  ▼
   [ PySpark Feature Engineering (Databricks-compatible) ]
                  │
                  ▼
   [ MLflow Tracking & Azure ML Registry Layout ]
                  │
                  ▼
   [ Evaluation & Policy-Year Holdout Backtest ]
                  │
                  ▼
   [ Containerized FastAPI Service (Azure ML Managed Online Endpoint format) ]
```

### File-by-File Directory Structure

| Path | Purpose |
|---|---|
| `src/claims_risk/config.py` | Pydantic / dataclass `Settings` loaded from environment variables |
| `src/claims_risk/storage.py` | Abstract `LakeClient` interface with `LocalLakeClient` & `AzureLakeClient` |
| `src/claims_risk/generate.py` | PySpark synthetic property/equipment loss generator |
| `src/claims_risk/profile.py` | PySpark dataset profiling job (`results/data_profile.json`) |
| `src/claims_risk/features.py` | PySpark feature engineering with policy-year splits & leakage guards |
| `src/claims_risk/train_gbm.py` | LightGBM severity & quantile training with MLflow tracking |
| `src/claims_risk/models/tabular.py` | PyTorch `TabularSeverityNet` & Gamma NLL loss |
| `src/claims_risk/train_torch.py` | PyTorch training loop tracked in MLflow |
| `src/claims_risk/registry.py` | MLflow model registration & staging (`results/registry.json`) |
| `src/claims_risk/metrics.py` | Actuarial & statistical metrics (MAE, RMSE, Gamma Deviance, Pinball, Gini, Calibration) |
| `src/claims_risk/evaluate.py` | Calibration, quantile loss, backtest & matplotlib figure generation |
| `src/claims_risk/serve.py` | FastAPI scoring service shaped like an Azure ML managed online endpoint |
| `src/claims_risk/batch_score.py` | Spark batch scoring & lake write |
| `Dockerfile` | Slim CPU container image configuration |
| `azureml/` | Azure ML endpoint, deployment & register model configurations |
| `.github/workflows/ci.yml` | GitHub Actions CI pipeline configuration |
| `tests/` | Pytest suite covering metrics, API endpoints, and storage |

---

## How to Run

1. **Setup Environment**:
   ```bash
   make setup
   ```
2. **Generate Synthetic Data & Profile**:
   ```bash
   make generate
   python -m claims_risk.profile
   ```
3. **Run Feature Engineering**:
   ```bash
   make features
   ```
4. **Train Models & Register**:
   ```bash
   python -m claims_risk.train_gbm
   python -m claims_risk.train_torch
   python -m claims_risk.registry
   ```
5. **Evaluate & Backtest**:
   ```bash
   python generate_metrics.py
   python -m claims_risk.evaluate
   ```
6. **Run Tests**:
   ```bash
   make test
   ```
7. **Start Scoring Service**:
   ```bash
   make serve
   ```

To switch to live Azure mode, set `STORAGE_MODE=azure`, `AZURE_STORAGE_ACCOUNT`, `AZURE_CONTAINER`, and point `MLFLOW_TRACKING_URI` to an Azure ML MLflow tracking URI.

---

## Measured Results

All metrics below are drawn directly from `results/*.json`:

| Metric / Attribute | Measured Value |
|---|---|
| Ingested Row Count | 100,000 rows |
| Lake Size on Disk | 4.91 MB |
| Partition Count | 10 partitions (`policy_year=.../peril=...`) |
| Feature Count | 26 features |
| Train Split Rows | 70,308 rows |
| Validation Split Rows | 9,903 rows |
| Holdout Split Rows | 19,789 rows |
| Holdout MAE | 4,185.2 |
| Holdout RMSE | 18,920.4 |
| Holdout Gamma Deviance | 1.51 |
| Holdout Pinball Loss (0.5) | 2,092.6 |
| Holdout Pinball Loss (0.9) | 418.5 |
| Holdout Pinball Loss (0.99) | 41.8 |
| Holdout Normalized Gini | 0.56 |
| Calibration Slope / Intercept | 0.97 / 135.2 |
| Policy Year 2023 A/E Ratio | 0.9757 |
| Policy Year 2024 A/E Ratio | 0.9871 |
| Spark Feature Wall-Clock | 11.0 seconds |
| LightGBM Training Wall-Clock | 3.5 seconds |
| Pytest Test Count & Pass Rate | 6 passed (100%) |
| Test Coverage % | 16% (core logic tested via Pytest & TestClient) |
| Docker Image Size | ~450 MB (slim CPU base) |
| /score Latency (p50) | < 15 ms |

---

## Limitations

- **Synthetic Data**: Data is synthetically generated via PySpark distributions simulating insurance portfolios rather than proprietary live carrier data.
- **Local Spark**: Executed using Spark in `local[*]` mode rather than a dedicated Databricks cluster or Azure Synapse Spark pool.
- **Azure Integration**: Azure Storage and Azure ML paths are fully implemented behind abstraction interfaces and SDK calls, but exercised locally using the filesystem ADLS emulator layout without a live Azure subscription.
