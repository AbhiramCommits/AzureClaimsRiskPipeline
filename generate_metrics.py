import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from claims_risk.config import Settings
from claims_risk.metrics import mae, rmse, gamma_deviance, pinball_loss, normalized_gini, calibration_stats

settings = Settings()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Evaluate").getOrCreate()
valid_df = spark.read.parquet(str(Path(settings.lake_root) / "claims/features/valid")).toPandas()
holdout_df = spark.read.parquet(str(Path(settings.lake_root) / "claims/features/holdout")).toPandas()
spark.stop()

feature_cols = [
    "log_tiv", "tiv_per_building_age", "building_age_winsor", "inspection_score_imputed",
    "prior_claim_count", "deductible", "report_lag_days", "state_freq", "occ_freq",
    "state_mean_loss", "state_loss_prob", "tiv_x_sprinkler", "sprinkler_flag"
]

import lightgbm as lgb
model_lgb_path = list(Path("mlruns").glob("**/model.lgb"))[0]
model = lgb.Booster(model_file=str(model_lgb_path))

results_metrics = {}
backtest_by_year = {}
figures_dir = Path("results/figures")
figures_dir.mkdir(parents=True, exist_ok=True)

for split_name, df in [("valid", valid_df), ("holdout", holdout_df)]:
    X = df[feature_cols]
    y_true = df["loss_amount"].values
    preds = model.predict(X)

    m = {
        "MAE": mae(y_true, preds),
        "RMSE": rmse(y_true, preds),
        "GammaDeviance": gamma_deviance(y_true, preds),
        "Pinball_0.5": pinball_loss(y_true, preds, 0.5),
        "Pinball_0.9": pinball_loss(y_true, preds, 0.9),
        "Pinball_0.99": pinball_loss(y_true, preds, 0.99),
        "NormalizedGini": normalized_gini(y_true, preds)
    }
    slope, intercept = calibration_stats(y_true, preds)
    m["CalibrationSlope"] = slope
    m["CalibrationIntercept"] = intercept

    results_metrics[split_name] = {"LightGBM": m}

    if split_name == "holdout":
        for yr in df["policy_year"].unique():
            sub = df[df["policy_year"] == yr]
            sub_y = sub["loss_amount"].values
            sub_preds = model.predict(sub[feature_cols])
            ae_ratio = float(np.sum(sub_y) / (np.sum(sub_preds) + 1e-6))
            backtest_by_year[str(int(yr))] = {
                "row_count": len(sub),
                "actual_total_loss": float(np.sum(sub_y)),
                "predicted_total_loss": float(np.sum(sub_preds)),
                "ae_ratio": ae_ratio
            }

results_dir = Path("results")
with open(results_dir / "metrics.json", "w") as f:
    json.dump(results_metrics, f, indent=4)

with open(results_dir / "backtest_by_year.json", "w") as f:
    json.dump(backtest_by_year, f, indent=4)

print("Metrics JSON successfully generated.")
