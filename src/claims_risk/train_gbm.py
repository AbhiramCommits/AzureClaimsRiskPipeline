import os
import json
import mlflow
import lightgbm as lgb
import pandas as pd
import numpy as np
from pathlib import Path
from pyspark.sql import SparkSession
from sklearn.linear_model import TweedieRegressor
from claims_risk.config import Settings
from claims_risk.storage import get_lake_client

def main():
    settings = Settings()
    lake_client = get_lake_client(settings)
    
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("claims-severity-risk")

    spark = SparkSession.builder.appName("AzureClaimsRisk-TrainGBM").getOrCreate()

    print("Reading feature splits from lake...")
    train_df = lake_client.read_parquet("claims/features/train").toPandas()
    valid_df = lake_client.read_parquet("claims/features/valid").toPandas()
    spark.stop()

    feature_cols = [
        "log_tiv", "tiv_per_building_age", "building_age_winsor", "inspection_score_imputed",
        "prior_claim_count", "deductible", "report_lag_days", "state_freq", "occ_freq",
        "state_mean_loss", "state_loss_prob", "tiv_x_sprinkler", "sprinkler_flag"
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["loss_amount"]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["loss_amount"]

    print("Training LightGBM Severity model...")
    with mlflow.start_run(run_name="lightgbm-severity") as run:
        run_id = run.info.run_id
        
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        model = lgb.train(
            {
                "objective": "regression",
                "metric": "rmse",
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1
            },
            train_data,
            num_boost_round=100,
            valid_sets=[valid_data]
        )

        preds = model.predict(X_valid)
        rmse = float(np.sqrt(np.mean((preds - y_valid) ** 2)))
        mae = float(np.mean(np.abs(preds - y_valid)))

        mlflow.log_metric("valid_rmse", rmse)
        mlflow.log_metric("valid_mae", mae)

        mlflow.lightgbm.log_model(model, "model")

    registry_data = {
        "best_run_id": run_id,
        "model_name": "claims-severity",
        "model_uri": f"runs:/{run_id}/model",
        "metrics": {
            "valid_rmse": rmse,
            "valid_mae": mae
        }
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "registry.json", "w") as f:
        json.dump(registry_data, f, indent=4)

    print("LightGBM training complete. Registry saved to results/registry.json")

if __name__ == "__main__":
    main()
