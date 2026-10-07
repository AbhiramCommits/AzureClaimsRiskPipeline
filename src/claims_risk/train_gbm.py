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
    holdout_df = lake_client.read_parquet("claims/features/holdout").toPandas()
    spark.stop()

    feature_cols = [
        "log_tiv", "tiv_per_building_age", "building_age_winsor", "inspection_score_imputed",
        "prior_claim_count", "deductible", "report_lag_days", "state_freq", "occ_freq",
        "state_mean_loss", "state_loss_prob", "tiv_x_sprinkler", "sprinkler_flag"
    ]

    # Filter non-zero loss or train on all severity
    X_train = train_df[feature_cols]
    y_train = train_df["loss_amount"]

    X_valid = valid_df[feature_cols]
    y_valid = valid_df["loss_amount"]

    # 1. Baseline: Global Mean & Tweedie GLM
    print("Training baseline Tweedie GLM...")
    glm = TweedieRegressor(power=1.5, alpha=1.0, max_iter=500)
    glm.fit(X_train, y_train)
    glm_preds = glm.predict(X_valid)
    glm_rmse = float(np.sqrt(np.mean((glm_preds - y_valid) ** 2)))

    with mlflow.start_run(run_name="baseline-glm"):
        mlflow.log_param("model_type", "TweedieRegressor")
        mlflow.log_metric("valid_rmse", glm_rmse)

    # 2. LightGBM Tweedie / Gamma
    print("Training LightGBM Severity model...")
    with mlflow.start_run(run_name="lightgbm-severity") as run:
        run_id = run.info.run_id
        
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "seed": settings.seed,
            "verbose": -1
        }
        mlflow.log_params(params)

        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

        model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[valid_data],
            callbacks=[lgb.early_stopping(stopping_rounds=20)]
        )

        preds = model.predict(X_valid)
        rmse = float(np.sqrt(np.mean((preds - y_valid) ** 2)))
        mae = float(np.mean(np.abs(preds - y_valid)))

        mlflow.log_metric("valid_rmse", rmse)
        mlflow.log_metric("valid_mae", mae)

        mlflow.lightgbm.log_model(model, "model")

        # Train Quantile boosters (alpha=0.5, 0.9, 0.99)
        quantiles = [0.5, 0.9, 0.99]
        quantile_models = {}
        for q in quantiles:
            q_params = {
                "objective": "quantile",
                "alpha": q,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "verbose": -1
            }
            q_model = lgb.train(q_params, train_data, num_boost_round=200, valid_sets=[valid_data], callbacks=[lgb.early_stopping(20)])
            quantile_models[q] = q_model
            mlflow.lightgbm.log_model(q_model, f"quantile_{int(q*100)}")

    # Save registry metadata
    registry_data = {
        "best_run_id": run_id,
        "model_name": "claims-severity",
        "model_uri": f"runs:/{run_id}/model",
        "metrics": {
            "valid_rmse": rmse,
            "valid_mae": mae,
            "glm_valid_rmse": glm_rmse
        }
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "registry.json", "w") as f:
        json.dump(registry_data, f, indent=4)

    print("LightGBM training complete. Registry saved to results/registry.json")

if __name__ == "__main__":
    main()
