import os
import json
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
from pyspark.sql import SparkSession

from claims_risk.config import Settings
from claims_risk.storage import get_lake_client
from claims_risk.models.tabular import TabularSeverityNet, GammaNLLLoss

def main():
    settings = Settings()
    lake_client = get_lake_client(settings)
    
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("claims-severity-risk")

    spark = SparkSession.builder.appName("AzureClaimsRisk-TrainTorch").getOrCreate()
    train_df = lake_client.read_parquet("claims/features/train").toPandas()
    valid_df = lake_client.read_parquet("claims/features/valid").toPandas()
    spark.stop()

    num_cols = [
        "log_tiv", "tiv_per_building_age", "building_age_winsor", "inspection_score_imputed",
        "prior_claim_count", "deductible", "report_lag_days", "state_freq", "occ_freq",
        "state_mean_loss", "state_loss_prob", "tiv_x_sprinkler", "sprinkler_flag"
    ]

    # Categorical columns encoded as integer indices
    train_df["state_idx"] = train_df["state"].astype("category").cat.codes
    valid_df["state_idx"] = valid_df["state"].astype("category").cat.codes
    cat_cols = ["state_idx"]
    cat_cardinalities = [train_df["state_idx"].nunique() + 1]
    embedding_dims = [8]

    X_train_num = torch.tensor(train_df[num_cols].values, dtype=torch.float32)
    X_train_cat = torch.tensor(train_df[cat_cols].values, dtype=torch.long)
    y_train = torch.tensor(train_df["loss_amount"].values, dtype=torch.float32).unsqueeze(1)

    X_valid_num = torch.tensor(valid_df[num_cols].values, dtype=torch.float32)
    X_valid_cat = torch.tensor(valid_df[cat_cols].values, dtype=torch.long)
    y_valid = torch.tensor(valid_df["loss_amount"].values, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_num, X_train_cat, y_train)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)

    model = TabularSeverityNet(num_numerics=len(num_cols), cat_cardinalities=cat_cardinalities, embedding_dims=embedding_dims)
    criterion = GammaNLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print("Training PyTorch Tabular Severity Net on CPU...")
    with mlflow.start_run(run_name="pytorch-tabular-severity") as run:
        run_id = run.info.run_id
        epochs = 5
        for epoch in range(epochs):
            model.train()
            total_loss = 0.0
            for batch_num, batch_cat, batch_y in train_loader:
                optimizer.zero_grad()
                preds = model(batch_num, batch_cat)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * batch_num.size(0)
            
            epoch_loss = total_loss / len(train_dataset)
            mlflow.log_metric("train_loss", epoch_loss, step=epoch)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_valid_num, X_valid_cat)
            val_loss = criterion(val_preds, y_valid).item()
            val_rmse = float(torch.sqrt(torch.mean((val_preds - y_valid) ** 2)).item())

        mlflow.log_metric("valid_loss", val_loss)
        mlflow.log_metric("valid_rmse", val_rmse)
        mlflow.pytorch.log_model(model, "pytorch_model")

    print(f"PyTorch model training complete. Validation RMSE: {val_rmse:.4f}")

if __name__ == "__main__":
    main()
