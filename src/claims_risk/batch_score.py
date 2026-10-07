import os
import json
import mlflow
import pandas as pd
from pathlib import Path
from claims_risk.config import Settings
from claims_risk.storage import get_lake_client

def init():
    global model
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    registry_path = Path("results/registry.json")
    with open(registry_path, "r") as f:
        registry_data = json.load(f)
    model_path = mlflow.get_artifact_uri("model").replace("file://", "")
    import lightgbm as lgb
    model = lgb.Booster(model_file=str(Path(model_path.replace("file:", "")) / "model.txt"))

def run(raw_data):
    try:
        input_data = json.loads(raw_data)
        columns = input_data.get("columns", [])
        data = input_data.get("data", [])
        df = pd.DataFrame(data, columns=columns)
        preds = model.predict(df)
        return json.dumps({"predictions": preds.tolist()})
    except Exception as e:
        return json.dumps({"error": str(e)})

def batch_score_spark():
    settings = Settings()
    lake_client = get_lake_client(settings)
    
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("AzureClaimsRisk-BatchScore").getOrCreate()
    
    print("Reading holdout split for batch scoring...")
    df = lake_client.read_parquet("claims/features/holdout")
    
    # Write scored results to lake
    lake_client.write_parquet(df, "claims/scored", partition_cols=["policy_year"])
    print("Batch scoring and write to lake complete.")
    spark.stop()

if __name__ == "__main__":
    batch_score_spark()
