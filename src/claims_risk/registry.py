import os
import json
import mlflow
from pathlib import Path
from claims_risk.config import Settings

def main():
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    registry_path = Path("results/registry.json")
    if not registry_path.exists():
        print("Registry file not found. Run training first.")
        return

    with open(registry_path, "r") as f:
        registry_data = json.load(f)

    run_id = registry_data["best_run_id"]
    model_name = registry_data["model_name"]
    model_uri = registry_data["model_uri"]

    print(f"Registering MLflow model from run {run_id} under name '{model_name}'...")
    
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(model_name)
    except Exception:
        pass

    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )

    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Staging"
    )

    print(f"Model version {model_version.version} registered and staged to Staging.")

if __name__ == "__main__":
    main()
