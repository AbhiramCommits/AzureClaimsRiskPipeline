import contextlib
import json
from pathlib import Path

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

from claims_risk.config import Settings


def main():
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    registry_path = Path("results/registry.json")
    if not registry_path.exists():
        print("results/registry.json not found. Run training first.")
        return
    registry = json.loads(registry_path.read_text())

    run_id = registry["best_run_id"]
    model_name = registry["model_name"]
    model_uri = registry["model_uri"]

    client = MlflowClient()
    with contextlib.suppress(MlflowException):  # already registered
        client.create_registered_model(model_name)

    version = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
    # MLflow 3.x prefers aliases over legacy stages.
    with contextlib.suppress(MlflowException):
        client.set_registered_model_alias(model_name, "champion", version.version)
    with contextlib.suppress(MlflowException):
        client.transition_model_version_stage(model_name, version.version, "Staging")

    registry["registered_version"] = version.version
    registry["alias"] = "champion"
    registry_path.write_text(json.dumps(registry, indent=4))
    print(f"Registered {model_name} version {version.version} (alias: champion) from run {run_id}")


if __name__ == "__main__":
    main()
