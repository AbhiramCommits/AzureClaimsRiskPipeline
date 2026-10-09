import json
from pathlib import Path

import mlflow
import mlflow.lightgbm

from claims_risk.config import Settings
from claims_risk.data import build_matrix, load_split


def export_artifacts(settings, registry_path="results/registry.json", out_dir="results/model_artifacts"):
    """Export the registered LightGBM boosters and category map for serving.

    Keeps the scoring service self-contained (no MLflow needed at runtime) while
    MLflow remains the source of truth for tracking and the model registry.
    """
    registry = json.loads(Path(registry_path).read_text())
    run_id = registry["best_run_id"]
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_df = load_split(settings, "train")
    _, card = build_matrix(train_df)
    (out / "card.json").write_text(json.dumps(card, indent=4))

    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    model.save_model(str(out / "severity.txt"), num_iteration=model.best_iteration)
    for q in (0.5, 0.9, 0.99):
        qm = mlflow.lightgbm.load_model(f"runs:/{run_id}/quantile_{int(q*100)}")
        qm.save_model(str(out / f"quantile_{int(q*100)}.txt"), num_iteration=qm.best_iteration)
    print(f"Exported boosters and category map to {out}")


if __name__ == "__main__":
    export_artifacts(Settings())
