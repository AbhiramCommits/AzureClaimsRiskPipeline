import json
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from claims_risk.config import Settings
from claims_risk.data import build_matrix, load_split


def main():
    settings = Settings()

    import mlflow
    import mlflow.lightgbm
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    with open("results/registry.json") as f:
        registry = json.load(f)
    run_id = registry["best_run_id"]
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")

    t0 = time.time()
    holdout = load_split(settings, "holdout")
    X, _ = build_matrix(holdout)
    y = holdout["loss_amount"].values.astype("float64")
    n = len(X)

    preds = model.predict(X)
    elapsed = time.time() - t0
    throughput = n / elapsed if elapsed else 0.0

    out = holdout[["policy_year", "loss_amount"]].copy()
    out["predicted_severity"] = preds
    out["residual"] = y - preds

    out_dir = Path(settings.lake_root) / "claims" / "scored"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(out, preserve_index=False)
    pq.write_to_dataset(table, root_path=str(out_dir), partition_cols=["policy_year"])

    summary = {
        "scored_rows": int(n),
        "wall_clock_seconds": elapsed,
        "throughput_rows_per_sec": throughput,
        "output_path": str(out_dir),
    }
    with open("results/batch_score.json", "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Batch scored {n:,} rows in {elapsed:.1f}s ({throughput:,.0f} rows/s) -> {out_dir}")


if __name__ == "__main__":
    main()
