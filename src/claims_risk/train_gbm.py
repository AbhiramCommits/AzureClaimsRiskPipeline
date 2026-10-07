import json
import tempfile
import time
from pathlib import Path

import numpy as np

# IMPORTANT: import LightGBM before anything that could pull in scikit-learn's
# OpenMP runtime. On macOS, loading both OpenMP runtimes in one process
# deadlocks LightGBM training (see results/ notes in README).
import lightgbm as lgb
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from claims_risk.config import Settings
from claims_risk.data import load_split, build_matrix, apply_matrix, CATEGORICAL_FEATURES, TARGET_COL


def _metrics(y_true, y_pred):
    r = np.sqrt(np.mean((y_pred - y_true) ** 2))
    m = np.mean(np.abs(y_pred - y_true))
    return float(r), float(m)


def main():
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("claims-severity-risk")

    baselines_path = Path("results/baselines.json")
    if not baselines_path.exists():
        raise SystemExit("results/baselines.json missing. Run `python -m claims_risk.baselines` first.")
    baselines = json.loads(baselines_path.read_text())

    t0 = time.time()
    train_df = load_split(settings, "train")
    valid_df = load_split(settings, "valid")
    X_train, card = build_matrix(train_df)
    y_train = train_df[TARGET_COL].values.astype("float64")
    del train_df
    import gc
    gc.collect()
    X_valid = apply_matrix(valid_df, card)
    y_valid = valid_df[TARGET_COL].values.astype("float64")
    del valid_df
    gc.collect()
    print(f"Loaded train={len(X_train):,} valid={len(X_valid):,} in {time.time()-t0:.1f}s", flush=True)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c in X_train.columns]
    train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols, free_raw_data=False)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data, categorical_feature=cat_cols, free_raw_data=False)

    with mlflow.start_run(run_name="lightgbm-severity") as run:
        run_id = run.info.run_id
        t_lgb = time.time()
        params = {
            "objective": "tweedie",
            "tweedie_variance_power": 1.5,
            "metric": "rmse",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "max_depth": 8,
            "min_data_in_leaf": 200,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "seed": settings.seed,
            "verbose": -1,
            "num_threads": 6,
        }
        mlflow.log_params(params)
        model = lgb.train(
            params, train_data, num_boost_round=1000,
            valid_sets=[valid_data], valid_names=["valid"],
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
        )
        severity_seconds = time.time() - t_lgb

        preds = model.predict(X_valid, num_iteration=model.best_iteration)
        rmse, mae = _metrics(y_valid, preds)
        mlflow.log_metric("valid_rmse", rmse)
        mlflow.log_metric("valid_mae", mae)
        mlflow.log_metric("best_iteration", model.best_iteration)
        mlflow.log_metric("train_seconds", severity_seconds)

        with tempfile.TemporaryDirectory() as td:
            fig, ax = plt.subplots(figsize=(8, 8))
            lgb.plot_importance(model, ax=ax, max_num_features=20)
            fi_path = Path(td) / "feature_importance.png"
            fig.savefig(fi_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(fi_path))

        mlflow.lightgbm.log_model(model, name="model")

        for q in (0.5, 0.9, 0.99):
            t_q = time.time()
            q_params = {
                "objective": "quantile", "alpha": q, "metric": "quantile",
                "learning_rate": 0.05, "num_leaves": 63, "max_depth": 8,
                "min_data_in_leaf": 200, "seed": settings.seed, "verbose": -1,
                "num_threads": 6,
            }
            qm = lgb.train(
                q_params, train_data, num_boost_round=300,
                valid_sets=[valid_data],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(100)],
            )
            mlflow.lightgbm.log_model(qm, name=f"quantile_{int(q*100)}")
            print(f"quantile {q} trained in {time.time()-t_q:.1f}s", flush=True)

    print(f"lightgbm valid RMSE={rmse:,.1f} MAE={mae:,.1f} best_iter={model.best_iteration} ({severity_seconds:.1f}s)", flush=True)

    results = {
        "global_mean": baselines["global_mean"],
        "tweedie_glm": baselines["tweedie_glm"],
        "lightgbm": {
            "valid_rmse": rmse, "valid_mae": mae,
            "best_iteration": int(model.best_iteration),
            "train_seconds": severity_seconds,
            "run_id": run_id,
            "model_uri": f"runs:/{run_id}/model",
        },
    }
    winner_key = min(("global_mean", "tweedie_glm", "lightgbm"), key=lambda k: results[k]["valid_rmse"])
    winner = results[winner_key]

    registry = {
        "best_run_id": winner.get("run_id"),
        "best_model": winner_key,
        "model_name": "claims-severity",
        "model_uri": winner.get("model_uri", f"runs:/{winner.get('run_id')}/model"),
        "metrics": {"valid_rmse": winner["valid_rmse"], "valid_mae": winner["valid_mae"]},
        "all_models": results,
    }
    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/registry.json", "w") as f:
        json.dump(registry, f, indent=4)
    print(f"Winner by valid RMSE: {winner_key} ({winner['valid_rmse']:,.1f})", flush=True)


if __name__ == "__main__":
    main()
