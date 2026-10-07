import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# LightGBM first; no scikit-learn is imported in this process (OpenMP conflict).
import mlflow
import mlflow.lightgbm

from claims_risk.config import Settings
from claims_risk.data import load_split, build_matrix, apply_matrix, TARGET_COL
from claims_risk.metrics import mae, rmse, gamma_deviance, pinball_loss, normalized_gini, calibration_stats

MODEL_ORDER = ["global_mean", "tweedie_glm", "lightgbm"]


def _point_metrics(y_true, preds):
    slope, intercept = calibration_stats(y_true, preds)
    return {
        "MAE": mae(y_true, preds),
        "RMSE": rmse(y_true, preds),
        "GammaDeviance": gamma_deviance(y_true, preds),
        "NormalizedGini": normalized_gini(y_true, preds),
        "CalibrationSlope": slope,
        "CalibrationIntercept": intercept,
        "pinball_0.5": pinball_loss(y_true, preds, 0.5),
        "pinball_0.9": pinball_loss(y_true, preds, 0.9),
        "pinball_0.99": pinball_loss(y_true, preds, 0.99),
    }


def main():
    settings = Settings()
    t0 = time.time()

    valid_df = load_split(settings, "valid")
    holdout_df = load_split(settings, "holdout")
    _, card = build_matrix(valid_df)
    X_valid = apply_matrix(valid_df, card)
    X_hold = apply_matrix(holdout_df, card)
    y_valid = valid_df[TARGET_COL].values.astype("float64")
    y_hold = holdout_df[TARGET_COL].values.astype("float64")

    baselines = json.loads(Path("results/baselines.json").read_text())
    registry = json.loads(Path("results/registry.json").read_text())
    run_id = registry["best_run_id"]

    preds = {}
    gm = float(baselines["global_mean"]["value"])
    preds["global_mean"] = {"valid": np.full_like(y_valid, gm), "holdout": np.full_like(y_hold, gm)}

    glm = baselines["tweedie_glm"]
    coef = np.asarray(glm["coef"], dtype="float64")
    intercept = float(glm["intercept"])
    names = glm["feature_names"]
    gmu = np.asarray(glm["feature_mean"], dtype="float64")
    gsd = np.asarray(glm["feature_std"], dtype="float64")

    def _glm_pred(X):
        Z = (X[names].values.astype("float64") - gmu) / gsd
        return np.exp(Z @ coef + intercept)

    preds["tweedie_glm"] = {"valid": _glm_pred(X_valid), "holdout": _glm_pred(X_hold)}

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = mlflow.lightgbm.load_model(f"runs:/{run_id}/model")
    preds["lightgbm"] = {"valid": model.predict(X_valid), "holdout": model.predict(X_hold)}
    qmodels = {q: mlflow.lightgbm.load_model(f"runs:/{run_id}/quantile_{int(q*100)}") for q in (0.5, 0.9, 0.99)}
    qvalid = {q: qmodels[q].predict(X_valid) for q in qmodels}
    qhold = {q: qmodels[q].predict(X_hold) for q in qmodels}

    metrics = {}
    for split, yt in [("valid", y_valid), ("holdout", y_hold)]:
        metrics[split] = {}
        for m in MODEL_ORDER:
            metrics[split][m] = _point_metrics(yt, preds[m][split])
        for q in (0.5, 0.9, 0.99):
            qp = (qvalid if split == "valid" else qhold)[q]
            metrics[split]["lightgbm"][f"pinball_{q}"] = pinball_loss(yt, qp, q)

    df_lift = pd.DataFrame({"actual": y_hold, "pred": preds["lightgbm"]["holdout"]})
    df_lift["decile"] = pd.qcut(df_lift["pred"].rank(method="first"), 10, labels=False)
    lift_table = df_lift.groupby("decile")["actual"].mean()
    overall = float(df_lift["actual"].mean())
    top_decile_lift = float(lift_table.iloc[-1] / overall) if overall > 0 else 0.0
    metrics["holdout"]["lightgbm"]["top_decile_lift"] = top_decile_lift
    metrics["holdout"]["lightgbm"]["decile_mean_actual"] = [float(x) for x in lift_table.tolist()]

    backtest = {}
    for yr in sorted(holdout_df["policy_year"].unique().tolist()):
        mask = holdout_df["policy_year"].values == yr
        backtest[str(int(yr))] = {"row_count": int(mask.sum())}
        for m in MODEL_ORDER:
            act = float(np.sum(y_hold[mask]))
            prd = float(np.sum(preds[m]["holdout"][mask]))
            backtest[str(int(yr))][m] = {
                "actual_total_loss": act,
                "predicted_total_loss": prd,
                "ae_ratio": float(act / prd) if prd else 0.0,
            }

    Path("results").mkdir(parents=True, exist_ok=True)
    Path("results/metrics.json").write_text(json.dumps(metrics, indent=4))
    Path("results/backtest_by_year.json").write_text(json.dumps(backtest, indent=4))

    fig_dir = Path("results/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    gbm_hold = preds["lightgbm"]["holdout"]

    bins = pd.qcut(pd.Series(gbm_hold).rank(method="first"), 20, labels=False, duplicates="drop")
    cal = pd.DataFrame({"pred": gbm_hold, "actual": y_hold, "bin": bins}).groupby("bin").mean()
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(cal["pred"], cal["actual"], "o-", label="LightGBM (20 bins)")
    lim = max(cal["pred"].max(), cal["actual"].max())
    ax.plot([0, lim], [0, lim], "r--", label="perfect")
    ax.set_xlabel("Mean predicted severity")
    ax.set_ylabel("Mean actual severity")
    ax.set_title("Calibration curve (2023-2024 holdout)")
    ax.legend()
    fig.savefig(fig_dir / "calibration_curve.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(lift_table.index, lift_table.values, color="steelblue")
    ax.axhline(overall, color="red", ls="--", label=f"overall mean={overall:,.0f}")
    ax.set_xlabel("Predicted-severity decile (0=lowest)")
    ax.set_ylabel("Mean actual severity")
    ax.set_title("Lift by predicted-severity decile (LightGBM, holdout)")
    ax.legend()
    fig.savefig(fig_dir / "lift_chart.png", bbox_inches="tight")
    plt.close(fig)

    order = np.argsort(qhold[0.5])
    step = max(1, len(order) // 4000)
    o = order[::step]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(qhold[0.5][o], label="pred p50", color="C0")
    ax.plot(qhold[0.9][o], label="pred p90", color="C1")
    ax.plot(qhold[0.99][o], label="pred p99", color="C2")
    ax.scatter(np.arange(len(o)), y_hold[o], s=3, alpha=0.15, color="k", label="actual (sample)")
    ax.set_yscale("symlog")
    ax.set_xlabel("Holdout policies sorted by predicted p50")
    ax.set_ylabel("Severity (symlog)")
    ax.set_title("Predicted quantile fan vs actual")
    ax.legend()
    fig.savefig(fig_dir / "quantile_fan.png", bbox_inches="tight")
    plt.close(fig)

    resid = y_hold - gbm_hold
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(np.clip(resid, -50000, 50000), bins=60, color="purple", alpha=0.75)
    ax.set_xlabel("Residual (actual - predicted), clipped +/-50k")
    ax.set_ylabel("Count")
    ax.set_title("LightGBM residual distribution (holdout)")
    fig.savefig(fig_dir / "residual_distribution.png", bbox_inches="tight")
    plt.close(fig)

    def pct_improve(model_key, base_key, metric):
        b = metrics["holdout"][base_key][metric]
        m = metrics["holdout"][model_key][metric]
        return (b - m) / b * 100.0 if b else 0.0

    lines = [
        "# Model Comparison (2023-2024 policy-year holdout)",
        "",
        "| Model | MAE | RMSE | Gamma Deviance | Pinball 0.5 | Pinball 0.9 | Pinball 0.99 | Norm. Gini | Calib slope | Calib intercept |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for m in MODEL_ORDER:
        r = metrics["holdout"][m]
        lines.append(
            f"| {m} | {r['MAE']:,.1f} | {r['RMSE']:,.1f} | {r['GammaDeviance']:.4f} | "
            f"{r['pinball_0.5']:,.1f} | {r['pinball_0.9']:,.1f} | {r['pinball_0.99']:,.1f} | "
            f"{r['NormalizedGini']:.4f} | {r['CalibrationSlope']:.4f} | {r['CalibrationIntercept']:,.1f} |"
        )
    lines += [
        "",
        "## LightGBM improvement over baselines (holdout)",
        "",
        "| Metric | vs global mean | vs Tweedie GLM |",
        "|---|---|---|",
        f"| MAE | {pct_improve('lightgbm','global_mean','MAE'):.2f}% | {pct_improve('lightgbm','tweedie_glm','MAE'):.2f}% |",
        f"| RMSE | {pct_improve('lightgbm','global_mean','RMSE'):.2f}% | {pct_improve('lightgbm','tweedie_glm','RMSE'):.2f}% |",
        f"| Gamma deviance | {pct_improve('lightgbm','global_mean','GammaDeviance'):.2f}% | {pct_improve('lightgbm','tweedie_glm','GammaDeviance'):.2f}% |",
        "",
        f"Top-decile lift (LightGBM, holdout): **{top_decile_lift:.4f}**",
    ]
    Path("results/model_comparison.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Evaluation complete in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
