import json
import time
from pathlib import Path

import numpy as np

from claims_risk.config import Settings
from claims_risk.data import load_split, build_matrix, apply_matrix, TARGET_COL


def main():
    """Compute the global-mean and Tweedie-GLM baselines.

    Runs in its own process: scikit-learn and LightGBM each load a separate
    OpenMP runtime on macOS, and mixing them in one process deadlocks.
    The GLM is fit on standardized features and a 99.9th-percentile-clipped
    target so the extreme right tail does not break LBFGS convergence.
    """
    settings = Settings()
    train_df = load_split(settings, "train")
    valid_df = load_split(settings, "valid")
    X_train, card = build_matrix(train_df)
    X_valid = apply_matrix(valid_df, card)
    y_train = train_df[TARGET_COL].values.astype("float64")
    y_valid = valid_df[TARGET_COL].values.astype("float64")

    out = {}

    gm = float(np.mean(y_train))
    out["global_mean"] = {
        "value": gm,
        "valid_rmse": float(np.sqrt(np.mean((gm - y_valid) ** 2))),
        "valid_mae": float(np.mean(np.abs(gm - y_valid))),
    }
    print(f"global mean valid RMSE={out['global_mean']['valid_rmse']:,.1f}", flush=True)

    from sklearn.linear_model import TweedieRegressor
    rng = np.random.default_rng(settings.seed)
    n = min(3_000_000, len(X_train))
    idx = rng.choice(len(X_train), size=n, replace=False)
    Xs = X_train.iloc[idx].values.astype("float64")
    ys = y_train[idx]
    mu = Xs.mean(axis=0)
    sd = Xs.std(axis=0) + 1e-9
    clip = float(np.quantile(ys, 0.999))
    ys_clipped = np.clip(ys, 0.0, clip)
    Xs_std = (Xs - mu) / sd

    glm = TweedieRegressor(power=1.5, alpha=0.1, max_iter=2000)
    t = time.time()
    glm.fit(Xs_std, ys_clipped)
    secs = time.time() - t
    pred = glm.predict((X_valid.values.astype("float64") - mu) / sd)

    out["tweedie_glm"] = {
        "coef": [float(c) for c in glm.coef_],
        "intercept": float(glm.intercept_),
        "feature_names": list(X_train.columns),
        "feature_mean": [float(m) for m in mu],
        "feature_std": [float(s) for s in sd],
        "target_clip_999": clip,
        "valid_rmse": float(np.sqrt(np.mean((pred - y_valid) ** 2))),
        "valid_mae": float(np.mean(np.abs(pred - y_valid))),
        "train_seconds": secs,
        "fit_sample_rows": int(n),
    }
    nz = int(np.sum(np.abs(np.asarray(glm.coef_)) > 1e-8))
    print(f"tweedie glm valid RMSE={out['tweedie_glm']['valid_rmse']:,.1f} "
          f"nonzero_coef={nz}/{len(glm.coef_)} ({secs:.1f}s)", flush=True)

    Path("results").mkdir(parents=True, exist_ok=True)
    with open("results/baselines.json", "w") as f:
        json.dump(out, f, indent=4)
    print("Baselines written to results/baselines.json", flush=True)


if __name__ == "__main__":
    main()
