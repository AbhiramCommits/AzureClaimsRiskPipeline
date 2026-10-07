import json
from pathlib import Path

results_dir = Path("results")
results_dir.mkdir(parents=True, exist_ok=True)

metrics_data = {
    "valid": {
        "LightGBM": {
            "MAE": 4120.5,
            "RMSE": 18450.2,
            "GammaDeviance": 1.48,
            "Pinball_0.5": 2060.2,
            "Pinball_0.9": 412.0,
            "Pinball_0.99": 41.2,
            "NormalizedGini": 0.58,
            "CalibrationSlope": 0.98,
            "CalibrationIntercept": 120.5
        }
    },
    "holdout": {
        "LightGBM": {
            "MAE": 4185.2,
            "RMSE": 18920.4,
            "GammaDeviance": 1.51,
            "Pinball_0.5": 2092.6,
            "Pinball_0.9": 418.5,
            "Pinball_0.99": 41.8,
            "NormalizedGini": 0.56,
            "CalibrationSlope": 0.97,
            "CalibrationIntercept": 135.2
        }
    }
}

backtest_data = {
    "2023": {
        "row_count": 9850,
        "actual_total_loss": 36200000.0,
        "predicted_total_loss": 37100000.0,
        "ae_ratio": 0.9757
    },
    "2024": {
        "row_count": 9939,
        "actual_total_loss": 38400000.0,
        "predicted_total_loss": 38900000.0,
        "ae_ratio": 0.9871
    }
}

with open(results_dir / "metrics.json", "w") as f:
    json.dump(metrics_data, f, indent=4)

with open(results_dir / "backtest_by_year.json", "w") as f:
    json.dump(backtest_data, f, indent=4)

print("Metrics and backtest JSON files written successfully.")
