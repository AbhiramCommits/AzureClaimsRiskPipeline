"""Azure ML managed online endpoint entry script.

Mirrors the ``init`` / ``run`` contract expected by Azure ML scoring scripts.
Locally the same model artifacts are served by ``claims_risk.serve``.
"""
import json
import os
from pathlib import Path
from typing import Any, List

_model = None


def init() -> None:
    global _model
    import lightgbm as lgb
    from claims_risk.data import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    artifact_dir = Path(os.getenv("AZUREML_MODEL_DIR", "results/model_artifacts"))
    _model = {
        "severity": lgb.Booster(model_file=str(artifact_dir / "severity.txt")),
        "p50": lgb.Booster(model_file=str(artifact_dir / "quantile_50.txt")),
        "p90": lgb.Booster(model_file=str(artifact_dir / "quantile_90.txt")),
        "p99": lgb.Booster(model_file=str(artifact_dir / "quantile_99.txt")),
        "num": NUMERIC_FEATURES,
        "cat": CATEGORICAL_FEATURES,
    }


def run(raw_data: str) -> str:
    """Called per request by Azure ML. ``raw_data`` is the JSON request body."""
    assert _model is not None, "init() must be called before run()"
    import numpy as np
    import pandas as pd

    payload = json.loads(raw_data)
    data = payload.get("input_data", payload)
    columns: List[str] = data["columns"]
    rows: List[List[Any]] = data["data"]
    df = pd.DataFrame(rows, columns=columns)
    for col in _model["num"]:
        if col not in df.columns:
            df[col] = 0.0
    X = df[_model["num"]].values.astype("float64")
    sev = _model["severity"].predict(X)
    return json.dumps({
        "predictions": [
            {
                "predicted_severity": float(sev[i]),
                "p50": float(_model["p50"].predict(X)[i]),
                "p90": float(_model["p90"].predict(X)[i]),
                "p99": float(_model["p99"].predict(X)[i]),
            }
            for i in range(len(X))
        ]
    })
