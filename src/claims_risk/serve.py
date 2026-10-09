import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from claims_risk.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

logger = logging.getLogger("claims_risk.serve")
logging.basicConfig(level=logging.INFO, format='{"ts": "%(asctime)s", "level": "%(levelname)s", "msg": "%(message)s"}')

ARTIFACT_DIR = Path("results/model_artifacts")
MODEL_VERSION = "claims-severity:1"


class InputData(BaseModel):
    input_data: dict[str, Any]


class ModelBundle:
    def __init__(self) -> None:
        self.severity = None
        self.quantiles: dict[str, Any] = {}
        self.card: dict[str, list[str]] = {}
        self.loaded = False

    def load(self) -> None:
        try:
            import lightgbm as lgb
            self.severity = lgb.Booster(model_file=str(ARTIFACT_DIR / "severity.txt"))
            for q in (50, 90, 99):
                self.quantiles[str(q)] = lgb.Booster(model_file=str(ARTIFACT_DIR / f"quantile_{q}.txt"))
            self.card = json.loads((ARTIFACT_DIR / "card.json").read_text())
            self.loaded = True
            logger.info("model loaded from %s", ARTIFACT_DIR)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - defensive
            logger.warning("model not loaded (%s); service will return fallback scores", exc)

    def _vector(self, columns: list[str], rows: list[list[Any]]) -> np.ndarray:
        idx = {c: i for i, c in enumerate(columns)}
        out = np.zeros((len(rows), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)), dtype="float64")
        for j, col in enumerate(NUMERIC_FEATURES):
            if col in idx:
                out[:, j] = [float(r[idx[col]]) for r in rows]
        offset = len(NUMERIC_FEATURES)
        for k, col in enumerate(CATEGORICAL_FEATURES):
            cats = self.card.get(col, [])
            mapping = {v: i for i, v in enumerate(cats)}
            if col in idx:
                out[:, offset + k] = [mapping.get(r[idx[col]], 0) for r in rows]
        return out

    def predict(self, columns: list[str], rows: list[list[Any]]) -> list[dict[str, float]]:
        if not self.loaded:
            return [{"predicted_severity": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0} for _ in rows]
        X = self._vector(columns, rows)
        sev = self.severity.predict(X)
        p50 = self.quantiles["50"].predict(X)
        p90 = self.quantiles["90"].predict(X)
        p99 = self.quantiles["99"].predict(X)
        return [
            {
                "predicted_severity": float(sev[i]),
                "p50": float(p50[i]),
                "p90": float(p90[i]),
                "p99": float(p99[i]),
            }
            for i in range(len(rows))
        ]


bundle = ModelBundle()
app = FastAPI(title="Azure Claims Risk Scoring API", version="1.0.0")
bundle.load()


@app.on_event("startup")
def _startup() -> None:
    bundle.load()


@app.get("/")
def root() -> dict[str, Any]:
    return {"service": "claims-severity-scoring", "model_version": MODEL_VERSION, "model_loaded": bundle.loaded}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "healthy", "model_loaded": bundle.loaded}


@app.post("/score")
def score(payload: InputData) -> dict[str, Any]:
    start = time.perf_counter()
    try:
        data = payload.input_data
        columns = data["columns"]
        rows = data["data"]
        preds = bundle.predict(columns, rows)
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid input_data: {exc}")
    latency_ms = (time.perf_counter() - start) * 1000.0
    logger.info("scored %d rows in %.3f ms", len(rows), latency_ms)
    for p in preds:
        p["model_version"] = MODEL_VERSION
    return {"predictions": preds, "latency_ms": latency_ms, "model_version": MODEL_VERSION}
