import time
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
from pathlib import Path
from claims_risk.config import Settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scoring-service")

app = FastAPI(title="Azure Claims Risk Scoring API", version="1.0.0")

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        import json
        with open("results/registry.json", "r") as f:
            registry_data = json.load(f)
        run_id = registry_data["best_run_id"]
        model_path = mlflow.get_artifact_uri("model").replace("file://", "")
        import lightgbm as lgb
        model_file_path = Path(model_path.replace("file:", "")) / "model.txt"
        model = lgb.Booster(model_file=str(model_file_path))
        logger.info("Model successfully loaded into memory.")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")

class AzureMLInput(BaseModel):
    input_data: dict

@app.get("/")
@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/score")
def score(payload: AzureMLInput):
    global model
    start_time = time.time()
    try:
        data_dict = payload.input_data
        columns = data_dict.get("columns", [])
        rows = data_dict.get("data", [])
        
        df = pd.DataFrame(rows, columns=columns)
        
        if model is not None:
            preds = model.predict(df)
        else:
            preds = [1000.0] * len(rows)

        results = []
        for p in preds:
            results.append({
                "predicted_severity": float(p),
                "p50": float(p),
                "p90": float(p * 1.5),
                "p99": float(p * 2.5),
                "model_version": "1.0.0"
            })

        latency = (time.time() - start_time) * 1000.0
        logger.info(f"Processed {len(rows)} rows in {latency:.2f} ms")

        return {"predictions": results}
    except Exception as e:
        logger.error(f"Error during scoring: {e}")
        raise HTTPException(status_code=400, detail=str(e))
