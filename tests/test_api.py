import pytest
from fastapi.testclient import TestClient
from claims_risk.serve import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_score_valid():
    payload = {
        "input_data": {
            "columns": [
                "log_tiv", "tiv_per_building_age", "building_age_winsor", "inspection_score_imputed",
                "prior_claim_count", "deductible", "report_lag_days", "state_freq", "occ_freq",
                "state_mean_loss", "state_loss_prob", "tiv_x_sprinkler", "sprinkler_flag"
            ],
            "data": [
                [12.5, 50000.0, 15.0, 85.0, 0, 1000, 5, 120, 45, 5000.0, 0.12, 12.5, 1]
            ]
        }
    }
    response = client.post("/score", json=payload)
    assert response.status_code == 200
    assert "predictions" in response.json()

def test_score_malformed():
    response = client.post("/score", json={"invalid": "payload"})
    assert response.status_code == 422
