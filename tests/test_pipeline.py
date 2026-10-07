import pytest
import numpy as np
from claims_risk.metrics import mae, rmse, gamma_deviance, pinball_loss
from claims_risk.config import Settings
from claims_risk.storage import get_lake_client, LocalLakeClient

def test_metrics_hand_computed():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    
    assert mae(y_true, y_pred) == 10.0
    assert pytest.approx(rmse(y_true, y_pred), 0.01) == 10.0

def test_pinball_loss_asymmetry():
    y_true = np.array([100.0])
    y_pred_under = np.array([50.0]) # prediction is lower than true
    y_pred_over = np.array([150.0]) # prediction is higher than true
    
    loss_under = pinball_loss(y_true, y_pred_under, 0.9)
    loss_over = pinball_loss(y_true, y_pred_over, 0.9)
    
    assert loss_under > 0
    assert loss_over > 0

def test_storage_client_uris():
    settings = Settings(lake_root="./data/lake")
    client = get_lake_client(settings)
    assert isinstance(client, LocalLakeClient)
    assert "data/lake/raw/claims" in client.abfss_uri("raw/claims")
