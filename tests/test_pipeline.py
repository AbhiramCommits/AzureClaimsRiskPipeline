import numpy as np
import pytest

from claims_risk.config import Settings
from claims_risk.metrics import (
    calibration_stats,
    gamma_deviance,
    mae,
    normalized_gini,
    pinball_loss,
    rmse,
)
from claims_risk.storage import LocalLakeClient, get_lake_client


def test_mae_rmse_hand_computed():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 310.0])
    assert mae(y_true, y_pred) == 10.0
    assert pytest.approx(rmse(y_true, y_pred), abs=1e-9) == 10.0


def test_gamma_deviance_zero_when_perfect():
    y = np.array([1.0, 10.0, 100.0, 1000.0])
    assert pytest.approx(gamma_deviance(y, y), abs=1e-9) == 0.0


def test_pinball_loss_hand_computed_and_asymmetry():
    # alpha=0.9: under-prediction (y=100,p=50) costs 0.9*50=45; over costs 0.1*50=5.
    y = np.array([100.0])
    assert pytest.approx(pinball_loss(y, np.array([50.0]), 0.9)) == 45.0
    assert pytest.approx(pinball_loss(y, np.array([150.0]), 0.9)) == 5.0
    assert pinball_loss(y, np.array([50.0]), 0.9) > pinball_loss(y, np.array([150.0]), 0.9)
    # median is symmetric
    assert pytest.approx(pinball_loss(y, np.array([50.0]), 0.5)) == 25.0
    assert pytest.approx(pinball_loss(y, np.array([150.0]), 0.5)) == 25.0


def test_normalized_gini_perfect_and_random():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_perfect = np.array([0.1, 0.15, 0.2, 0.8, 0.85, 0.9])
    assert pytest.approx(normalized_gini(y_true, y_perfect), abs=1e-9) == 1.0
    y_rev = np.array([0.9, 0.85, 0.8, 0.2, 0.15, 0.1])
    assert pytest.approx(normalized_gini(y_true, y_rev), abs=1e-9) == -1.0


def test_calibration_slope_intercept():
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])
    y_true = 2.0 * y_pred + 1.0
    slope, intercept = calibration_stats(y_true, y_pred)
    assert pytest.approx(slope, abs=1e-6) == 2.0
    assert pytest.approx(intercept, abs=1e-6) == 1.0


def test_storage_client_and_abfss_uri_agree(tmp_path):
    settings = Settings(lake_root=str(tmp_path / "lake"))
    client = get_lake_client(settings)
    assert isinstance(client, LocalLakeClient)
    assert client.abfss_uri("raw/claims").endswith("lake/raw/claims")


@pytest.mark.skipif(
    not __import__("pathlib").Path("data/lake/claims/features/train").exists(),
    reason="feature lake not present",
)
def test_policy_year_leakage_guards():
    import pyarrow.dataset as ds

    def years_and_ids(split):
        d = ds.dataset(f"data/lake/claims/features/{split}", format="parquet", partitioning="hive")
        t = d.to_table(columns=["policy_year", "policy_id"])
        return t.column("policy_year").to_numpy(), t.column("policy_id").to_pylist()

    tr_years, tr_ids = years_and_ids("train")
    va_years, va_ids = years_and_ids("valid")
    ho_years, ho_ids = years_and_ids("holdout")

    assert tr_years.max() <= 2021
    assert va_years.min() == 2022 and va_years.max() == 2022
    assert ho_years.min() >= 2023

    sample = set(tr_ids[:200_000])
    assert sample.isdisjoint(set(va_ids[:200_000]))
    assert sample.isdisjoint(set(ho_ids[:200_000]))
