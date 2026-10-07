import numpy as np
import pytest

pytest.importorskip("pyspark")
pytest.importorskip("lightgbm")


def test_small_end_to_end(tmp_path, monkeypatch):
    """50k rows -> raw lake -> features -> tiny GBM -> score, all locally."""
    monkeypatch.setenv("LAKE_ROOT", str(tmp_path / "lake"))
    monkeypatch.setenv("N_ROWS", "50000")
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir(exist_ok=True)

    from pyspark.sql import SparkSession
    spark = (
        SparkSession.builder.master("local[2]")
        .appName("claims-e2e")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    from claims_risk.config import Settings
    from claims_risk.storage import get_lake_client
    from claims_risk import generate, features

    settings = Settings()
    raw = generate.generate_chunk(spark, 50_000, settings.seed)
    get_lake_client(settings).write_parquet(
        raw, "claims/raw/claims", partition_cols=["policy_year", "peril"], mode="overwrite"
    )
    features.main(spark)

    from claims_risk.data import load_split, build_matrix, apply_matrix, TARGET_COL
    import lightgbm as lgb

    train = load_split(settings, "train")
    valid = load_split(settings, "valid")
    assert len(train) > 0 and len(valid) > 0

    X_train, card = build_matrix(train)
    X_valid = apply_matrix(valid, card)
    booster = lgb.train(
        {"objective": "regression", "verbose": -1, "num_threads": 2},
        lgb.Dataset(X_train, label=train[TARGET_COL].values),
        num_boost_round=10,
    )
    preds = booster.predict(X_valid)
    assert len(preds) == len(valid)
    assert np.all(np.isfinite(preds))
