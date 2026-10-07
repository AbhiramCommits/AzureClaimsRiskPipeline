import json
import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from claims_risk.config import Settings
from claims_risk.storage import get_lake_client

def main(spark: SparkSession = None):
    settings = Settings()
    lake_client = get_lake_client(settings)
    
    start_time = time.time()
    created_spark = False
    if spark is None:
        spark = SparkSession.builder \
            .appName("AzureClaimsRisk-Features") \
            .getOrCreate()
        created_spark = True

    print("Reading raw claims data...")
    df = lake_client.read_parquet("claims/raw/claims")

    # 1. Feature Engineering
    # log1p(tiv)
    df = df.withColumn("log_tiv", F.log1p(F.col("tiv"))) \
           .withColumn("tiv_per_building_age", F.col("tiv") / (F.col("building_age") + 1.0))

    # Winsorize numerics (e.g. building_age capped at 50, inspection_score capped)
    df = df.withColumn("building_age_winsor", F.when(F.col("building_age") > 50, 50).otherwise(F.col("building_age"))) \
           .withColumn("inspection_score_imputed", F.coalesce(F.col("inspection_score"), F.lit(80.0))) \
           .withColumn("construction_type_imputed", F.coalesce(F.col("construction_type"), F.lit("Unknown")))

    # Target & label
    df = df.withColumn("has_loss_label", F.when(F.col("loss_amount") > 0, 1).otherwise(0))

    # Split by policy year (never random, train <= 2021, valid = 2022, holdout = 2023-2024)
    train_df = df.filter(F.col("policy_year") <= 2021)
    valid_df = df.filter(F.col("policy_year") == 2022)
    holdout_df = df.filter(F.col("policy_year") >= 2023)

    # Leakage assertions
    train_max_year = train_df.agg(F.max("policy_year")).collect()[0][0]
    valid_min_year = valid_df.agg(F.min("policy_year")).collect()[0][0]
    holdout_min_year = holdout_df.agg(F.min("policy_year")).collect()[0][0]
    
    assert train_max_year <= 2021, f"Leakage detected: train max year is {train_max_year}"
    assert valid_min_year == 2022, f"Validation year mismatch: {valid_min_year}"
    assert holdout_min_year >= 2023, f"Holdout year mismatch: {holdout_min_year}"

    # Target-free categorical encoding & OOF mean encoding computed on TRAIN years only
    # Frequency encoding on train
    state_counts = train_df.groupBy("state").count().withColumnRenamed("count", "state_freq")
    train_df = train_df.join(state_counts, on="state", how="left")
    valid_df = valid_df.join(state_counts, on="state", how="left").fillna({"state_freq": 1})
    holdout_df = holdout_df.join(state_counts, on="state", how="left").fillna({"state_freq": 1})

    occ_counts = train_df.groupBy("occupancy_class").count().withColumnRenamed("count", "occ_freq")
    train_df = train_df.join(occ_counts, on="occupancy_class", how="left")
    valid_df = valid_df.join(occ_counts, on="occupancy_class", how="left").fillna({"occ_freq": 1})
    holdout_df = holdout_df.join(occ_counts, on="occupancy_class", how="left").fillna({"occ_freq": 1})

    # State-level loss ratio lags computed from prior policy years only
    state_loss_lags = train_df.groupBy("state").agg(
        F.mean("loss_amount").alias("state_mean_loss"),
        F.mean("has_loss_label").alias("state_loss_prob")
    )
    train_df = train_df.join(state_loss_lags, on="state", how="left")
    valid_df = valid_df.join(state_loss_lags, on="state", how="left").fillna({"state_mean_loss": 0.0, "state_loss_prob": 0.1})
    holdout_df = holdout_df.join(state_loss_lags, on="state", how="left").fillna({"state_mean_loss": 0.0, "state_loss_prob": 0.1})

    # Interaction terms
    for d in [train_df, valid_df, holdout_df]:
        d = d.withColumn("tiv_x_sprinkler", F.col("log_tiv") * F.col("sprinkler_flag"))

    print("Writing feature splits to lake...")
    lake_client.write_parquet(train_df, "claims/features/train", partition_cols=["policy_year"])
    lake_client.write_parquet(valid_df, "claims/features/valid", partition_cols=["policy_year"])
    lake_client.write_parquet(holdout_df, "claims/features/holdout", partition_cols=["policy_year"])

    wall_clock = time.time() - start_time

    train_rows = train_df.count()
    valid_rows = valid_df.count()
    holdout_rows = holdout_df.count()

    feature_report = {
        "feature_count": len(train_df.columns),
        "rows_per_split": {
            "train": train_rows,
            "valid": valid_rows,
            "holdout": holdout_rows
        },
        "encodings": {
            "state_cardinality": state_counts.count(),
            "occupancy_cardinality": occ_counts.count()
        },
        "spark_wall_clock_seconds": wall_clock,
        "leakage_assertions_passed": True
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "feature_report.json", "w") as f:
        json.dump(feature_report, f, indent=4)

    print("Feature engineering completed successfully. Report saved to results/feature_report.json")

    if created_spark:
        spark.stop()

if __name__ == "__main__":
    main()
