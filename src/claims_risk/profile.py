import json
import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from claims_risk.config import Settings
from claims_risk.storage import get_lake_client

def main():
    settings = Settings()
    lake_client = get_lake_client(settings)
    target_relpath = "claims/raw/claims"

    print("Starting data profiling...")
    start_time = time.time()

    spark = SparkSession.builder \
        .appName("AzureClaimsRisk-Profile") \
        .getOrCreate()

    df = lake_client.read_parquet(target_relpath)

    total_rows = df.count()
    
    # Calculate bytes on disk and partition count
    full_path = Path(settings.lake_root) / target_relpath.lstrip("/")
    total_bytes = sum(f.stat().st_size for f in full_path.glob("**/*") if f.is_file())
    
    # Partition count (number of parquet files or directories)
    parquet_files = list(full_path.glob("**/*.parquet")) + list(full_path.glob("**/*.snappy.parquet"))
    partition_count = len(list(full_path.glob("policy_year=*")))

    # Null rates per column
    null_counts = df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in df.columns]).collect()[0].asDict()
    null_rates = {k: v / total_rows for k, v in null_counts.items()}

    # Loss amount statistics (mean, median, p90, p99, max, zero-loss share)
    loss_stats = df.approxQuantile("loss_amount", [0.5, 0.9, 0.99], 0.01)
    loss_summary = df.select(
        F.mean("loss_amount").alias("mean"),
        F.sum(F.when(F.col("loss_amount") == 0, 1).otherwise(0)).alias("zero_count")
    ).collect()[0]

    mean_loss = float(loss_summary["mean"])
    median_loss = float(loss_stats[0])
    p90_loss = float(loss_stats[1])
    p99_loss = float(loss_stats[2])
    max_loss = float(df.agg(F.max("loss_amount")).collect()[0][0])
    zero_loss_share = float(loss_summary["zero_count"]) / total_rows

    # Gini coefficient of loss concentration
    # We compute Gini on non-zero losses or overall loss_amount
    losses = [row["loss_amount"] for row in df.select("loss_amount").filter(F.col("loss_amount") > 0).limit(100000).collect()]
    gini = compute_gini(losses)

    wall_clock = time.time() - start_time

    profile_data = {
        "row_count": total_rows,
        "bytes_on_disk": total_bytes,
        "partition_count": partition_count,
        "file_count": len(parquet_files),
        "null_rates": null_rates,
        "loss_amount_stats": {
            "mean": mean_loss,
            "median": median_loss,
            "p90": p90_loss,
            "p99": p99_loss,
            "max": max_loss,
            "zero_loss_share": zero_loss_share,
            "gini_concentration": gini
        },
        "generation_wall_clock_seconds": wall_clock
    }

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "data_profile.json", "w") as f:
        json.dump(profile_data, f, indent=4)

    print("Data profile successfully written to results/data_profile.json")
    spark.stop()

def compute_gini(x):
    import numpy as np
    x = np.array(x)
    if len(x) == 0 or np.amin(x) < 0:
        return 0.0
    x = np.sort(x)
    n = len(x)
    index = np.arange(1, n + 1)
    return float((np.sum((2 * index - n - 1) * x)) / (n * np.sum(x)))

if __name__ == "__main__":
    main()
