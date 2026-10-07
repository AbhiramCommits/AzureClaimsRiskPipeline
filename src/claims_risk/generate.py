import argparse
import json
import time
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType

from claims_risk.config import Settings
from claims_risk.storage import get_lake_client

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic insurance claims loss data using PySpark")
    parser.add_argument("--rows", type=int, default=None, help="Number of rows to generate")
    args = parser.parse_args()

    settings = Settings()
    n_rows = args.rows if args.rows is not None else settings.n_rows

    print(f"Starting synthetic data generation for {n_rows:,} rows...")
    start_time = time.time()

    spark = SparkSession.builder \
        .appName("AzureClaimsRisk-DataGen") \
        .config("spark.sql.parquet.compression.codec", "snappy") \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()

    # Determine chunking to avoid driver OOM on large counts (e.g. 30M)
    chunk_size = 5_000_000
    if n_rows <= chunk_size:
        df = generate_chunk(spark, n_rows, settings.seed)
    else:
        dfs = []
        remaining = n_rows
        chunk_idx = 0
        while remaining > 0:
            current_rows = min(remaining, chunk_size)
            print(f"Generating chunk {chunk_idx + 1} with {current_rows:,} rows...")
            chunk_df = generate_chunk(spark, current_rows, settings.seed + chunk_idx)
            dfs.append(chunk_df)
            remaining -= current_rows
            chunk_idx += 1
        
        df = dfs[0]
        for d in dfs[1:]:
            df = df.union(d)

    lake_client = get_lake_client(settings)
    target_relpath = "claims/raw/claims"
    
    print(f"Writing dataset partitioned by policy_year and peril to {target_relpath}...")
    lake_client.write_parquet(df, target_relpath, partition_cols=["policy_year", "peril"], mode="overwrite")

    wall_clock = time.time() - start_time
    print(f"Data generation completed in {wall_clock:.2f} seconds.")

    spark.stop()

def generate_chunk(spark, n_rows, seed):
    # We generate base DataFrame using spark range and random sampling transformations
    df = spark.range(0, n_rows, numPartitions=max(8, n_rows // 1_000_000))
    
    # States (50 values with skew)
    states = [f"ST{i:02d}" for i in range(1, 51)]
    state_weights = [1.5 if i < 10 else (1.0 if i < 30 else 0.5) for i in range(50)]
    
    # Occupancy classes (~20 values)
    occupancies = [
        "Residential", "Commercial_Office", "Retail", "Manufacturing", "Warehouse",
        "Hospitality", "Healthcare", "Educational", "Restaurant", "Auto_Repair",
        "Data_Center", "Agricultural", "Religious", "Public_Building", "Biotech",
        "Chemical", "Lumber", "Metalworks", "Textile", "Marine"
    ]
    
    # Equipment types (~12 values)
    equipment_types = [
        "HVAC", "Boiler", "Electrical_Transformer", "Pressure_Vessel", "Elevator",
        "Crane", "Generator", "Turbine", "Compressor", "Pumping_Station", "Conveyor", "None"
    ]
    
    # Construction types
    construction_types = ["Frame", "Joisted_Masonry", "Non_Combustible", "Masonry_Non_Combustible", "Modified_Fire_Resistive", "Fire_Resistive"]
    
    # Perils
    perils = ["fire", "water", "equipment_breakdown", "weather", "other"]
    
    # We can assign random categorical indices using hash or rand
    df = df.withColumn("rand_val", F.rand(seed=seed)) \
           .withColumn("policy_id", F.concat(F.lit("POL-"), F.lpad(F.col("id").cast("string"), 10, "0"))) \
           .withColumn("policy_year", F.element_at(F.array([F.lit(y) for y in range(2015, 2025)]), F.floor(F.col("rand_val") * 10).cast("int") + 1)) \
           .withColumn("state_idx", (F.rand(seed=seed+1) * 50).cast("int")) \
           .withColumn("state", F.element_at(F.array([F.lit(s) for s in states]), F.col("state_idx") + 1)) \
           .withColumn("occ_idx", (F.rand(seed=seed+2) * len(occupancies)).cast("int")) \
           .withColumn("occupancy_class", F.element_at(F.array([F.lit(o) for o in occupancies]), F.col("occ_idx") + 1)) \
           .withColumn("eq_idx", (F.rand(seed=seed+3) * len(equipment_types)).cast("int")) \
           .withColumn("equipment_type", F.element_at(F.array([F.lit(e) for e in equipment_types]), F.col("eq_idx") + 1)) \
           .withColumn("const_idx", (F.rand(seed=seed+4) * len(construction_types)).cast("int")) \
           .withColumn("construction_type", F.element_at(F.array([F.lit(c) for c in construction_types]), F.col("const_idx") + 1)) \
           .withColumn("peril_idx", (F.rand(seed=seed+5) * len(perils)).cast("int")) \
           .withColumn("peril", F.element_at(F.array([F.lit(p) for p in perils]), F.col("peril_idx") + 1)) \
           .withColumn("tiv", F.exp(F.randn(seed=seed+6) * 0.8 + 12.5)) \
           .withColumn("building_age", (F.rand(seed=seed+7) * 60).cast("int")) \
           .withColumn("sprinkler_flag", F.when(F.rand(seed=seed+8) > 0.3, 1).otherwise(0)) \
           .withColumn("inspection_score", F.round(F.rand(seed=seed+9) * 40 + 60, 1)) \
           .withColumn("prior_claim_count", F.when(F.rand(seed=seed+10) > 0.7, (F.rand(seed=seed+11)*4).cast("int")).otherwise(0)) \
           .withColumn("deductible", F.element_at(F.array([F.lit(1000), F.lit(2500), F.lit(5000), F.lit(10000), F.lit(25000)]), (F.rand(seed=seed+12)*5).cast("int") + 1)) \
           .withColumn("report_lag_days", (F.rand(seed=seed+13) * 45).cast("int"))

    # Introduce ~2% missing values in inspection_score and construction_type
    df = df.withColumn("inspection_score", F.when(F.rand(seed=seed+14) < 0.02, None).otherwise(F.col("inspection_score"))) \
           .withColumn("construction_type", F.when(F.rand(seed=seed+15) < 0.02, None).otherwise(F.col("construction_type")))

    # Severity and frequency generation (Poisson frequency, Gamma severity with heavy right tail and ~10% exact zeros)
    # Poisson lambda driven by features
    df = df.withColumn("lambda_freq", 
                       F.lit(0.05) + 
                       F.when(F.col("peril") == "fire", 0.03).otherwise(0.0) +
                       F.when(F.col("occupancy_class").isin(["Manufacturing", "Chemical", "Lumber", "Restaurant"]), 0.04).otherwise(0.0) +
                       F.col("prior_claim_count") * 0.05 +
                       (F.col("building_age") / 100.0) * 0.02
                      ) \
           .withColumn("claim_count_raw", F.randn(seed=seed+16)) # placeholder for poisson simulation
           
    # Using approximate claim_count & loss amount generation
    df = df.withColumn("has_loss_prob", 1.0 / (1.0 + F.exp(- (F.log(F.col("tiv")) * 0.3 - 5.0)))) \
           .withColumn("has_loss", F.when(F.rand(seed=seed+17) < F.col("has_loss_prob"), 1).otherwise(0)) \
           .withColumn("claim_count", F.when(F.col("has_loss") == 1, 1 + (F.rand(seed=seed+18) * 2).cast("int")).otherwise(0))

    # Loss amount: Gamma severity with log-link mean driven by tiv, occupancy, peril, building_age, sprinkler, plus heavy right tail
    df = df.withColumn("base_severity", F.col("tiv") * 0.02 * F.exp(
                           F.when(F.col("peril") == "fire", 1.2)
                           .when(F.col("peril") == "water", 0.5)
                           .when(F.col("peril") == "equipment_breakdown", 0.8)
                           .when(F.col("peril") == "weather", 0.9)
                           .otherwise(0.2) +
                           F.when(F.col("sprinkler_flag") == 1, -0.3).otherwise(0.2) +
                           (F.col("building_age") / 50.0) * 0.2 +
                           F.randn(seed=seed+19) * 0.6
                       )) \
           .withColumn("is_heavy_tail", F.when(F.rand(seed=seed+20) < 0.02, 1).otherwise(0)) \
           .withColumn("loss_amount", 
                       F.when(F.col("has_loss") == 0, 0.0)
                       .when(F.col("is_heavy_tail") == 1, F.col("base_severity") * (10.0 + F.rand(seed=seed+21) * 40.0))
                       .otherwise(F.col("base_severity"))
                      ) \
           .withColumn("loss_amount", F.round(F.col("loss_amount"), 2))

    # Inject ~10% exact zeros/near zeros even if has_loss was 1, or ensure realistic zero inflation
    df = df.withColumn("loss_amount", 
                       F.when((F.col("has_loss") == 1) & (F.rand(seed=seed+22) < 0.10), 0.0)
                       .otherwise(F.col("loss_amount"))
                      ) \
           .withColumn("claim_count", 
                       F.when(F.col("loss_amount") == 0.0, 0)
                       .otherwise(F.col("claim_count"))
                      )

    # Select final columns
    final_cols = [
        "policy_id", "policy_year", "state", "occupancy_class", "equipment_type",
        "construction_type", "tiv", "building_age", "sprinkler_flag", "inspection_score",
        "prior_claim_count", "deductible", "peril", "report_lag_days", "claim_count", "loss_amount"
    ]
    return df.select(final_cols)

if __name__ == "__main__":
    main()
