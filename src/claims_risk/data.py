
import pandas as pd
import pyarrow.dataset as ds

from claims_risk.config import Settings

NUMERIC_FEATURES: list[str] = [
    "log_tiv", "tiv_per_building_age", "building_age_winsor", "inspection_score_imputed",
    "prior_claim_count", "deductible", "report_lag_days", "state_freq", "occ_freq",
    "state_mean_loss", "state_loss_prob", "tiv_x_sprinkler", "sprinkler_flag",
]

CATEGORICAL_FEATURES: list[str] = [
    "state", "occupancy_class", "equipment_type", "construction_type_imputed", "peril",
]

TARGET_COL = "loss_amount"


def split_path(settings: Settings, split: str) -> str:
    return f"{settings.lake_root.rstrip('/')}/claims/features/{split}"


def load_split(
    settings: Settings,
    split: str,
    columns: list[str] | None = None,
    float32: bool = True,
) -> pd.DataFrame:
    """Read a feature split from the lake as a pandas DataFrame using pyarrow.

    Avoids the Spark driver serialization overhead of ``toPandas`` on tens of
    millions of rows by reading column subsets directly from parquet.
    """
    dataset = ds.dataset(split_path(settings, split), format="parquet", partitioning="hive")
    available = set(dataset.schema.names)
    cols = [c for c in (columns or NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COL, "policy_year"]) if c in available]
    table = dataset.to_table(columns=cols)
    df = table.to_pandas()
    if float32:
        for c in NUMERIC_FEATURES:
            if c in df.columns:
                df[c] = df[c].astype("float32")
    return df


def build_matrix(df: pd.DataFrame, cat_cardinalities: dict | None = None):
    """Return (X, cat_cardinalities) where categoricals are integer codes.

    Encoders are fit on the frame passed in (callers pass the train frame first
    and then re-use the returned cardinality map for valid/holdout to avoid
    leakage).
    """
    import pandas as pd
    X = df[NUMERIC_FEATURES].copy()
    card = {} if cat_cardinalities is None else cat_cardinalities
    for col in CATEGORICAL_FEATURES:
        if col not in df.columns:
            continue
        if cat_cardinalities is None:
            cats = pd.Categorical(df[col])
            codes = cats.codes
            card[col] = list(cats.categories)
        else:
            mapping = {v: i for i, v in enumerate(card[col])}
            codes = df[col].map(mapping).fillna(-1).astype(int).values
        X[col] = codes
    return X, card


def apply_matrix(df: pd.DataFrame, card: dict):
    X, _ = build_matrix(df, cat_cardinalities=card)
    return X
