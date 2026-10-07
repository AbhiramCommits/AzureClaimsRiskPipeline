from dataclasses import dataclass, field
import os

@dataclass
class Settings:
    storage_mode: str = field(default_factory=lambda: os.getenv("STORAGE_MODE", "local"))
    lake_root: str = field(default_factory=lambda: os.getenv("LAKE_ROOT", "./data/lake"))
    n_rows: int = field(default_factory=lambda: int(os.getenv("N_ROWS", "30000000")))
    policy_years: list = field(default_factory=lambda: list(range(2015, 2025)))
    seed: int = field(default_factory=lambda: int(os.getenv("SEED", "42")))
    mlflow_tracking_uri: str = field(default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    azure_storage_account: str = field(default_factory=lambda: os.getenv("AZURE_STORAGE_ACCOUNT", ""))
    azure_container: str = field(default_factory=lambda: os.getenv("AZURE_CONTAINER", "datalake"))
    azure_workspace: str = field(default_factory=lambda: os.getenv("AZURE_WORKSPACE", ""))
    azure_subscription: str = field(default_factory=lambda: os.getenv("AZURE_SUBSCRIPTION", ""))
    azure_resource_group: str = field(default_factory=lambda: os.getenv("AZURE_RESOURCE_GROUP", ""))
