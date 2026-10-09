from abc import ABC, abstractmethod
from pathlib import Path

from claims_risk.config import Settings


class LakeClient(ABC):
    @abstractmethod
    def write_parquet(self, df, relpath: str, partition_cols: list | None = None, mode: str = "overwrite"):
        pass

    @abstractmethod
    def read_parquet(self, relpath: str):
        pass

    @abstractmethod
    def abfss_uri(self, relpath: str) -> str:
        pass

    @abstractmethod
    def exists(self, relpath: str) -> bool:
        pass


class LocalLakeClient(LakeClient):
    def __init__(self, root: str = "./data/lake"):
        self.root = Path(root).absolute()
        self.root.mkdir(parents=True, exist_ok=True)

    def _full_path(self, relpath: str) -> str:
        return str(self.root / relpath.lstrip("/"))

    def write_parquet(self, df, relpath: str, partition_cols: list | None = None, mode: str = "overwrite"):
        path = self._full_path(relpath)
        writer = df.write.mode(mode)
        if partition_cols:
            writer.partitionBy(partition_cols).parquet(path)
        else:
            writer.parquet(path)

    def read_parquet(self, relpath: str):
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        path = self._full_path(relpath)
        return spark.read.parquet(path)

    def abfss_uri(self, relpath: str) -> str:
        path = relpath.lstrip("/")
        return f"file://{self.root}/{path}"

    def exists(self, relpath: str) -> bool:
        path = Path(self._full_path(relpath))
        return path.exists()


class AzureLakeClient(LakeClient):
    def __init__(self, account_name: str, container_name: str):
        self.account_name = account_name
        self.container_name = container_name
        # Azure SDK integration code path (guarded)
        try:
            from azure.identity import DefaultAzureCredential
            from azure.storage.filedatalake import DataLakeServiceClient
            self.credential = DefaultAzureCredential()
            self.service_client = DataLakeServiceClient(
                account_url=f"https://{account_name}.dfs.core.windows.net",
                credential=self.credential
            )
        except Exception:  # noqa: BLE001 - fall back to Spark abfss:// access only
            self.service_client = None

    def write_parquet(self, df, relpath: str, partition_cols: list | None = None, mode: str = "overwrite"):
        # When using Spark on Azure Databricks/Synapse, writing directly via Spark abfss:// is standard.
        # If falling back locally or running azure client directly:
        uri = self.abfss_uri(relpath)
        writer = df.write.mode(mode)
        if partition_cols:
            writer.partitionBy(partition_cols).parquet(uri)
        else:
            writer.parquet(uri)

    def read_parquet(self, relpath: str):
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()
        uri = self.abfss_uri(relpath)
        return spark.read.parquet(uri)

    def abfss_uri(self, relpath: str) -> str:
        path = relpath.lstrip("/")
        return f"abfss://{self.container_name}@{self.account_name}.dfs.core.windows.net/{path}"

    def exists(self, relpath: str) -> bool:
        if not self.service_client:
            return False
        try:
            filesystem_client = self.service_client.get_file_system_client(self.container_name)
            directory_client = filesystem_client.get_directory_client(relpath.lstrip("/"))
            return directory_client.exists()
        except Exception:  # noqa: BLE001 - treat any SDK/auth error as missing
            return False


def get_lake_client(settings: Settings | None = None) -> LakeClient:
    if settings is None:
        settings = Settings()
    if settings.storage_mode.lower() == "azure":
        return AzureLakeClient(settings.azure_storage_account, settings.azure_container)
    else:
        return LocalLakeClient(settings.lake_root)
