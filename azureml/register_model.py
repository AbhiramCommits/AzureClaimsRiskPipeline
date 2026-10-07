import os
from azureml.core import Workspace
from azureml.core.model import Model

def register_model_azure():
    subscription_id = os.getenv("AZURE_SUBSCRIPTION_ID")
    resource_group = os.getenv("AZURE_RESOURCE_GROUP")
    workspace_name = os.getenv("AZURE_WORKSPACE_NAME")

    if not all([subscription_id, resource_group, workspace_name]):
        print("Azure ML environment variables not set. Skipping Azure ML registration.")
        return

    ws = Workspace(
        subscription_id=subscription_id,
        resource_group=resource_group,
        workspace_name=workspace_name
    )

    model = Model.register(
        workspace=ws,
        model_path="mlruns/",
        model_name="claims-severity-azureml",
        tags={"framework": "LightGBM", "task": "severity-prediction"}
    )
    print(f"Registered model in Azure ML workspace: {model.name}, version {model.version}")

if __name__ == "__main__":
    register_model_azure()
