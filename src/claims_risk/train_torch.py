import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import mlflow

from claims_risk.config import Settings
from claims_risk.data import load_split, build_matrix, apply_matrix, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET_COL
from claims_risk.models.tabular import TabularSeverityNet, GammaNLLLoss


def _standardize(train_num: np.ndarray, other_num: np.ndarray):
    mu = train_num.mean(axis=0)
    sd = train_num.std(axis=0) + 1e-6
    return (train_num - mu) / sd, (other_num - mu) / sd, mu, sd


def main():
    settings = Settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("claims-severity-risk")
    torch.manual_seed(settings.seed)
    np.random.seed(settings.seed)

    t0 = time.time()
    train_df = load_split(settings, "train")
    valid_df = load_split(settings, "valid")

    # Bound CPU training time with a documented subsample of the train split.
    TORCH_TRAIN_ROWS = 3_000_000
    if len(train_df) > TORCH_TRAIN_ROWS:
        train_df = train_df.sample(TORCH_TRAIN_ROWS, random_state=settings.seed)
    Xtr, card = build_matrix(train_df)
    Xva = apply_matrix(valid_df, card)

    num_tr = Xtr[NUMERIC_FEATURES].values.astype("float32")
    num_va = Xva[NUMERIC_FEATURES].values.astype("float32")
    num_tr, num_va, mu, sd = _standardize(num_tr, num_va)

    cat_names = [c for c in CATEGORICAL_FEATURES if c in Xtr.columns]
    cat_tr = np.stack([Xtr[c].values for c in cat_names], axis=1).astype("int64")
    cat_va = np.stack([Xva[c].values for c in cat_names], axis=1).astype("int64")
    cardinalities = [int(cat_tr[:, i].max()) + 1 for i in range(cat_tr.shape[1])]
    embed_dims = [min(16, max(2, (c + 1) // 2)) for c in cardinalities]

    y_tr = train_df[TARGET_COL].values.astype("float32")
    y_va = valid_df[TARGET_COL].values.astype("float32")
    # Clip the extreme right tail for training stability of the Gamma NLL.
    clip_val = float(np.quantile(y_tr, 0.999))
    y_tr_clipped = np.clip(y_tr, 0.0, clip_val)

    ds = TensorDataset(
        torch.from_numpy(num_tr), torch.from_numpy(cat_tr), torch.from_numpy(y_tr_clipped).unsqueeze(1)
    )
    loader = DataLoader(ds, batch_size=8192, shuffle=True, num_workers=0)

    model = TabularSeverityNet(len(NUMERIC_FEATURES), cardinalities, embed_dims)
    criterion = GammaNLLLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    num_va_t = torch.from_numpy(num_va)
    cat_va_t = torch.from_numpy(cat_va)
    y_va_t = torch.from_numpy(y_va).unsqueeze(1)
    y_va_clip_t = torch.from_numpy(np.clip(y_va, 0.0, clip_val)).unsqueeze(1)

    with mlflow.start_run(run_name="pytorch-tabular-severity") as run:
        run_id = run.info.run_id
        mlflow.log_param("train_rows", len(train_df))
        mlflow.log_param("target_clip_999", clip_val)
        mlflow.log_param("batch_size", 8192)
        mlflow.log_param("architecture", "embeddings+bn-mlp+log-link")
        best_val = float("inf")
        patience, bad = 2, 0
        for epoch in range(5):
            model.train()
            total, seen = 0.0, 0
            for bn, bc, by in loader:
                optimizer.zero_grad()
                preds = model(bn, bc)
                loss = criterion(preds, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                total += loss.item() * bn.size(0)
                seen += bn.size(0)
            scheduler.step()
            train_loss = total / seen
            model.eval()
            with torch.no_grad():
                val_pred = model(num_va_t, cat_va_t)
                val_loss = criterion(val_pred, y_va_clip_t).item()
                val_rmse = float(torch.sqrt(torch.mean((val_pred - y_va_t) ** 2)).item())
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_rmse", val_rmse, step=epoch)
            print(f"epoch {epoch}: train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_rmse={val_rmse:,.1f}", flush=True)
            if val_loss < best_val - 1e-4:
                best_val = val_loss
                bad = 0
                mlflow.pytorch.log_model(model, name="pytorch_model", serialization_format="pickle")
            else:
                bad += 1
                if bad >= patience:
                    break
        mlflow.log_metric("best_val_loss", best_val)

    Path("results").mkdir(parents=True, exist_ok=True)
    meta = {
        "run_id": run_id,
        "best_val_loss": best_val,
        "train_rows": len(train_df),
        "target_clip_999": clip_val,
        "wall_clock_seconds": time.time() - t0,
    }
    with open("results/torch_run.json", "w") as f:
        json.dump(meta, f, indent=4)
    print(f"PyTorch training complete. best_val_loss={best_val:.4f}, wall={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
