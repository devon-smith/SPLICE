"""v1: a 2-layer MLP head on the frozen-DINOv2 2305-d pair feature.

Same backbone and features as the v0 logistic model -- only the head changes
(linear -> 2305-512-128-1 MLP with ReLU + dropout), to test whether
non-linearity over the pair feature helps. Trains on the cached features
(GPU-light), Adam + cosine schedule, early-stopping on val AUPRC.

Example:
  python scripts/train/v1_mlp.py --features /mnt/disks/splice-data/pairs/dino_v0_boundary
"""

import argparse
import copy
import json
import logging
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v1_mlp")

DEFAULT_FEATURES = "/mnt/disks/splice-data/pairs/dino_v0_boundary"
DEFAULT_OUT = "/mnt/disks/splice-data/outputs/v1_mlp"


class MLPHead(nn.Module):
    """2305 -> 512 -> 128 -> 1 with ReLU and dropout."""

    def __init__(self, in_dim: int = 2305, hidden=(512, 128), dropout: float = 0.2) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        dim = in_dim
        for width in hidden:
            layers += [nn.Linear(dim, width), nn.ReLU(), nn.Dropout(dropout)]
            dim = width
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def open_wandb(project: str, mode: str, config: dict):
    """Best-effort W&B run; returns the run or None (logging is never fatal)."""
    if mode == "disabled":
        return None
    try:
        import wandb

        has_key = bool(os.environ.get("WANDB_API_KEY")) or bool(wandb.api.api_key)
        resolved = mode if (mode != "online" or has_key) else "offline"
        return wandb.init(
            project=project,
            name=f"v1_mlp_{date.today().isoformat()}",
            mode=resolved,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B init failed: %s", exc)
        return None


def wandb_log(run, payload: dict) -> None:
    if run is None:
        return
    try:
        run.log(payload)
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B log failed: %s", exc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=5, help="early-stop patience on val AUPRC")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="Adam weight decay (v1.5 uses 1e-4; default 0 preserves legacy behaviour)")
    ap.add_argument("--seed", type=int, default=231)
    ap.add_argument("--wandb_project", default="splice-v0")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = np.load(feat_dir / "features.npy")
    meta = pd.read_parquet(feat_dir / "meta.parquet")
    labels = meta["y_inconsistent"].to_numpy().astype(np.float32)
    masks = {s: (meta["split"] == s).to_numpy() for s in ("train", "val", "test")}

    scaler = StandardScaler().fit(features[masks["train"]])

    def to_device(mask: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x = scaler.transform(features[mask]).astype(np.float32)
        return torch.from_numpy(x).to(device), torch.from_numpy(labels[mask]).to(device)

    x_tr, y_tr = to_device(masks["train"])
    x_va, y_va = to_device(masks["val"])
    x_te, y_te = to_device(masks["test"])
    log.info(
        "train %d | val %d | test %d | feature dim %d",
        len(y_tr),
        len(y_va),
        len(y_te),
        features.shape[1],
    )

    model = MLPHead(in_dim=features.shape[1], dropout=args.dropout).to(device)
    pos_weight = torch.tensor(
        [float((y_tr == 0).sum()) / float(max((y_tr == 1).sum(), 1))], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
    log.info("class-balancing pos_weight=%.2f", pos_weight.item())

    run = open_wandb(args.wandb_project, args.wandb_mode, vars(args))

    n = len(y_tr)
    best_auprc, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        running = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
            optim.zero_grad()
            loss = criterion(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optim.step()
            running += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.inference_mode():
            val_scores = torch.sigmoid(model(x_va)).cpu().numpy()
        val_auprc = ranking_metrics(y_va.cpu().numpy(), val_scores)["auprc"]
        train_loss = running / n
        log.info("epoch %2d  train_loss %.4f  val_auprc %.4f", epoch, train_loss, val_auprc)
        wandb_log(
            run,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_auprc": val_auprc,
                "lr": sched.get_last_lr()[0],
            },
        )

        if val_auprc > best_auprc:
            best_auprc, best_epoch, stale = val_auprc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                log.info("early stop at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_scores = torch.sigmoid(model(x_va)).cpu().numpy()
        test_scores = torch.sigmoid(model(x_te)).cpu().numpy()
    y_va_np, y_te_np = y_va.cpu().numpy(), y_te.cpu().numpy()

    thr = best_f1_threshold(y_va_np, val_scores)
    rank = ranking_metrics(y_te_np, test_scores)
    hard = threshold_metrics(y_te_np, test_scores, thr)
    row = {
        "model": "v1_mlp",
        "features": feat_dir.name,
        "best_epoch": best_epoch,
        "val_auprc": best_auprc,
        "auroc": rank["auroc"],
        "auprc": rank["auprc"],
        "f1_at_val_thr": hard["f1"],
        "precision_at_val_thr": hard["precision"],
        "recall_at_val_thr": hard["recall"],
        "accuracy_at_val_thr": hard["accuracy"],
        "val_thr": thr,
        "n_test": rank["n"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    torch.save(best_state, out_dir / "v1_mlp.pt")
    joblib.dump(scaler, out_dir / "scaler.joblib")
    np.savez(
        out_dir / "scores.npz", val_s=val_scores, val_y=y_va_np, test_s=test_scores, test_y=y_te_np
    )
    (out_dir / "results.json").write_text(json.dumps(row, indent=2))
    wandb_log(run, {f"test_{k}": v for k, v in rank.items() if isinstance(v, float)})
    if run is not None:
        try:
            run.finish()
        except Exception as exc:  # noqa: BLE001
            log.warning("W&B finish failed: %s", exc)

    print("\n=== v1 MLP (test split) ===")
    for k in ("auroc", "auprc", "f1_at_val_thr", "precision_at_val_thr", "recall_at_val_thr"):
        print(f"  {k:22s} {row[k]:.4f}")
    print(f"  best epoch {best_epoch} (val AUPRC {best_auprc:.4f})\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
