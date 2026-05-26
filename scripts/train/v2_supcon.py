"""v2: Supervised Contrastive Learning on the frozen DINOv2 pair feature.

Trains two heads jointly on the cached 2305-d pair feature:
  - projection head  2305 -> 512 -> 128 (L2-normalised) -- SupCon loss only
  - classification head  2305 -> 512 -> 128 -> 1 (same as v1) -- BCE loss

  total loss = lambda_sc * L_supcon + (1 - lambda_sc) * L_bce

The projection head is discarded at inference. Class-balanced sampling
(WeightedRandomSampler) ensures each batch is ~50/50 positive/negative so the
contrastive loss always has meaningful positive pairs despite the 7.5% base rate.

This is a standalone experiment on frozen features -- no backbone fine-tuning.
Compare test AUPRC against v1.5 (0.409) to isolate the effect of the loss.

Example:
  python scripts/train/v2_supcon.py \\
      --features /mnt/disks/splice-data/pairs/dino_v0_boundary \\
      --out /mnt/disks/splice-data/outputs/v2_supcon
"""

import argparse
import copy
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics  # noqa: E402
from src.losses.supcon import SupConLoss  # noqa: E402
from v1_mlp import MLPHead  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2_supcon")

DEFAULT_FEATURES = "/mnt/disks/splice-data/pairs/dino_v0_boundary"
DEFAULT_OUT = "/mnt/disks/splice-data/outputs/v2_supcon"


class ProjectionHead(nn.Module):
    """2305 -> 512 -> 128, L2-normalised output for SupCon loss."""

    def __init__(self, in_dim: int = 2305, hidden: int = 512, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


def _balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Sample with replacement; each class gets equal expected frequency per batch."""
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    w = np.where(labels == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
    return WeightedRandomSampler(torch.from_numpy(w), num_samples=len(labels), replacement=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--temperature", type=float, default=0.07,
                    help="SupCon temperature τ")
    ap.add_argument("--lambda_sc", type=float, default=0.5,
                    help="SupCon loss weight; 0 = pure BCE (degrades to v1)")
    ap.add_argument("--seed", type=int, default=231)
    ap.add_argument("--wandb_project", default="splice-v2-supcon")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    features = np.load(Path(args.features) / "features.npy")
    meta = pd.read_parquet(Path(args.features) / "meta.parquet")
    labels = meta["y_inconsistent"].to_numpy().astype(np.float32)
    masks = {s: (meta["split"] == s).to_numpy() for s in ("train", "val", "test")}
    log.info("train %d | val %d | test %d | pos %.2f%%",
             masks["train"].sum(), masks["val"].sum(), masks["test"].sum(),
             100 * labels.mean())

    scaler = StandardScaler().fit(features[masks["train"]])
    X = torch.from_numpy(scaler.transform(features).astype(np.float32))
    Y = torch.from_numpy(labels)

    def split_tensors(mask):
        return X[mask].to(device), Y[mask].to(device)

    x_va, y_va = split_tensors(masks["val"])
    x_te, y_te = split_tensors(masks["test"])

    # Training DataLoader with balanced sampler (all on CPU, moved per batch)
    train_ds = TensorDataset(X[masks["train"]], Y[masks["train"]])
    sampler = _balanced_sampler(labels[masks["train"]])
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                              drop_last=True)

    in_dim = features.shape[1]
    proj = ProjectionHead(in_dim=in_dim).to(device)
    clf = MLPHead(in_dim=in_dim, dropout=args.dropout).to(device)

    supcon = SupConLoss(temperature=args.temperature)
    bce = nn.BCEWithLogitsLoss()  # balanced sampler handles class weighting

    optim = torch.optim.Adam(
        list(proj.parameters()) + list(clf.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    run = _open_wandb(args.wandb_project, args.wandb_mode, args)
    log.info("lambda_sc=%.2f  temperature=%.3f  batch_size=%d",
             args.lambda_sc, args.temperature, args.batch_size)

    best_auprc, best_proj, best_clf, best_epoch, stale = -1.0, None, None, -1, 0

    for epoch in range(args.epochs):
        proj.train()
        clf.train()
        running_sc, running_bce, n_seen = 0.0, 0.0, 0

        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optim.zero_grad()
            z = proj(x_b)
            logit = clf(x_b)
            l_sc = supcon(z, y_b.long())
            l_bce = bce(logit, y_b)
            loss = args.lambda_sc * l_sc + (1 - args.lambda_sc) * l_bce
            loss.backward()
            optim.step()
            running_sc += l_sc.item() * len(y_b)
            running_bce += l_bce.item() * len(y_b)
            n_seen += len(y_b)
        sched.step()

        proj.eval()
        clf.eval()
        with torch.inference_mode():
            val_scores = torch.sigmoid(clf(x_va)).cpu().numpy()
        val_auprc = ranking_metrics(y_va.cpu().numpy().astype(int), val_scores)["auprc"]
        log.info("epoch %2d  sc %.4f  bce %.4f  val_auprc %.4f  (best %.4f ep %d)",
                 epoch, running_sc / n_seen, running_bce / n_seen,
                 val_auprc, best_auprc, best_epoch)
        _wandb_log(run, {"epoch": epoch, "loss_sc": running_sc / n_seen,
                         "loss_bce": running_bce / n_seen, "val_auprc": val_auprc})

        if val_auprc > best_auprc:
            best_auprc, best_epoch, stale = val_auprc, epoch, 0
            best_proj = copy.deepcopy(proj.state_dict())
            best_clf = copy.deepcopy(clf.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                log.info("early stop at epoch %d (best ep %d)", epoch, best_epoch)
                break

    proj.load_state_dict(best_proj)
    clf.load_state_dict(best_clf)
    proj.eval()
    clf.eval()
    with torch.inference_mode():
        val_scores = torch.sigmoid(clf(x_va)).cpu().numpy()
        test_scores = torch.sigmoid(clf(x_te)).cpu().numpy()
    val_y = y_va.cpu().numpy().astype(int)
    test_y = y_te.cpu().numpy().astype(int)

    thr = best_f1_threshold(val_y, val_scores)
    rank = ranking_metrics(test_y, test_scores)
    hard = threshold_metrics(test_y, test_scores, thr)

    import joblib
    joblib.dump(scaler, out_dir / "scaler.joblib")
    torch.save(best_clf, out_dir / "clf.pt")
    torch.save(best_proj, out_dir / "proj.pt")
    np.savez(out_dir / "scores.npz", val_s=val_scores, val_y=val_y,
             test_s=test_scores, test_y=test_y)

    result = {
        "model": "v2_supcon",
        "lambda_sc": args.lambda_sc,
        "temperature": args.temperature,
        "best_epoch": best_epoch,
        "val_auprc": float(best_auprc),
        "test_auroc": rank["auroc"],
        "test_auprc": rank["auprc"],
        "f1_at_val_thr": hard["f1"],
        "precision_at_val_thr": hard["precision"],
        "recall_at_val_thr": hard["recall"],
        "val_thr": float(thr),
        "n_test": rank["n"],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))
    _wandb_log(run, {f"test_{k}": v for k, v in rank.items() if isinstance(v, float)})
    if run is not None:
        try:
            run.finish()
        except Exception:  # noqa: BLE001
            pass

    print("\n=== v2 SupCon (test split) ===")
    for k in ("test_auprc", "test_auroc", "f1_at_val_thr"):
        print(f"  {k:22s}  {result[k]:.4f}")
    print(f"  best epoch  {best_epoch}  val AUPRC {best_auprc:.4f}  (v1 baseline: 0.409)")
    print(f"  saved -> {out_dir}")


def _open_wandb(project, mode, args):
    if mode == "disabled":
        return None
    try:
        import os
        import wandb
        has_key = bool(os.environ.get("WANDB_API_KEY")) or bool(wandb.api.api_key)
        resolved = mode if (mode != "online" or has_key) else "offline"
        return wandb.init(project=project, name=f"supcon_lsc{args.lambda_sc}_t{args.temperature}",
                          mode=resolved, config=vars(args))
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B init failed: %s", exc)
        return None


def _wandb_log(run, payload):
    if run is None:
        return
    try:
        run.log(payload)
    except Exception:  # noqa: BLE001
        pass


if __name__ == "__main__":
    main()
