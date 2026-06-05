# v2: Supervised Contrastive Learning on the frozen 2305-d DINOv2 pair feature
# trains a projection head (SupCon loss) jointly with an MLP classifier (BCE);
# projection head is discarded at inference. compare test AUPRC vs v1.5 0.409
# to isolate the effect of the loss. usage: python scripts/train/v2_supcon.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics
from src.losses.supcon import SupConLoss

FEATURES = "/mnt/disks/splice-data/pairs/dino_v0_boundary"
OUT_DIR = "/mnt/disks/splice-data/outputs/v2_supcon"

EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 512
LR = 1e-3
DROPOUT = 0.1
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07  # SupCon tau
LAMBDA_SC = 0.5     # SupCon loss weight; 0 = pure BCE (degrades to v1)


# 2305 -> 512 -> 128 -> 1
class MLPHead(nn.Module):
    def __init__(self, in_dim=2305, hidden=(512, 128), dropout=0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for w in hidden:
            layers += [nn.Linear(dim, w), nn.ReLU(), nn.Dropout(dropout)]
            dim = w
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# CS231N Lec 12: Supervised contrastive loss — projection head maps features to embedding space
class ProjectionHead(nn.Module):
    def __init__(self, in_dim=2305, hidden=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--lambda_sc", type=float, default=LAMBDA_SC)
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--seed", type=int, default=231)
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
    print(f"train {masks['train'].sum()} | val {masks['val'].sum()} | "
          f"test {masks['test'].sum()} | pos {100 * labels.mean():.2f}%")

    scaler = StandardScaler().fit(features[masks["train"]])
    X = torch.from_numpy(scaler.transform(features).astype(np.float32))
    Y = torch.from_numpy(labels)
    x_va, y_va = X[masks["val"]].to(device), Y[masks["val"]].to(device)
    x_te, y_te = X[masks["test"]].to(device), Y[masks["test"]].to(device)

    # balanced sampler: each class ~50/50 per batch so SupCon always has +ve pairs
    train_labels = labels[masks["train"]]
    n_pos = int((train_labels == 1).sum())
    n_neg = int((train_labels == 0).sum())
    w = np.where(train_labels == 1, 1.0 / n_pos, 1.0 / n_neg).astype(np.float64)
    sampler = WeightedRandomSampler(torch.from_numpy(w), num_samples=len(train_labels),
                                     replacement=True)
    train_loader = DataLoader(TensorDataset(X[masks["train"]], Y[masks["train"]]),
                              batch_size=BATCH_SIZE, sampler=sampler, drop_last=True)

    in_dim = features.shape[1]
    proj = ProjectionHead(in_dim=in_dim).to(device)
    clf = MLPHead(in_dim=in_dim, dropout=DROPOUT).to(device)

    supcon = SupConLoss(temperature=args.temperature)  # CS231N Lec 12: Supervised contrastive loss (SupConLoss)
    bce = nn.BCEWithLogitsLoss()  # balanced sampler handles class weighting

    optim = torch.optim.Adam(  # CS231N Lec 3: Adam optimizer usage
        list(proj.parameters()) + list(clf.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,  # CS231N Lec 3: Weight decay / L2 regularization on MLP head
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)  # CS231N Lec 3: Learning rate schedules (cosine LR schedule)
    print(f"lambda_sc={args.lambda_sc:.2f}  temperature={args.temperature:.3f}  batch={BATCH_SIZE}")

    best_auprc, best_proj, best_clf, best_epoch, stale = -1.0, None, None, -1, 0

    for epoch in range(EPOCHS):
        proj.train()
        clf.train()
        running_sc, running_bce, n_seen = 0.0, 0.0, 0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(device), y_b.to(device)
            optim.zero_grad()
            z = proj(x_b)
            logit = clf(x_b)
            l_sc = supcon(z, y_b.long())  # CS231N Lec 12: Contrastive positive/negative pair construction from scene labels
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
        print(f"epoch {epoch:2d}  sc {running_sc/n_seen:.4f}  bce {running_bce/n_seen:.4f}  "
              f"val_auprc {val_auprc:.4f}  (best {best_auprc:.4f} ep {best_epoch})")

        if val_auprc > best_auprc:
            best_auprc, best_epoch, stale = val_auprc, epoch, 0
            best_proj = copy.deepcopy(proj.state_dict())
            best_clf = copy.deepcopy(clf.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop at epoch {epoch} (best ep {best_epoch})")
                break

    proj.load_state_dict(best_proj)
    clf.load_state_dict(best_clf)
    proj.eval()
    clf.eval()
    with torch.inference_mode():
        val_scores = torch.sigmoid(clf(x_va)).cpu().numpy()
        test_scores = torch.sigmoid(clf(x_te)).cpu().numpy()
    val_y, test_y = y_va.cpu().numpy().astype(int), y_te.cpu().numpy().astype(int)

    thr = best_f1_threshold(val_y, val_scores)
    rank = ranking_metrics(test_y, test_scores)
    hard = threshold_metrics(test_y, test_scores, thr)

    joblib.dump(scaler, out_dir / "scaler.joblib")
    torch.save(best_clf, out_dir / "clf.pt")
    torch.save(best_proj, out_dir / "proj.pt")
    np.savez(out_dir / "scores.npz", val_s=val_scores, val_y=val_y,
             test_s=test_scores, test_y=test_y)

    result = {
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
    }
    (out_dir / "results.json").write_text(json.dumps(result, indent=2))

    print(f"\nv2 SupCon test_auprc {rank['auprc']:.4f}  test_auroc {rank['auroc']:.4f}  "
          f"f1 {hard['f1']:.4f}  best_epoch {best_epoch}  (v1.5 baseline 0.409)")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
