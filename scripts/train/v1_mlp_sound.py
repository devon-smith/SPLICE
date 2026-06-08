# v1.5: MLP head on the 2305-d DINOv2 pair feature with a dropout x weight_decay sweep
# selects best (dropout, wd) by mean val AUPRC across 3 seeds, then evals test once
# usage: python scripts/train/v1_mlp_sound.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import copy
import itertools
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

from src.eval.metrics import ranking_metrics

FEATURES = "/mnt/disks/splice-data/pairs/dino_v0_boundary"
OUT_DIR = "/mnt/disks/splice-data/outputs/v1_sound"

DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3]
SEEDS = [0, 1, 2]
EPOCHS = 50
PATIENCE = 5
BATCH_SIZE = 1024
LR = 1e-3


# 2305 -> 512 -> 128 -> 1
class MLPHead(nn.Module):
    def __init__(self, in_dim=2305, hidden=(512, 128), dropout=0.2):
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


# train one MLP with given dropout/wd/seed; return best val AUPRC, state dict, epoch
def train_one(x_tr, y_tr, x_va, y_va, dropout, wd, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPHead(in_dim=x_tr.shape[1], dropout=dropout).to(device)
    pos_weight = torch.tensor(
        [float((y_tr == 0).sum()) / float(max((y_tr == 1).sum(), 1))], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)

    n = len(y_tr)
    best_auprc, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(EPOCHS):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            optim.zero_grad()
            loss = criterion(model(x_tr[idx]), y_tr[idx])
            loss.backward()
            optim.step()
        sched.step()
        model.eval()
        with torch.inference_mode():
            val_scores = torch.sigmoid(model(x_va)).cpu().numpy()
        val_auprc = ranking_metrics(y_va.cpu().numpy(), val_scores)["auprc"]
        if val_auprc > best_auprc:
            best_auprc, best_epoch, stale = val_auprc, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    return best_auprc, best_state, best_epoch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    features = np.load(feat_dir / "features.npy")
    meta = pd.read_parquet(feat_dir / "meta.parquet")
    labels = meta["y_inconsistent"].to_numpy().astype(np.float32)
    masks = {s: (meta["split"] == s).to_numpy() for s in ("train", "val", "test")}
    scaler = StandardScaler().fit(features[masks["train"]])

    def to_device(mask):
        x = scaler.transform(features[mask]).astype(np.float32)
        return torch.from_numpy(x).to(device), torch.from_numpy(labels[mask]).to(device)

    x_tr, y_tr = to_device(masks["train"])
    x_va, y_va = to_device(masks["val"])
    x_te, y_te = to_device(masks["test"])
    n_runs = len(DROPOUTS) * len(WEIGHT_DECAYS) * len(SEEDS)
    print(f"train {len(y_tr)} | val {len(y_va)} | test {len(y_te)} | grid = {n_runs} runs")

    # sweep over all dropout x wd x seed; val-only, test untouched
    records = []
    for dropout, wd, seed in itertools.product(DROPOUTS, WEIGHT_DECAYS, SEEDS):
        val_auprc, state, best_epoch = train_one(x_tr, y_tr, x_va, y_va,
                                                  dropout, wd, seed, device)
        records.append({"dropout": dropout, "weight_decay": wd, "seed": seed,
                        "val_auprc": val_auprc, "best_epoch": best_epoch, "state": state})
        print(f"dropout={dropout:.1f} wd={wd:.0e} seed={seed} -> "
              f"val_auprc {val_auprc:.4f} (epoch {best_epoch})")

    grid = pd.DataFrame([
        {k: r[k] for k in ("dropout", "weight_decay", "seed", "val_auprc", "best_epoch")}
        for r in records
    ])
    grid.to_csv(out_dir / "sweep_grid.csv", index=False)
    cell_mean = grid.groupby(["dropout", "weight_decay"])["val_auprc"].mean()
    best_dropout, best_wd = cell_mean.idxmax()
    print(f"\nselected dropout={best_dropout} wd={best_wd:.0e} "
          f"(mean val AUPRC {cell_mean.max():.4f})")

    # eval the 3 winning seeds on test (one and done)
    winners = [r for r in records if r["dropout"] == best_dropout and r["weight_decay"] == best_wd]
    val_y, test_y = y_va.cpu().numpy(), y_te.cpu().numpy()
    per_seed = []
    for rec in sorted(winners, key=lambda r: r["seed"]):
        model = MLPHead(in_dim=x_tr.shape[1], dropout=best_dropout).to(device)
        model.load_state_dict(rec["state"])
        model.eval()
        with torch.inference_mode():
            val_s = torch.sigmoid(model(x_va)).cpu().numpy()
            test_s = torch.sigmoid(model(x_te)).cpu().numpy()
        per_seed.append({
            "seed": rec["seed"], "val_s": val_s, "test_s": test_s,
            "val_auprc": rec["val_auprc"],
            "test_auprc": ranking_metrics(test_y, test_s)["auprc"],
        })
        torch.save(rec["state"], out_dir / f"v1_sound_seed{rec['seed']}.pt")

    ens_val = np.mean([p["val_s"] for p in per_seed], axis=0)
    ens_test = np.mean([p["test_s"] for p in per_seed], axis=0)
    test_auprcs = np.array([p["test_auprc"] for p in per_seed])
    val_auprcs = np.array([p["val_auprc"] for p in per_seed])
    ens_test_auprc = ranking_metrics(test_y, ens_test)["auprc"]

    np.savez(
        out_dir / "scores.npz",
        val_s=ens_val, val_y=val_y, test_s=ens_test, test_y=test_y,
        **{f"seed{p['seed']}_test_s": p["test_s"] for p in per_seed},
        **{f"seed{p['seed']}_val_s": p["val_s"] for p in per_seed},
    )
    joblib.dump(scaler, out_dir / "scaler.joblib")

    summary = {
        "best_dropout": float(best_dropout),
        "best_weight_decay": float(best_wd),
        "val_auprc_mean": val_auprcs.mean(),
        "val_auprc_std": val_auprcs.std(),
        "test_auprc_mean": test_auprcs.mean(),
        "test_auprc_std": test_auprcs.std(),
        "test_auprc_per_seed": test_auprcs.tolist(),
        "ensemble_test_auprc": float(ens_test_auprc),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))

    print(f"\nv1.5 selected: dropout={best_dropout} weight_decay={best_wd:.0e}")
    print(f"val  AUPRC {val_auprcs.mean():.4f} +/- {val_auprcs.std():.4f}")
    print(f"test AUPRC {test_auprcs.mean():.4f} +/- {test_auprcs.std():.4f}  "
          f"(per-seed {np.round(test_auprcs, 4).tolist()})")
    print(f"3-seed ensemble test AUPRC {ens_test_auprc:.4f}")
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
