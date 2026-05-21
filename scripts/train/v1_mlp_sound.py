"""Phase-3 P1: a soundly-regularized v1 MLP ("v1.5").

Full grid sweep over dropout x weight_decay x seed (6 x 4 x 3 = 72 runs). Each
run trains with early stopping on val AUPRC and keeps the best-val checkpoint
(not the last epoch). The (dropout, weight_decay) cell with the best mean val
AUPRC across its 3 seeds is the selected v1.5 config; only then is the held-out
test set scored -- once, for the 3 seeds of that cell.

Outputs:
  outputs/v1_sound/   3 checkpoints, scaler, scores.npz (per-seed + 3-seed
                      mean-probability ensemble), sweep_grid.csv, results.json
  configs/v1_sound.yaml
  reports/v1_sound_results.md
W&B project splice-v1-sweep -- one run per (dropout, weight_decay, seed).

Example:
  python scripts/train/v1_mlp_sound.py
"""

import argparse
import copy
import itertools
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.preprocessing import StandardScaler
from torch import nn

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from src.eval.metrics import ranking_metrics  # noqa: E402
from v1_mlp import MLPHead  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v1_mlp_sound")

REPO = HERE.parents[2]
DEFAULT_FEATURES = "/mnt/disks/splice-data/pairs/dino_v0_boundary"
DEFAULT_OUT = "/mnt/disks/splice-data/outputs/v1_sound"
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
WEIGHT_DECAYS = [0.0, 1e-5, 1e-4, 1e-3]
SEEDS = [0, 1, 2]


def train_one(data, dropout: float, wd: float, seed: int, args, device: str):
    """Train one MLP; return (best_val_auprc, best_state_dict, best_epoch)."""
    x_tr, y_tr, x_va, y_va = data
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLPHead(in_dim=x_tr.shape[1], dropout=dropout).to(device)
    pos_weight = torch.tensor(
        [float((y_tr == 0).sum()) / float(max((y_tr == 1).sum(), 1))], device=device
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)

    n = len(y_tr)
    best_auprc, best_state, best_epoch, stale = -1.0, None, -1, 0
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, args.batch_size):
            idx = perm[i : i + args.batch_size]
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
            if stale >= args.patience:
                break
    return best_auprc, best_state, best_epoch


def log_wandb_run(project: str, mode: str, name: str, config: dict, summary: dict) -> None:
    """Best-effort single-run W&B logging (never fatal)."""
    if mode == "disabled":
        return
    try:
        import wandb

        run = wandb.init(project=project, name=name, config=config, mode=mode, reinit=True)
        run.log(summary)
        run.finish()
    except Exception as exc:  # noqa: BLE001
        log.warning("W&B failed for %s: %s", name, exc)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wandb_project", default="splice-v1-sweep")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    wandb_mode = args.wandb_mode
    if wandb_mode == "online":
        try:
            import wandb

            if not (os.environ.get("WANDB_API_KEY") or wandb.api.api_key):
                wandb_mode = "offline"
                log.warning("no W&B key -> offline")
        except Exception:  # noqa: BLE001
            wandb_mode = "disabled"

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
    data = (x_tr, y_tr, x_va, y_va)
    log.info(
        "train %d | val %d | test %d | grid = %d runs",
        len(y_tr),
        len(y_va),
        len(y_te),
        len(DROPOUTS) * len(WEIGHT_DECAYS) * len(SEEDS),
    )

    # ---- grid sweep (val only -- test is untouched here) --------------------
    records = []
    for dropout, wd, seed in itertools.product(DROPOUTS, WEIGHT_DECAYS, SEEDS):
        val_auprc, state, best_epoch = train_one(data, dropout, wd, seed, args, device)
        records.append(
            {
                "dropout": dropout,
                "weight_decay": wd,
                "seed": seed,
                "val_auprc": val_auprc,
                "best_epoch": best_epoch,
                "state": state,
            }
        )
        log.info(
            "dropout=%.1f wd=%.0e seed=%d -> val_auprc %.4f (epoch %d)",
            dropout,
            wd,
            seed,
            val_auprc,
            best_epoch,
        )
        log_wandb_run(
            args.wandb_project,
            wandb_mode,
            f"d{dropout}_wd{wd:.0e}_s{seed}",
            {"dropout": dropout, "weight_decay": wd, "seed": seed, "lr": args.lr},
            {"val_auprc": val_auprc, "best_epoch": best_epoch},
        )

    grid = pd.DataFrame(
        [
            {k: r[k] for k in ("dropout", "weight_decay", "seed", "val_auprc", "best_epoch")}
            for r in records
        ]
    )
    grid.to_csv(out_dir / "sweep_grid.csv", index=False)
    cell_mean = grid.groupby(["dropout", "weight_decay"])["val_auprc"].mean()
    best_dropout, best_wd = cell_mean.idxmax()
    log.info(
        "SELECTED dropout=%.1f weight_decay=%.0e (mean val AUPRC %.4f)",
        best_dropout,
        best_wd,
        cell_mean.max(),
    )

    # ---- one-shot test eval of the selected config's 3 seeds ----------------
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
        per_seed.append(
            {
                "seed": rec["seed"],
                "val_s": val_s,
                "test_s": test_s,
                "val_auprc": rec["val_auprc"],
                "test_auprc": ranking_metrics(test_y, test_s)["auprc"],
            }
        )
        torch.save(rec["state"], out_dir / f"v1_sound_seed{rec['seed']}.pt")

    ens_val = np.mean([p["val_s"] for p in per_seed], axis=0)
    ens_test = np.mean([p["test_s"] for p in per_seed], axis=0)
    test_auprcs = np.array([p["test_auprc"] for p in per_seed])
    val_auprcs = np.array([p["val_auprc"] for p in per_seed])
    ens_test_auprc = ranking_metrics(test_y, ens_test)["auprc"]

    np.savez(
        out_dir / "scores.npz",
        val_s=ens_val,
        val_y=val_y,
        test_s=ens_test,
        test_y=test_y,
        **{f"seed{p['seed']}_test_s": p["test_s"] for p in per_seed},
        **{f"seed{p['seed']}_val_s": p["val_s"] for p in per_seed},
    )
    joblib.dump(scaler, out_dir / "scaler.joblib")

    config = {
        "model": "v1.5_mlp_sound",
        "features": feat_dir.name,
        "head": {"hidden": [512, 128], "dropout": float(best_dropout)},
        "optim": {
            "name": "adam",
            "lr": args.lr,
            "weight_decay": float(best_wd),
            "schedule": "cosine",
        },
        "train": {"epochs": args.epochs, "patience": args.patience, "batch_size": args.batch_size},
        "seeds": SEEDS,
        "selected_by": "mean val AUPRC over 3 seeds, 6x4 dropout x weight_decay grid",
    }
    (REPO / "configs" / "v1_sound.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    summary = {
        "best_dropout": float(best_dropout),
        "best_weight_decay": float(best_wd),
        "val_auprc_mean": float(val_auprcs.mean()),
        "val_auprc_std": float(val_auprcs.std()),
        "test_auprc_mean": float(test_auprcs.mean()),
        "test_auprc_std": float(test_auprcs.std()),
        "test_auprc_per_seed": test_auprcs.tolist(),
        "ensemble_test_auprc": float(ens_test_auprc),
        "v1_reference_auprc": 0.409,
        "v0_reference_auprc": 0.356,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2))
    _write_report(out_dir, cell_mean, summary, args)

    print("\n=== v1.5 sound MLP ===")
    print(f"  selected: dropout={best_dropout}  weight_decay={best_wd:.0e}")
    print(f"  val  AUPRC {val_auprcs.mean():.4f} +/- {val_auprcs.std():.4f}")
    print(
        f"  test AUPRC {test_auprcs.mean():.4f} +/- {test_auprcs.std():.4f}"
        f"  (per-seed {np.round(test_auprcs, 4).tolist()})"
    )
    print(f"  3-seed ensemble test AUPRC {ens_test_auprc:.4f}")
    print(f"  v1 reference 0.409 | v0 reference 0.356\nsaved -> {out_dir}")


def _write_report(out_dir, cell_mean, s, args) -> None:
    md = ["# v1.5 — Soundly-Regularized MLP (Phase 3, P1)\n"]
    md.append(
        f"Full grid sweep: dropout {DROPOUTS} x weight_decay {WEIGHT_DECAYS} x "
        f"{len(SEEDS)} seeds = {len(DROPOUTS) * len(WEIGHT_DECAYS) * len(SEEDS)} runs. Each run "
        f"early-stops on val AUPRC (patience {args.patience}, best checkpoint kept). The held-out "
        "test set is scored once, after the config is selected on val.\n"
    )
    md.append(
        f"**Selected config:** dropout = {s['best_dropout']}, "
        f"weight_decay = {s['best_weight_decay']:.0e}.\n"
    )
    md.append("## Mean val AUPRC per (dropout, weight_decay) cell\n")
    pivot = cell_mean.unstack("weight_decay")
    md.append("| dropout \\ wd | " + " | ".join(f"{c:.0e}" for c in pivot.columns) + " |")
    md.append("|" + "---|" * (len(pivot.columns) + 1))
    for d, rowvals in pivot.iterrows():
        md.append(f"| {d} | " + " | ".join(f"{v:.4f}" for v in rowvals) + " |")
    md.append("")
    md.append("## Selected config — 3-seed results\n")
    md.append("| metric | value |")
    md.append("|---|---|")
    md.append(f"| val AUPRC (mean ± std) | {s['val_auprc_mean']:.4f} ± {s['val_auprc_std']:.4f} |")
    md.append(
        f"| **test AUPRC (mean ± std)** | **{s['test_auprc_mean']:.4f} ± "
        f"{s['test_auprc_std']:.4f}** |"
    )
    md.append(
        f"| test AUPRC per seed | {', '.join(f'{v:.4f}' for v in s['test_auprc_per_seed'])} |"
    )
    md.append(f"| 3-seed ensemble test AUPRC | {s['ensemble_test_auprc']:.4f} |")
    md.append(f"| v1 MLP reference (single seed, dropout 0.2, wd 0) | {s['v1_reference_auprc']} |")
    md.append(f"| v0 logistic reference | {s['v0_reference_auprc']} |")
    md.append("")
    delta = s["test_auprc_mean"] - s["v1_reference_auprc"]
    verdict = (
        "within noise of the original v1 — the v1 number was sound, not a lucky overfitting seed"
        if abs(delta) <= 0.01
        else (
            "below the original v1 by >0.01 — the original v1 likely benefited from a lucky "
            "overfitting seed; this regularized number is the honest one"
            if delta < 0
            else "above the original v1 — regularization genuinely helped"
        )
    )
    md.append("## Verdict\n")
    md.append(
        f"v1.5 test AUPRC {s['test_auprc_mean']:.4f} vs original v1 {s['v1_reference_auprc']} "
        f"(Δ {delta:+.4f}): {verdict}. Downstream Phase-3 analyses use the 3-seed "
        "mean-probability ensemble as the canonical v1.5 prediction.\n"
    )
    (REPO / "reports" / "v1_sound_results.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
