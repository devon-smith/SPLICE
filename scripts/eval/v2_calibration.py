"""Calibration: reliability diagrams + ECE + Brier for v0, v1.5, v2.

All three heads were trained with class-weighted BCE (pos_weight ≈ 12) on a
~7.5%-positive distribution, so the raw scores are *not* calibrated probabilities
— they're upweighted to reach a usable F1. This script quantifies the gap (and
how it shifts between models) on val and test.

Outputs reliability diagrams + Brier/ECE per model + a markdown report.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

DEFAULT_V0 = "/mnt/disks/splice-data/outputs/v0/scores.npz"
DEFAULT_V15 = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
DEFAULT_V2 = "/mnt/disks/splice-data/outputs/v2_lora_extended/seed0/r8_a16/scores.npz"


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(p, bins, right=False) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    total = 0.0
    n = len(p)
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        conf = p[sel].mean()
        acc = y[sel].mean()
        total += (sel.sum() / n) * abs(conf - acc)
    return float(total)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(((p - y) ** 2).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0_scores", default=DEFAULT_V0)
    ap.add_argument("--v1_5_scores", default=DEFAULT_V15)
    ap.add_argument("--v2_scores", default=DEFAULT_V2)
    ap.add_argument("--out_fig", default=str(REPO / "reports/figures/calibration_v0_v15_v2.png"))
    ap.add_argument("--out_json", default=str(REPO / "reports/v2_calibration_metrics.json"))
    ap.add_argument("--n_bins", type=int, default=15)
    args = ap.parse_args()

    v0 = np.load(args.v0_scores)
    v15 = np.load(args.v1_5_scores)
    v2 = np.load(args.v2_scores)

    models = {
        "v0 logistic": {
            "val_s": v0["logistic__val_s"].astype(float),
            "val_y": v0["logistic__val_y"].astype(int),
            "test_s": v0["logistic__test_s"].astype(float),
            "test_y": v0["logistic__test_y"].astype(int),
        },
        "v1.5 MLP (seed 2)": {
            "val_s": v15["seed2_val_s"].astype(float),
            "val_y": v15["val_y"].astype(int),
            "test_s": v15["seed2_test_s"].astype(float),
            "test_y": v15["test_y"].astype(int),
        },
        "v2 LoRA (seed 0)": {
            "val_s": v2["val_s"].astype(float),
            "val_y": v2["val_y"].astype(int),
            "test_s": v2["test_s"].astype(float),
            "test_y": v2["test_y"].astype(int),
        },
    }

    results: dict[str, dict] = {}
    for name, d in models.items():
        results[name] = {
            "val_brier": brier(d["val_y"], d["val_s"]),
            "val_ece": ece(d["val_y"], d["val_s"], args.n_bins),
            "val_mean_pred": float(d["val_s"].mean()),
            "val_pos_rate": float(d["val_y"].mean()),
            "test_brier": brier(d["test_y"], d["test_s"]),
            "test_ece": ece(d["test_y"], d["test_s"], args.n_bins),
            "test_mean_pred": float(d["test_s"].mean()),
            "test_pos_rate": float(d["test_y"].mean()),
        }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(results, indent=2))
    print(f"wrote {args.out_json}")

    for name, r in results.items():
        print(f"\n{name}")
        print(f"  val:  brier={r['val_brier']:.4f}  ECE={r['val_ece']:.4f}  "
              f"mean_pred={r['val_mean_pred']:.4f}  pos_rate={r['val_pos_rate']:.4f}")
        print(f"  test: brier={r['test_brier']:.4f}  ECE={r['test_ece']:.4f}  "
              f"mean_pred={r['test_mean_pred']:.4f}  pos_rate={r['test_pos_rate']:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, split in zip(axes, ("val", "test")):
        for name, d in models.items():
            y = d[f"{split}_y"]
            s = d[f"{split}_s"]
            prob_true, prob_pred = calibration_curve(y, s, n_bins=args.n_bins, strategy="uniform")
            ax.plot(prob_pred, prob_true, marker="o", markersize=4, label=name, linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="perfectly calibrated")
        ax.set_xlabel("mean predicted probability (bin)")
        ax.set_ylabel("observed positive rate (bin)")
        ax.set_title(f"Reliability diagram — {split}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=9, loc="upper left")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out_fig, dpi=130)
    print(f"\nwrote {args.out_fig}")


if __name__ == "__main__":
    main()
