# reliability diagrams + ECE + Brier for v0, v1.5, v2
# all three are trained with class-weighted BCE (pos_weight ~= 12) so the raw
# sigmoid outputs are not calibrated probabilities; this quantifies the gap.
# usage: python scripts/eval/v2_calibration.py

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve

V0_NPZ = "/mnt/disks/splice-data/outputs/v0/scores.npz"
V15_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
V2_NPZ = "/mnt/disks/splice-data/outputs/v2_lora_extended/seed0/r8_a16/scores.npz"
N_BINS = 15


# expected calibration error: weighted L1 between predicted prob and observed rate per bin
def ece(y, p, n_bins=N_BINS):
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bins, right=False) - 1, 0, n_bins - 1)
    n = len(p)
    total = 0.0
    for b in range(n_bins):
        sel = idx == b
        if not sel.any():
            continue
        total += (sel.sum() / n) * abs(p[sel].mean() - y[sel].mean())
    return float(total)


def brier(y, p):
    return float(((p - y) ** 2).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0_scores", default=V0_NPZ)
    ap.add_argument("--v1_5_scores", default=V15_NPZ)
    ap.add_argument("--v2_scores", default=V2_NPZ)
    ap.add_argument("--out_fig", default="reports/figures/calibration_v0_v15_v2.png")
    ap.add_argument("--out_json", default="reports/v2_calibration_metrics.json")
    args = ap.parse_args()

    v0, v15, v2 = np.load(args.v0_scores), np.load(args.v1_5_scores), np.load(args.v2_scores)
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

    results = {}
    for name, d in models.items():
        results[name] = {
            "val_brier": brier(d["val_y"], d["val_s"]),
            "val_ece": ece(d["val_y"], d["val_s"]),
            "val_mean_pred": float(d["val_s"].mean()),
            "val_pos_rate": float(d["val_y"].mean()),
            "test_brier": brier(d["test_y"], d["test_s"]),
            "test_ece": ece(d["test_y"], d["test_s"]),
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
            prob_true, prob_pred = calibration_curve(d[f"{split}_y"], d[f"{split}_s"],
                                                     n_bins=N_BINS, strategy="uniform")
            ax.plot(prob_pred, prob_true, marker="o", markersize=4, label=name, linewidth=1.5)
        ax.plot([0, 1], [0, 1], "k:", alpha=0.5, label="perfectly calibrated")
        ax.set_xlabel("mean predicted probability (bin)")
        ax.set_ylabel("observed positive rate (bin)")
        ax.set_title(f"Reliability diagram -- {split}")
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
