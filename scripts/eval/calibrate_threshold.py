# calibrate an inference threshold from within-shot visual variation
# the 3 keyframes of a single shot are the same continuous scene by definition,
# so scoring within-shot pairs gives a consistent variation distribution.
# tau_95 = 95th percentile

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.pairs import build_pair_features_batch, load_embeddings
from src.eval.metrics import threshold_metrics

CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
EMB_DIR = "/mnt/disks/splice-data/embeddings/dinov2_base"
V0_DIR = "/mnt/disks/splice-data/outputs/v0"
OUT_DIR = "/mnt/disks/splice-data/outputs/calibration"
WITHIN_SHOT_PAIRS = [(0, 1), (1, 2), (0, 2)]


# build (eA, eB) for every within-shot keyframe pair in `split`
def within_shot_embeddings(cut_index, split, emb, key2row):
    df = pd.read_parquet(cut_index,
                         columns=["movie_id", "shot_left_idx", "shot_right_idx", "split"])
    df = df[df["split"] == split]
    shots = set()
    for col in ("shot_left_idx", "shot_right_idx"):
        shots.update(zip(df["movie_id"], df[col]))
    rows_a, rows_b = [], []
    for movie_id, shot_idx in shots:
        keys = [f"{movie_id}/shot_{shot_idx:04d}_img_{i}.jpg" for i in range(3)]
        rows = [key2row.get(k, -1) for k in keys]
        if any(r < 0 for r in rows):
            continue
        for i, j in WITHIN_SHOT_PAIRS:
            rows_a.append(rows[i])
            rows_b.append(rows[j])
    print(f"{split}: {len(shots)} shots -> {len(rows_a)} within-shot pairs")
    return emb[np.array(rows_a)], emb[np.array(rows_b)]


def cosine_inconsistency(e_a, e_b):
    num = np.sum(e_a * e_b, axis=1)
    den = np.linalg.norm(e_a, axis=1) * np.linalg.norm(e_b, axis=1)
    return 1.0 - num / np.clip(den, 1e-8, None)


# positive-class probability from the trained logistic, in chunks to keep memory bounded
def logistic_scores(clf, e_a, e_b, chunk=50_000):
    out = np.empty(len(e_a), dtype=np.float64)
    for i in range(0, len(e_a), chunk):
        feats = build_pair_features_batch(e_a[i : i + chunk], e_b[i : i + chunk])
        out[i : i + chunk] = clf.predict_proba(feats)[:, 1]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut_index", default=CUT_INDEX)
    ap.add_argument("--embeddings", default=EMB_DIR)
    ap.add_argument("--v0_dir", default=V0_DIR)
    ap.add_argument("--split", default="val")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    v0_dir = Path(args.v0_dir)

    emb, key2row = load_embeddings(args.embeddings)
    e_a, e_b = within_shot_embeddings(args.cut_index, args.split, emb, key2row)

    clf = joblib.load(v0_dir / "v0_logistic.joblib")
    within = {
        "raw_dino_cosine": cosine_inconsistency(e_a, e_b),
        "logistic": logistic_scores(clf, e_a, e_b),
    }

    npz = np.load(v0_dir / "scores.npz")
    cut_level = {n: npz[f"{n}__val_s"] for n in within if f"{n}__val_s" in npz.files}

    q_pct = int(args.quantile * 100)
    calibration = {}
    for name, w in within.items():
        tau = float(np.quantile(w, args.quantile))
        calibration[name] = {
            f"tau_{q_pct}": tau,
            "distribution_stats": {
                "mean": np.mean(w),
                "std": np.std(w),
                "p50": np.percentile(w, 50),
                "p75": np.percentile(w, 75),
                "p90": np.percentile(w, 90),
                "p95": np.percentile(w, 95),
                "p99": np.percentile(w, 99),
            },
        }
        print(f"{name}: tau_{q_pct} = {tau:.4f}")

    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2, default=float))

    # within-shot vs cut-level distribution figure
    fig, axes = plt.subplots(1, len(within), figsize=(6 * len(within), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, w) in zip(axes, within.items()):
        ax.hist(w, bins=60, density=True, alpha=0.6, label="within-shot (consistent)")
        if name in cut_level:
            ax.hist(cut_level[name], bins=60, density=True, alpha=0.6, label="cut-level")
        tau = calibration[name][f"tau_{q_pct}"]
        ax.axvline(tau, color="r", ls="--", label=f"tau_{q_pct}={tau:.3f}")
        ax.set(xlabel="inconsistency score", ylabel="density", title=name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "within_shot_vs_cut.png", dpi=110)
    plt.close(fig)

    # augment the v0 results table with F1@tau_95
    results_path = v0_dir / "results.json"
    if results_path.exists():
        rows = json.loads(results_path.read_text())
        for row in rows:
            name = row["model"]
            key = f"tau_{q_pct}"
            if name in calibration and f"{name}__test_s" in npz.files:
                tau = calibration[name][key]
                m = threshold_metrics(npz[f"{name}__test_y"], npz[f"{name}__test_s"], tau)
                row[f"f1_at_{key}"] = m["f1"]
                row[f"precision_at_{key}"] = m["precision"]
                row[f"recall_at_{key}"] = m["recall"]
            else:
                row[f"f1_at_{key}"] = None
        (v0_dir / "results_calibrated.json").write_text(json.dumps(rows, indent=2, default=float))
        table = pd.DataFrame(rows)
        cols = ["model", "auroc", "auprc", "f1_at_val_thr", f"f1_at_tau_{q_pct}"]
        table = table[[c for c in cols if c in table.columns]]
        table.to_csv(v0_dir / "results_calibrated.csv", index=False)
        print("\nv0 results with F1@tau_95")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
