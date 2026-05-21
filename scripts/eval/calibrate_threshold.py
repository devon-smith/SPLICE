"""Calibrate an inference threshold from natural within-shot visual variation.

The three keyframes of a single shot (img_0, img_1, img_2) are, by definition,
the same continuous scene. Scoring those within-shot pairs gives a distribution
of "consistent" variation; its 95th percentile (tau_95) is a principled cutoff:
a cut scoring above tau_95 is more discontinuous than 95% of within-shot motion.

Calibrated scorers: ``raw_dino_cosine`` and ``logistic`` (handoff Task 6). The
script writes the calibration JSON + a distribution figure, and augments the
Task 5 results table with an F1@tau_95 column.

Example:
  python scripts/eval/calibrate_threshold.py \\
      --cut_index /mnt/disks/splice-data/outputs/cut_index/cuts.parquet \\
      --embeddings /mnt/disks/splice-data/embeddings/dinov2_base \\
      --v0_dir /mnt/disks/splice-data/outputs/v0
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.pairs import build_pair_features_batch, load_embeddings  # noqa: E402
from src.eval.metrics import threshold_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibrate_threshold")

DEFAULT_CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
DEFAULT_EMB = "/mnt/disks/splice-data/embeddings/dinov2_base"
DEFAULT_V0 = "/mnt/disks/splice-data/outputs/v0"
DEFAULT_OUT = "/mnt/disks/splice-data/outputs/calibration"
WITHIN_SHOT_PAIRS = [(0, 1), (1, 2), (0, 2)]


def within_shot_embeddings(
    cut_index: str, split: str, emb: np.ndarray, key2row: dict[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Build (eA, eB) for every within-shot keyframe pair of every shot in ``split``."""
    df = pd.read_parquet(
        cut_index, columns=["movie_id", "shot_left_idx", "shot_right_idx", "split"]
    )
    df = df[df["split"] == split]
    shots: set[tuple[str, int]] = set()
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
    log.info("%s: %d shots -> %d within-shot pairs", split, len(shots), len(rows_a))
    return emb[np.array(rows_a)], emb[np.array(rows_b)]


def cosine_inconsistency(e_a: np.ndarray, e_b: np.ndarray) -> np.ndarray:
    """1 - cosine similarity, row-wise."""
    num = np.sum(e_a * e_b, axis=1)
    den = np.linalg.norm(e_a, axis=1) * np.linalg.norm(e_b, axis=1)
    return 1.0 - num / np.clip(den, 1e-8, None)


def logistic_scores(clf, e_a: np.ndarray, e_b: np.ndarray, chunk: int = 50_000) -> np.ndarray:
    """Positive-class probability of the trained logistic model, chunked."""
    out = np.empty(len(e_a), dtype=np.float64)
    for i in range(0, len(e_a), chunk):
        feats = build_pair_features_batch(e_a[i : i + chunk], e_b[i : i + chunk])
        out[i : i + chunk] = clf.predict_proba(feats)[:, 1]
    return out


def dist_stats(values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "p50": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut_index", default=DEFAULT_CUT_INDEX)
    ap.add_argument("--embeddings", default=DEFAULT_EMB)
    ap.add_argument("--v0_dir", default=DEFAULT_V0, help="Task 5 outputs (model + scores.npz)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--quantile", type=float, default=0.95)
    ap.add_argument("--out", default=DEFAULT_OUT)
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

    # cut-level scores for the same split (from Task 5) for the comparison plot
    npz = np.load(v0_dir / "scores.npz")
    cut_level = {name: npz[f"{name}__val_s"] for name in within if f"{name}__val_s" in npz.files}

    calibration = {}
    for name, w in within.items():
        tau = float(np.quantile(w, args.quantile))
        calibration[name] = {
            f"tau_{int(args.quantile * 100)}": tau,
            "distribution_stats": dist_stats(w),
        }
        log.info("%s: tau_%d = %.4f", name, int(args.quantile * 100), tau)

    (out_dir / "calibration.json").write_text(json.dumps(calibration, indent=2))

    # ---- figure: within-shot vs cut-level distributions ---------------------
    fig, axes = plt.subplots(1, len(within), figsize=(6 * len(within), 4))
    axes = np.atleast_1d(axes)
    for ax, (name, w) in zip(axes, within.items()):
        ax.hist(w, bins=60, density=True, alpha=0.6, label="within-shot (consistent)")
        if name in cut_level:
            ax.hist(cut_level[name], bins=60, density=True, alpha=0.6, label="cut-level")
        tau = calibration[name][f"tau_{int(args.quantile * 100)}"]
        ax.axvline(tau, color="r", ls="--", label=f"tau_{int(args.quantile * 100)}={tau:.3f}")
        ax.set(xlabel="inconsistency score", ylabel="density", title=name)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "within_shot_vs_cut.png", dpi=110)
    plt.close(fig)

    # ---- augment the Task 5 results table with F1@tau_95 --------------------
    results_path = v0_dir / "results.json"
    if results_path.exists():
        rows = json.loads(results_path.read_text())
        for row in rows:
            name = row["model"]
            key = f"tau_{int(args.quantile * 100)}"
            if name in calibration and f"{name}__test_s" in npz.files:
                tau = calibration[name][key]
                m = threshold_metrics(npz[f"{name}__test_y"], npz[f"{name}__test_s"], tau)
                row[f"f1_at_{key}"] = m["f1"]
                row[f"precision_at_{key}"] = m["precision"]
                row[f"recall_at_{key}"] = m["recall"]
            else:
                row[f"f1_at_{key}"] = None
        (v0_dir / "results_calibrated.json").write_text(json.dumps(rows, indent=2))
        table = pd.DataFrame(rows)
        cols = ["model", "auroc", "auprc", "f1_at_val_thr", f"f1_at_tau_{int(args.quantile*100)}"]
        table = table[[c for c in cols if c in table.columns]]
        table.to_csv(v0_dir / "results_calibrated.csv", index=False)
        print("\n=== v0 results with F1@tau_95 ===")
        print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print(f"\nsaved calibration -> {out_dir}")


if __name__ == "__main__":
    main()
