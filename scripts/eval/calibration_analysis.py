"""Calibration analysis: where the within-shot thresholds sit, and why tau_95 is
a high-recall operating point.

Diagnostics only -- reads cached v0 artifacts; does not retrain or rescore v0.
Compares the within-shot score distribution (genuinely-continuous frame pairs)
with the cut-level distribution, locates tau_95 / tau_99 / the val-optimal
threshold within the cut-level distribution, and reports test precision/recall/F1
at each so a deployable operating point can be chosen.

Writes reports/figures/score_distributions.png and prints a markdown section to
paste into reports/v0_results.md.
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

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from calibrate_threshold import (  # noqa: E402
    cosine_inconsistency,
    logistic_scores,
    within_shot_embeddings,
)
from src.data.pairs import load_embeddings  # noqa: E402
from src.eval.metrics import threshold_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibration_analysis")

REPO = HERE.parents[2]
DEFAULT_CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
DEFAULT_EMB = "/mnt/disks/splice-data/embeddings/dinov2_base"
DEFAULT_V0 = "/mnt/disks/splice-data/outputs/v0"
SCORERS = ["logistic", "raw_dino_cosine"]
LABELS = {"logistic": "logistic (v0)", "raw_dino_cosine": "raw DINOv2 cosine"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut_index", default=DEFAULT_CUT_INDEX)
    ap.add_argument("--embeddings", default=DEFAULT_EMB)
    ap.add_argument("--v0_dir", default=DEFAULT_V0)
    args = ap.parse_args()

    v0 = Path(args.v0_dir)
    emb, key2row = load_embeddings(args.embeddings)
    e_a, e_b = within_shot_embeddings(args.cut_index, "val", emb, key2row)
    clf = joblib.load(v0 / "v0_logistic.joblib")
    within = {
        "logistic": logistic_scores(clf, e_a, e_b),
        "raw_dino_cosine": cosine_inconsistency(e_a, e_b),
    }

    npz = np.load(v0 / "scores.npz")
    results = {r["model"]: r for r in json.loads((v0 / "results.json").read_text())}

    fig_dir = REPO / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    md: list[str] = []
    for ax, scorer in zip(axes, SCORERS):
        w = within[scorer]
        cut_val = npz[f"{scorer}__val_s"]
        test_y, test_s = npz[f"{scorer}__test_y"], npz[f"{scorer}__test_s"]
        tau95 = float(np.quantile(w, 0.95))
        tau99 = float(np.quantile(w, 0.99))
        val_thr = float(results[scorer]["val_thr"])

        ops = []
        for name, thr in [("tau_95", tau95), ("tau_99", tau99), ("val-optimal", val_thr)]:
            pct = float((cut_val < thr).mean() * 100)  # where thr sits in cut-level scores
            m = threshold_metrics(test_y, test_s, thr)
            ops.append((name, thr, pct, m["precision"], m["recall"], m["f1"]))

        md.append(f"\n**{LABELS[scorer]}**\n")
        md.append(
            "| operating point | threshold | cut-level percentile | precision | recall | F1 |"
        )
        md.append("|---|---|---|---|---|---|")
        for name, thr, pct, p, r, f1 in ops:
            md.append(f"| {name} | {thr:.3f} | {pct:.1f} | {p:.3f} | {r:.3f} | {f1:.3f} |")

        ax.hist(
            w, bins=60, range=(0, 1), density=True, alpha=0.55, label="within-shot (consistent)"
        )
        ax.hist(cut_val, bins=60, range=(0, 1), density=True, alpha=0.55, label="cut-level (val)")
        for name, thr, *_ in ops:
            ax.axvline(
                thr,
                ls="--",
                lw=1.2,
                color={"tau_95": "tab:red", "tau_99": "tab:purple", "val-optimal": "tab:green"}[
                    name
                ],
            )
        ax.set(xlabel="inconsistency score", ylabel="density", title=LABELS[scorer], xlim=(0, 1))
        ax.legend(fontsize=8)

    fig.suptitle(
        "Within-shot vs cut-level score distributions "
        "(dashed: red tau_95, purple tau_99, green val-optimal)"
    )
    fig.tight_layout()
    fig.savefig(fig_dir / "score_distributions.png", dpi=120)
    plt.close(fig)

    print("\n".join(md))
    print("\nwrote reports/figures/score_distributions.png")


if __name__ == "__main__":
    main()
