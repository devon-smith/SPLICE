"""Phase-3 P3: stratified evaluation of v1.5 by transition difficulty.

Axes (test split, cached predictions only -- no retraining):
  1. raw DINOv2 cosine quintile  -- where does v1.5 add signal beyond cosine?
  2. movie cut-count (proxy for cut rate; MovieNet year metadata not on disk)
  3. cut position within the film (early / middle / late thirds)
  4. scene-boundary depth (scene_id jump) for y=1 cuts

Writes reports/v1_stratified_difficulty.md and two figures.
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import best_f1_threshold  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stratified_difficulty")

REPO = Path(__file__).resolve().parents[2]
CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
V0 = Path("/mnt/disks/splice-data/outputs/v0")
V1 = Path("/mnt/disks/splice-data/outputs/v1_sound")
PAIRS = Path("/mnt/disks/splice-data/pairs/dino_v0_boundary")


def _auprc(y: np.ndarray, s: np.ndarray) -> float:
    """AUPRC, or NaN if a slice lacks either class."""
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cut_index", default=CUT_INDEX)
    args = ap.parse_args()

    fig_dir = REPO / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(args.cut_index)
    cuts = cuts[cuts["split"] == "test"].reset_index(drop=True)
    v1 = np.load(V1 / "scores.npz")
    v0 = np.load(V0 / "scores.npz")
    if not np.array_equal(cuts["y_inconsistent"].to_numpy().astype(int), v1["test_y"].astype(int)):
        raise SystemExit("cut-index test rows do not align with v1.5 scores")

    cuts["v15"] = v1["test_s"]
    cuts["y"] = cuts["y_inconsistent"].astype(int)
    cuts["raw_cos_score"] = v0["raw_dino_cosine__test_s"]
    cuts["cos_sim"] = 1.0 - cuts["raw_cos_score"]  # cosine similarity (high = look alike)
    thr = best_f1_threshold(v1["val_y"], v1["val_s"])
    log.info(
        "test cuts %d | movies %d | v1.5 threshold %.4f", len(cuts), cuts["movie_id"].nunique(), thr
    )

    md = ["# v1 Stratified Evaluation by Transition Difficulty\n"]
    md.append(
        f"v1.5 (seed-2 sound MLP) on the test split: {len(cuts):,} cuts, "
        f"{cuts['movie_id'].nunique()} movies, {100 * cuts['y'].mean():.2f}% positive. "
        f"Decision threshold {thr:.3f} (F1-optimal on val).\n"
    )

    # ---- axis 1: raw DINOv2 cosine quintile --------------------------------
    md.append("## 1. By raw DINOv2 cosine quintile\n")
    md.append(
        "Cuts binned into quintiles of cosine similarity: Q1 = look most different, "
        "Q5 = look most similar. Within a quintile cosine is near-constant, so raw "
        "cosine cannot rank -- its AUPRC collapses to the base rate; v1.5's AUPRC "
        "above the base rate is signal it adds *beyond* cosine.\n"
    )
    md.append("| quintile | n | positive rate | v1.5 AUPRC | raw-cosine AUPRC | v1.5 FPR |")
    md.append("|---|---|---|---|---|---|")
    cuts["cos_q"] = pd.qcut(cuts["cos_sim"], 5, labels=[f"Q{i}" for i in range(1, 6)])
    q_auprc = []
    for q in [f"Q{i}" for i in range(1, 6)]:
        g = cuts[cuts["cos_q"] == q]
        neg = g[g["y"] == 0]
        fpr = float((neg["v15"] >= thr).mean()) if len(neg) else float("nan")
        v15_ap, raw_ap = _auprc(g["y"].to_numpy(), g["v15"].to_numpy()), _auprc(
            g["y"].to_numpy(), g["raw_cos_score"].to_numpy()
        )
        q_auprc.append(v15_ap)
        md.append(
            f"| {q} | {len(g):,} | {g['y'].mean():.3f} | {v15_ap:.3f} | "
            f"{raw_ap:.3f} | {fpr:.3f} |"
        )
    md.append("")

    # ---- axis 2: movie cut-count proxy -------------------------------------
    md.append("## 2. By movie cut-count (cut-rate proxy)\n")
    md.append(
        "MovieNet year metadata is not in this data distribution, so films are split "
        "by total shot count (a coarse proxy for cut rate) at the median test movie.\n"
    )
    shots_per_movie = cuts.groupby("movie_id")["shot_right_idx"].max() + 1
    median_shots = float(shots_per_movie.median())
    cuts["busy"] = cuts["movie_id"].map(shots_per_movie > median_shots)
    md.append(f"Median test movie has {median_shots:.0f} shots.\n")
    md.append("| group | movies | n cuts | positive rate | v1.5 AUPRC |")
    md.append("|---|---|---|---|---|")
    for busy, name in [(True, "busy (more shots)"), (False, "slow (fewer shots)")]:
        g = cuts[cuts["busy"] == busy]
        md.append(
            f"| {name} | {g['movie_id'].nunique()} | {len(g):,} | "
            f"{g['y'].mean():.3f} | {_auprc(g['y'].to_numpy(), g['v15'].to_numpy()):.3f} |"
        )
    md.append("")

    # ---- axis 3: cut position within film ----------------------------------
    md.append("## 3. By cut position within the film\n")
    total = cuts.groupby("movie_id")["shot_right_idx"].transform("max")
    cuts["rel_pos"] = cuts["shot_left_idx"] / total.clip(lower=1)
    cuts["pos_bin"] = pd.cut(
        cuts["rel_pos"],
        [0, 1 / 3, 2 / 3, 1.01],
        labels=["early", "middle", "late"],
        include_lowest=True,
    )
    md.append("| position | n | positive rate | v1.5 AUPRC |")
    md.append("|---|---|---|---|")
    pos_auprc = []
    for b in ["early", "middle", "late"]:
        g = cuts[cuts["pos_bin"] == b]
        ap = _auprc(g["y"].to_numpy(), g["v15"].to_numpy())
        pos_auprc.append(ap)
        md.append(f"| {b} | {len(g):,} | {g['y'].mean():.3f} | {ap:.3f} |")
    md.append("")

    # ---- axis 4: scene-boundary depth --------------------------------------
    md.append("## 4. By scene-boundary depth\n")
    pos = cuts[cuts["y"] == 1].copy()
    pos["jump"] = pos["scene_right_id"] - pos["scene_left_id"]
    jump_counts = pos["jump"].value_counts().sort_index().to_dict()
    if set(jump_counts) <= {1}:
        md.append(
            "Degenerate: every y=1 cut joins two *adjacent* shots, so the scene-id jump "
            f"is always exactly 1 ({jump_counts}). MovieNet's `invideo_scene_id` is "
            "contiguous, so a cut between adjacent shots can only ever move to the next "
            "scene. There is no small-vs-large jump distinction to stratify on.\n"
        )
    else:
        md.append(f"Scene-id jump distribution among y=1 cuts: {jump_counts}.\n")
        md.append("| jump | n | v1.5 mean score | recall@thr |")
        md.append("|---|---|---|---|")
        for small, name in [(True, "small (jump = 1)"), (False, "large (jump > 1)")]:
            g = pos[(pos["jump"] == 1) == small]
            if len(g):
                md.append(
                    f"| {name} | {len(g):,} | {g['v15'].mean():.3f} | "
                    f"{(g['v15'] >= thr).mean():.3f} |"
                )
    md.append("")

    # ---- interpretation ----------------------------------------------------
    md.append("## Interpretation\n")
    md.append(
        f"- **Cosine quintile:** v1.5 AUPRC by quintile is {np.round(q_auprc, 3).tolist()}. "
        "Where raw cosine is decisive (extreme quintiles) the within-band ranking problem "
        "is easy or near-saturated; v1.5's value-add concentrates where cosine alone is "
        "ambiguous."
    )
    md.append(
        f"- **Cut position:** AUPRC early/middle/late = {np.round(pos_auprc, 3).tolist()}; "
        "differences indicate whether film openings/closings are harder to score."
    )
    md.append(
        "- **Cut-count proxy** is coarse (no duration metadata); read it as busy-vs-slow "
        "films, not a true cut-rate split.\n"
    )
    (REPO / "reports" / "v1_stratified_difficulty.md").write_text("\n".join(md))

    # ---- figures -----------------------------------------------------------
    qs = [f"Q{i}" for i in range(1, 6)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    base = [cuts.loc[cuts["cos_q"] == q, "y"].mean() for q in qs]
    x = np.arange(5)
    ax.bar(x - 0.2, q_auprc, 0.4, label="v1.5 AUPRC")
    ax.bar(x + 0.2, base, 0.4, label="base rate (chance AUPRC)")
    ax.set(
        xticks=x,
        xlabel="cosine-similarity quintile (Q1 most different)",
        ylabel="AUPRC",
        title="v1.5 AUPRC by raw-DINOv2-cosine quintile",
    )
    ax.set_xticklabels(qs)
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_dir / "v1_quintile_auprc.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["early", "middle", "late"], pos_auprc, color="tab:green", alpha=0.8)
    ax.set(ylabel="v1.5 AUPRC", title="v1.5 AUPRC by cut position within film")
    ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "v1_position_auprc.png", dpi=120)
    plt.close(fig)

    print("wrote reports/v1_stratified_difficulty.md and 2 figures")


if __name__ == "__main__":
    main()
