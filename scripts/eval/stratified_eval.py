"""Stratified evaluation of the v0 test-set predictions.

Reads the cached v0 scores (no retraining, no rescoring) and slices them along:
  - class       y=0 (consistent) vs y=1 (scene-boundary cut)
  - same_movie  within-film vs cross-film
  - by movie    per-movie AUPRC + 95% movie-level bootstrap CI

Shot-scale transition stratification is intentionally skipped: MovieNet cinematic
-style annotations are not present in this data distribution (the BaSSL `anno`
files carry scene-boundary fields only). See reports/v0_stratified.md.

Writes reports/v0_stratified.md and reports/figures/v0_stratified_by_movie.png.

Example:
  python scripts/eval/stratified_eval.py
"""

import argparse
import json
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
from src.eval.metrics import threshold_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("stratified_eval")

REPO = Path(__file__).resolve().parents[2]
DEFAULT_V0 = "/mnt/disks/splice-data/outputs/v0"
DEFAULT_PAIRS = "/mnt/disks/splice-data/pairs/dino_v0_boundary"
MODELS = ["logistic", "raw_dino_cosine"]  # v0 model + handoff-specified baseline
LABELS = {"logistic": "logistic (v0)", "raw_dino_cosine": "raw DINOv2 cosine"}


def load_test_predictions(v0_dir: Path, pairs_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Test-split predictions joined with movie_id; plus per-model val thresholds."""
    npz = np.load(v0_dir / "scores.npz")
    meta = pd.read_parquet(pairs_dir / "meta.parquet")
    test_meta = meta[meta["split"] == "test"].reset_index(drop=True)

    df = pd.DataFrame({"movie_id": test_meta["movie_id"], "y": npz["logistic__test_y"]})
    if not np.array_equal(df["y"].to_numpy(), test_meta["y_inconsistent"].to_numpy()):
        raise SystemExit("scores.npz test labels do not match meta -- alignment broken")
    for m in MODELS:
        df[m] = npz[f"{m}__test_s"]
    results = {r["model"]: r for r in json.loads((v0_dir / "results.json").read_text())}
    thresholds = {m: results[m]["val_thr"] for m in MODELS}
    log.info("loaded %d test cuts across %d movies", len(df), df["movie_id"].nunique())
    return df, thresholds


def per_movie_auprc(df: pd.DataFrame, model: str) -> pd.Series:
    """AUPRC within each movie (movies lacking either class are dropped)."""
    out = {}
    for movie_id, g in df.groupby("movie_id"):
        pos = int(g["y"].sum())
        if 0 < pos < len(g):
            out[movie_id] = average_precision_score(g["y"], g[model])
    return pd.Series(out, name=model)


def bootstrap_ci(
    values: np.ndarray, n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean and 95% percentile CI from resampling the unit (movies) with replacement."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    boot_means = values[idx].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(boot_means, 2.5)),
        float(np.percentile(boot_means, 97.5)),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0_dir", default=DEFAULT_V0)
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    df, thresholds = load_test_predictions(Path(args.v0_dir), Path(args.pairs))
    fig_dir = REPO / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    md: list[str] = ["# v0 Stratified Evaluation\n"]
    md.append(
        "Slices of the v0 **test split** predictions "
        f"({len(df):,} cuts, {df['movie_id'].nunique()} movies, "
        f"{100 * df['y'].mean():.2f}% positive). Compares the v0 logistic model "
        "against the raw DINOv2 cosine baseline. No retraining -- cached scores only.\n"
    )

    # ---- axis 1: by class ---------------------------------------------------
    md.append("## By class\n")
    md.append(
        "Per-class count, mean predicted score, and hit-rate at the val-optimal "
        "threshold (recall for y=1, specificity for y=0). Hit-rates re-aggregate "
        "to the overall accuracy reported in v0_results.\n"
    )
    md.append("| model | class | n | mean score | hit-rate@thr |")
    md.append("|---|---|---|---|---|")
    for m in MODELS:
        for cls, name in [(1, "y=1 inconsistent"), (0, "y=0 consistent")]:
            sub = df[df["y"] == cls]
            preds = (sub[m] >= thresholds[m]).astype(int)
            hit = float((preds == cls).mean())
            md.append(
                f"| {LABELS[m]} | {name} | {len(sub):,} | " f"{sub[m].mean():.4f} | {hit:.4f} |"
            )
    md.append("")

    # ---- axis 2: same-movie -------------------------------------------------
    md.append("## Within-film vs cross-film\n")
    md.append(
        "Every cut in the index is a pair of **adjacent shots from the same movie** "
        "by construction, so `same_movie` is uniformly true: there are "
        f"{len(df):,} within-film cuts and 0 cross-film cuts. This axis is therefore "
        "degenerate for the current dataset and cannot be stratified. A cross-film "
        "evaluation would require synthesising cross-film shot pairs (out of scope "
        "for this phase); flagged for the team. All metrics below are within-film.\n"
    )

    # ---- axis 3: by movie ---------------------------------------------------
    md.append("## By movie\n")
    md.append(
        "Per-movie AUPRC (movie is the resampling unit, since cuts within a film "
        "are correlated). Mean is the macro average over movies; the 95% CI is a "
        f"movie-level bootstrap ({args.n_boot:,} resamples). Pooled AUPRC is the "
        "micro average over all test cuts (the v0_results headline number).\n"
    )
    md.append("| model | n movies | mean per-movie AUPRC | 95% CI | pooled AUPRC | pooled F1@thr |")
    md.append("|---|---|---|---|---|---|")
    per_movie = {}
    for m in MODELS:
        pm = per_movie_auprc(df, m)
        per_movie[m] = pm
        mean, lo, hi = bootstrap_ci(pm.to_numpy(), n_boot=args.n_boot)
        pooled_ap = average_precision_score(df["y"], df[m])
        pooled_f1 = threshold_metrics(df["y"], df[m], thresholds[m])["f1"]
        md.append(
            f"| {LABELS[m]} | {len(pm)} | {mean:.4f} | "
            f"[{lo:.4f}, {hi:.4f}] | {pooled_ap:.4f} | {pooled_f1:.4f} |"
        )
        log.info("%s: mean per-movie AUPRC %.4f  95%% CI [%.4f, %.4f]", m, mean, lo, hi)
    md.append("")

    # ---- axis 4: shot-scale (skipped) --------------------------------------
    md.append("## Shot-scale transitions (skipped)\n")
    md.append(
        "MovieNet cinematic-style annotations (shot scale: long / full / medium / "
        "close-up / extreme-close-up) are **not present** in this data "
        "distribution -- the BaSSL `anno` files carry scene-boundary fields only "
        "(`video_id, shot_id, boundary_label, invideo_scene_id, ...`). The 5x5 "
        "scale-transition matrix is therefore skipped rather than approximated. "
        "It can be added later if the official MovieNet meta package is fetched.\n"
    )

    # ---- analysis -----------------------------------------------------------
    log_mean = per_movie["logistic"].mean()
    raw_mean = per_movie["raw_dino_cosine"].mean()
    spread = per_movie["logistic"].max() - per_movie["logistic"].min()
    md.append("## Analysis\n")
    md.append(
        f"The logistic model leads the raw-cosine baseline at the movie level too "
        f"(mean per-movie AUPRC {log_mean:.3f} vs {raw_mean:.3f}), consistent with the "
        f"pooled result. Per-movie AUPRC varies widely (range ~{spread:.2f} across test "
        "movies): continuity is far easier to score in some films than others, so the "
        "movie-level bootstrap CI -- not the point estimate -- is the honest summary. "
        "The within-film/cross-film axis is degenerate here because every labelled cut "
        "is an adjacent same-movie shot pair; a cross-film split is future work.\n"
    )

    (REPO / "reports" / "v0_stratified.md").write_text("\n".join(md))

    # ---- figure -------------------------------------------------------------
    fig, axx = plt.subplots(figsize=(6, 5))
    data = [per_movie[m].to_numpy() for m in MODELS]
    axx.boxplot(data, tick_labels=[LABELS[m] for m in MODELS], showmeans=True)
    axx.set_ylabel("per-movie AUPRC (test split)")
    axx.set_title("v0 per-movie AUPRC: logistic vs raw DINOv2 cosine")
    axx.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "v0_stratified_by_movie.png", dpi=120)
    plt.close(fig)

    print("\nwrote reports/v0_stratified.md and reports/figures/v0_stratified_by_movie.png")


if __name__ == "__main__":
    main()
