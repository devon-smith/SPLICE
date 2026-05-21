"""Phase-3 P4: per-cut error analysis of v1.5.

Identifies the most confident mistakes and successes on the test split:
  - top-50 false positives  (high v1.5 score, y=0)
  - top-50 false negatives  (low  v1.5 score, y=1)
  - top-20 well-handled positives / negatives (for contrast)

Writes per-category CSVs (all model scores + 6 keyframe paths), copies each
example's keyframes under reports/figures/error_grid/ (git-ignored), renders two
composite grid figures, and writes a descriptive analysis to
reports/v1_error_analysis.md. Describes the data only -- no causal speculation.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import best_f1_threshold  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("error_analysis")

REPO = Path(__file__).resolve().parents[2]
CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
V0 = Path("/mnt/disks/splice-data/outputs/v0")
V1 = Path("/mnt/disks/splice-data/outputs/v1_sound")
KEYFRAMES = [
    "left_img0_path",
    "left_img1_path",
    "left_img2_path",
    "right_img0_path",
    "right_img1_path",
    "right_img2_path",
]
CSV_COLS = [
    "cut_id",
    "movie_id",
    "shot_left_idx",
    "shot_right_idx",
    "y",
    "v15_score",
    "logistic_score",
    "raw_dino_cosine_score",
    "hsv_score",
    "clip_score",
    "cos_sim",
] + KEYFRAMES


def load_test_df() -> tuple[pd.DataFrame, float]:
    cuts = pd.read_parquet(CUT_INDEX)
    cuts = cuts[cuts["split"] == "test"].reset_index(drop=True)
    v1, v0 = np.load(V1 / "scores.npz"), np.load(V0 / "scores.npz")
    if not np.array_equal(cuts["y_inconsistent"].to_numpy().astype(int), v1["test_y"].astype(int)):
        raise SystemExit("cut-index test rows do not align with v1.5 scores")
    cuts["y"] = cuts["y_inconsistent"].astype(int)
    cuts["v15_score"] = v1["test_s"]
    cuts["logistic_score"] = v0["logistic__test_s"]
    cuts["raw_dino_cosine_score"] = v0["raw_dino_cosine__test_s"]
    cuts["hsv_score"] = v0["hsv_chisq__test_s"]
    cuts["clip_score"] = v0["clip_cosine__test_s"]
    cuts["cos_sim"] = 1.0 - cuts["raw_dino_cosine_score"]
    cuts["cut_id"] = cuts["movie_id"] + "_" + cuts["shot_left_idx"].astype(str).str.zfill(4)
    return cuts, best_f1_threshold(v1["val_y"], v1["val_s"])


def copy_keyframes(subset: pd.DataFrame, prefix: str, grid_dir: Path) -> int:
    """Copy each example's 6 keyframes under grid_dir/<prefix>_<i>/; returns # missing."""
    missing = 0
    for i, (_, row) in enumerate(subset.iterrows()):
        dst = grid_dir / f"{prefix}_{i:02d}"
        dst.mkdir(parents=True, exist_ok=True)
        for col in KEYFRAMES:
            src = Path(row[col])
            if src.exists():
                shutil.copy(src, dst / src.name)
            else:
                missing += 1
                log.warning("missing keyframe %s", src)
    return missing


def make_grid(subset: pd.DataFrame, title: str, out_path: Path, n: int = 6) -> None:
    """Composite grid: n examples, each the two boundary frames (left img2, right img0)."""
    rows = subset.head(n)
    fig, axes = plt.subplots(len(rows), 2, figsize=(6, 2.7 * len(rows)))
    for r, (_, row) in enumerate(rows.iterrows()):
        for c, (col, side) in enumerate(
            [("left_img2_path", "left img2"), ("right_img0_path", "right img0")]
        ):
            ax = axes[r, c]
            ax.axis("off")
            p = Path(row[col])
            if p.exists():
                ax.imshow(Image.open(p).convert("RGB"))
            else:
                ax.text(0.5, 0.5, "missing", ha="center", va="center")
            if c == 0:
                ax.set_title(
                    f"{row['movie_id']} sh{row['shot_left_idx']} "
                    f"v1.5={row['v15_score']:.2f} y={row['y']}",
                    fontsize=8,
                    loc="left",
                )
            else:
                ax.set_title(side, fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def describe(subset: pd.DataFrame, thr: float, kind: str) -> list[str]:
    """Descriptive stats for one error category -- data only, no causal claims."""
    s = subset["v15_score"]
    by_movie = subset["movie_id"].value_counts()
    top = ", ".join(f"{m} ({c})" for m, c in by_movie.head(3).items())
    md = [
        f"### {kind}  (n = {len(subset):,})\n",
        f"- distinct movies: {subset['movie_id'].nunique()} of {64}; " f"top contributors: {top}",
        f"- single-movie max share: {by_movie.iloc[0]}/{len(subset)} "
        f"= {100 * by_movie.iloc[0] / len(subset):.1f}%",
        f"- v1.5 score: min {s.min():.3f}, median {s.median():.3f}, "
        f"mean {s.mean():.3f}, max {s.max():.3f}",
        f"- mean cosine similarity of the cut: {subset['cos_sim'].mean():.3f}",
    ]
    if kind.startswith("False positive"):
        margin = s - thr
        md.append(f"- distance above threshold ({thr:.3f}): median {margin.median():.3f}")
    if kind.startswith("False negative"):
        margin = thr - s
        near = float((margin < 0.1).mean())
        md.append(
            f"- distance below threshold ({thr:.3f}): median {margin.median():.3f}; "
            f"{100 * near:.1f}% are within 0.1 of the threshold (near-misses)"
        )
    md.append("")
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_errors", type=int, default=50)
    ap.add_argument("--n_well", type=int, default=20)
    args = ap.parse_args()

    df, thr = load_test_df()
    log.info("test cuts %d | v1.5 threshold %.4f", len(df), thr)
    grid_dir = REPO / "reports" / "figures" / "error_grid"
    grid_dir.mkdir(parents=True, exist_ok=True)

    all_fp = df[(df["y"] == 0) & (df["v15_score"] >= thr)]
    all_fn = df[(df["y"] == 1) & (df["v15_score"] < thr)]
    top_fp = df[df["y"] == 0].nlargest(args.n_errors, "v15_score")
    top_fn = df[df["y"] == 1].nsmallest(args.n_errors, "v15_score")
    well_pos = df[df["y"] == 1].nlargest(args.n_well, "v15_score")
    well_neg = df[df["y"] == 0].nsmallest(args.n_well, "v15_score")

    top_fp[CSV_COLS].to_csv(REPO / "reports" / "v1_false_positives.csv", index=False)
    top_fn[CSV_COLS].to_csv(REPO / "reports" / "v1_false_negatives.csv", index=False)
    pd.concat(
        [
            well_pos.assign(kind="well_handled_positive"),
            well_neg.assign(kind="well_handled_negative"),
        ]
    )[CSV_COLS + ["kind"]].to_csv(REPO / "reports" / "v1_well_handled.csv", index=False)

    miss_fp = copy_keyframes(top_fp, "fp", grid_dir)
    miss_fn = copy_keyframes(top_fn, "fn", grid_dir)
    log.info("keyframes copied; missing: %d (fp) + %d (fn)", miss_fp, miss_fn)

    make_grid(
        top_fp,
        "v1.5 false positives (most confident, y=0)",
        REPO / "reports" / "figures" / "v1_error_grid_fp.png",
    )
    make_grid(
        top_fn,
        "v1.5 false negatives (most confident misses, y=1)",
        REPO / "reports" / "figures" / "v1_error_grid_fn.png",
    )

    md = ["# v1.5 Error Analysis (Phase 3, P4)\n"]
    md.append(
        f"v1.5 (seed-2 sound MLP) on the test split, decision threshold {thr:.3f}. "
        f"Of {len(df):,} test cuts: {len(all_fp):,} false positives "
        f"({100 * len(all_fp) / (df['y'] == 0).sum():.1f}% of negatives) and "
        f"{len(all_fn):,} false negatives ({100 * len(all_fn) / (df['y'] == 1).sum():.1f}% "
        "of positives). The CSVs list the most confident 50 of each with all five model "
        "scores and keyframe paths; this section describes the *full* error sets. "
        "Description only -- qualitative interpretation is left to the team.\n"
    )
    md.append("## Error categories\n")
    md += describe(all_fp, thr, "False positives (y=0 scored above threshold)")
    md += describe(all_fn, thr, "False negatives (y=1 scored below threshold)")
    md.append("## Well-handled, for contrast\n")
    md.append(
        f"- top-{args.n_well} correct positives: mean v1.5 score "
        f"{well_pos['v15_score'].mean():.3f}"
    )
    md.append(
        f"- top-{args.n_well} correct negatives: mean v1.5 score "
        f"{well_neg['v15_score'].mean():.3f}\n"
    )
    md.append("## Open question for the team\n")
    md.append(
        "Which of these errors are genuine model failures versus MovieNet labelling "
        "noise? The cut index had ~0.86% boundary_label/scene-id disagreement at the "
        "source; some confident false positives/negatives may be mislabelled cuts. The "
        "grid figures (`v1_error_grid_fp.png`, `v1_error_grid_fn.png`) and the CSVs are "
        "for that qualitative review.\n"
    )
    (REPO / "reports" / "v1_error_analysis.md").write_text("\n".join(md))
    print(
        "wrote v1_error_analysis.md, 3 CSVs, 2 grid figures; "
        f"per-example keyframes under {grid_dir} (git-ignored)"
    )


if __name__ == "__main__":
    main()
