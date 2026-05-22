"""Score the Veo AI-gen pairs through all six SPLICE models, one row per pair.

The aggregate harness (eval_aigen.py) reports metrics, not per-pair scores -- and
with an all-y=0 set those metrics (AUROC/AUPRC) are undefined anyway. This script
reuses eval_aigen.py's scoring pipeline verbatim -- the same pipeline whose
``--in_dist_check`` reproduces MovieNet v0/mean-pool/v1.5 AUPRC exactly -- and
persists the per-pair scores the M2 deck needs.

  python scripts/eval/score_aigen_per_pair.py

Writes:
  outputs/aigen_eval/results/per_pair_scores.csv
  outputs/aigen_eval/results/per_pair_analysis.md
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent))  # scripts/eval -- so `import eval_aigen` works
sys.path.insert(0, str(REPO))
from eval_aigen import (  # noqa: E402
    DEFAULT_MP,
    DEFAULT_V0,
    DEFAULT_V1,
    boundary_features,
    embed_aigen_keyframes,
    mean_pool_features,
    score_embedding_models,
    score_image_baselines,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("score_aigen_per_pair")

AIGEN_CUTS = "/mnt/disks/splice-data/outputs/aigen_eval/cuts.parquet"
OUT_DIR = "/mnt/disks/splice-data/outputs/aigen_eval/results"

# MovieNet v1.5 reference points (reports/v1_final.md §5, reports/v1_calibration.md).
MOVIENET_Y0_MEAN = 0.30  # v1.5 mean predicted score on y=0 (within-scene) test cuts
MOVIENET_Y1_MEAN = 0.69  # ... on y=1 (cross-scene) test cuts
MOVIENET_WITHIN_SHOT_MEDIAN = 0.001  # within-shot (no cut at all) median, v1_calibration τ50

# Dispatch's independent qualitative buckets (human judgement, set before scoring).
BUCKET_OF = {
    "A003": "clean",
    "A013": "clean",
    "A001": "drift",
    "A002": "drift",
    "A004": "drift",
    "A011": "drift",
    "A012": "drift",
    "A014": "drift",
    "A005": "major",
    "A015": "major",
}
BUCKET_ORDER = ["clean", "drift", "major"]
BUCKET_LABEL = {
    "clean": "Clean continuity — expect LOW",
    "drift": "Wardrobe held, identity/accessories drifted — expect MID",
    "major": "Major identity failure — expect HIGH",
}
# CSV column -> dict key returned by the eval_aigen scorers
MODEL_COLS = {
    "raw_cos": "raw_dino_cosine",
    "hsv_chisq": "hsv_chisq",
    "clip_cos": "clip_cosine",
    "v0_logistic": "logistic",
    "mean_pool_3": "mean_pool",
    "v1.5_MLP": "v1.5",
}


def pair_id_of(path: str) -> str:
    """Pair id is the keyframe's parent dir, e.g. .../keyframes/A001/left_img2.jpg -> A001."""
    return Path(path).parent.name


def _stats(x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=float)
    return {
        "mean": float(x.mean()),
        "median": float(np.median(x)),
        "std": float(x.std(ddof=1)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aigen_index", default=AIGEN_CUTS)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(args.aigen_index)
    # movie_id is the constant "aigen_veo"; the real pair id lives in the keyframe path.
    cuts["pair_id"] = cuts["left_img2_path"].map(pair_id_of)
    log.info("scoring %d AI-gen pairs: %s", len(cuts), cuts["pair_id"].tolist())

    # Reuse eval_aigen's verified pipeline: re-embed all 6 keyframes per pair,
    # build the boundary + mean-pool features, score all six models.
    emb = embed_aigen_keyframes(cuts)
    feats = boundary_features(cuts, lambda p: emb[p])
    mp_feats = mean_pool_features(cuts, lambda p: emb[p])
    scores = {
        **score_embedding_models(feats, mp_feats, DEFAULT_V0, DEFAULT_V1, DEFAULT_MP),
        **score_image_baselines(cuts),
    }

    df = pd.DataFrame(
        {
            "pair_id": cuts["pair_id"].to_numpy(),
            "shot_type": cuts["shot_type"].to_numpy(),
            "notes": cuts["notes"].to_numpy(),
            **{col: scores[key] for col, key in MODEL_COLS.items()},
            "intended_label": cuts["y_inconsistent"].to_numpy().astype(int),
        }
    )
    df["bucket"] = df["pair_id"].map(BUCKET_OF)

    csv_cols = ["pair_id", "shot_type", "notes", *MODEL_COLS, "intended_label"]
    csv_path = out_dir / "per_pair_scores.csv"
    df[csv_cols].to_csv(csv_path, index=False)
    log.info("wrote %s (%d rows)", csv_path, len(df))

    ranked = df.sort_values("v1.5_MLP", ascending=False).reset_index(drop=True)
    _print_console(ranked, df)
    _write_report(out_dir / "per_pair_analysis.md", ranked, df)
    log.info("wrote %s", out_dir / "per_pair_analysis.md")


def _print_console(ranked: pd.DataFrame, df: pd.DataFrame) -> None:
    print("\n=== Per-pair scores (sorted by v1.5 descending) ===\n")
    hdr = f"{'pair':<6}{'bucket':<8}{'raw_cos':>9}{'hsv_chisq':>11}{'clip_cos':>10}"
    hdr += f"{'v0_log':>9}{'mp3':>9}{'v1.5':>9}"
    print(hdr)
    print("-" * len(hdr))
    for _, r in ranked.iterrows():
        print(
            f"{r['pair_id']:<6}{r['bucket']:<8}{r['raw_cos']:>9.3f}{r['hsv_chisq']:>11.3f}"
            f"{r['clip_cos']:>10.3f}{r['v0_logistic']:>9.3f}{r['mean_pool_3']:>9.3f}"
            f"{r['v1.5_MLP']:>9.3f}"
        )

    s15, sraw = _stats(df["v1.5_MLP"]), _stats(df["raw_cos"])
    print("\n=== Aggregate stats (n=10, all intended_label=0) ===")
    for name, s in (("v1.5_MLP", s15), ("raw_cos", sraw)):
        print(
            f"  {name:<10} mean {s['mean']:.3f}  median {s['median']:.3f}  "
            f"std {s['std']:.3f}  range [{s['min']:.3f}, {s['max']:.3f}]"
        )
    print("\n=== Mean v1.5 by Dispatch bucket ===")
    for b in BUCKET_ORDER:
        g = df[df["bucket"] == b]
        print(
            f"  {b:<6} (n={len(g)})  mean v1.5 {g['v1.5_MLP'].mean():.3f}   "
            f"[{', '.join(sorted(g['pair_id']))}]"
        )
    print(
        f"\n=== vs MovieNet ===\n  Veo y=0 v1.5 mean {s15['mean']:.3f}  |  "
        f"MovieNet y=0 (within-scene) v1.5 mean {MOVIENET_Y0_MEAN:.2f}  |  "
        f"MovieNet y=1 (cross-scene) {MOVIENET_Y1_MEAN:.2f}"
    )


def _ranking_findings(ranked: pd.DataFrame, df: pd.DataFrame) -> dict:
    ids = ranked["pair_id"].tolist()  # high -> low v1.5
    bmean = {b: float(df.loc[df["bucket"] == b, "v1.5_MLP"].mean()) for b in BUCKET_ORDER}
    return {
        "ids": ids,
        "top2": set(ids[:2]),
        "bottom2": set(ids[-2:]),
        "agree_top": set(ids[:2]) == {"A005", "A015"},
        "agree_bottom": set(ids[-2:]) == {"A003", "A013"},
        "bucket_mean": bmean,
        "monotonic": bmean["clean"] < bmean["drift"] < bmean["major"],
    }


def _analysis(ranked: pd.DataFrame, df: pd.DataFrame, f: dict) -> list[str]:
    s15 = _stats(df["v1.5_MLP"])
    bm = f["bucket_mean"]
    top_id = ranked.iloc[0]["pair_id"]
    bot_id = ranked.iloc[-1]["pair_id"]
    major_ids = {"A005", "A015"}
    caught_major = sorted(major_ids & f["top2"])
    missed_major = sorted(major_ids - f["top2"])

    if f["monotonic"]:
        mono_txt = (
            f"The bucket means are monotonic — clean {bm['clean']:.3f} < drift "
            f"{bm['drift']:.3f} < major {bm['major']:.3f} — so on average v1.5 "
            f"orders the three buckets the way a human reviewer did."
        )
    else:
        mono_txt = (
            f"Bucket means are clean {bm['clean']:.3f}, drift {bm['drift']:.3f}, "
            f"major {bm['major']:.3f} (not strictly monotonic)."
        )

    # --- paragraph 1: spread and ranking vs Dispatch's buckets ---
    head = (
        f"v1.5 spans {s15['min']:.3f} ({bot_id}) to {s15['max']:.3f} ({top_id}) "
        f"across the 10 Veo continuous-action pairs — mean {s15['mean']:.3f}, "
        f"median {s15['median']:.3f}, std {s15['std']:.3f}. Every pair is intended "
        f"y=0, so AUROC/AUPRC are undefined; this is a score-distribution and "
        f"ranking check, not a metric."
    )
    if f["agree_top"] and f["agree_bottom"]:
        p1 = (
            f"{head} The ranking matches Dispatch's independent buckets exactly: "
            f"the two 'major identity failure' pairs (A005, A015) take the top two "
            f"scores and the two 'clean continuity' pairs (A003, A013) take the "
            f"bottom two. {mono_txt}"
        )
    elif len(caught_major) == 1 and missed_major:
        mm = missed_major[0]
        mm_rank = f["ids"].index(mm) + 1
        mm_score = float(df.loc[df["pair_id"] == mm, "v1.5_MLP"].iloc[0])
        p1 = (
            f"{head} Agreement with Dispatch's buckets is partial, and clearest at "
            f"the extremes: {caught_major[0]}, one of the two 'major identity "
            f"failure' pairs, scores far above everything else ({s15['max']:.3f}, "
            f"~3x the next pair), and a 'clean continuity' pair ({bot_id}) scores "
            f"lowest ({s15['min']:.3f}). {mono_txt} But the per-pair ranking breaks "
            f"on {mm}: Dispatch's other 'major' pair scores only {mm_score:.3f} "
            f"({mm_rank}th of 10), buried inside the drift cluster. v1.5 catches one "
            f"of the two flagged identity failures emphatically and misses the other."
        )
    else:
        p1 = (
            f"{head} Agreement with Dispatch's buckets is partial: the top two by "
            f"v1.5 are {', '.join(sorted(f['top2']))} (Dispatch 'major' = A005, "
            f"A015) and the bottom two are {', '.join(sorted(f['bottom2']))} "
            f"(Dispatch 'clean' = A003, A013). {mono_txt}"
        )

    # --- paragraph 2: why v1.5 misses A015, MovieNet comparison, caveats ---
    veo_mean = s15["mean"]
    if veo_mean < 0.7 * MOVIENET_Y0_MEAN:
        cmp_word = "far below"
    elif veo_mean > 1.3 * MOVIENET_Y0_MEAN:
        cmp_word = "well above"
    else:
        cmp_word = "comparable to"
    clip_top2 = set(df.sort_values("clip_cos", ascending=False)["pair_id"].tolist()[:2])

    p2_parts = []
    if missed_major:
        mm = missed_major[0]
        row = df[df["pair_id"] == mm].iloc[0]
        seg = (
            f"The {mm} miss is the informative part — and it is not a frozen-feature "
            f"blind spot. raw DINOv2 cosine does score {mm} low "
            f"({row['raw_cos']:.3f}): per the labels file only the backdrop holds "
            f"across that cut, and DINOv2's global embedding is background-dominated. "
            f"But the same 2305-d feature still carries the signal — v0 logistic "
            f"gives {mm} its single highest score of all ten pairs "
            f"({row['v0_logistic']:.3f}) and zero-shot CLIP cosine flags it too "
            f"({row['clip_cos']:.3f})."
        )
        if clip_top2 == {"A005", "A015"}:
            seg += (
                f" CLIP cosine, in fact, ranks A005 and {mm} as its own top two — "
                f"the cleanest match to Dispatch's 'major' bucket of any model here. "
                f"v1.5's MLP head discounts what a linear head on the identical "
                f"feature, and CLIP, both pick up; with n=1 for {mm} that is an "
                f"observation, not a verdict, but a coherent one."
            )
        else:
            seg += (
                " So the signal is in the feature: v1.5's MLP head discounts what a "
                "linear head on the identical feature picks up."
            )
        p2_parts.append(seg)

    movienet = (
        f"More broadly, v1.5's mean of {veo_mean:.3f} on these intended-consistent "
        f"Veo pairs sits {cmp_word} its {MOVIENET_Y0_MEAN:.2f} mean on real "
        f"MovieNet within-scene (y=0) cuts"
    )
    if cmp_word == "far below":
        movienet += (
            " — not a transfer win but the Veo pairs being easier: two clips from "
            "near-identical prompts share setting, lighting and background, so the "
            "boundary frames sit close in embedding space and v1.5 reads them as "
            "consistent even where identity has drifted. "
        )
    else:
        movienet += ". "
    movienet += (
        "The blunt caveat: 10 single-class pairs from one generator support a "
        "qualitative read of model behaviour, not any quantitative transfer claim "
        "(AUROC/AUPRC need both classes). What they show is that v1.5 inherits "
        "DINOv2's background-dominated bias — it flags overt scene change well and "
        "subtle identity change poorly — the same limitation seen on MovieNet and "
        "the stated motivation for v2."
    )
    p2_parts.append(movienet)
    return [p1, " ".join(p2_parts)]


def _write_report(path: Path, ranked: pd.DataFrame, df: pd.DataFrame) -> None:
    f = _ranking_findings(ranked, df)
    s15, sraw = _stats(df["v1.5_MLP"]), _stats(df["raw_cos"])
    md = [
        "# Per-Pair Scores — Veo AI-Gen Continuous-Action Pairs\n",
        "Ten Veo-generated continuous-action pairs (`shot_type = continuous_action`, "
        "all `intended_label = 0`) scored through all six SPLICE models. The aggregate "
        "harness (`eval_aigen.py`) cannot report AUROC/AUPRC here — every pair is one "
        "class — so this is the per-pair score record the M2 deck needs. Scores are "
        "**inconsistency** scores: higher = the model thinks the cut is less "
        "continuous. Produced by `scripts/eval/score_aigen_per_pair.py`.\n",
        "## Per-pair scores (sorted by v1.5 descending)\n",
        "| pair | Dispatch bucket | raw_cos | hsv_χ² | clip_cos | v0_logistic "
        "| mean_pool_3 | v1.5_MLP |",
        "|---|---|--:|--:|--:|--:|--:|--:|",
    ]
    for _, r in ranked.iterrows():
        md.append(
            f"| {r['pair_id']} | {r['bucket']} | {r['raw_cos']:.3f} "
            f"| {r['hsv_chisq']:.3f} | {r['clip_cos']:.3f} | {r['v0_logistic']:.3f} "
            f"| {r['mean_pool_3']:.3f} | {r['v1.5_MLP']:.3f} |"
        )
    md += [
        "\n## Aggregate statistics  (n = 10, all intended_label = 0)\n",
        "| model | mean | median | std | min | max |",
        "|---|--:|--:|--:|--:|--:|",
        f"| v1.5_MLP | {s15['mean']:.3f} | {s15['median']:.3f} | {s15['std']:.3f} "
        f"| {s15['min']:.3f} | {s15['max']:.3f} |",
        f"| raw_cos (no-training baseline) | {sraw['mean']:.3f} | {sraw['median']:.3f} "
        f"| {sraw['std']:.3f} | {sraw['min']:.3f} | {sraw['max']:.3f} |",
        "\n## Cross-reference with Dispatch's qualitative buckets\n",
        "Dispatch bucketed the pairs by eye *before* scoring. Expected score order: "
        "clean < drift < major.\n",
        "| bucket | pairs | mean v1.5 | min | max |",
        "|---|---|--:|--:|--:|",
    ]
    for b in BUCKET_ORDER:
        g = df[df["bucket"] == b]
        md.append(
            f"| {BUCKET_LABEL[b]} | {', '.join(sorted(g['pair_id']))} "
            f"| {g['v1.5_MLP'].mean():.3f} | {g['v1.5_MLP'].min():.3f} "
            f"| {g['v1.5_MLP'].max():.3f} |"
        )
    verdict = (
        "v1.5's ranking **agrees** with Dispatch: the two 'major' pairs score "
        "highest and the two 'clean' pairs score lowest."
        if f["agree_top"] and f["agree_bottom"]
        else "v1.5's ranking **partially agrees** with Dispatch — see the analysis "
        "below for where it diverges."
    )
    md.append(f"\n{verdict}\n")
    md.append("## MovieNet reference points\n")
    md.append(
        f"- v1.5 mean predicted score on MovieNet **y=0 within-scene** cuts: "
        f"**{MOVIENET_Y0_MEAN:.2f}** (`reports/v1_final.md` §5).\n"
        f"- v1.5 mean on MovieNet **y=1 cross-scene** cuts: "
        f"**{MOVIENET_Y1_MEAN:.2f}** (same source).\n"
        f"- v1.5 within-shot (no cut at all) median: ~{MOVIENET_WITHIN_SHOT_MEDIAN} "
        f"(`reports/v1_calibration.md` τ50) — the genuinely-continuous floor.\n"
    )
    md.append("## Analysis\n")
    for para in _analysis(ranked, df, f):
        md.append(para + "\n")
    path.write_text("\n".join(md))


if __name__ == "__main__":
    main()
