# score the 10 Veo AI-gen pairs through all 6 SPLICE models, one row per pair
# the aggregate harness (eval_aigen.py) reports metrics, not per-pair scores;
# with an all-y=0 set AUROC/AUPRC are undefined anyway.
# usage: python scripts/eval/score_aigen_per_pair.py

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from eval_aigen import (DEFAULT_MP, DEFAULT_V0, DEFAULT_V1,
                        boundary_features, embed_aigen_keyframes,
                        mean_pool_features, score_embedding_models, score_image_baselines)

AIGEN_CUTS = "/mnt/disks/splice-data/outputs/aigen_eval/cuts.parquet"
OUT_DIR = "/mnt/disks/splice-data/outputs/aigen_eval/results"

# Dispatch's qualitative buckets (human judgement set before scoring)
BUCKET_OF = {"A003": "clean", "A013": "clean",
             "A001": "drift", "A002": "drift", "A004": "drift",
             "A011": "drift", "A012": "drift", "A014": "drift",
             "A005": "major", "A015": "major"}
BUCKET_ORDER = ["clean", "drift", "major"]

# CSV column -> dict key returned by eval_aigen's scorers
MODEL_COLS = {"raw_cos": "raw_dino_cosine", "hsv_chisq": "hsv_chisq",
              "clip_cos": "clip_cosine", "v0_logistic": "logistic",
              "mean_pool_3": "mean_pool", "v1.5_MLP": "v1.5"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aigen_index", default=AIGEN_CUTS)
    ap.add_argument("--out", default=OUT_DIR)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    cuts = pd.read_parquet(args.aigen_index)
    # movie_id is constant "aigen_veo"; the real pair id lives in the keyframe path
    cuts["pair_id"] = cuts["left_img2_path"].map(lambda p: Path(p).parent.name)
    print(f"scoring {len(cuts)} AI-gen pairs: {cuts['pair_id'].tolist()}")

    # reuse eval_aigen's verified pipeline
    emb = embed_aigen_keyframes(cuts)
    feats = boundary_features(cuts, lambda p: emb[p])
    mp_feats = mean_pool_features(cuts, lambda p: emb[p])
    scores = {
        **score_embedding_models(feats, mp_feats, DEFAULT_V0, DEFAULT_V1, DEFAULT_MP),
        **score_image_baselines(cuts),
    }

    df = pd.DataFrame({
        "pair_id": cuts["pair_id"].to_numpy(),
        "shot_type": cuts["shot_type"].to_numpy(),
        "notes": cuts["notes"].to_numpy(),
        **{col: scores[key] for col, key in MODEL_COLS.items()},
        "intended_label": cuts["y_inconsistent"].to_numpy().astype(int),
    })
    df["bucket"] = df["pair_id"].map(BUCKET_OF)

    csv_cols = ["pair_id", "shot_type", "notes", *MODEL_COLS, "intended_label"]
    csv_path = out_dir / "per_pair_scores.csv"
    df[csv_cols].to_csv(csv_path, index=False)
    print(f"wrote {csv_path} ({len(df)} rows)")

    # console: per-pair table sorted by v1.5 desc
    ranked = df.sort_values("v1.5_MLP", ascending=False).reset_index(drop=True)
    print("\nper-pair scores (sorted by v1.5 desc)")
    print(f"{'pair':<6}{'bucket':<8}{'raw_cos':>9}{'hsv_chisq':>11}{'clip_cos':>10}"
          f"{'v0_log':>9}{'mp3':>9}{'v1.5':>9}")
    for _, r in ranked.iterrows():
        print(f"{r['pair_id']:<6}{r['bucket']:<8}{r['raw_cos']:>9.3f}{r['hsv_chisq']:>11.3f}"
              f"{r['clip_cos']:>10.3f}{r['v0_logistic']:>9.3f}{r['mean_pool_3']:>9.3f}"
              f"{r['v1.5_MLP']:>9.3f}")

    print("\nmean v1.5 by Dispatch bucket")
    for b in BUCKET_ORDER:
        g = df[df["bucket"] == b]
        print(f"  {b:<6} (n={len(g)})  mean v1.5 {g['v1.5_MLP'].mean():.3f}   "
              f"[{', '.join(sorted(g['pair_id']))}]")


if __name__ == "__main__":
    main()
