"""Diagnostic for v2's underperformance on the Veo pilot.

Tests four hypotheses for the Spearman inversion (v0 +0.66, v1.5 +0.44, v2 +0.39
vs Dispatch buckets clean<drift<major):
  1. Real finding -- LoRA suppressed identity features.
  2. Frame selection -- boundary frames don't show the drift mid-clip.
  3. Label-score mismatch -- models agree among themselves, all disagree with Dispatch.
  4. Pipeline bug -- wrong frames / wrong labels.

Reads cached scores + keyframes only. Produces:
  reports/figures/a005_frame_inspection.png
  reports/aigen_full_ranking_table.csv
  reports/aigen_diagnostic.md
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]

BUCKET_OF = {
    "A003": "clean", "A013": "clean",
    "A001": "drift", "A002": "drift", "A004": "drift",
    "A011": "drift", "A012": "drift", "A014": "drift",
    "A005": "major", "A015": "major",
}
BUCKET_RANK = {"clean": 1, "drift": 2, "major": 3}
BUCKET_ORDER = ["clean", "drift", "major"]
KEYFRAMES = Path("/mnt/disks/splice-data/outputs/aigen_eval/keyframes")
V2_CSV = REPO / "reports/aigen_v2_pilot.csv"
PRIOR_CSV = Path("/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv")


def task1_frame_inspection(out_fig: Path) -> dict:
    """Side-by-side: A005's 3 left keyframes (img0/1/2) and 3 right keyframes (img0/1/2).
    The scorer sees left_img2 (last of left) + right_img0 (first of right) -- the
    middle column boundary. Adjacent images show what's happening within each clip."""
    pid = "A005"
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for col, idx in enumerate([0, 1, 2]):
        for row, side in enumerate(["left", "right"]):
            img = Image.open(KEYFRAMES / pid / f"{side}_img{idx}.jpg")
            axes[row, col].imshow(img)
            label = f"{side} clip — img{idx}"
            if (row == 0 and col == 2) or (row == 1 and col == 0):
                label += "  ← SCORED"
            axes[row, col].set_title(label, fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle(
        "A005 keyframes (Dispatch label: MAJOR identity failure)\n"
        "Scorer sees the boundary pair (left img2 + right img0)",
        fontsize=12,
    )
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    print(f"wrote {out_fig}")
    # also report file sizes as a sanity hint that frames load
    sizes = {f"{side}_img{i}": (KEYFRAMES / pid / f"{side}_img{i}.jpg").stat().st_size
             for side in ("left", "right") for i in range(3)}
    return {"figure": str(out_fig), "file_sizes": sizes}


def task2_ranking(prior_csv: Path, v2_csv: Path) -> tuple[pd.DataFrame, dict]:
    prior = pd.read_csv(prior_csv).set_index("pair_id")
    v2 = pd.read_csv(v2_csv).set_index("pair_id")
    rows = []
    for pid in sorted(BUCKET_OF):
        rows.append({
            "pair_id": pid,
            "dispatch_bucket": BUCKET_OF[pid],
            "bucket_rank": BUCKET_RANK[BUCKET_OF[pid]],
            "v0_score": float(prior.loc[pid, "v0_logistic"]),
            "v1_5_score": float(prior.loc[pid, "v1.5_MLP"]),
            "v2_score": float(v2.loc[pid, "v2_3seed_mean"]),
            "clip_score": float(prior.loc[pid, "clip_cos"]),
        })
    df = pd.DataFrame(rows)
    # ranks (1 = highest score = most discontinuous)
    for col in ("v0_score", "v1_5_score", "v2_score", "clip_score"):
        df[col.replace("_score", "_rank")] = df[col].rank(ascending=False, method="min").astype(int)
    return df, _ranking_metrics(df)


def _ranking_metrics(df: pd.DataFrame) -> dict:
    # cross-model agreement on rank (high = models agree among themselves)
    cross = {}
    for a, b in [("v0_rank", "v1_5_rank"), ("v0_rank", "v2_rank"), ("v1_5_rank", "v2_rank"),
                 ("v0_rank", "clip_rank"), ("v1_5_rank", "clip_rank"), ("v2_rank", "clip_rank")]:
        rho, p = spearmanr(df[a], df[b])
        cross[f"{a} vs {b}"] = {"rho": float(rho), "p": float(p)}
    # each model vs Dispatch bucket rank
    vs_bucket = {}
    for col in ("v0_score", "v1_5_score", "v2_score", "clip_score"):
        rho, p = spearmanr(df[col], df["bucket_rank"])
        vs_bucket[col.replace("_score", "")] = {"rho": float(rho), "p": float(p)}
    return {"cross_model": cross, "vs_dispatch": vs_bucket}


def task2_bucket_breakdown(df: pd.DataFrame) -> dict:
    median = {m: float(df[f"{m}_score"].median()) for m in ("v0", "v1_5", "v2", "clip")}
    breakdown = {}
    for b in BUCKET_ORDER:
        g = df[df["dispatch_bucket"] == b]
        breakdown[b] = {
            "n": int(len(g)),
            "v0_mean": float(g["v0_score"].mean()),
            "v1_5_mean": float(g["v1_5_score"].mean()),
            "v2_mean": float(g["v2_score"].mean()),
            "v0_above_med": int((g["v0_score"] > median["v0"]).sum()),
            "v1_5_above_med": int((g["v1_5_score"] > median["v1_5"]).sum()),
            "v2_above_med": int((g["v2_score"] > median["v2"]).sum()),
        }
    # which pair did each model rank highest within each bucket?
    top_by_bucket = {}
    for b in BUCKET_ORDER:
        g = df[df["dispatch_bucket"] == b]
        top_by_bucket[b] = {
            m: g.sort_values(f"{m}_score", ascending=False)["pair_id"].tolist()
            for m in ("v0", "v1_5", "v2")
        }
    return {"median": median, "breakdown": breakdown, "top_within_bucket": top_by_bucket}


def task3_leave_one_out(df: pd.DataFrame) -> dict:
    """If we drop each pair in turn, how does Spearman vs Dispatch change?"""
    out = {}
    for pid in df["pair_id"]:
        sub = df[df["pair_id"] != pid]
        bucket_ranks = sub["bucket_rank"].to_numpy()
        out[pid] = {
            "v0": float(spearmanr(sub["v0_score"], bucket_ranks)[0]),
            "v1_5": float(spearmanr(sub["v1_5_score"], bucket_ranks)[0]),
            "v2": float(spearmanr(sub["v2_score"], bucket_ranks)[0]),
        }
    # find the pair whose removal lifts v2's rho the MOST -- that's the "v2 hates this pair"
    v2_lifts = sorted(out.items(), key=lambda x: -x[1]["v2"])
    return {"loo_spearman": out, "v2_most_lifted_by_dropping": v2_lifts[:3]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_fig", default=str(REPO / "reports/figures/a005_frame_inspection.png"))
    ap.add_argument("--out_csv", default=str(REPO / "reports/aigen_full_ranking_table.csv"))
    ap.add_argument("--out_json", default=str(REPO / "reports/aigen_diagnostic_metrics.json"))
    args = ap.parse_args()

    print("=== Task 1: A005 frame inspection ===")
    t1 = task1_frame_inspection(Path(args.out_fig))

    print("\n=== Task 2: cross-model ranking ===")
    df, metrics = task2_ranking(PRIOR_CSV, V2_CSV)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")
    bucket = task2_bucket_breakdown(df)
    loo = task3_leave_one_out(df)

    print("\n=== Verify Spearman from scratch (Task 3) ===")
    for m in ("v0", "v1_5", "v2", "clip"):
        rho, p = spearmanr(df[f"{m}_score"], df["bucket_rank"])
        print(f"  {m:6s}  rho={rho:+.4f}  p={p:.4f}  (joined fresh from CSVs)")

    print("\n=== Cross-model rank agreement ===")
    for k, v in metrics["cross_model"].items():
        print(f"  {k:30s}  rho={v['rho']:+.4f}  p={v['p']:.4f}")

    print("\n=== Bucket means + above-median counts ===")
    for b, m in bucket["breakdown"].items():
        print(f"  {b:<6} n={m['n']}  v0 {m['v0_mean']:.4f} ({m['v0_above_med']}/{m['n']} >med)  "
              f"v1.5 {m['v1_5_mean']:.4f} ({m['v1_5_above_med']}/{m['n']} >med)  "
              f"v2 {m['v2_mean']:.4f} ({m['v2_above_med']}/{m['n']} >med)")

    print("\n=== Leave-one-out: drop pair X, what's v2's rho? ===")
    print(f"  (full) v2 rho = {metrics['vs_dispatch']['v2']['rho']:+.4f}")
    for pid, lifts in loo["v2_most_lifted_by_dropping"]:
        print(f"  drop {pid} ({BUCKET_OF[pid]:<6}): v0 {lifts['v0']:+.4f}  "
              f"v1.5 {lifts['v1_5']:+.4f}  v2 {lifts['v2']:+.4f}")

    Path(args.out_json).write_text(json.dumps({
        "task1": t1,
        "task2_ranking_metrics": metrics,
        "task2_bucket_breakdown": bucket,
        "task3_leave_one_out": loo,
    }, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
