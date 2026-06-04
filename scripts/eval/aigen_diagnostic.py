# diagnostic for v2's underperformance on the Veo pilot.
# tests 4 hypotheses: pipeline bug, label-score mismatch, frame selection, real finding.
# usage: python scripts/eval/aigen_diagnostic.py

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[2]
KEYFRAMES = Path("/mnt/disks/splice-data/outputs/aigen_eval/keyframes")
V2_CSV = REPO / "reports/aigen_v2_pilot.csv"
PRIOR_CSV = Path("/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv")

BUCKET_OF = {"A003": "clean", "A013": "clean",
             "A001": "drift", "A002": "drift", "A004": "drift",
             "A011": "drift", "A012": "drift", "A014": "drift",
             "A005": "major", "A015": "major"}
BUCKET_RANK = {"clean": 1, "drift": 2, "major": 3}
BUCKET_ORDER = ["clean", "drift", "major"]


# render A005's 6 keyframes side-by-side; mark which 2 the scorer actually sees
def frame_inspection(out_fig):
    pid = "A005"
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))
    for col, idx in enumerate([0, 1, 2]):
        for row, side in enumerate(["left", "right"]):
            axes[row, col].imshow(Image.open(KEYFRAMES / pid / f"{side}_img{idx}.jpg"))
            label = f"{side} clip -- img{idx}"
            if (row == 0 and col == 2) or (row == 1 and col == 0):
                label += "  <- SCORED"
            axes[row, col].set_title(label, fontsize=10)
            axes[row, col].axis("off")
    fig.suptitle("A005 keyframes (Dispatch: MAJOR identity failure) -- scorer sees boundary pair",
                 fontsize=12)
    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig, dpi=130, bbox_inches="tight")
    print(f"wrote {out_fig}")


def ranking_table():
    prior = pd.read_csv(PRIOR_CSV).set_index("pair_id")
    v2 = pd.read_csv(V2_CSV).set_index("pair_id")
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
    # 1 = highest score = most discontinuous
    for col in ("v0_score", "v1_5_score", "v2_score", "clip_score"):
        df[col.replace("_score", "_rank")] = df[col].rank(ascending=False, method="min").astype(int)
    return df


def cross_model_agreement(df):
    out = {}
    for a, b in [("v0_rank", "v1_5_rank"), ("v0_rank", "v2_rank"), ("v1_5_rank", "v2_rank"),
                 ("v0_rank", "clip_rank"), ("v1_5_rank", "clip_rank"), ("v2_rank", "clip_rank")]:
        rho, p = spearmanr(df[a], df[b])
        out[f"{a} vs {b}"] = {"rho": float(rho), "p": float(p)}
    return out


def vs_dispatch(df):
    out = {}
    for col in ("v0_score", "v1_5_score", "v2_score", "clip_score"):
        rho, p = spearmanr(df[col], df["bucket_rank"])
        out[col.replace("_score", "")] = {"rho": float(rho), "p": float(p)}
    return out


def bucket_breakdown(df):
    median = {m: float(df[f"{m}_score"].median()) for m in ("v0", "v1_5", "v2", "clip")}
    out = {}
    for b in BUCKET_ORDER:
        g = df[df["dispatch_bucket"] == b]
        out[b] = {"n": int(len(g))}
        for m in ("v0", "v1_5", "v2"):
            out[b][f"{m}_mean"] = float(g[f"{m}_score"].mean())
            out[b][f"{m}_above_med"] = int((g[f"{m}_score"] > median[m]).sum())
    return median, out


# if we drop each pair in turn, how does v2's Spearman vs Dispatch change?
def leave_one_out(df):
    out = {}
    for pid in df["pair_id"]:
        sub = df[df["pair_id"] != pid]
        out[pid] = {
            "v0": float(spearmanr(sub["v0_score"], sub["bucket_rank"])[0]),
            "v1_5": float(spearmanr(sub["v1_5_score"], sub["bucket_rank"])[0]),
            "v2": float(spearmanr(sub["v2_score"], sub["bucket_rank"])[0]),
        }
    # which pair, when removed, lifts v2's rho the most? (that's the "v2 hates this pair")
    return out, sorted(out.items(), key=lambda x: -x[1]["v2"])[:3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_fig", default=str(REPO / "reports/figures/a005_frame_inspection.png"))
    ap.add_argument("--out_csv", default=str(REPO / "reports/aigen_full_ranking_table.csv"))
    ap.add_argument("--out_json", default=str(REPO / "reports/aigen_diagnostic_metrics.json"))
    args = ap.parse_args()

    print("Task 1: A005 frame inspection")
    frame_inspection(Path(args.out_fig))

    print("\nTask 2: cross-model ranking")
    df = ranking_table()
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")
    cross = cross_model_agreement(df)
    vs_b = vs_dispatch(df)
    median, breakdown = bucket_breakdown(df)
    loo, top_loo = leave_one_out(df)

    # verify Spearman from scratch (sanity check)
    print("\nSpearman vs Dispatch (re-computed from CSVs)")
    for m in ("v0", "v1_5", "v2", "clip"):
        rho, p = spearmanr(df[f"{m}_score"], df["bucket_rank"])
        print(f"  {m:6s}  rho={rho:+.4f}  p={p:.4f}")

    print("\ncross-model rank agreement (do models agree among themselves?)")
    for k, v in cross.items():
        print(f"  {k:30s}  rho={v['rho']:+.4f}  p={v['p']:.4f}")

    print("\nbucket means + above-median counts")
    for b, m in breakdown.items():
        print(f"  {b:<6} n={m['n']}  v0 {m['v0_mean']:.4f} ({m['v0_above_med']}/{m['n']} >med)  "
              f"v1.5 {m['v1_5_mean']:.4f} ({m['v1_5_above_med']}/{m['n']} >med)  "
              f"v2 {m['v2_mean']:.4f} ({m['v2_above_med']}/{m['n']} >med)")

    print(f"\nleave-one-out: drop X, what's v2's rho? (full = {vs_b['v2']['rho']:+.4f})")
    for pid, lifts in top_loo:
        print(f"  drop {pid} ({BUCKET_OF[pid]:<6}): v0 {lifts['v0']:+.4f}  "
              f"v1.5 {lifts['v1_5']:+.4f}  v2 {lifts['v2']:+.4f}")

    Path(args.out_json).write_text(json.dumps({
        "vs_dispatch": vs_b,
        "cross_model": cross,
        "bucket_median": median,
        "bucket_breakdown": breakdown,
        "leave_one_out": loo,
    }, indent=2))
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
