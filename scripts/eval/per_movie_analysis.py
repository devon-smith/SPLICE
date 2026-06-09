# per-movie test diagnostics

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

V0_NPZ = "/mnt/disks/splice-data/outputs/v0/scores.npz"
V0_KEY = "logistic__test_s"
V15_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
V2_NPZ = "/mnt/disks/splice-data/outputs/v2_lora_extended/seed0/r8_a16/scores.npz"
CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
OUT_CSV = "reports/per_movie_analysis.csv"
OUT_FIG = "reports/per_movie_analysis.png"


def per_movie_ap(scores, labels, movie_ids, min_cuts=5):
    out = {}
    for mid in np.unique(movie_ids):
        m = movie_ids == mid
        if m.sum() < min_cuts:
            continue
        y = labels[m]
        if y.sum() == 0 or y.sum() == m.sum():
            continue
        out[str(mid)] = float(average_precision_score(y, scores[m]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v0_scores", default=V0_NPZ)
    ap.add_argument("--v0_key", default=V0_KEY)
    ap.add_argument("--v1_5_scores", default=V15_NPZ)
    ap.add_argument("--v2_scores", default=V2_NPZ)
    ap.add_argument("--cut_index", default=CUT_INDEX)
    ap.add_argument("--out_csv", default=OUT_CSV)
    ap.add_argument("--out_fig", default=OUT_FIG)
    args = ap.parse_args()

    v0 = np.load(args.v0_scores)
    v0_s = v0[args.v0_key].astype(float)
    y = v0[args.v0_key.replace("__test_s", "__test_y")].astype(int)

    cuts = pd.read_parquet(args.cut_index, columns=["split", "movie_id", "y_inconsistent"])
    test = cuts[cuts["split"] == "test"].reset_index(drop=True)
    assert len(test) == len(y)
    movie_ids = test["movie_id"].to_numpy()

    v15 = np.load(args.v1_5_scores)
    v15_seeds = {s: v15[f"seed{s}_test_s"].astype(float) for s in range(3)}
    assert np.array_equal(y, v15["test_y"].astype(int))

    v2 = np.load(args.v2_scores)
    v2_s = v2["test_s"].astype(float)
    assert np.array_equal(y, v2["test_y"].astype(int))

    ap0 = per_movie_ap(v0_s, y, movie_ids)
    # v1.5: average per-movie AP across seeds, NOT score-averaged ensemble.
    # this matches the canonical headline 0.418.
    ap15_per_seed = {s: per_movie_ap(v15_seeds[s], y, movie_ids) for s in range(3)}
    common = set.intersection(*(set(ap15_per_seed[s]) for s in range(3)))
    ap15 = {m: float(np.mean([ap15_per_seed[s][m] for s in range(3)])) for m in common}
    ap2 = per_movie_ap(v2_s, y, movie_ids)

    # per-movie metadata (n_cuts, pos_rate) for context
    meta = test.groupby("movie_id").agg(
        n_cuts=("y_inconsistent", "size"),
        n_pos=("y_inconsistent", "sum"),
    ).reset_index()
    meta["pos_rate"] = meta["n_pos"] / meta["n_cuts"]

    rows = []
    for mid in sorted(set(ap0) & set(ap15) & set(ap2)):
        m = meta[meta["movie_id"] == mid].iloc[0]
        rows.append({
            "movie_id": mid,
            "n_cuts": int(m["n_cuts"]),
            "n_pos": int(m["n_pos"]),
            "pos_rate": float(m["pos_rate"]),
            "v0": ap0[mid], "v1_5": ap15[mid], "v2": ap2[mid],
            "v2_gain_over_v1_5": ap2[mid] - ap15[mid],
            "v1_5_gain_over_v0": ap15[mid] - ap0[mid],
        })
    out = pd.DataFrame(rows).sort_values("v2_gain_over_v1_5", ascending=False)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(out)} movies)")

    print("\nMacro means:")
    for col in ("v0", "v1_5", "v2"):
        print(f"  {col:6s} mean={out[col].mean():.4f}  median={out[col].median():.4f}")
    print(f"  delta v2-v1.5 mean={out['v2_gain_over_v1_5'].mean():+.4f}")
    print(f"  movies where v2>v1.5: {(out['v2_gain_over_v1_5']>0).sum()}/{len(out)}")

    print("\nTop-10 v2 gain over v1.5:")
    print(out.head(10).to_string(index=False))
    print("\nBottom-10 (regressions):")
    print(out.tail(10).to_string(index=False))

    # bar chart: v1.5 vs v2 per movie, sorted by v2 AP
    by_v2 = out.sort_values("v2").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    x = np.arange(len(by_v2))
    ax.bar(x - 0.18, by_v2["v1_5"], width=0.36, label="v1.5 (3-seed mean)",
           alpha=0.85, color="#888")
    ax.bar(x + 0.18, by_v2["v2"], width=0.36, label="v2 LoRA (seed 0)",
           alpha=0.85, color="#1f77b4")
    ax.axhline(by_v2["v1_5"].mean(), color="#888", linestyle=":", linewidth=1, alpha=0.7,
               label=f"v1.5 macro={by_v2['v1_5'].mean():.3f}")
    ax.axhline(by_v2["v2"].mean(), color="#1f77b4", linestyle=":", linewidth=1, alpha=0.7,
               label=f"v2 macro={by_v2['v2'].mean():.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(by_v2["movie_id"], rotation=90, fontsize=6)
    ax.set_ylabel("per-movie AP")
    ax.set_xlabel("test movie (sorted by v2 AP)")
    ax.set_title("Per-movie test AP: v1.5 vs v2 LoRA")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=130)
    print(f"\nwrote {args.out_fig}")


if __name__ == "__main__":
    main()
