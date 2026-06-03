"""Per-movie test-set diagnostic: where does v2 help, where does it not?

Joins per-movie macro AP from the three trained heads (v0, v1.5, v2) on the 64
MovieNet test movies. v1.5 = mean of 3 seeds. v2 = seed-0 proxy (until Phase 2
completes; re-run with 3-seed mean afterward by swapping --v2_scores).

Writes a CSV + a bar-chart figure and prints headline ranks (best/worst movies
for v2 vs v1.5). Reads cached score files only; trains nothing.
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def per_movie_ap(scores: np.ndarray, labels: np.ndarray, movie_ids: np.ndarray,
                  min_cuts: int = 5) -> dict[str, float]:
    out = {}
    for mid in np.unique(movie_ids):
        idx = movie_ids == mid
        if idx.sum() < min_cuts:
            continue
        pos = labels[idx].sum()
        if pos == 0 or pos == idx.sum():
            continue
        out[str(mid)] = float(average_precision_score(labels[idx], scores[idx]))
    return out


def load_test_movie_ids(cut_index: str, n: int) -> np.ndarray:
    df = pd.read_parquet(cut_index, columns=["split", "movie_id"])
    test = df[df["split"] == "test"].reset_index(drop=True)
    assert len(test) == n, f"len(test)={len(test)} vs n_scores={n}"
    return test["movie_id"].to_numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v0_scores", default="/mnt/disks/splice-data/outputs/v0/scores.npz")
    ap.add_argument("--v0_key", default="logistic__test_s")
    ap.add_argument("--v1_5_scores", default="/mnt/disks/splice-data/outputs/v1_sound/scores.npz")
    ap.add_argument("--v2_scores",
                    default="/mnt/disks/splice-data/outputs/v2_lora_extended/seed0/r8_a16/scores.npz")
    ap.add_argument("--cut_index", default="/mnt/disks/splice-data/outputs/cut_index/cuts.parquet")
    ap.add_argument("--out_csv", default=str(REPO / "reports/per_movie_analysis.csv"))
    ap.add_argument("--out_fig", default=str(REPO / "reports/per_movie_analysis.png"))
    args = ap.parse_args()

    v0_npz = np.load(args.v0_scores)
    v0_s = v0_npz[args.v0_key].astype(float)
    v0_y_key = args.v0_key.replace("__test_s", "__test_y")
    y = v0_npz[v0_y_key].astype(int)
    n = len(y)
    movie_ids = load_test_movie_ids(args.cut_index, n)

    v15_npz = np.load(args.v1_5_scores)
    v15_seeds = {s: v15_npz[f"seed{s}_test_s"].astype(float) for s in range(3)}
    assert np.array_equal(y, v15_npz["test_y"].astype(int)), "v0/v1.5 label mismatch"

    v2_npz = np.load(args.v2_scores)
    v2_s = v2_npz["test_s"].astype(float)
    assert np.array_equal(y, v2_npz["test_y"].astype(int)), "v0/v2 label mismatch"

    ap0 = per_movie_ap(v0_s, y, movie_ids)
    # v1.5: per-movie AP averaged across the 3 seeds -- matches the canonical
    # 3-seed-mean macro AP (0.418) reported in reports/macro_ap.md. NOT the
    # score-ensemble (which would be a different, ensemble-quality number).
    ap15_per_seed = {s: per_movie_ap(v15_seeds[s], y, movie_ids) for s in range(3)}
    common15 = set.intersection(*(set(ap15_per_seed[s]) for s in range(3)))
    ap15 = {m: float(np.mean([ap15_per_seed[s][m] for s in range(3)])) for m in common15}
    ap2 = per_movie_ap(v2_s, y, movie_ids)

    df = pd.read_parquet(args.cut_index, columns=["split", "movie_id", "y_inconsistent"])
    test = df[df["split"] == "test"]
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
            "v0": ap0[mid],
            "v1_5": ap15[mid],
            "v2": ap2[mid],
            "v2_gain_over_v1_5": ap2[mid] - ap15[mid],
            "v1_5_gain_over_v0": ap15[mid] - ap0[mid],
        })
    out = pd.DataFrame(rows).sort_values("v2_gain_over_v1_5", ascending=False)
    out.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv} ({len(out)} movies)")

    print("\nMacro means (this script's filter set):")
    for col in ("v0", "v1_5", "v2"):
        print(f"  {col:6s} mean={out[col].mean():.4f}  median={out[col].median():.4f}")
    print(f"  Δ v2-v1.5 mean={out['v2_gain_over_v1_5'].mean():+.4f}")
    print(f"  movies where v2>v1.5: {(out['v2_gain_over_v1_5']>0).sum()}/{len(out)}")

    print("\nTop-10 v2 gain over v1.5:")
    print(out.head(10).to_string(index=False))
    print("\nBottom-10 (regressions):")
    print(out.tail(10).to_string(index=False))

    # bar chart: per-movie v1.5 vs v2, sorted by v2 macro AP
    by_v2 = out.sort_values("v2").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(13, 4.5))
    x = np.arange(len(by_v2))
    ax.bar(x - 0.18, by_v2["v1_5"], width=0.36, label="v1.5 (3-seed mean)", alpha=0.85, color="#888")
    ax.bar(x + 0.18, by_v2["v2"], width=0.36, label="v2 LoRA (seed 0)", alpha=0.85, color="#1f77b4")
    ax.axhline(by_v2["v1_5"].mean(), color="#888", linestyle=":", linewidth=1, alpha=0.7,
               label=f"v1.5 macro={by_v2['v1_5'].mean():.3f}")
    ax.axhline(by_v2["v2"].mean(), color="#1f77b4", linestyle=":", linewidth=1, alpha=0.7,
               label=f"v2 macro={by_v2['v2'].mean():.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(by_v2["movie_id"], rotation=90, fontsize=6)
    ax.set_ylabel("per-movie AP")
    ax.set_xlabel("test movie (sorted by v2 AP)")
    ax.set_title("Per-movie test AP: v1.5 (3-seed mean) vs v2 LoRA (seed 0)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=130)
    print(f"\nwrote {args.out_fig}")


if __name__ == "__main__":
    main()
