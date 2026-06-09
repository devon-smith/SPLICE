#rank-based flagging analysis for the fixed 10-pair Veo/Dispatch pilot


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

V0_NPZ = "/mnt/disks/splice-data/outputs/v0/scores.npz"
V1_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
PER_PAIR_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
OUT_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/rank_based_scores.csv"

FLAG_THRESHOLD = 95.0
FLAG_MODELS = {"v1.5": "v1.5_MLP", "v0": "v0_logistic", "clip": "clip_cos"}

BUCKET_OF = {"A003": "clean", "A013": "clean",
             "A001": "drift", "A002": "drift", "A004": "drift",
             "A011": "drift", "A012": "drift", "A014": "drift",
             "A005": "major", "A015": "major"}
MAJOR = ["A005", "A015"]
CLEAN = ["A003", "A013"]
DRIFT = ["A001", "A002", "A004", "A011", "A012", "A014"]


# MovieNet within-scene (y=0) test scores for the 3 flag models
def within_scene_reference():
    v0, v1 = np.load(V0_NPZ), np.load(V1_NPZ)
    y0 = v0["logistic__test_y"].astype(int) == 0
    yc = v0["clip_cosine__test_y"].astype(int) == 0
    return {
        "v1.5": v1["test_s"][v1["test_y"].astype(int) == 0].astype(float),
        "v0": v0["logistic__test_s"][y0].astype(float),
        "clip": v0["clip_cosine__test_s"][yc].astype(float),
    }


def flag_eval(df, flag_col):
    flagged = set(df.loc[df[flag_col], "pair_id"])
    return {
        "major_caught": sum(p in flagged for p in MAJOR),
        "clean_cleared": sum(p not in flagged for p in CLEAN),
        "drift_flagged": sum(p in flagged for p in DRIFT),
    }


# can a single threshold on a percentile column isolate the major pairs from all others?
def separation(df, pct_col):
    by = df.set_index("pair_id")[pct_col]
    nonmajor = by[CLEAN + DRIFT]
    major_min = float(by[MAJOR].min())
    nonmajor_max = float(nonmajor.max())
    isolates = major_min > nonmajor_max
    return {
        "major_min": major_min,
        "clean_max": float(by[CLEAN].max()),
        "nonmajor_max": nonmajor_max,
        "nonmajor_max_id": str(nonmajor.idxmax()),
        "isolates_major": isolates,
        "margin": major_min - nonmajor_max,
        "threshold": (major_min + nonmajor_max) / 2 if isolates else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_pair_csv", default=PER_PAIR_CSV)
    ap.add_argument("--out_csv", default=OUT_CSV)
    args = ap.parse_args()

    ref = within_scene_reference()
    print(f"reference: MovieNet within-scene y=0, n={len(ref['v1.5'])}")
    pp = pd.read_csv(args.per_pair_csv, dtype={"pair_id": str})

    rows = []
    for _, r in pp.iterrows():
        pct = {m: float(percentileofscore(ref[m], float(r[col]), kind="mean"))
               for m, col in FLAG_MODELS.items()}
        max_pct = max(pct.values())
        rows.append({
            "pair_id": r["pair_id"], "bucket": BUCKET_OF[r["pair_id"]],
            "v1.5_percentile": pct["v1.5"], "v0_percentile": pct["v0"],
            "clip_percentile": pct["clip"], "max_percentile": max_pct,
            "flag_v1.5": pct["v1.5"] > FLAG_THRESHOLD,
            "flag_v0": pct["v0"] > FLAG_THRESHOLD,
            "flag_clip": pct["clip"] > FLAG_THRESHOLD,
            "flag_ensemble": max_pct > FLAG_THRESHOLD,
        })
    df = pd.DataFrame(rows)
    csv_cols = ["pair_id", "v1.5_percentile", "v0_percentile", "clip_percentile",
                "max_percentile", "flag_v1.5", "flag_v0", "flag_clip", "flag_ensemble"]
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df[csv_cols].to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}")

    print("\nper-pair percentiles vs MovieNet within-scene")
    print(f"  {'pair':<7}{'bucket':<8}{'v1.5':>8}{'v0':>8}{'clip':>8}{'max':>8}{'flag?':>8}")
    for _, r in df.sort_values("max_percentile", ascending=False).iterrows():
        print(f"  {r['pair_id']:<7}{r['bucket']:<8}{r['v1.5_percentile']:>8.1f}"
              f"{r['v0_percentile']:>8.1f}{r['clip_percentile']:>8.1f}"
              f"{r['max_percentile']:>8.1f}{('YES' if r['flag_ensemble'] else 'no'):>8}")

    print(f"\nflag evaluation (threshold = {FLAG_THRESHOLD:.0f}th percentile)")
    for fc in ("flag_v1.5", "flag_v0", "flag_clip", "flag_ensemble"):
        e = flag_eval(df, fc)
        print(f"  {fc:<14} major caught {e['major_caught']}/2  "
              f"clean cleared {e['clean_cleared']}/2  drift flagged {e['drift_flagged']}/6")

    print("\nseparation: can one threshold on a percentile column isolate the major pairs?")
    pct_cols = {"flag_v1.5": "v1.5_percentile", "flag_v0": "v0_percentile",
                "flag_clip": "clip_percentile", "flag_ensemble": "max_percentile"}
    for fc, col in pct_cols.items():
        s = separation(df, col)
        thr = f"thr={s['threshold']:.0f}" if s["isolates_major"] else "no"
        print(f"  {col:18s} major_min {s['major_min']:.1f}  "
              f"nonmajor_max {s['nonmajor_max']:.1f} ({s['nonmajor_max_id']})  "
              f"isolates={s['isolates_major']}  margin {s['margin']:+.1f}  {thr}")


if __name__ == "__main__":
    main()
