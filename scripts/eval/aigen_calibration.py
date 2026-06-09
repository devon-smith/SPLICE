# AI-gen calibration... domain-specific score percentiles for all 6 scorers


import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

REPO = Path(__file__).resolve().parents[2]
V0_NPZ = "/mnt/disks/splice-data/outputs/v0/scores.npz"
V1_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
MP_NPZ = "/mnt/disks/splice-data/outputs/v0_mean_pool/scores.npz"
PER_PAIR_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
CALIB_JSON = REPO / "configs" / "aigen_calibration.json"

PERCENTILES = [50, 75, 90, 95, 99]
MODELS = ["v1.5", "v0_logistic", "mean_pool_3", "raw_dino_cosine", "hsv_chisq", "clip_cosine"]
VEO_COL = {
    "v1.5": "v1.5_MLP", "v0_logistic": "v0_logistic", "mean_pool_3": "mean_pool_3",
    "raw_dino_cosine": "raw_cos", "hsv_chisq": "hsv_chisq", "clip_cosine": "clip_cos",
}
MAJOR = ["A005", "A015"]  # Dispatch's "major identity failure" pairs


def pctiles(arr):
    return {f"p{p}": float(np.percentile(arr, p)) for p in PERCENTILES}


# {model: (test scores, test labels)} for all 6 scorers
def load_movienet():
    v0, v1, mp = np.load(V0_NPZ), np.load(V1_NPZ), np.load(MP_NPZ)
    return {
        "v1.5": (v1["test_s"].astype(float), v1["test_y"].astype(int)),
        "v0_logistic": (v0["logistic__test_s"].astype(float),
                        v0["logistic__test_y"].astype(int)),
        "mean_pool_3": (mp["logistic__test_s"].astype(float),
                        mp["logistic__test_y"].astype(int)),
        "raw_dino_cosine": (v0["raw_dino_cosine__test_s"].astype(float),
                            v0["raw_dino_cosine__test_y"].astype(int)),
        "hsv_chisq": (v0["hsv_chisq__test_s"].astype(float),
                      v0["hsv_chisq__test_y"].astype(int)),
        "clip_cosine": (v0["clip_cosine__test_s"].astype(float),
                        v0["clip_cosine__test_y"].astype(int)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_pair_csv", default=PER_PAIR_CSV)
    args = ap.parse_args()

    pp = pd.read_csv(args.per_pair_csv, dtype={"pair_id": str})
    veo = {m: pp[VEO_COL[m]].to_numpy(dtype=float) for m in MODELS}
    mn = load_movienet()

    calib = {
        "_comment": (
            "AI-gen score-calibration percentiles. veo_continuous_action is the "
            "10-pair Veo y=0 pilot. A Veo score above its p90/p95 is unusual "
            "relative to other Veo continuous-action pairs."
        ),
        "percentiles": PERCENTILES,
        "veo_continuous_action": {m: pctiles(veo[m]) for m in MODELS},
        "movienet_within_scene": {m: pctiles(s[y == 0]) for m, (s, y) in mn.items()},
        "movienet_scene_boundary": {m: pctiles(s[y == 1]) for m, (s, y) in mn.items()},
    }
    CALIB_JSON.parent.mkdir(parents=True, exist_ok=True)
    CALIB_JSON.write_text(json.dumps(calib, indent=2))
    print(f"wrote {CALIB_JSON}")

    # console: Veo p50/p90/p95 + where the major-identity pairs land
    vc = calib["veo_continuous_action"]
    print("\nVeo y=0 percentiles:")
    print(f"  {'model':18s}  {'p50':>7s}  {'p75':>7s}  {'p90':>7s}  {'p95':>7s}")
    for m in MODELS:
        d = vc[m]
        print(f"  {m:18s}  {d['p50']:.4f}  {d['p75']:.4f}  {d['p90']:.4f}  {d['p95']:.4f}")

    print("\nWhere the major-identity pairs land (Veo percentile rank):")
    for pid in MAJOR:
        row = pp[pp["pair_id"] == pid].iloc[0]
        bits = []
        for m in MODELS:
            s = float(row[VEO_COL[m]])
            rank = float(percentileofscore(veo[m], s, kind="mean"))
            bits.append(f"{m.split('_')[0]}:{rank:.0f}")
        print(f"  {pid}:  " + "  ".join(bits))


if __name__ == "__main__":
    main()
