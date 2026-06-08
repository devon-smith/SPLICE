# logistic regression on the 4610-d fused DINOv2+CLIP boundary feature (v2 fusion try)
# trains on MovieNet, evaluates on test + Veo pilot, bootstrap CI vs v1.5
# usage: python scripts/train/fused_logistic.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.movienet import keyframe_key
from src.data.pairs import build_fused_pair_features, load_embeddings
from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics
from src.models.dinov2_encoder import DINOv2Encoder

FUSED_DIR = "/mnt/disks/splice-data/pairs/fused_boundary"
CLIP_DIR = "/mnt/disks/splice-data/embeddings/clip_vitl14"
V0_BOUNDARY_META = "/mnt/disks/splice-data/pairs/dino_v0_boundary/meta.parquet"
V1_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
AIGEN_CUTS = "/mnt/disks/splice-data/outputs/aigen_eval/cuts.parquet"
PER_PAIR_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
OUT_DIR = "/mnt/disks/splice-data/outputs/fused_logistic"

V0_AUPRC = 0.356   # published v0 test AUPRC for reference
V1_AUPRC = 0.4045  # v1.5 seed 2 test AUPRC for reference

BUCKET_OF = {"A003": "clean", "A013": "clean",
             "A001": "drift", "A002": "drift", "A004": "drift",
             "A011": "drift", "A012": "drift", "A014": "drift",
             "A005": "major", "A015": "major"}
BUCKET_CODE = {"clean": 0, "drift": 1, "major": 2}


# embed a list of keyframe paths with DINOv2 (Veo frames are not in the MovieNet cache)
def embed_dino(paths, encoder, batch=64):
    out = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        pv = torch.stack([
            encoder.processor(images=Image.open(p).convert("RGB"),
                              return_tensors="pt")["pixel_values"][0]
            for p in chunk
        ])
        emb = encoder.encode(pv).numpy()
        for p, e in zip(chunk, emb):
            out[p] = e
    return out


# movie-level paired bootstrap CI for AUPRC(A) - AUPRC(B)
def bootstrap_auprc_diff(y, sa, sb, movie_ids, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    by_movie = {mid: np.where(movie_ids == mid)[0] for mid in np.unique(movie_ids)}
    movies = list(by_movie)
    obs = average_precision_score(y, sa) - average_precision_score(y, sb)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(movies, size=len(movies), replace=True)
        idx = np.concatenate([by_movie[m] for m in pick])
        diffs[b] = (average_precision_score(y[idx], sa[idx])
                    - average_precision_score(y[idx], sb[idx]))
    return {"diff": float(obs),
            "ci": (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fused_dir", default=FUSED_DIR)
    ap.add_argument("--clip_dir", default=CLIP_DIR)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # train fused logistic on MovieNet
    feat = np.load(Path(args.fused_dir) / "features.npy")
    meta = pd.read_parquet(Path(args.fused_dir) / "meta.parquet")
    print(f"fused features {feat.shape} | positive rate {100 * meta['y_inconsistent'].mean():.2f}%")
    y = meta["y_inconsistent"].to_numpy().astype(int)
    is_train = (meta["split"] == "train").to_numpy()
    is_val = (meta["split"] == "val").to_numpy()
    is_test = (meta["split"] == "test").to_numpy()

    print(f"training fused logistic on {is_train.sum()} cuts ...")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    clf.fit(feat[is_train], y[is_train])
    joblib.dump(clf, out_dir / "model.pkl")

    val_s = clf.predict_proba(feat[is_val])[:, 1]
    test_s = clf.predict_proba(feat[is_test])[:, 1]
    np.savez(out_dir / "scores.npz",
             val_s=val_s, val_y=y[is_val], test_s=test_s, test_y=y[is_test])

    # MovieNet test metrics
    thr = best_f1_threshold(y[is_val], val_s)
    rank = ranking_metrics(y[is_test], test_s)
    tm = threshold_metrics(y[is_test], test_s, thr)
    mn = {
        "auprc": rank["auprc"], "auroc": rank["auroc"],
        "f1_at_val": tm["f1"], "precision": tm["precision"], "recall": tm["recall"],
        "val_thr": thr,
    }
    print(f"MovieNet test: AUPRC {mn['auprc']:.4f}  AUROC {mn['auroc']:.4f}  F1 {mn['f1_at_val']:.4f}")

    # significance vs v1.5: align by cut_id, movie-level bootstrap on AUPRC diff
    v1 = np.load(V1_NPZ)
    dino_meta = pd.read_parquet(V0_BOUNDARY_META)
    dino_test = dino_meta[dino_meta["split"] == "test"].reset_index(drop=True)
    v15_by_cut = dict(zip(dino_test["cut_id"], v1["test_s"].astype(float)))
    y_by_cut = dict(zip(dino_test["cut_id"], v1["test_y"].astype(int)))

    test_meta = meta[is_test].reset_index(drop=True)
    cut_ids = test_meta["cut_id"].to_numpy()
    keep = np.array([c in v15_by_cut for c in cut_ids])
    if not keep.all():
        print(f"warning: {(~keep).sum()} test cuts not in v1.5 -- dropped")
    cut_ids = cut_ids[keep]
    fused_s = test_s[keep]
    v15_s = np.array([v15_by_cut[c] for c in cut_ids])
    y_aligned = np.array([y_by_cut[c] for c in cut_ids])
    movie_ids = test_meta.loc[keep, "movie_id"].to_numpy()

    boot = bootstrap_auprc_diff(y_aligned, fused_s, v15_s, movie_ids, n_boot=args.n_boot)
    boot["n"] = len(y_aligned)
    print(f"bootstrap AUPRC(fused) - AUPRC(v1.5) = {boot['diff']:+.4f}  "
          f"CI [{boot['ci'][0]:+.4f}, {boot['ci'][1]:+.4f}]")

    # Veo pilot: embed boundary frames fresh with DINOv2, use cached CLIP, score with fused clf
    veo_cuts = pd.read_parquet(AIGEN_CUTS)
    veo_cuts["pair_id"] = veo_cuts["left_img2_path"].map(lambda p: Path(p).parent.name)
    paths = sorted(set(veo_cuts[["left_img2_path", "right_img0_path"]].to_numpy().ravel()))
    dino = embed_dino(paths, DINOv2Encoder())
    clip_emb, clip_k2r = load_embeddings(args.clip_dir)
    d_left = np.stack([dino[p] for p in veo_cuts["left_img2_path"]])
    d_right = np.stack([dino[p] for p in veo_cuts["right_img0_path"]])
    c_left = np.stack([clip_emb[clip_k2r[keyframe_key(p)]] for p in veo_cuts["left_img2_path"]])
    c_right = np.stack([clip_emb[clip_k2r[keyframe_key(p)]] for p in veo_cuts["right_img0_path"]])
    veo_feats = build_fused_pair_features(d_left, d_right, c_left, c_right)
    veo_scores = clf.predict_proba(veo_feats)[:, 1]

    veo_df = pd.DataFrame({"pair_id": veo_cuts["pair_id"], "fused": veo_scores})
    veo_df["bucket"] = veo_df["pair_id"].map(BUCKET_OF)
    code = veo_df["bucket"].map(BUCKET_CODE).to_numpy()
    rho = float(spearmanr(veo_df["fused"].to_numpy(), code)[0])
    bucket_mean = {b: float(veo_df.loc[veo_df["bucket"] == b, "fused"].mean())
                   for b in ("clean", "drift", "major")}

    # v1.5 Spearman on the same buckets for like-for-like compare
    pp = pd.read_csv(PER_PAIR_CSV, dtype={"pair_id": str})
    pp_code = pp["pair_id"].map(BUCKET_OF).map(BUCKET_CODE).to_numpy()
    v15_rho = float(spearmanr(pp["v1.5_MLP"].to_numpy(), pp_code)[0])

    print("\nVeo pilot")
    print(f"  bucket means: clean {bucket_mean['clean']:.3f}  "
          f"drift {bucket_mean['drift']:.3f}  major {bucket_mean['major']:.3f}")
    print(f"  Spearman vs Dispatch: fused {rho:+.3f}  (v1.5 {v15_rho:+.3f})")
    print(f"\nv0 ref {V0_AUPRC:.3f} | v1.5 ref {V1_AUPRC:.3f} | fused {mn['auprc']:.3f}")

    (out_dir / "results.json").write_text(json.dumps({
        "movienet": mn,
        "bootstrap_vs_v15": boot,
        "veo": {"bucket_mean": bucket_mean, "spearman": rho, "v15_spearman": v15_rho,
                "per_pair": veo_df.to_dict(orient="records")},
    }, indent=2, default=float))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
