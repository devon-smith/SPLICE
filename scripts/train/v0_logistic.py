# v0: logistic regression on the 2305-d DINOv2 pair feature + three non-learned baselines
# (raw DINOv2 cosine, HSV chi-square, CLIP cosine). all four thresholded at F1-optimal on val.
# usage: python scripts/train/v0_logistic.py

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics
from src.models.baselines import (CLIPImageEncoder, chisq_scores,
                                   compute_hsv_histograms, cosine_distance_scores)

FEATURES = "/mnt/disks/splice-data/pairs/dino_v0_boundary"
CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
OUT_DIR = "/mnt/disks/splice-data/outputs/v0"
CLIP_MODEL = "openai/clip-vit-large-patch14"
MODEL_ORDER = ["logistic", "raw_dino_cosine", "hsv_chisq", "clip_cosine"]


# threshold on val (F1-optimal), then report test ranking + threshold metrics
def evaluate(name, val_y, val_s, test_y, test_s):
    thr = best_f1_threshold(val_y, val_s)
    test_rank = ranking_metrics(test_y, test_s)
    test_thr = threshold_metrics(test_y, test_s, thr)
    return {
        "model": name,
        "n_test": test_rank["n"],
        "auroc": test_rank["auroc"],
        "auprc": test_rank["auprc"],
        "f1_at_val_thr": test_thr["f1"],
        "precision_at_val_thr": test_thr["precision"],
        "recall_at_val_thr": test_thr["recall"],
        "accuracy_at_val_thr": test_thr["accuracy"],
        "val_thr": thr,
        "f1_on_val": threshold_metrics(val_y, val_s, thr)["f1"],
    }


def drop_nan(y, s):
    ok = ~np.isnan(s)
    return y[ok], s[ok]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=FEATURES)
    ap.add_argument("--cut_index", default=CUT_INDEX)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--skip_baselines", action="store_true",
                    help="only logistic + dino cosine (skips HSV + CLIP which are slow)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features)

    features = np.load(feat_dir / "features.npy")
    meta = pd.read_parquet(feat_dir / "meta.parquet")
    print(f"features {features.shape} | positive rate {100 * meta['y_inconsistent'].mean():.2f}%")

    is_train = (meta["split"] == "train").to_numpy()
    is_val = (meta["split"] == "val").to_numpy()
    is_test = (meta["split"] == "test").to_numpy()
    y = meta["y_inconsistent"].to_numpy().astype(int)

    scores = {}

    # logistic on the 2305-d pair feature
    print(f"training logistic regression on {is_train.sum()} cuts ...")
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    clf.fit(features[is_train], y[is_train])
    joblib.dump(clf, out_dir / "v0_logistic.joblib")
    scores["logistic"] = {
        "val_y": y[is_val], "val_s": clf.predict_proba(features[is_val])[:, 1],
        "test_y": y[is_test], "test_s": clf.predict_proba(features[is_test])[:, 1],
    }

    # raw DINOv2 cosine: 1 - cos(eL, eR) -- cosine is the last column of the pair feature
    raw = 1.0 - features[:, -1]
    scores["raw_dino_cosine"] = {
        "val_y": y[is_val], "val_s": raw[is_val],
        "test_y": y[is_test], "test_s": raw[is_test],
    }

    # HSV chi-square + CLIP cosine baselines need the actual frames (not cached features)
    models = list(MODEL_ORDER)
    if not args.skip_baselines:
        cuts = pd.read_parquet(args.cut_index, columns=["movie_id", "shot_left_idx",
                                                         "left_img2_path", "right_img0_path"])
        cuts["cut_id"] = cuts["movie_id"] + "_" + cuts["shot_left_idx"].astype(str).str.zfill(4)
        cuts = cuts.set_index("cut_id")
        eval_mask = is_val | is_test
        eval_ids = meta.loc[eval_mask, "cut_id"].to_numpy()
        left = cuts.loc[eval_ids, "left_img2_path"].to_numpy()
        right = cuts.loc[eval_ids, "right_img0_path"].to_numpy()
        uniq = sorted(set(left) | set(right))
        print(f"baselines: {len(eval_ids)} cuts, {len(uniq)} unique frames")

        print("computing HSV histograms ...")
        hists = compute_hsv_histograms(uniq, bins=(8, 8, 8))
        hsv_all = chisq_scores(hists, left, right)

        print("computing CLIP embeddings ...")
        clip_emb = CLIPImageEncoder(model_id=CLIP_MODEL).encode_paths(uniq)
        clip_all = cosine_distance_scores(clip_emb, left, right)

        eval_split = meta.loc[eval_mask, "split"].to_numpy()
        for name, allvals in (("hsv_chisq", hsv_all), ("clip_cosine", clip_all)):
            scores[name] = {
                "val_y": y[is_val], "val_s": allvals[eval_split == "val"],
                "test_y": y[is_test], "test_s": allvals[eval_split == "test"],
            }
    else:
        models = ["logistic", "raw_dino_cosine"]

    # persist scores first -- the CLIP pass is expensive, don't lose it on a later crash
    np.savez(out_dir / "scores.npz",
             **{f"{n}__{k}": scores[n][k] for n in scores for k in scores[n]})

    rows = []
    for name in models:
        s = scores[name]
        val_y, val_s = drop_nan(s["val_y"], s["val_s"])
        test_y, test_s = drop_nan(s["test_y"], s["test_s"])
        rows.append(evaluate(name, val_y, val_s, test_y, test_s))

    table = pd.DataFrame(rows)[[
        "model", "auroc", "auprc", "f1_at_val_thr", "f1_on_val",
        "precision_at_val_thr", "recall_at_val_thr", "accuracy_at_val_thr", "n_test",
    ]]
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    table.to_csv(out_dir / "results.csv", index=False)

    print("\nv0 results (test split; threshold = F1-optimal on val)")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
