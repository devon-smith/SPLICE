"""v0: train the logistic cut-continuity classifier and evaluate it against three
non-learned baselines.

Models
  logistic         logistic regression on the 2305-d DINOv2 pair feature
  raw_dino_cosine  1 - cos(eL, eR), read straight off the pair feature
  hsv_chisq        HSV colour-histogram chi-square distance
  clip_cosine      1 - cos of CLIP ViT-L image embeddings

Every model is thresholded at its F1-optimal point on val and reported on test.
Per-model val/test scores are saved to scores.npz so calibrate_threshold.py can
add the tau_95 column without recomputation.

Example:
  python scripts/train/v0_logistic.py \\
      --features /mnt/disks/splice-data/pairs/dino_v0_boundary \\
      --out /mnt/disks/splice-data/outputs/v0
"""

import argparse
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics  # noqa: E402
from src.models.baselines import (  # noqa: E402
    CLIPImageEncoder,
    chisq_scores,
    compute_hsv_histograms,
    cosine_distance_scores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v0_logistic")

DEFAULT_FEATURES = "/mnt/disks/splice-data/pairs/dino_v0_boundary"
DEFAULT_CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
DEFAULT_OUT = "/mnt/disks/splice-data/outputs/v0"
MODEL_ORDER = ["logistic", "raw_dino_cosine", "hsv_chisq", "clip_cosine"]


# ---------------------------------------------------------------------------
# Weights & Biases (optional; falls back to offline if not authenticated)
# ---------------------------------------------------------------------------
def resolve_wandb_mode(requested: str) -> str:
    """Pick a usable W&B mode: honour 'disabled', else online iff a key is found."""
    if requested == "disabled":
        return "disabled"
    import wandb

    has_key = bool(os.environ.get("WANDB_API_KEY")) or bool(wandb.api.api_key)
    if requested == "online" and not has_key:
        log.warning("W&B: no API key found -> falling back to offline mode")
        return "offline"
    return requested


def log_wandb(project: str, name: str, mode: str, config: dict, metrics: dict, fig_path: Path):
    if mode == "disabled":
        return
    try:
        import wandb

        run = wandb.init(project=project, name=name, mode=mode, config=config, reinit=True)
        run.log({**metrics, "curves": wandb.Image(str(fig_path))})
        run.finish()
    except Exception as exc:  # noqa: BLE001 - W&B logging must never lose computed results
        log.warning("W&B logging failed for %s: %s", name, exc)


# ---------------------------------------------------------------------------
def plot_curves(y_true: np.ndarray, scores: np.ndarray, title: str, path: Path) -> None:
    """Save a 2-panel ROC + precision-recall figure."""
    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(fpr, tpr)
    ax1.plot([0, 1], [0, 1], "k--", lw=0.7)
    ax1.set(xlabel="FPR", ylabel="TPR", title=f"{title}  ROC")
    ax2.plot(rec, prec)
    ax2.axhline(y_true.mean(), color="k", ls="--", lw=0.7)
    ax2.set(xlabel="recall", ylabel="precision", title=f"{title}  PR")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def evaluate(name: str, val_y, val_s, test_y, test_s) -> dict:
    """Threshold on val (F1-optimal), report ranking + threshold metrics on test."""
    thr = best_f1_threshold(val_y, val_s)
    val_thr = threshold_metrics(val_y, val_s, thr)
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
        "f1_on_val": val_thr["f1"],
    }


def _drop_nan(y: np.ndarray, s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ok = ~np.isnan(s)
    return y[ok], s[ok]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=DEFAULT_FEATURES)
    ap.add_argument("--cut_index", default=DEFAULT_CUT_INDEX)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--clip_model_id", default="openai/clip-vit-large-patch14")
    ap.add_argument("--wandb_project", default="splice-v0")
    ap.add_argument("--wandb_mode", default="online", choices=["online", "offline", "disabled"])
    ap.add_argument("--skip_baselines", action="store_true", help="logistic + dino cosine only")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_dir = Path(args.features)

    # ---- load pair features + metadata --------------------------------------
    features = np.load(feat_dir / "features.npy")
    meta = pd.read_parquet(feat_dir / "meta.parquet")
    log.info(
        "features %s | positive rate %.2f%%", features.shape, 100 * meta["y_inconsistent"].mean()
    )
    is_train = (meta["split"] == "train").to_numpy()
    is_val = (meta["split"] == "val").to_numpy()
    is_test = (meta["split"] == "test").to_numpy()
    y = meta["y_inconsistent"].to_numpy().astype(int)

    scores: dict[str, dict] = {}

    # ---- model 1: logistic regression ---------------------------------------
    log.info("training logistic regression on %d cuts ...", int(is_train.sum()))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    clf.fit(features[is_train], y[is_train])
    joblib.dump(clf, out_dir / "v0_logistic.joblib")
    scores["logistic"] = {
        "val_y": y[is_val],
        "val_s": clf.predict_proba(features[is_val])[:, 1],
        "test_y": y[is_test],
        "test_s": clf.predict_proba(features[is_test])[:, 1],
    }

    # ---- model 2: raw DINOv2 cosine (cosine is the last feature column) ------
    raw = 1.0 - features[:, -1]  # higher = more inconsistent
    scores["raw_dino_cosine"] = {
        "val_y": y[is_val],
        "val_s": raw[is_val],
        "test_y": y[is_test],
        "test_s": raw[is_test],
    }

    # ---- models 3 & 4: HSV chi-square and CLIP cosine (need the frames) ------
    if not args.skip_baselines:
        cuts = pd.read_parquet(
            args.cut_index,
            columns=["movie_id", "shot_left_idx", "left_img2_path", "right_img0_path"],
        )
        cuts["cut_id"] = cuts["movie_id"] + "_" + cuts["shot_left_idx"].astype(str).str.zfill(4)
        cuts = cuts.set_index("cut_id")
        eval_mask = is_val | is_test
        eval_ids = meta.loc[eval_mask, "cut_id"].to_numpy()
        left = cuts.loc[eval_ids, "left_img2_path"].to_numpy()
        right = cuts.loc[eval_ids, "right_img0_path"].to_numpy()
        uniq = sorted(set(left) | set(right))
        log.info("baselines: %d cuts, %d unique frames", len(eval_ids), len(uniq))

        log.info("computing HSV histograms ...")
        hists = compute_hsv_histograms(uniq, bins=(8, 8, 8))
        hsv_all = chisq_scores(hists, left, right)

        log.info("computing CLIP embeddings ...")
        clip_emb = CLIPImageEncoder(model_id=args.clip_model_id).encode_paths(uniq)
        clip_all = cosine_distance_scores(clip_emb, left, right)

        # eval rows are movie-sorted (val/test interleaved) -> split by the split column
        eval_split = meta.loc[eval_mask, "split"].to_numpy()
        for name, allvals in (("hsv_chisq", hsv_all), ("clip_cosine", clip_all)):
            scores[name] = {
                "val_y": y[is_val],
                "val_s": allvals[eval_split == "val"],
                "test_y": y[is_test],
                "test_s": allvals[eval_split == "test"],
            }
    else:
        MODEL_ORDER[:] = ["logistic", "raw_dino_cosine"]

    # persist raw scores first -- the CLIP pass is expensive, never lose it
    np.savez(
        out_dir / "scores.npz",
        **{f"{n}__{k}": scores[n][k] for n in scores for k in scores[n]},
    )

    # ---- evaluate, plot, log ------------------------------------------------
    rows = []
    for name in MODEL_ORDER:
        s = scores[name]
        val_y, val_s = _drop_nan(s["val_y"], s["val_s"])
        test_y, test_s = _drop_nan(s["test_y"], s["test_s"])
        row = evaluate(name, val_y, val_s, test_y, test_s)
        rows.append(row)
        fig_path = out_dir / f"curves_{name}.png"
        plot_curves(test_y, test_s, name, fig_path)
        log_wandb(
            args.wandb_project,
            f"{name}-{date.today().isoformat()}-boundary",
            resolve_wandb_mode(args.wandb_mode),
            {"model": name, "features": str(feat_dir)},
            {k: v for k, v in row.items() if isinstance(v, (int, float))},
            fig_path,
        )

    table = pd.DataFrame(rows)[
        [
            "model",
            "auroc",
            "auprc",
            "f1_at_val_thr",
            "f1_on_val",
            "precision_at_val_thr",
            "recall_at_val_thr",
            "accuracy_at_val_thr",
            "n_test",
        ]
    ]
    (out_dir / "results.json").write_text(json.dumps(rows, indent=2))
    table.to_csv(out_dir / "results.csv", index=False)

    print("\n=== v0 results (test split; threshold = F1-optimal on val) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved -> {out_dir}")


if __name__ == "__main__":
    main()
