# AI-USE: This file was generated with Claude (claude-sonnet-4-6) via Claude Code.
# Prompt: "write a script that loads cached scores for raw DINOv2 cosine, v0 logistic,
# v1.5 MLP, and v2 LoRA from existing score files, optionally computes HSV and CLIP
# on-the-fly, and plots all models on a single precision-recall curve figure saved
# to reports/figures/pr_curves.png."

"""Plot precision-recall curves for all 6 models on the MovieNet test split.

Fast models (raw DINOv2 cosine, v0 logistic, v1.5 MLP, v2 LoRA) load from
cached score files and run in seconds.  Slow models (HSV chi-sq, CLIP cosine)
are computed on-the-fly from images/embeddings; use --skip_slow to omit them.

Saves reports/figures/pr_curves.png.

Usage:
  python scripts/eval/plot_pr_curves.py
  python scripts/eval/plot_pr_curves.py --skip_slow
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "train"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("plot_pr_curves")

PAIR_FEATURES  = Path("/mnt/disks/splice-data/pairs/dino_v0_boundary")
V0_MODEL       = Path("/mnt/disks/splice-data/outputs/v0/v0_logistic.joblib")
V1_SCALER      = Path("/mnt/disks/splice-data/outputs/v1_sound/scaler.joblib")
V1_MODEL       = Path("/mnt/disks/splice-data/outputs/v1_sound/v1_sound_seed2.pt")
V1_SCORES_NPZ  = Path("/mnt/disks/splice-data/outputs/v1_sound/scores.npz")
V2_SWEEP_DIR   = Path("/mnt/disks/splice-data/outputs/v2_lora_20260530")
CUTS_INDEX     = Path("/mnt/disks/splice-data/outputs/cut_index/cuts.parquet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _best_v2_dir(sweep_dir: Path) -> Path:
    best_path, best_auprc = None, -1.0
    for rj in sorted(sweep_dir.glob("*/results.json")):
        r = json.loads(rj.read_text())
        if r.get("test_auprc", -1) > best_auprc:
            best_auprc = r["test_auprc"]
            best_path = rj.parent
    if best_path is None:
        raise SystemExit(f"no results.json found under {sweep_dir}")
    log.info("v2 best config: %s  test AUPRC %.4f", best_path.name, best_auprc)
    return best_path


def _load_pair_features(split: str = "test"):
    import pandas as pd
    features = np.load(PAIR_FEATURES / "features.npy")
    meta = pd.read_parquet(PAIR_FEATURES / "meta.parquet")
    mask = (meta["split"] == split).to_numpy()
    return features[mask], meta.loc[mask, "y_inconsistent"].to_numpy().astype(int)


def _v0_scores(feats: np.ndarray) -> np.ndarray:
    log.info("scoring v0 logistic...")
    return joblib.load(V0_MODEL).predict_proba(feats)[:, 1]


def _v1_scores(feats: np.ndarray) -> np.ndarray:
    # Use pre-saved scores if available (faster)
    if V1_SCORES_NPZ.exists():
        log.info("loading v1.5 scores from npz...")
        return np.load(V1_SCORES_NPZ)["test_s"]
    log.info("scoring v1.5 MLP from model weights...")
    from v1_mlp import MLPHead
    cfg = yaml.safe_load((REPO / "configs" / "v1_sound.yaml").read_text())
    model = MLPHead(in_dim=2305, dropout=cfg["head"]["dropout"])
    model.load_state_dict(torch.load(V1_MODEL, map_location="cpu", weights_only=True))
    model.eval()
    scaler = joblib.load(V1_SCALER)
    with torch.inference_mode():
        x = torch.from_numpy(scaler.transform(feats).astype(np.float32))
        return torch.sigmoid(model(x)).numpy().ravel()


def _v2_scores(sweep_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    run_dir = _best_v2_dir(sweep_dir)
    npz = np.load(run_dir / "scores.npz")
    return npz["test_s"], npz["test_y"].astype(int)


def _hsv_scores(cuts_index: Path, y: np.ndarray) -> np.ndarray:
    import pandas as pd
    from src.models.baselines import chisq_scores, compute_hsv_histograms
    log.info("computing HSV chi-sq scores (slow)...")
    cuts = pd.read_parquet(cuts_index)
    test = cuts[cuts["split"] == "test"].reset_index(drop=True)
    left  = test["left_img2_path"].to_numpy()
    right = test["right_img0_path"].to_numpy()
    uniq  = sorted(set(left) | set(right))
    hists = compute_hsv_histograms(uniq, n_workers=8)
    return chisq_scores(hists, left, right)


def _clip_scores(cuts_index: Path) -> np.ndarray:
    import pandas as pd
    from src.models.baselines import CLIPImageEncoder, cosine_distance_scores
    log.info("computing CLIP cosine scores (slow)...")
    cuts = pd.read_parquet(cuts_index)
    test = cuts[cuts["split"] == "test"].reset_index(drop=True)
    left  = test["left_img2_path"].to_numpy()
    right = test["right_img0_path"].to_numpy()
    uniq  = sorted(set(left) | set(right))
    clip  = CLIPImageEncoder().encode_paths(uniq, num_workers=4)
    return cosine_distance_scores(clip, left, right)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _plot(model_scores: list[tuple[str, np.ndarray, np.ndarray]], out_path: Path) -> None:
    colours = {
        "CLIP cosine":       "#aaaaaa",
        "HSV $\\chi^2$":     "#888888",
        "Raw DINOv2 cosine": "#4daf4a",
        "v0 logistic":       "#377eb8",
        "v1.5 MLP":          "#ff7f00",
        "v2 LoRA":           "#e41a1c",
    }
    styles = {
        "CLIP cosine":       "--",
        "HSV $\\chi^2$":     "--",
        "Raw DINOv2 cosine": "-.",
        "v0 logistic":       ":",
        "v1.5 MLP":          "-",
        "v2 LoRA":           "-",
    }

    fig, ax = plt.subplots(figsize=(5, 4))
    for name, y, s in model_scores:
        p, r, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(r, p,
                label=f"{name}  (AP={ap:.3f})",
                color=colours.get(name, "#333333"),
                linestyle=styles.get(name, "-"),
                linewidth=1.8)

    # Random baseline
    pos_rate = model_scores[0][1].mean()
    ax.axhline(pos_rate, color="black", linestyle=":", linewidth=0.8,
               label=f"Random  (AP={pos_rate:.3f})")

    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall Curves — MovieNet-318 Test Split", fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("saved -> %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip_slow", action="store_true",
                    help="skip HSV and CLIP (they take 10-30 min to compute)")
    ap.add_argument("--v2_sweep_dir", type=Path, default=V2_SWEEP_DIR)
    ap.add_argument("--out", type=Path,
                    default=REPO / "reports" / "figures" / "pr_curves.png")
    args = ap.parse_args()

    log.info("loading pair features...")
    feats, y = _load_pair_features("test")
    log.info("test set: %d cuts, %.2f%% positive", len(y), 100 * y.mean())

    # Verify v2 labels align
    v2_s, v2_y = _v2_scores(args.v2_sweep_dir)
    if not np.array_equal(y, v2_y):
        raise SystemExit("label mismatch between pair features and v2 scores.npz")

    model_scores = []

    if not args.skip_slow:
        model_scores.append(("CLIP cosine",       y, _clip_scores(CUTS_INDEX)))
        model_scores.append(("HSV $\\chi^2$",     y, _hsv_scores(CUTS_INDEX, y)))

    model_scores += [
        ("Raw DINOv2 cosine", y, 1.0 - feats[:, -1]),
        ("v0 logistic",       y, _v0_scores(feats)),
        ("v1.5 MLP",          y, _v1_scores(feats)),
        ("v2 LoRA",           y, v2_s),
    ]

    _plot(model_scores, args.out)

    print("\nAUPRC summary:")
    for name, y_, s in model_scores:
        print(f"  {name:22s}  {average_precision_score(y_, s):.4f}")


if __name__ == "__main__":
    main()
