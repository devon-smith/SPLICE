"""Phase-3 P6: feature importance and interpretability for the v0/v1 pair feature.

  1. coefficient norms of the v0 logistic head, by feature slice
  2. permutation importance of each slice (AUPRC drop when the slice is shuffled)
  3. ablation models -- logistic re-trained on feature subsets
  4. linear probes -- can DINOv2 embeddings predict low-level image attributes?

The 2305-d pair feature is [ eL(768) | eR(768) | |eL-eR|(768) | cos(1) ].

Writes reports/v1_feature_importance.md and reports/figures/v1_ablation_auprc.png.
"""

import argparse
import logging
import sys
from pathlib import Path

import cv2
import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import average_precision_score, r2_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.pairs import load_embeddings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("feature_importance")

REPO = Path(__file__).resolve().parents[2]
PAIRS = Path("/mnt/disks/splice-data/pairs/dino_v0_boundary")
V0 = Path("/mnt/disks/splice-data/outputs/v0")
EMB = "/mnt/disks/splice-data/embeddings/dinov2_base"
FRAMES = Path("/mnt/disks/splice-data/datasets/movienet/240P_frames")
SLICES = {  # name -> (start, end) columns of the 2305-d feature
    "left embedding": (0, 768),
    "right embedding": (768, 1536),
    "|eL - eR|": (1536, 2304),
    "cosine": (2304, 2305),
}
ABLATIONS = {  # name -> column ranges
    "full (2305-d)": [(0, 2305)],
    "concat only": [(0, 1536)],
    "|eL-eR| only": [(1536, 2304)],
    "cosine only": [(2304, 2305)],
    "concat + |eL-eR|": [(0, 2304)],
    "|eL-eR| + cosine": [(1536, 2305)],
}


def _cols(ranges: list[tuple[int, int]]) -> np.ndarray:
    return np.concatenate([np.arange(a, b) for a, b in ranges])


def analysis_coefficients(clf) -> tuple[list[str], dict]:
    coef = clf[-1].coef_[0]  # logistic head, on standardised features
    md = [
        "## 1. Coefficient analysis (v0 logistic head)\n",
        "Coefficients are on standardised features, so magnitudes compare across "
        "slices. `mean |coef|` (per-dimension) is the fair cross-slice measure.\n",
        "| slice | dims | L1 norm | L2 norm | mean \\|coef\\| |",
        "|---|---|---|---|---|",
    ]
    l2 = {}
    for name, (a, b) in SLICES.items():
        s = coef[a:b]
        l2[name] = float(np.linalg.norm(s))
        md.append(
            f"| {name} | {b - a} | {np.abs(s).sum():.2f} | "
            f"{np.linalg.norm(s):.2f} | {np.abs(s).mean():.4f} |"
        )
    md.append("")
    return md, l2


def analysis_permutation(clf, x_test, y_test, n_repeat=5) -> tuple[list[str], dict]:
    base = average_precision_score(y_test, clf.predict_proba(x_test)[:, 1])
    rng = np.random.default_rng(0)
    md = [
        "## 2. Permutation importance\n",
        f"Each slice's rows are shuffled jointly ({n_repeat} repeats); the AUPRC drop "
        f"is its marginal importance. Baseline test AUPRC = {base:.4f}.\n",
        "| slice | AUPRC after shuffle | AUPRC drop |",
        "|---|---|---|",
    ]
    drops = {}
    for name, (a, b) in SLICES.items():
        vals = []
        for _ in range(n_repeat):
            xp = x_test.copy()
            xp[:, a:b] = xp[rng.permutation(len(xp)), a:b]
            vals.append(average_precision_score(y_test, clf.predict_proba(xp)[:, 1]))
        shuffled = float(np.mean(vals))
        drops[name] = base - shuffled
        md.append(f"| {name} | {shuffled:.4f} | {base - shuffled:.4f} |")
    md.append("")
    return md, drops


def analysis_ablations(features, y, masks) -> tuple[list[str], dict]:
    md = [
        "## 3. Ablation models (logistic re-trained on feature subsets)\n",
        "| feature subset | dims | test AUPRC |",
        "|---|---|---|",
    ]
    aurpc = {}
    for name, ranges in ABLATIONS.items():
        cols = _cols(ranges)
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"),
        )
        clf.fit(features[masks["train"]][:, cols], y[masks["train"]])
        ap = average_precision_score(
            y[masks["test"]], clf.predict_proba(features[masks["test"]][:, cols])[:, 1]
        )
        aurpc[name] = ap
        md.append(f"| {name} | {len(cols)} | {ap:.4f} |")
        log.info("ablation %-22s dims=%-5d test AUPRC %.4f", name, len(cols), ap)
    md.append("")
    return md, aurpc


def analysis_probes(n_sample=8000) -> list[str]:
    emb, key2row = load_embeddings(EMB)
    rng = np.random.default_rng(0)
    train_movies = set(
        pd.read_parquet(
            "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet", columns=["movie_id", "split"]
        ).query("split == 'train'")["movie_id"]
    )
    keys = [k for k in key2row if k.split("/")[0] in train_movies]
    sample = rng.choice(keys, size=min(n_sample, len(keys)), replace=False)

    rows, vecs = [], []
    for k in sample:
        img = cv2.imread(str(FRAMES / k))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        rows.append((gray.mean() / 255, gray.std() / 255, hsv[:, :, 1].mean() / 255))
        vecs.append(emb[key2row[k]])
    attrs = np.array(rows)
    x = np.array(vecs, dtype=np.float32)
    n_tr = int(0.8 * len(x))
    md = [
        "## 4. Linear probes -- what is in the DINOv2 embedding?\n",
        f"Ridge probes from frozen DINOv2 embeddings to low-level image attributes "
        f"({len(x):,} train keyframes, 80/20 split). High R^2 means the attribute is "
        "linearly decodable from the embedding.\n",
        "| attribute | probe R^2 |",
        "|---|---|",
    ]
    for j, attr in enumerate(["mean luminance", "luminance contrast (std)", "mean saturation"]):
        probe = Ridge(alpha=1.0).fit(x[:n_tr], attrs[:n_tr, j])
        r2 = r2_score(attrs[n_tr:, j], probe.predict(x[n_tr:]))
        md.append(f"| {attr} | {r2:.3f} |")
        log.info("probe %-26s R^2 %.3f", attr, r2)
    md.append("")
    return md


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--probe_sample", type=int, default=8000)
    args = ap.parse_args()

    features = np.load(PAIRS / "features.npy")
    meta = pd.read_parquet(PAIRS / "meta.parquet")
    y = meta["y_inconsistent"].to_numpy().astype(int)
    masks = {s: (meta["split"] == s).to_numpy() for s in ("train", "val", "test")}
    clf = joblib.load(V0 / "v0_logistic.joblib")

    md = ["# v1 Feature Importance & Interpretability (Phase 3, P6)\n"]
    md.append(
        "Which parts of the 2305-d pair feature carry the signal? "
        "Feature layout: `[ eL(768) | eR(768) | |eL-eR|(768) | cos(1) ]`.\n"
    )
    coef_md, coef_l2 = analysis_coefficients(clf)
    md += coef_md
    perm_md, drops = analysis_permutation(clf, features[masks["test"]], y[masks["test"]])
    md += perm_md
    abl_md, ablation = analysis_ablations(features, y, masks)
    md += abl_md
    md += analysis_probes(args.probe_sample)

    top_perm = max(drops, key=drops.get)
    md.append("## Interpretation\n")
    md.append(
        f"- **Permutation importance** ranks `{top_perm}` first "
        f"(AUPRC drop {drops[top_perm]:.4f}) -- the slice the trained model leans on most."
    )
    md.append(
        f"- **Ablations**: `|eL-eR|` alone reaches AUPRC {ablation['|eL-eR| only']:.3f} vs "
        f"the full feature's {ablation['full (2305-d)']:.3f}; `cosine` alone "
        f"{ablation['cosine only']:.3f} (= the raw-cosine baseline). The difference "
        "signal carries most of the value; concat and cosine add the remainder."
    )
    md.append(
        "- **Probes** show the DINOv2 embedding linearly encodes low-level appearance "
        "(luminance, contrast, saturation) -- exactly the cues a visual cut disturbs, "
        "which is why difference-of-embeddings is a strong continuity feature.\n"
    )
    (REPO / "reports" / "v1_feature_importance.md").write_text("\n".join(md))

    fig, ax = plt.subplots(figsize=(7, 4))
    names = list(ABLATIONS)
    ax.bar(range(len(names)), [ablation[n] for n in names], color="tab:blue", alpha=0.85)
    ax.axhline(0.255, color="r", ls="--", lw=1, label="raw DINOv2 cosine baseline")
    ax.set(
        xticks=range(len(names)),
        ylabel="test AUPRC",
        title="v0 logistic AUPRC by feature subset (ablation)",
    )
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(REPO / "reports" / "figures" / "v1_ablation_auprc.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    slices = list(coef_l2)
    ax.bar(range(len(slices)), [coef_l2[s] for s in slices], color="tab:purple", alpha=0.85)
    ax.set(
        xticks=range(len(slices)),
        ylabel="L2 norm of coefficients",
        title="v0 logistic coefficient L2 norm by feature slice",
    )
    ax.set_xticklabels(slices, rotation=15, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(REPO / "reports" / "figures" / "v1_coefficient_norms.png", dpi=120)
    plt.close(fig)
    print("wrote v1_feature_importance.md + v1_ablation_auprc.png + v1_coefficient_norms.png")


if __name__ == "__main__":
    main()
