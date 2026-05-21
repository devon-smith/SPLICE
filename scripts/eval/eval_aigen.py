"""Block 3c: AI-generated-video evaluation harness.

Takes an AI-gen cut index (built by build_aigen_eval.py), runs the *exact* v0/v1
scoring pipeline — frozen DINOv2 embedding, 2305-d pair feature, the six scorers
— and reports metrics overall and per source, against the MovieNet operating
thresholds in configs/operating_thresholds.json.

  python scripts/eval/eval_aigen.py --aigen_index <cuts.parquet> --out <dir>/

`--in_dist_check` runs the same pipeline on the MovieNet test split (using the
cached MovieNet embeddings) and asserts it reproduces v0 AUPRC 0.356, v0
mean-pool-3 AUPRC 0.388 and v1.5 AUPRC 0.405 within 0.005 — a self-test of the
harness.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "train"))
from src.data.movienet import keyframe_key  # noqa: E402
from src.data.pairs import build_pair_features_batch, load_embeddings  # noqa: E402
from src.eval.metrics import ranking_metrics, threshold_metrics  # noqa: E402
from src.models.baselines import (  # noqa: E402
    CLIPImageEncoder,
    chisq_scores,
    compute_hsv_histograms,
    cosine_distance_scores,
)
from src.models.dinov2_encoder import DINOv2Encoder  # noqa: E402
from v1_mlp import MLPHead  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("eval_aigen")

REPO = HERE.parents[2]
MOVIENET_CUTS = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
MOVIENET_EMB = "/mnt/disks/splice-data/embeddings/dinov2_base"
DEFAULT_V0 = "/mnt/disks/splice-data/outputs/v0/v0_logistic.joblib"
DEFAULT_MP = "/mnt/disks/splice-data/outputs/v0_mean_pool/v0_logistic.joblib"
DEFAULT_V1 = "/mnt/disks/splice-data/outputs/v1_sound/v1_sound_seed2.pt"
DEFAULT_V1_SCALER = "/mnt/disks/splice-data/outputs/v1_sound/scaler.joblib"
BOUNDARY = ("left_img2_path", "right_img0_path")  # frames flanking the cut
LEFT_KF = ("left_img0_path", "left_img1_path", "left_img2_path")
RIGHT_KF = ("right_img0_path", "right_img1_path", "right_img2_path")
ALL_KF = LEFT_KF + RIGHT_KF  # all six keyframes -- mean-pool-3 needs every one


def embed_aigen_keyframes(cuts: pd.DataFrame, batch: int = 64) -> dict[str, np.ndarray]:
    """Encode every unique AI-gen keyframe with frozen DINOv2; returns {path: emb}."""
    from PIL import Image

    paths = sorted(set(cuts[list(ALL_KF)].to_numpy().ravel()))
    encoder = DINOv2Encoder()
    out: dict[str, np.ndarray] = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i : i + batch]
        pv = torch.stack(
            [
                encoder.processor(images=Image.open(p).convert("RGB"), return_tensors="pt")[
                    "pixel_values"
                ][0]
                for p in chunk
            ]
        )
        emb = encoder.encode(pv).numpy()
        for p, e in zip(chunk, emb):
            out[p] = e
    log.info("embedded %d AI-gen keyframes", len(out))
    return out


def boundary_features(cuts: pd.DataFrame, emb_of) -> np.ndarray:
    """2305-d boundary pair feature per cut, given a path->embedding callable."""
    e_left = np.stack([emb_of(p) for p in cuts["left_img2_path"]])
    e_right = np.stack([emb_of(p) for p in cuts["right_img0_path"]])
    return build_pair_features_batch(e_left, e_right)


def mean_pool_features(cuts: pd.DataFrame, emb_of) -> np.ndarray:
    """2305-d mean-pool-3 pair feature: each side is the mean of its 3 keyframe embeddings.

    Mirrors build_pair_features.py --mode mean_pool_3 exactly.
    """
    e_left = np.mean([[emb_of(p) for p in cuts[c]] for c in LEFT_KF], axis=0)
    e_right = np.mean([[emb_of(p) for p in cuts[c]] for c in RIGHT_KF], axis=0)
    return build_pair_features_batch(e_left, e_right)


def score_embedding_models(
    feats: np.ndarray, mp_feats: np.ndarray, v0_path: str, v1_path: str, mp_path: str
) -> dict:
    """raw DINOv2 cosine, v0 logistic, v0 mean-pool-3, v1.5 MLP -- from the cached features."""
    scores = {"raw_dino_cosine": 1.0 - feats[:, -1]}
    scores["logistic"] = joblib.load(v0_path).predict_proba(feats)[:, 1]
    scores["mean_pool"] = joblib.load(mp_path).predict_proba(mp_feats)[:, 1]
    cfg = yaml.safe_load((REPO / "configs" / "v1_sound.yaml").read_text())
    model = MLPHead(in_dim=2305, dropout=cfg["head"]["dropout"])
    model.load_state_dict(torch.load(v1_path, map_location="cpu", weights_only=True))
    model.eval()
    scaler = joblib.load(DEFAULT_V1_SCALER)
    with torch.inference_mode():
        x = torch.from_numpy(scaler.transform(feats).astype(np.float32))
        scores["v1.5"] = torch.sigmoid(model(x)).numpy()
    return scores


def score_image_baselines(cuts: pd.DataFrame) -> dict:
    """HSV chi-square and CLIP cosine -- recomputed from the keyframe images."""
    left = cuts["left_img2_path"].to_numpy()
    right = cuts["right_img0_path"].to_numpy()
    uniq = sorted(set(left) | set(right))
    hists = compute_hsv_histograms(uniq, n_workers=8)
    clip = CLIPImageEncoder().encode_paths(uniq, num_workers=4)
    return {
        "hsv_chisq": chisq_scores(hists, left, right),
        "clip_cosine": cosine_distance_scores(clip, left, right),
    }


def metric_row(name: str, y: np.ndarray, s: np.ndarray, thr: dict) -> dict:
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    rank = ranking_metrics(y, s)
    row = {"model": name, "n": int(len(y)), "auroc": rank["auroc"], "auprc": rank["auprc"]}
    if name in thr and "val_thr" in thr[name]:
        row["f1_at_val"] = threshold_metrics(y, s, thr[name]["val_thr"])["f1"]
    if name in thr and "tau99" in thr[name]:
        row["f1_at_tau99"] = threshold_metrics(y, s, thr[name]["tau99"])["f1"]
    return row


def _fmt(row: dict, key: str) -> str:
    v = row.get(key)
    return f"{v:.3f}" if isinstance(v, (int, float)) else "—"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--aigen_index", default=None)
    ap.add_argument("--out", default="/mnt/disks/splice-data/outputs/aigen_eval/results/")
    ap.add_argument("--v0_model", default=DEFAULT_V0)
    ap.add_argument("--mp_model", default=DEFAULT_MP)
    ap.add_argument("--v1_model", default=DEFAULT_V1)
    ap.add_argument("--in_dist_check", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = json.loads((REPO / "configs" / "operating_thresholds.json").read_text())

    if args.in_dist_check:
        run_in_dist_check(args, thr, out_dir)
        return

    if not args.aigen_index:
        raise SystemExit("--aigen_index is required (or use --in_dist_check)")
    cuts = pd.read_parquet(args.aigen_index)
    log.info("AI-gen cuts: %d, sources %s", len(cuts), sorted(cuts["source"].unique()))
    emb = embed_aigen_keyframes(cuts)
    feats = boundary_features(cuts, lambda p: emb[p])
    mp_feats = mean_pool_features(cuts, lambda p: emb[p])
    scores = {
        **score_embedding_models(feats, mp_feats, args.v0_model, args.v1_model, args.mp_model),
        **score_image_baselines(cuts),
    }
    y = cuts["y_inconsistent"].to_numpy().astype(int)

    order = ["raw_dino_cosine", "hsv_chisq", "clip_cosine", "logistic", "mean_pool", "v1.5"]
    overall = [metric_row(m, y, scores[m], thr) for m in order]
    per_source = {}
    for src, g in cuts.groupby("source"):
        idx = g.index.to_numpy()
        per_source[src] = [metric_row(m, y[idx], scores[m][idx], thr) for m in order]

    results = {
        "n_pairs": int(len(cuts)),
        "positive_rate": float(y.mean()),
        "overall": overall,
        "per_source": per_source,
    }
    (out_dir / "aigen_results.json").write_text(json.dumps(results, indent=2, default=float))
    _write_md(out_dir, results, order)
    _plot(out_dir, per_source, order)
    print(f"wrote AI-gen results -> {out_dir}")


def run_in_dist_check(args, thr, out_dir) -> None:
    """Re-score the MovieNet test split through this harness; assert it reproduces v0/v1.5."""
    log.info("in-distribution check on MovieNet test split")
    cuts = pd.read_parquet(MOVIENET_CUTS)
    cuts = cuts[cuts["split"] == "test"].reset_index(drop=True)
    emb, key2row = load_embeddings(MOVIENET_EMB)
    feats = boundary_features(cuts, lambda p: emb[key2row[keyframe_key(p)]])
    mp_feats = mean_pool_features(cuts, lambda p: emb[key2row[keyframe_key(p)]])
    scores = score_embedding_models(feats, mp_feats, args.v0_model, args.v1_model, args.mp_model)
    y = cuts["y_inconsistent"].to_numpy().astype(int)

    expect = {"logistic": 0.3561, "mean_pool": 0.3882, "v1.5": 0.4045}
    ok = True
    lines = [
        "# AI-gen Harness In-Distribution Check\n",
        "Re-scoring the MovieNet test split through the eval_aigen.py pipeline.\n",
        "| model | reproduced AUPRC | expected | Δ | pass |",
        "|---|---|---|---|---|",
    ]
    for m, exp in expect.items():
        got = ranking_metrics(y, scores[m])["auprc"]
        passed = abs(got - exp) < 0.005
        ok &= passed
        lines.append(
            f"| {m} | {got:.4f} | {exp:.4f} | {got - exp:+.4f} | "
            f"{'PASS' if passed else 'FAIL'} |"
        )
        log.info(
            "%s: reproduced AUPRC %.4f (expected %.4f) -> %s",
            m,
            got,
            exp,
            "PASS" if passed else "FAIL",
        )
    (out_dir / "in_dist_check.md").write_text("\n".join(lines) + "\n")
    if not ok:
        raise SystemExit("IN-DIST CHECK FAILED: harness does not reproduce known numbers")
    print("in-distribution check PASSED — harness reproduces v0/v1.5 within 0.005")


def _write_md(out_dir, results, order) -> None:
    lab = {
        "raw_dino_cosine": "Raw DINOv2 cosine",
        "hsv_chisq": "HSV χ²",
        "clip_cosine": "CLIP ViT-L cosine",
        "logistic": "v0 logistic",
        "mean_pool": "v0 mean-pool-3",
        "v1.5": "v1.5 MLP",
    }
    md = [
        "# AI-Gen Evaluation Results\n",
        "## Setup\n",
        f"- N pairs: {results['n_pairs']}",
        f"- positive (inconsistent) rate: {100 * results['positive_rate']:.1f}%",
        f"- sources: {', '.join(sorted(results['per_source']))}\n",
        "## Aggregate results\n",
        "| Model | AUROC | AUPRC | F1@val | F1@τ99 |",
        "|---|---|---|---|---|",
    ]
    for r in results["overall"]:
        md.append(
            f"| {lab[r['model']]} | {r['auroc']:.3f} | {r['auprc']:.3f} | "
            f"{_fmt(r, 'f1_at_val')} | {_fmt(r, 'f1_at_tau99')} |"
        )
    md.append("\n## Per-source\n")
    for src, rows in results["per_source"].items():
        md.append(f"### {src}  (n = {rows[0]['n']})\n")
        md.append("| Model | AUROC | AUPRC |")
        md.append("|---|---|---|")
        for r in rows:
            md.append(f"| {lab[r['model']]} | {r['auroc']:.3f} | {r['auprc']:.3f} |")
        md.append("")
    md.append("## MovieNet comparison\n")
    md.append("| Model | MovieNet AUPRC | AI-gen AUPRC | Δ |")
    md.append("|---|---|---|---|")
    mn = {"logistic": 0.356, "mean_pool": 0.388, "v1.5": 0.403}
    by = {r["model"]: r for r in results["overall"]}
    for m in ("logistic", "mean_pool", "v1.5"):
        md.append(
            f"| {lab[m]} | {mn[m]:.3f} | {by[m]['auprc']:.3f} | " f"{by[m]['auprc'] - mn[m]:+.3f} |"
        )
    md.append("\n## Analysis\n")
    md.append(
        "_DEVON FILLS IN: Did MovieNet-trained features transfer to AI-gen? Which "
        "source was easiest / hardest? Did v1.5 keep its lift over v0? Note that "
        "per-source metrics with n < 20 are not reliable._\n"
    )
    (out_dir / "aigen_results.md").write_text("\n".join(md))


def _plot(out_dir, per_source, order) -> None:
    lab = {
        "raw_dino_cosine": "raw cos",
        "hsv_chisq": "HSV",
        "clip_cosine": "CLIP",
        "logistic": "v0",
        "mean_pool": "v0 MP3",
        "v1.5": "v1.5",
    }
    sources = sorted(per_source)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(order)
    x = np.arange(len(sources))
    for j, m in enumerate(order):
        vals = [next(r["auprc"] for r in per_source[s] if r["model"] == m) for s in sources]
        ax.bar(x + j * width, vals, width, label=lab[m])
    ax.set(
        xticks=x + 0.4 - width / 2,
        xlabel="source",
        ylabel="AUPRC",
        title="AI-gen AUPRC by source and model",
    )
    ax.set_xticklabels(sources)
    ax.legend(fontsize=8, ncol=6)
    fig.tight_layout()
    fig.savefig(out_dir / "aigen_auprc_by_source.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    main()
