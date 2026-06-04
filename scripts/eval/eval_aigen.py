# AI-generated-video evaluation harness
# takes an aigen cut index, runs the same v0/v1 scoring pipeline (DINOv2 + 2305-d
# pair feature + 6 scorers), reports metrics overall and per source.
# --in_dist_check sanity-checks the harness on MovieNet test (should reproduce v0/v1.5 AUPRCs).
# usage: python scripts/eval/eval_aigen.py --aigen_index PATH --out DIR

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from PIL import Image

from src.data.movienet import keyframe_key
from src.data.pairs import build_pair_features_batch, load_embeddings
from src.eval.metrics import ranking_metrics, threshold_metrics
from src.models.baselines import (CLIPImageEncoder, chisq_scores,
                                   compute_hsv_histograms, cosine_distance_scores)
from src.models.dinov2_encoder import DINOv2Encoder

REPO = Path(__file__).resolve().parents[2]
MOVIENET_CUTS = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
MOVIENET_EMB = "/mnt/disks/splice-data/embeddings/dinov2_base"
DEFAULT_V0 = "/mnt/disks/splice-data/outputs/v0/v0_logistic.joblib"
DEFAULT_MP = "/mnt/disks/splice-data/outputs/v0_mean_pool/v0_logistic.joblib"
DEFAULT_V1 = "/mnt/disks/splice-data/outputs/v1_sound/v1_sound_seed2.pt"
DEFAULT_V1_SCALER = "/mnt/disks/splice-data/outputs/v1_sound/scaler.joblib"

LEFT_KF = ("left_img0_path", "left_img1_path", "left_img2_path")
RIGHT_KF = ("right_img0_path", "right_img1_path", "right_img2_path")
ALL_KF = LEFT_KF + RIGHT_KF


# inlined here so eval_aigen doesn't depend on the v1.5 training script
class MLPHead(nn.Module):
    def __init__(self, in_dim=2305, hidden=(512, 128), dropout=0.1):
        super().__init__()
        layers = []
        dim = in_dim
        for w in hidden:
            layers += [nn.Linear(dim, w), nn.ReLU(), nn.Dropout(dropout)]
            dim = w
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# encode every unique AI-gen keyframe with frozen DINOv2 -> {path: emb}
def embed_aigen_keyframes(cuts, batch=64):
    paths = sorted(set(cuts[list(ALL_KF)].to_numpy().ravel()))
    encoder = DINOv2Encoder()
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
    print(f"embedded {len(out)} AI-gen keyframes")
    return out


def boundary_features(cuts, emb_of):
    e_left = np.stack([emb_of(p) for p in cuts["left_img2_path"]])
    e_right = np.stack([emb_of(p) for p in cuts["right_img0_path"]])
    return build_pair_features_batch(e_left, e_right)


# mirrors build_pair_features.py --mode mean_pool_3
def mean_pool_features(cuts, emb_of):
    e_left = np.mean([[emb_of(p) for p in cuts[c]] for c in LEFT_KF], axis=0)
    e_right = np.mean([[emb_of(p) for p in cuts[c]] for c in RIGHT_KF], axis=0)
    return build_pair_features_batch(e_left, e_right)


# raw DINOv2 cosine + v0 logistic + v0 mean-pool-3 + v1.5 MLP
def score_embedding_models(feats, mp_feats, v0_path, v1_path, mp_path):
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


# HSV chi-square + CLIP cosine -- recomputed from keyframe images
def score_image_baselines(cuts):
    left = cuts["left_img2_path"].to_numpy()
    right = cuts["right_img0_path"].to_numpy()
    uniq = sorted(set(left) | set(right))
    hists = compute_hsv_histograms(uniq, n_workers=8)
    clip = CLIPImageEncoder().encode_paths(uniq, num_workers=4)
    return {"hsv_chisq": chisq_scores(hists, left, right),
            "clip_cosine": cosine_distance_scores(clip, left, right)}


def metric_row(name, y, s, thr):
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    row = {"model": name, "n": int(len(y))}
    if len(np.unique(y)) < 2:
        row.update({"auroc": float("nan"), "auprc": float("nan"), "note": "single-class"})
    else:
        rank = ranking_metrics(y, s)
        row.update({"auroc": rank["auroc"], "auprc": rank["auprc"]})
    if name in thr and "val_thr" in thr[name]:
        row["f1_at_val"] = threshold_metrics(y, s, thr[name]["val_thr"])["f1"]
    if name in thr and "tau99" in thr[name]:
        row["f1_at_tau99"] = threshold_metrics(y, s, thr[name]["tau99"])["f1"]
    return row


# self-test: re-score MovieNet test through this harness and assert it reproduces v0/v1.5
def run_in_dist_check(args, thr, out_dir):
    print("in-distribution check on MovieNet test split")
    cuts = pd.read_parquet(MOVIENET_CUTS)
    cuts = cuts[cuts["split"] == "test"].reset_index(drop=True)
    emb, key2row = load_embeddings(MOVIENET_EMB)
    feats = boundary_features(cuts, lambda p: emb[key2row[keyframe_key(p)]])
    mp_feats = mean_pool_features(cuts, lambda p: emb[key2row[keyframe_key(p)]])
    scores = score_embedding_models(feats, mp_feats, args.v0_model, args.v1_model, args.mp_model)
    y = cuts["y_inconsistent"].to_numpy().astype(int)

    expect = {"logistic": 0.3561, "mean_pool": 0.3882, "v1.5": 0.4045}
    ok = True
    for m, exp in expect.items():
        got = ranking_metrics(y, scores[m])["auprc"]
        passed = abs(got - exp) < 0.005
        ok &= passed
        print(f"  {m}: {got:.4f} (expected {exp:.4f}, delta {got - exp:+.4f}) -> "
              f"{'PASS' if passed else 'FAIL'}")
    if not ok:
        raise SystemExit("IN-DIST CHECK FAILED: harness does not reproduce known numbers")
    print("in-distribution check PASSED -- harness reproduces v0/v1.5 within 0.005")


def main():
    ap = argparse.ArgumentParser()
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
    print(f"AI-gen cuts: {len(cuts)}, sources {sorted(cuts['source'].unique())}")
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
    print(f"wrote {out_dir / 'aigen_results.json'}")

    # per-source AUPRC bar chart
    sources = sorted(per_source)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    width = 0.8 / len(order)
    x = np.arange(len(sources))
    for j, m in enumerate(order):
        vals = [next(r["auprc"] for r in per_source[s] if r["model"] == m) for s in sources]
        ax.bar(x + j * width, vals, width, label=m)
    ax.set(xticks=x + 0.4 - width / 2, xlabel="source", ylabel="AUPRC",
           title="AI-gen AUPRC by source and model")
    ax.set_xticklabels(sources)
    ax.legend(fontsize=8, ncol=6)
    fig.tight_layout()
    fig.savefig(out_dir / "aigen_auprc_by_source.png", dpi=120)
    plt.close(fig)

    # console summary
    print("\nOverall:")
    for r in overall:
        print(f"  {r['model']:18s}  AUROC {r.get('auroc', float('nan')):.3f}  "
              f"AUPRC {r.get('auprc', float('nan')):.3f}  n={r['n']}")


if __name__ == "__main__":
    main()
