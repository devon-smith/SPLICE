"""Action 3: train a logistic head on the 4610-d fused DINOv2+CLIP feature.

The v2 architecture experiment. Trains logistic regression with v0_logistic.py's
recipe (StandardScaler + balanced LogisticRegression) on the fused boundary
feature, evaluates on MovieNet test and the Veo pilot, and bootstraps the AUPRC
difference against v1.5. Outcome verdict (A/B/C) in reports/fused_results.md.

Prerequisite: pairs/fused_boundary/ built by scripts/prep/build_fused_features.py
(which needs the full CLIP cache).

  python scripts/train/fused_logistic.py
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts" / "eval"))
from significance_tests import bootstrap_auprc_diff  # noqa: E402
from src.data.movienet import keyframe_key  # noqa: E402
from src.data.pairs import build_fused_pair_features, load_embeddings  # noqa: E402
from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics  # noqa: E402
from src.models.dinov2_encoder import DINOv2Encoder  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fused_logistic")

FUSED_DIR = "/mnt/disks/splice-data/pairs/fused_boundary"
CLIP_DIR = "/mnt/disks/splice-data/embeddings/clip_vitl14"
V0_BOUNDARY_META = "/mnt/disks/splice-data/pairs/dino_v0_boundary/meta.parquet"
V1_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
AIGEN_CUTS = "/mnt/disks/splice-data/outputs/aigen_eval/cuts.parquet"
PER_PAIR_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
DEFAULT_OUT = "/mnt/disks/splice-data/outputs/fused_logistic"

V0_AUPRC = 0.356  # published MovieNet test AUPRC, v0 logistic (reports/v0_results.md)
V1_AUPRC = 0.4045  # published MovieNet test AUPRC, v1.5 MLP seed 2

BUCKET_OF = {
    "A003": "clean",
    "A013": "clean",
    "A001": "drift",
    "A002": "drift",
    "A004": "drift",
    "A011": "drift",
    "A012": "drift",
    "A014": "drift",
    "A005": "major",
    "A015": "major",
}
BUCKET_CODE = {"clean": 0, "drift": 1, "major": 2}
BOUNDARY = ("left_img2_path", "right_img0_path")


def embed_dino(paths: list[str], encoder: DINOv2Encoder, batch: int = 64) -> dict:
    """Fresh DINOv2 embeddings for a set of keyframe paths -> {path: emb}."""
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
    return out


def veo_fused_features(cuts: pd.DataFrame, clip_dir: str) -> np.ndarray:
    """Build the 4610-d fused feature for the Veo pilot cuts.

    DINOv2 embeddings are computed fresh (the AI-gen frames are not in the
    MovieNet cache); CLIP embeddings come from the cached clip_vitl14 store.
    """
    paths = sorted(set(cuts[list(BOUNDARY)].to_numpy().ravel()))
    dino = embed_dino(paths, DINOv2Encoder())
    clip_emb, clip_k2r = load_embeddings(clip_dir)

    d_left = np.stack([dino[p] for p in cuts["left_img2_path"]])
    d_right = np.stack([dino[p] for p in cuts["right_img0_path"]])
    c_left = np.stack([clip_emb[clip_k2r[keyframe_key(p)]] for p in cuts["left_img2_path"]])
    c_right = np.stack([clip_emb[clip_k2r[keyframe_key(p)]] for p in cuts["right_img0_path"]])
    return build_fused_pair_features(d_left, d_right, c_left, c_right)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fused_dir", default=FUSED_DIR)
    ap.add_argument("--clip_dir", default=CLIP_DIR)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n_boot", type=int, default=1000)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load fused features + train ----------------------------------------
    feat = np.load(Path(args.fused_dir) / "features.npy")
    meta = pd.read_parquet(Path(args.fused_dir) / "meta.parquet")
    log.info(
        "fused features %s | positive rate %.2f%%", feat.shape, 100 * meta["y_inconsistent"].mean()
    )
    y = meta["y_inconsistent"].to_numpy().astype(int)
    is_train = (meta["split"] == "train").to_numpy()
    is_val = (meta["split"] == "val").to_numpy()
    is_test = (meta["split"] == "test").to_numpy()

    log.info("training fused logistic on %d cuts ...", int(is_train.sum()))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced", solver="lbfgs"),
    )
    clf.fit(feat[is_train], y[is_train])
    joblib.dump(clf, out_dir / "model.pkl")

    val_s = clf.predict_proba(feat[is_val])[:, 1]
    test_s = clf.predict_proba(feat[is_test])[:, 1]
    np.savez(
        out_dir / "scores.npz",
        val_s=val_s,
        val_y=y[is_val],
        test_s=test_s,
        test_y=y[is_test],
    )

    # ---- MovieNet test metrics ----------------------------------------------
    thr = best_f1_threshold(y[is_val], val_s)
    rank = ranking_metrics(y[is_test], test_s)
    tm = threshold_metrics(y[is_test], test_s, thr)
    mn = {
        "auprc": rank["auprc"],
        "auroc": rank["auroc"],
        "f1_at_val": tm["f1"],
        "precision": tm["precision"],
        "recall": tm["recall"],
        "val_thr": thr,
    }
    log.info(
        "MovieNet test: AUPRC %.4f  AUROC %.4f  F1 %.4f", mn["auprc"], mn["auroc"], mn["f1_at_val"]
    )

    # ---- significance vs v1.5 (movie-level bootstrap, aligned by cut_id) -----
    boot = _bootstrap_vs_v15(meta[is_test].reset_index(drop=True), test_s, args.n_boot)

    # ---- Veo pilot ----------------------------------------------------------
    veo = _veo_eval(args.clip_dir, clf)

    _write_report(out_dir, mn, boot, veo)
    _print_console(mn, boot, veo)
    log.info("wrote %s", out_dir / "model.pkl")


def _bootstrap_vs_v15(test_meta: pd.DataFrame, fused_test_s: np.ndarray, n_boot: int) -> dict:
    """Movie-level bootstrap CI for AUPRC(fused) - AUPRC(v1.5) on the test split."""
    v1 = np.load(V1_NPZ)
    dino_meta = pd.read_parquet(V0_BOUNDARY_META)
    dino_test = dino_meta[dino_meta["split"] == "test"].reset_index(drop=True)
    # v1.5 test scores are row-aligned to dino_v0_boundary test rows -> map by cut_id
    v15_by_cut = dict(zip(dino_test["cut_id"], v1["test_s"].astype(float)))
    y_by_cut = dict(zip(dino_test["cut_id"], v1["test_y"].astype(int)))

    cut_ids = test_meta["cut_id"].to_numpy()
    keep = np.array([c in v15_by_cut for c in cut_ids])
    if not keep.all():
        log.warning("%d test cuts not matched to v1.5 scores -- dropped", int((~keep).sum()))
    cut_ids = cut_ids[keep]
    fused_s = fused_test_s[keep]
    v15_s = np.array([v15_by_cut[c] for c in cut_ids])
    y = np.array([y_by_cut[c] for c in cut_ids])
    movie_ids = test_meta.loc[keep, "movie_id"].to_numpy()

    res = bootstrap_auprc_diff(y, fused_s, v15_s, movie_ids, n_boot=n_boot)
    log.info(
        "bootstrap AUPRC(fused) - AUPRC(v1.5) = %+.4f  CI [%+.4f, %+.4f]",
        res["diff"],
        res["ci"][0],
        res["ci"][1],
    )
    return {"diff": res["diff"], "ci": res["ci"], "n": int(len(y))}


def _veo_eval(clip_dir: str, clf) -> dict:
    """Score the Veo pilot with the fused model; bucket means + Spearman."""
    cuts = pd.read_parquet(AIGEN_CUTS)
    cuts["pair_id"] = cuts["left_img2_path"].map(lambda p: Path(p).parent.name)
    feats = veo_fused_features(cuts, clip_dir)
    scores = clf.predict_proba(feats)[:, 1]

    df = pd.DataFrame({"pair_id": cuts["pair_id"], "fused": scores})
    df["bucket"] = df["pair_id"].map(BUCKET_OF)
    code = df["bucket"].map(BUCKET_CODE).to_numpy()
    rho = float(spearmanr(df["fused"].to_numpy(), code)[0])

    # v1.5's Spearman on the same buckets, recomputed for a like-for-like compare
    pp = pd.read_csv(PER_PAIR_CSV, dtype={"pair_id": str})
    pp_code = pp["pair_id"].map(BUCKET_OF).map(BUCKET_CODE).to_numpy()
    v15_rho = float(spearmanr(pp["v1.5_MLP"].to_numpy(), pp_code)[0])

    bucket_mean = {
        b: float(df.loc[df["bucket"] == b, "fused"].mean()) for b in ("clean", "drift", "major")
    }
    return {
        "per_pair": df.sort_values("fused", ascending=False),
        "bucket_mean": bucket_mean,
        "spearman": rho,
        "v15_spearman": v15_rho,
    }


def _verdict(mn: dict, boot: dict, veo: dict) -> tuple[str, str]:
    """Map results to outcome A / B / C."""
    lo, hi = boot["ci"]
    veo_better = veo["spearman"] > veo["v15_spearman"]
    mn_sig_better = mn["auprc"] > V1_AUPRC and lo > 0
    mn_sig_worse = mn["auprc"] < V1_AUPRC and hi < 0
    mn_tie = lo <= 0 <= hi

    if mn_sig_better and veo_better:
        v = "A"
        why = (
            "fused beats v1.5 on MovieNet (bootstrap CI excludes 0) and aligns "
            "better with Dispatch's buckets on Veo — fusion is the v2 architecture."
        )
    elif mn_tie and veo_better:
        v = "B"
        why = (
            "fused is statistically indistinguishable from v1.5 on MovieNet "
            "(bootstrap CI includes 0) but aligns better with Dispatch's buckets "
            "on Veo — fusion is an AI-gen-specific specialization, not a general "
            "upgrade."
        )
    elif mn_sig_worse or mn["auprc"] < V1_AUPRC:
        v = "C"
        why = (
            "fused does not beat v1.5 on MovieNet — fusion of two frozen "
            "backbones does not clear the frozen-feature ceiling. LoRA is the "
            "remaining v2 lever."
        )
    else:
        v = "B*"
        why = (
            "fused matches v1.5 on MovieNet but does NOT improve Dispatch "
            "alignment on Veo — fusion neither helps nor hurts; not worth the "
            "doubled feature."
        )
    return v, why


def _print_console(mn: dict, boot: dict, veo: dict) -> None:
    verdict, why = _verdict(mn, boot, veo)
    print("\n=== Fused logistic — MovieNet test ===")
    print(f"  AUPRC {mn['auprc']:.4f}  (v0 {V0_AUPRC:.3f}, v1.5 {V1_AUPRC:.3f})")
    print(f"  AUROC {mn['auroc']:.4f}   F1@val {mn['f1_at_val']:.4f}")
    print(
        f"  bootstrap AUPRC(fused)-AUPRC(v1.5) = {boot['diff']:+.4f} "
        f"CI [{boot['ci'][0]:+.4f}, {boot['ci'][1]:+.4f}]"
    )
    print("\n=== Fused logistic — Veo pilot ===")
    bm = veo["bucket_mean"]
    print(
        f"  bucket means: clean {bm['clean']:.3f}  drift {bm['drift']:.3f}  major {bm['major']:.3f}"
    )
    print(
        f"  Spearman vs Dispatch buckets: fused {veo['spearman']:+.3f}  (v1.5 {veo['v15_spearman']:+.3f})"
    )
    print(f"\n=== OUTCOME: {verdict} ===\n  {why}")


def _write_report(out_dir: Path, mn: dict, boot: dict, veo: dict) -> None:
    verdict, why = _verdict(mn, boot, veo)
    lo, hi = boot["ci"]
    md = [
        "# Fused DINOv2+CLIP Logistic — Results (Action 3)\n",
        "Logistic regression on the 4610-d fused boundary feature "
        "`[DINOv2 2305-d | CLIP 2305-d]`, trained with v0_logistic.py's recipe "
        "(StandardScaler + balanced LogisticRegression, C=1.0, lbfgs, "
        "max_iter=2000). The v2 architecture experiment. Produced by "
        "`scripts/train/fused_logistic.py`.\n",
        "## MovieNet test\n",
        "| model | AUPRC | AUROC | F1@val-thr |",
        "|---|--:|--:|--:|",
        f"| v0 logistic (DINOv2 2305-d) | {V0_AUPRC:.3f} | 0.849 | 0.388 |",
        f"| v1.5 MLP (DINOv2 2305-d) | {V1_AUPRC:.3f} | 0.859 | 0.424 |",
        f"| **fused logistic (4610-d)** | **{mn['auprc']:.3f}** | **{mn['auroc']:.3f}** "
        f"| **{mn['f1_at_val']:.3f}** |",
        "\n## Significance vs v1.5\n",
        f"Movie-level paired bootstrap ({boot['n']:,} test cuts, "
        f"{len(boot['ci'])}-sided 95% CI, 1000 resamples; protocol of "
        f"`v1_significance.md`):\n",
        f"- AUPRC(fused) − AUPRC(v1.5) = **{boot['diff']:+.4f}**, 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]",
        f"- {'CI excludes 0 — significant.' if (lo > 0 or hi < 0) else 'CI includes 0 — not significant.'}\n",
        "## Veo pilot\n",
        "Fused model scored on the 10 Veo continuous-action pairs (DINOv2 "
        "embedded fresh, CLIP from cache).\n",
        "| pair | bucket | fused score |",
        "|---|---|--:|",
    ]
    for _, r in veo["per_pair"].iterrows():
        md.append(f"| {r['pair_id']} | {r['bucket']} | {r['fused']:.3f} |")
    bm = veo["bucket_mean"]
    mono = "monotonic" if bm["clean"] < bm["drift"] < bm["major"] else "not monotonic"
    md += [
        "",
        f"Bucket means: clean **{bm['clean']:.3f}** / drift **{bm['drift']:.3f}** "
        f"/ major **{bm['major']:.3f}** ({mono}).",
        f"Spearman vs Dispatch buckets: fused **{veo['spearman']:+.3f}** vs v1.5 "
        f"**{veo['v15_spearman']:+.3f}**.\n",
        "## Outcome\n",
        f"**Outcome {verdict}.** {why}\n",
        _discussion(mn, boot, veo, verdict),
        "",
    ]
    (out_dir / "results.json").write_text(
        json.dumps(
            {
                "movienet": mn,
                "bootstrap_vs_v15": boot,
                "veo": {k: v for k, v in veo.items() if k != "per_pair"},
                "verdict": verdict,
            },
            indent=2,
            default=float,
        )
    )
    (REPO / "reports" / "fused_results.md").write_text("\n".join(md))


def _discussion(mn: dict, boot: dict, veo: dict, verdict: str) -> str:
    lo, hi = boot["ci"]
    delta = mn["auprc"] - V1_AUPRC
    veo_delta = veo["spearman"] - veo["v15_spearman"]
    return (
        "## Read\n\n"
        f"On MovieNet the fused logistic scores {mn['auprc']:.3f} AUPRC against "
        f"v1.5's {V1_AUPRC:.3f} ({delta:+.3f}); the movie-level bootstrap puts "
        f"the difference at {boot['diff']:+.4f} with 95% CI "
        f"[{lo:+.4f}, {hi:+.4f}]. Note the comparison is not perfectly "
        f"controlled — the fused model is a *logistic* head and v1.5 is a "
        f"2-layer MLP — so a fair architecture read also weighs fused logistic "
        f"against v0 logistic ({V0_AUPRC:.3f}), the same head on DINOv2 alone: "
        f"adding the CLIP half moves a logistic head from {V0_AUPRC:.3f} to "
        f"{mn['auprc']:.3f}. On the Veo pilot the fused model correlates "
        f"{veo['spearman']:+.3f} with Dispatch's buckets vs v1.5's "
        f"{veo['v15_spearman']:+.3f} ({veo_delta:+.3f}). "
        + (
            "Fusion clears the frozen-feature ceiling — pursue it as v2. "
            if verdict == "A"
            else (
                "Fusion does not move the MovieNet number but does help on "
                "AI-gen — a specialization worth keeping for AI-gen scoring while "
                "v1.5 stays the MovieNet model. "
                if verdict == "B"
                else (
                    "Fusion neither helps MovieNet nor AI-gen alignment — not worth "
                    "the doubled feature. "
                    if verdict == "B*"
                    else "Concatenating a second frozen backbone does not beat v1.5 — "
                    "consistent with the frozen-feature ceiling in v1_final.md. The "
                    "remaining v2 lever is LoRA fine-tuning, which unfreezes the "
                    "backbone rather than widening a frozen feature. "
                )
            )
        )
        + "The v2 architecture decision (fusion vs LoRA vs both) needs this "
        "result plus human input — n=10 on the Veo side remains the binding "
        "limitation on the AI-gen claim."
    )


if __name__ == "__main__":
    main()
