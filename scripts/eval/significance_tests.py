# AI-USE: This file was AI-assisted with Claude (claude-sonnet-4-6) via Claude Code.
# Prompt summary: "write Phase-3 P2: significance tests for five model comparisons
# using DeLong's AUROC test, movie-level bootstrap CI for AUPRC, and paired
# permutation test for F1, reading cached predictions only."

"""Phase-3 P2: significance tests for the five key model comparisons.

For each comparison, on the shared test split:
  - DeLong's test for the AUROC difference (correlated ROC AUCs, same eval set)
  - a movie-level bootstrap 95% CI for the AUPRC difference (1000 resamples)
  - a paired permutation test for the F1 difference at each model's val-optimal
    threshold (10000 permutations)

All six models score the same 105,095 test cuts in the same order, so the tests
are properly paired. Reads cached predictions only -- no retraining.

Writes reports/v1_significance.md.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.eval.metrics import best_f1_threshold  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("significance_tests")

REPO = Path(__file__).resolve().parents[2]
V0 = Path("/mnt/disks/splice-data/outputs/v0")
V0_MP = Path("/mnt/disks/splice-data/outputs/v0_mean_pool")
V1 = Path("/mnt/disks/splice-data/outputs/v1_sound")
PAIRS = Path("/mnt/disks/splice-data/pairs/dino_v0_boundary")


# ---------------------------------------------------------------------------
# DeLong's test (fast algorithm, Sun & Xu 2014) for two correlated AUROCs
# ---------------------------------------------------------------------------
def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranked = x[order]
    n = len(x)
    out = np.zeros(n)
    i = 0
    while i < n:
        j = i
        while j < n and ranked[j] == ranked[i]:
            j += 1
        out[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    res = np.empty(n)
    res[order] = out
    return res


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict:
    """AUROC of A and B, the difference, its DeLong variance, z and two-sided p."""
    y = np.asarray(y_true).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives (label 1) first
    m = int((y == 1).sum())
    n = len(y) - m
    preds = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])[:, order]
    pos, neg = preds[:, :m], preds[:, m:]
    tx = np.vstack([_midrank(pos[r]) for r in range(2)])
    ty = np.vstack([_midrank(neg[r]) for r in range(2)])
    tz = np.vstack([_midrank(preds[r]) for r in range(2)])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    var_diff = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var_diff) if var_diff > 0 else 0.0
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "diff": float(aucs[0] - aucs[1]),
        "ci": (
            float(aucs[0] - aucs[1] - 1.96 * np.sqrt(max(var_diff, 0))),
            float(aucs[0] - aucs[1] + 1.96 * np.sqrt(max(var_diff, 0))),
        ),
        "p": float(2 * norm.sf(abs(z))),
    }


# ---------------------------------------------------------------------------
def bootstrap_auprc_diff(
    y: np.ndarray,
    sa: np.ndarray,
    sb: np.ndarray,
    movie_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict:
    """Movie-level bootstrap CI for AUPRC(A) - AUPRC(B)."""
    rng = np.random.default_rng(seed)
    by_movie = {mid: np.where(movie_ids == mid)[0] for mid in np.unique(movie_ids)}
    movies = list(by_movie)
    obs = average_precision_score(y, sa) - average_precision_score(y, sb)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(movies, size=len(movies), replace=True)
        idx = np.concatenate([by_movie[m] for m in pick])
        diffs[b] = average_precision_score(y[idx], sa[idx]) - average_precision_score(
            y[idx], sb[idx]
        )
    return {
        "diff": float(obs),
        "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
    }


def _f1(y: np.ndarray, pred: np.ndarray) -> float:
    tp = float(np.sum(pred & (y == 1)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y == 1)))
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else 0.0


def bootstrap_f1_diff(
    y: np.ndarray,
    dec_a: np.ndarray,
    dec_b: np.ndarray,
    movie_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 1,
) -> dict:
    """Movie-level bootstrap CI for F1(A) - F1(B) at fixed decision thresholds."""
    rng = np.random.default_rng(seed)
    by_movie = {mid: np.where(movie_ids == mid)[0] for mid in np.unique(movie_ids)}
    movies = list(by_movie)
    obs = _f1(y, dec_a) - _f1(y, dec_b)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(movies, size=len(movies), replace=True)
        idx = np.concatenate([by_movie[m] for m in pick])
        diffs[b] = _f1(y[idx], dec_a[idx]) - _f1(y[idx], dec_b[idx])
    return {
        "diff": float(obs),
        "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
    }


def permutation_f1_test(
    y: np.ndarray,
    dec_a: np.ndarray,
    dec_b: np.ndarray,
    n_perm: int = 10000,
    seed: int = 0,
) -> dict:
    """Paired permutation test for F1(A) - F1(B); per-cut decisions swapped at random."""
    rng = np.random.default_rng(seed)
    obs = _f1(y, dec_a) - _f1(y, dec_b)
    perm = np.empty(n_perm)
    n = len(y)
    for i in range(n_perm):
        swap = rng.random(n) < 0.5
        pa = np.where(swap, dec_b, dec_a)
        pb = np.where(swap, dec_a, dec_b)
        perm[i] = _f1(y, pa) - _f1(y, pb)
    p = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    return {
        "diff": float(obs),
        "ci": (float(np.percentile(perm, 2.5)), float(np.percentile(perm, 97.5))),
        "p": p,
    }


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=10000)
    args = ap.parse_args()

    v0 = np.load(V0 / "scores.npz")
    mp = np.load(V0_MP / "scores.npz")
    v1 = np.load(V1 / "scores.npz")
    meta = pd.read_parquet(PAIRS / "meta.parquet")
    movie_ids = meta.loc[meta["split"] == "test", "movie_id"].to_numpy()

    y = v0["logistic__test_y"].astype(int)
    for name, arr in [("mean_pool", mp["logistic__test_y"]), ("v1.5", v1["test_y"])]:
        if not np.array_equal(arr.astype(int), y):
            raise SystemExit(f"test labels for {name} do not match v0 -- alignment broken")

    res = {m["model"]: m for m in json.loads((V0 / "results.json").read_text())}
    mp_thr = {m["model"]: m for m in json.loads((V0_MP / "results.json").read_text())}
    scores = {
        "v1.5": (v1["test_s"], best_f1_threshold(v1["val_y"], v1["val_s"])),
        "v0 logistic": (v0["logistic__test_s"], res["logistic"]["val_thr"]),
        "raw DINOv2 cosine": (v0["raw_dino_cosine__test_s"], res["raw_dino_cosine"]["val_thr"]),
        "HSV chi-square": (v0["hsv_chisq__test_s"], res["hsv_chisq"]["val_thr"]),
        "CLIP cosine": (v0["clip_cosine__test_s"], res["clip_cosine"]["val_thr"]),
        "mean-pool-3": (mp["logistic__test_s"], mp_thr["logistic"]["val_thr"]),
    }

    comparisons = [
        ("v1.5 - v0 logistic", "v1.5", "v0 logistic"),
        ("v0 logistic - raw DINOv2 cosine", "v0 logistic", "raw DINOv2 cosine"),
        ("mean-pool-3 - v0 logistic", "mean-pool-3", "v0 logistic"),
        ("v0 logistic - HSV chi-square", "v0 logistic", "HSV chi-square"),
        ("raw DINOv2 cosine - CLIP cosine", "raw DINOv2 cosine", "CLIP cosine"),
    ]

    rows = []
    for label, a, b in comparisons:
        sa, ta = scores[a]
        sb, tb = scores[b]
        ok = ~(np.isnan(sa) | np.isnan(sb))  # HSV/CLIP may have missing-frame NaNs
        yy, saa, sbb, mids = y[ok], sa[ok], sb[ok], movie_ids[ok]
        n_drop = int((~ok).sum())
        dec_a, dec_b = saa >= ta, sbb >= tb
        d = delong_test(yy, saa, sbb)
        bp = bootstrap_auprc_diff(yy, saa, sbb, mids, n_boot=args.n_boot)
        bf = bootstrap_f1_diff(yy, dec_a, dec_b, mids, n_boot=args.n_boot)
        pf = permutation_f1_test(yy, dec_a, dec_b, n_perm=args.n_perm)
        rows.append(
            {
                "comparison": label,
                "n": len(yy),
                "n_drop": n_drop,
                **{f"auroc_{k}": v for k, v in d.items()},
                "auprc_diff": bp["diff"],
                "auprc_ci": bp["ci"],
                "f1_diff": bf["diff"],
                "f1_ci": bf["ci"],
                "f1_p": pf["p"],
            }
        )
        log.info(
            "%s: dAUROC %.4f (p=%.2e)  dAUPRC %.4f  dF1 %.4f (perm p=%.4f)",
            label,
            d["diff"],
            d["p"],
            bp["diff"],
            pf["diff"],
            pf["p"],
        )

    _write_report(rows, args)
    print("\nwrote reports/v1_significance.md")


def _sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _write_report(rows: list[dict], args) -> None:
    md = ["# v1 Significance Tests (Phase 3, P2)\n"]
    md.append(
        "Five model comparisons on the shared test split. DeLong's test for the "
        f"AUROC difference; movie-level bootstrap ({args.n_boot} resamples) for the "
        f"AUPRC-difference CI; paired permutation test ({args.n_perm:,} permutations) "
        "for the F1 difference. A 95% CI excluding 0, or p < 0.05, is significant. "
        "v1.5 = the seed-2 sound MLP.\n"
    )
    md.append(
        "| Comparison | ΔAUROC (95% CI) | DeLong p | ΔAUPRC (95% CI) | ΔF1 (95% CI) | F1 perm p |"
    )
    md.append("|---|---|---|---|---|---|")
    for r in rows:
        lo, hi = r["auroc_ci"]
        alo, ahi = r["auprc_ci"]
        flo, fhi = r["f1_ci"]
        md.append(
            f"| {r['comparison']} | {r['auroc_diff']:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
            f"{r['auroc_p']:.2e} {_sig(r['auroc_p'])} | "
            f"{r['auprc_diff']:+.4f} [{alo:+.4f}, {ahi:+.4f}] | "
            f"{r['f1_diff']:+.4f} [{flo:+.4f}, {fhi:+.4f}] | "
            f"{r['f1_p']:.4f} {_sig(r['f1_p'])} |"
        )
    md.append("\n*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant.\n")

    md.append("## Interpretation\n")
    for r in rows:
        auprc_sig = (r["auprc_ci"][0] > 0) or (r["auprc_ci"][1] < 0)
        verdict = (
            "significant"
            if (r["auroc_p"] < 0.05 and auprc_sig)
            else (
                "significant on AUROC but not AUPRC"
                if r["auroc_p"] < 0.05
                else "not statistically significant"
            )
        )
        md.append(
            f"- **{r['comparison']}** — {verdict}: ΔAUROC {r['auroc_diff']:+.4f} "
            f"(DeLong p={r['auroc_p']:.1e}), ΔAUPRC {r['auprc_diff']:+.4f} "
            f"[{r['auprc_ci'][0]:+.4f}, {r['auprc_ci'][1]:+.4f}]."
        )
    md.append("")
    (REPO / "reports" / "v1_significance.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
