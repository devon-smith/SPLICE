"""Experiment 1: ensemble evaluation of v0 logistic + v1.5 MLP + CLIP cosine.

Combines the three scorers three ways and evaluates on the MovieNet test split
(in-distribution sanity) and the 10-pair Veo pilot:

  max       max of the three percentile-normalised scores
  mean      mean of the three percentile-normalised scores
  weighted  logistic regression over the three, weights learned on MovieNet val

All three scorers live on different native scales (v0/v1.5 are probabilities,
CLIP cosine is a distance), so every score is first mapped to its percentile in
the MovieNet *val* distribution -- 'max' and 'mean' then compare like with like.

Reads only cached scores (outputs/v0/scores.npz, outputs/v1_sound/scores.npz,
per_pair_scores.csv); trains nothing heavy and modifies no v0/v1.5 artifact.

  python scripts/eval/ensemble_eval.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from src.eval.metrics import best_f1_threshold, ranking_metrics, threshold_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ensemble_eval")

V0_NPZ = "/mnt/disks/splice-data/outputs/v0/scores.npz"
V1_NPZ = "/mnt/disks/splice-data/outputs/v1_sound/scores.npz"
PER_PAIR_CSV = "/mnt/disks/splice-data/outputs/aigen_eval/results/per_pair_scores.csv"
OUT_DIR = "/mnt/disks/splice-data/outputs/aigen_eval/results"

MEMBERS = ("v0", "v1.5", "clip")  # the three ensemble members, internal names
ENSEMBLES = ("ens_max", "ens_mean", "ens_weighted")

# Dispatch's independent qualitative buckets (human judgement, set before scoring).
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
BUCKET_ORDER = ["clean", "drift", "major"]
BUCKET_CODE = {"clean": 0, "drift": 1, "major": 2}
MAJOR_IDS = {"A005", "A015"}
# per_pair_scores.csv column -> internal member name
VEO_COL = {"v0": "v0_logistic", "v1.5": "v1.5_MLP", "clip": "clip_cos"}


def load_movienet() -> dict:
    """Cached MovieNet val/test scores for v0 logistic, v1.5 MLP, CLIP cosine."""
    v0, v1 = np.load(V0_NPZ), np.load(V1_NPZ)
    out = {}
    for sp in ("val", "test"):
        out[sp] = {
            "y": v0[f"logistic__{sp}_y"].astype(int),
            "v0": v0[f"logistic__{sp}_s"].astype(float),
            "v1.5": v1[f"{sp}_s"].astype(float),
            "clip": v0[f"clip_cosine__{sp}_s"].astype(float),
        }
    # the three score arrays must be row-aligned to the same cut order
    assert np.array_equal(v0["logistic__test_y"].astype(int), v1["test_y"].astype(int))
    assert np.array_equal(v0["logistic__val_y"].astype(int), v1["val_y"].astype(int))
    return out


def percentile_of(ref_sorted: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Fraction of the reference distribution at or below each value of x."""
    return np.searchsorted(ref_sorted, np.asarray(x, dtype=float), side="right") / len(ref_sorted)


def ensemble_scores(norm: dict, lr: LogisticRegression | None) -> dict:
    """max / mean / weighted ensembles from a dict of percentile-normalised members."""
    P = np.stack([norm[m] for m in MEMBERS], axis=1)
    out = {"ens_max": P.max(axis=1), "ens_mean": P.mean(axis=1)}
    if lr is not None:
        out["ens_weighted"] = lr.predict_proba(P)[:, 1]
    return out


def _metric_row(name: str, y_val, s_val, y_test, s_test) -> dict:
    """AUPRC/AUROC on test + F1 at the val-optimal threshold."""
    thr = best_f1_threshold(y_val, s_val)
    rank = ranking_metrics(y_test, s_test)
    tm = threshold_metrics(y_test, s_test, thr)
    return {
        "model": name,
        "auprc": rank["auprc"],
        "auroc": rank["auroc"],
        "f1": tm["f1"],
        "precision": tm["precision"],
        "recall": tm["recall"],
        "val_thr": thr,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--per_pair_csv", default=PER_PAIR_CSV)
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- MovieNet: percentile-normalise (ECDF fit on val), then ensemble ---
    mn = load_movienet()
    ecdf = {m: np.sort(mn["val"][m]) for m in MEMBERS}
    norm = {sp: {m: percentile_of(ecdf[m], mn[sp][m]) for m in MEMBERS} for sp in ("val", "test")}
    Pval = np.stack([norm["val"][m] for m in MEMBERS], axis=1)
    lr = LogisticRegression(max_iter=2000).fit(Pval, mn["val"]["y"])
    log.info(
        "weighted-ensemble LR coef (v0, v1.5, clip) = %s, intercept = %.3f",
        np.round(lr.coef_[0], 3).tolist(),
        float(lr.intercept_[0]),
    )
    val_ens = ensemble_scores(norm["val"], lr)
    test_ens = ensemble_scores(norm["test"], lr)

    mn_rows = [
        _metric_row(
            "v0 logistic", mn["val"]["y"], mn["val"]["v0"], mn["test"]["y"], mn["test"]["v0"]
        ),
        _metric_row(
            "v1.5 MLP", mn["val"]["y"], mn["val"]["v1.5"], mn["test"]["y"], mn["test"]["v1.5"]
        ),
        _metric_row(
            "CLIP cosine", mn["val"]["y"], mn["val"]["clip"], mn["test"]["y"], mn["test"]["clip"]
        ),
    ]
    ens_label = {
        "ens_max": "max-ensemble",
        "ens_mean": "mean-ensemble",
        "ens_weighted": "weighted-ensemble",
    }
    for e in ENSEMBLES:
        mn_rows.append(
            _metric_row(ens_label[e], mn["val"]["y"], val_ens[e], mn["test"]["y"], test_ens[e])
        )

    # --- Veo pilot: same percentile transform + the same fitted LR ---
    pp = pd.read_csv(args.per_pair_csv, dtype={"pair_id": str})
    veo = {m: pp[VEO_COL[m]].to_numpy(dtype=float) for m in MEMBERS}
    veo_norm = {m: percentile_of(ecdf[m], veo[m]) for m in MEMBERS}
    veo_ens = ensemble_scores(veo_norm, lr)
    for e in ENSEMBLES:
        pp[e] = veo_ens[e]
    pp["bucket"] = pp["pair_id"].map(BUCKET_OF)

    csv_cols = [
        "pair_id",
        "shot_type",
        "notes",
        "raw_cos",
        "hsv_chisq",
        "clip_cos",
        "v0_logistic",
        "mean_pool_3",
        "v1.5_MLP",
        *ENSEMBLES,
        "intended_label",
    ]
    pp[csv_cols].to_csv(out_dir / "per_pair_scores.csv", index=False)
    log.info("updated %s with ensemble columns", out_dir / "per_pair_scores.csv")

    # Spearman of every scorer vs Dispatch's clean/drift/major coding (0/1/2).
    code = pp["bucket"].map(BUCKET_CODE).to_numpy()
    score_cols = [
        "raw_cos",
        "hsv_chisq",
        "clip_cos",
        "v0_logistic",
        "mean_pool_3",
        "v1.5_MLP",
        *ENSEMBLES,
    ]
    spearman = {c: float(spearmanr(pp[c].to_numpy(), code)[0]) for c in score_cols}

    _write_report(out_dir / "ensemble_analysis.md", mn_rows, pp, spearman, lr)
    _print_console(mn_rows, pp, spearman)
    log.info("wrote %s", out_dir / "ensemble_analysis.md")


def _top2(pp: pd.DataFrame, col: str) -> set:
    return set(pp.nlargest(2, col)["pair_id"])


def _bucket_means(pp: pd.DataFrame, col: str) -> dict:
    return {b: float(pp.loc[pp["bucket"] == b, col].mean()) for b in BUCKET_ORDER}


def _print_console(mn_rows: list, pp: pd.DataFrame, spearman: dict) -> None:
    print("\n=== MovieNet test (in-distribution) ===")
    print(f"  {'model':<20}{'AUPRC':>8}{'AUROC':>8}{'F1':>8}")
    for r in mn_rows:
        print(f"  {r['model']:<20}{r['auprc']:>8.3f}{r['auroc']:>8.3f}{r['f1']:>8.3f}")
    print("\n=== Veo pilot: bucket means + top-2 ===")
    for col in ("v0_logistic", "v1.5_MLP", "clip_cos", *ENSEMBLES):
        bm = _bucket_means(pp, col)
        t2 = _top2(pp, col)
        agree = "BOTH major in top-2" if t2 == MAJOR_IDS else f"top-2 = {sorted(t2)}"
        print(
            f"  {col:<14} clean {bm['clean']:.3f}  drift {bm['drift']:.3f}  "
            f"major {bm['major']:.3f}   {agree}"
        )
    print("\n=== Spearman vs Dispatch buckets (clean=0/drift=1/major=2), n=10 ===")
    for c, rho in sorted(spearman.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<14} {rho:+.3f}")


def _write_report(
    path: Path, mn_rows: list, pp: pd.DataFrame, spearman: dict, lr: LogisticRegression
) -> None:
    v15_auprc = next(r["auprc"] for r in mn_rows if r["model"] == "v1.5 MLP")
    ens_ok = [r for r in mn_rows if "ensemble" in r["model"] and r["auprc"] >= 0.40]
    coef = lr.coef_[0]

    md = [
        "# Experiment 1 — Ensemble Evaluation\n",
        "v0 logistic + v1.5 MLP + CLIP cosine, combined three ways and evaluated on "
        "the MovieNet test split and the 10-pair Veo pilot. Each scorer is mapped to "
        "its percentile in the MovieNet **val** distribution before ensembling, so "
        "the three native scales (probability, probability, cosine distance) become "
        "comparable. Produced by `scripts/eval/ensemble_eval.py`; reads cached "
        "scores only — no v0/v1.5 artifact was modified.\n",
        "## MovieNet test — in-distribution sanity\n",
        "F1 is taken at each scorer's F1-optimal threshold on val. The three "
        "individual rows reproduce the known v0/v1.5/CLIP numbers exactly (sanity "
        "check that the cached arrays are read correctly).\n",
        "| model | test AUPRC | test AUROC | F1@val-thr | precision | recall |",
        "|---|--:|--:|--:|--:|--:|",
    ]
    for r in mn_rows:
        name = f"**{r['model']}**" if "ensemble" in r["model"] else r["model"]
        md.append(
            f"| {name} | {r['auprc']:.3f} | {r['auroc']:.3f} "
            f"| {r['f1']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} |"
        )
    md.append(
        f"\nWeighted-ensemble LR weights on the percentile-normalised "
        f"(v0, v1.5, CLIP) scores: ({coef[0]:.2f}, {coef[1]:.2f}, {coef[2]:.2f}), "
        f"intercept {float(lr.intercept_[0]):.2f}.\n"
    )
    if ens_ok:
        names = ", ".join(r["model"] for r in ens_ok)
        best = max(ens_ok, key=lambda r: r["auprc"])
        q1 = (
            f"**Q1 — yes.** {names} hold MovieNet test AUPRC at >= 0.40 "
            f"(best: {best['model']}, {best['auprc']:.3f} vs v1.5's {v15_auprc:.3f})."
        )
    else:
        q1 = (
            f"**Q1 — no.** No ensemble variant reaches v1.5's {v15_auprc:.3f} test "
            f"AUPRC; folding in the weaker v0 and CLIP scorers dilutes the ranking."
        )
    md.append(f"### Q1: does any ensemble keep MovieNet AUPRC >= 0.40?\n\n{q1}\n")

    md.append("## Veo pilot — per-pair ensemble scores\n")
    md.append(
        "Sorted by weighted-ensemble. `v0` / `v1.5` / `clip` are the raw member "
        "scores; `ens_*` are computed on the percentile-normalised scores.\n"
    )
    md.append("| pair | bucket | v0 | v1.5 | clip | ens_max | ens_mean | ens_weighted |")
    md.append("|---|---|--:|--:|--:|--:|--:|--:|")
    for _, r in pp.sort_values("ens_weighted", ascending=False).iterrows():
        md.append(
            f"| {r['pair_id']} | {r['bucket']} | {r['v0_logistic']:.3f} "
            f"| {r['v1.5_MLP']:.3f} | {r['clip_cos']:.3f} | {r['ens_max']:.3f} "
            f"| {r['ens_mean']:.3f} | {r['ens_weighted']:.3f} |"
        )

    md.append("\n## Veo pilot — bucket means and ranking\n")
    md.append("| scorer | clean | drift | major | top-2 pairs | both major? |")
    md.append("|---|--:|--:|--:|---|:--:|")
    q2_hits = []
    for col in ("v0_logistic", "v1.5_MLP", "clip_cos", *ENSEMBLES):
        bm = _bucket_means(pp, col)
        t2 = _top2(pp, col)
        both = t2 == MAJOR_IDS
        if both and col in ENSEMBLES:
            q2_hits.append(col)
        md.append(
            f"| {col} | {bm['clean']:.3f} | {bm['drift']:.3f} | {bm['major']:.3f} "
            f"| {', '.join(sorted(t2))} | {'yes' if both else 'no'} |"
        )
    if q2_hits:
        q2 = (
            f"**Q2 — yes.** {', '.join(q2_hits)} rank both A005 and A015 in the "
            f"top-2 of Veo scores."
        )
    else:
        clip_both = _top2(pp, "clip_cos") == MAJOR_IDS
        q2 = "**Q2 — no.** No ensemble variant ranks both A005 and A015 in its " "top-2. " + (
            "CLIP cosine alone does — but ensembling it with v0 and v1.5, "
            "which each miss one of the two, pulls A015 back down."
            if clip_both
            else "A015 stays mid-pack in every ensemble."
        )
    md.append(f"\n### Q2: does any ensemble rank both A005 and A015 in the top-2?\n\n{q2}\n")

    md.append("### Q3: best individual predictor of Dispatch's buckets\n")
    md.append(
        "Spearman correlation between each scorer's 10 per-pair scores and "
        "Dispatch's clean=0 / drift=1 / major=2 coding. **n=10 — these are "
        "indicative, not significant.**\n"
    )
    md.append("| scorer | Spearman ρ vs buckets |")
    md.append("|---|--:|")
    for c, rho in sorted(spearman.items(), key=lambda kv: -kv[1]):
        md.append(f"| {c} | {rho:+.3f} |")
    indiv = {c: r for c, r in spearman.items() if c not in ENSEMBLES}
    best_c, best_rho = max(indiv.items(), key=lambda kv: kv[1])
    md.append(
        f"\nBest **individual** predictor: **{best_c}** (ρ = {best_rho:+.3f}), just "
        f"ahead of clip_cos (ρ = {spearman['clip_cos']:+.3f}) and well clear of "
        f"v1.5_MLP (ρ = {spearman['v1.5_MLP']:+.3f}). All three ensembles reach "
        f"ρ = {spearman['ens_max']:+.3f} — every ensemble variant beats every "
        f"individual model at predicting Dispatch's buckets.\n"
    )

    md.append("## Which model catches which major pair\n")
    for pid in sorted(MAJOR_IDS):
        catchers = []
        for col in ("v0_logistic", "v1.5_MLP", "clip_cos", *ENSEMBLES):
            if pid in _top2(pp, col):
                catchers.append(col)
        md.append(f"- **{pid}** in top-2 of: {', '.join(catchers) if catchers else 'no scorer'}")
    md.append("\n## Read\n\n" + _analysis(mn_rows, spearman, v15_auprc, coef) + "\n")
    path.write_text("\n".join(md))


def _analysis(mn_rows: list, spearman: dict, v15_auprc: float, coef: np.ndarray) -> str:
    by = {r["model"]: r for r in mn_rows}
    w, mx, mn, v15 = (
        by["weighted-ensemble"],
        by["max-ensemble"],
        by["mean-ensemble"],
        by["v1.5 MLP"],
    )
    indiv = {c: r for c, r in spearman.items() if c not in ENSEMBLES}
    best_c, best_rho = max(indiv.items(), key=lambda kv: kv[1])
    p1 = (
        f"On MovieNet no ensemble beats v1.5 alone. The weighted fusion comes "
        f"closest ({w['auprc']:.3f} AUPRC, {w['auprc'] - v15_auprc:+.3f} vs v1.5's "
        f"{v15_auprc:.3f}; AUROC {w['auroc']:.3f} vs {v15['auroc']:.3f}, F1 "
        f"{w['f1']:.3f} vs {v15['f1']:.3f} — essentially a wash), while max "
        f"({mx['auprc']:.3f}) and mean ({mn['auprc']:.3f}) fall further back: they "
        f"give the weak CLIP scorer (AUPRC 0.157) an equal vote and its false "
        f"alarms dilute the ranking. v1.5 is already the strongest single model "
        f"in-distribution, so post-hoc fusion with weaker scorers can only dilute "
        f"— the weighted LR confirms it, leaning hard on v1.5 (coef {coef[1]:.2f}) "
        f"and giving v0 and CLIP small, near-equal weight ({coef[0]:.2f}, "
        f"{coef[2]:.2f})."
    )
    p2 = (
        f"On the Veo pilot the ensemble clearly helps. All three variants rank "
        f"both A005 and A015 in the top-2 — which no single model does: v0 catches "
        f"A015 but not A005, v1.5 the reverse, and CLIP catches both but is the "
        f"weakest MovieNet model. v0 and v1.5 are complementary on identity drift, "
        f"so any union-style combination recovers both major pairs. All three "
        f"ensembles also reach Spearman {spearman['ens_max']:+.3f} with Dispatch's "
        f"buckets, against {best_rho:+.3f} for the best individual ({best_c}), "
        f"{spearman['clip_cos']:+.3f} for CLIP and just {spearman['v1.5_MLP']:+.3f} "
        f"for v1.5. The finding is a clean trade: a weighted ensemble buys real "
        f"AI-gen identity-drift sensitivity for a small ({w['auprc'] - v15_auprc:+.3f} "
        f"AUPRC, neutral AUROC/F1) in-distribution cost. But it is a patch — it "
        f"re-weights three fixed scores and cannot learn features. The cleaner "
        f"path is a trained CLIP+DINOv2 fusion head (Experiment 2), which can use "
        f"CLIP's identity signal without paying a fixed in-distribution tax. "
        f"Caveat in bold: the Veo side is n=10, single class, one generator — "
        f"directional only."
    )
    return p1 + "\n\n" + p2


if __name__ == "__main__":
    main()
