# AI-USE: This file was AI-assisted with Claude (claude-sonnet-4-6) via Claude Code.
# Prompt summary: "write Phase-3 P5: comprehensive calibration analysis of v1.5
# including multi-threshold sweep, reliability diagram, Brier score/ECE, Platt
# scaling, and tau95 robustness across splits."

"""Phase-3 P5: comprehensive calibration analysis of v1.5.

  1. multi-threshold sweep (tau50/75/90/95/99 from within-shot variation)
  2. reliability diagram (predicted probability vs observed frequency)
  3. Brier score and Expected Calibration Error (ECE)
  4. Platt scaling fitted on val predictions, evaluated on test
  5. within-shot tau95 robustness across train / val / test movies

Diagnostics only -- v1.5 is the seed-2 sound MLP; predictions are re-scored only
for the within-shot pairs (the test-set numbers come from the cached scores).

Writes reports/v1_calibration.md, reports/figures/v1_reliability.png,
reports/figures/v1_threshold_sweep.png.
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[2] / "scripts" / "train"))
from calibrate_threshold import within_shot_embeddings  # noqa: E402
from src.data.pairs import build_pair_features_batch, load_embeddings  # noqa: E402
from src.eval.metrics import threshold_metrics  # noqa: E402
from v1_mlp import MLPHead  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("calibration_comprehensive")

REPO = HERE.parents[2]
CUT_INDEX = "/mnt/disks/splice-data/outputs/cut_index/cuts.parquet"
EMB = "/mnt/disks/splice-data/embeddings/dinov2_base"
V1 = Path("/mnt/disks/splice-data/outputs/v1_sound")
USE_CASE = {
    50: "very aggressive flag (most cuts reviewed)",
    75: "aggressive review",
    90: "broad review",
    95: "high-recall review (handoff default)",
    99: "balanced -- reproduces the F1-optimal point",
}


def v15_within_shot(e_a, e_b, model, scaler, device, chunk=100_000) -> np.ndarray:
    """Score within-shot pairs with the v1.5 MLP (chunked)."""
    out = []
    for i in range(0, len(e_a), chunk):
        feats = build_pair_features_batch(e_a[i : i + chunk], e_b[i : i + chunk])
        feats = scaler.transform(feats).astype(np.float32)
        with torch.inference_mode():
            s = torch.sigmoid(model(torch.from_numpy(feats).to(device))).cpu().numpy()
        out.append(s)
    return np.concatenate(out)


def reliability(y: np.ndarray, p: np.ndarray, n_bins: int = 10):
    """Decile reliability bins and Expected Calibration Error."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    for i in range(n_bins):
        hi = p <= edges[i + 1] if i == n_bins - 1 else p < edges[i + 1]
        mask = (p >= edges[i]) & hi
        if mask.sum() == 0:
            continue
        mean_pred, frac_pos = float(p[mask].mean()), float(y[mask].mean())
        rows.append((mean_pred, frac_pos, int(mask.sum())))
        ece += abs(mean_pred - frac_pos) * mask.sum() / len(y)
    return rows, float(ece)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quantile", type=float, default=0.95)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fig_dir = REPO / "reports" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load((REPO / "configs" / "v1_sound.yaml").read_text())
    model = MLPHead(in_dim=2305, dropout=cfg["head"]["dropout"]).to(device)
    model.load_state_dict(
        torch.load(V1 / "v1_sound_seed2.pt", map_location=device, weights_only=True)
    )
    model.eval()
    scaler = joblib.load(V1 / "scaler.joblib")

    npz = np.load(V1 / "scores.npz")
    test_y, test_s = npz["test_y"].astype(int), npz["test_s"]
    val_y, val_s = npz["val_y"].astype(int), npz["val_s"]

    emb, key2row = load_embeddings(EMB)
    within = {}
    for split in ("train", "val", "test"):
        e_a, e_b = within_shot_embeddings(CUT_INDEX, split, emb, key2row)
        within[split] = v15_within_shot(e_a, e_b, model, scaler, device)
        log.info("%s within-shot pairs scored: %d", split, len(within[split]))

    md = ["# v1.5 Comprehensive Calibration (Phase 3, P5)\n"]
    md.append(
        "v1.5 = seed-2 sound MLP. Within-shot scores are the v1.5 model applied to "
        "img0/1/2 pairs of single shots (genuinely continuous); test metrics use the "
        "cached test predictions.\n"
    )

    # ---- 1. multi-threshold sweep ------------------------------------------
    md.append("## 1. Multi-threshold sweep\n")
    md.append("Thresholds are percentiles of the v1.5 within-shot (val) score distribution.\n")
    md.append("| τ | threshold | test precision | test recall | test F1 | implied use case |")
    md.append("|---|---|---|---|---|---|")
    sweep = []
    for q in (50, 75, 90, 95, 99):
        tau = float(np.quantile(within["val"], q / 100))
        m = threshold_metrics(test_y, test_s, tau)
        sweep.append((q, tau, m["precision"], m["recall"], m["f1"]))
        md.append(
            f"| τ{q} | {tau:.3f} | {m['precision']:.3f} | {m['recall']:.3f} | "
            f"{m['f1']:.3f} | {USE_CASE[q]} |"
        )
    md.append("")

    # ---- 2 & 3. reliability, Brier, ECE -----------------------------------
    rows, ece = reliability(test_y, test_s)
    bs = brier(test_y, test_s)
    md.append("## 2-3. Reliability, Brier score, ECE\n")
    md.append(
        f"- Brier score: **{bs:.4f}**  (lower is better; base-rate-only ≈ "
        f"{brier(test_y, np.full_like(test_s, test_y.mean())):.4f})"
    )
    md.append(f"- Expected Calibration Error (10 bins): **{ece:.4f}**")
    over = sum(mp > fp for mp, fp, _ in rows)
    md.append(
        f"- of {len(rows)} populated deciles, {over} sit above the diagonal "
        f"(predicted > observed = over-confident)."
    )
    md.append("See `reports/figures/v1_reliability.png`.\n")

    # ---- 4. Platt scaling -------------------------------------------------
    platt = LogisticRegression(C=1e6, solver="lbfgs")
    platt.fit(val_s.reshape(-1, 1), val_y)
    test_s_platt = platt.predict_proba(test_s.reshape(-1, 1))[:, 1]
    rows_p, ece_p = reliability(test_y, test_s_platt)
    bs_p = brier(test_y, test_s_platt)
    md.append("## 4. Platt scaling (fitted on val, applied to test)\n")
    md.append("| | Brier | ECE |")
    md.append("|---|---|---|")
    md.append(f"| raw v1.5 | {bs:.4f} | {ece:.4f} |")
    md.append(f"| Platt-scaled | {bs_p:.4f} | {ece_p:.4f} |")
    verdict = (
        "improves"
        if (bs_p < bs and ece_p < ece)
        else ("does not improve" if (bs_p >= bs and ece_p >= ece) else "is mixed on")
    )
    md.append(
        f"\nPost-hoc Platt scaling {verdict} calibration; AUROC/AUPRC are unchanged "
        "(monotonic transform).\n"
    )

    # ---- 5. within-shot tau95 robustness ---------------------------------
    md.append("## 5. Within-shot τ95 robustness across splits\n")
    md.append("| split computed on | τ95 | within-shot pairs |")
    md.append("|---|---|---|")
    taus = {}
    for split in ("train", "val", "test"):
        taus[split] = float(np.quantile(within[split], args.quantile))
        md.append(f"| {split} | {taus[split]:.4f} | {len(within[split]):,} |")
    spread = max(taus.values()) - min(taus.values())
    stable = spread <= 0.05
    md.append(
        f"\nτ95 spread across splits = {spread:.4f} — "
        + (
            "**stable** (< 0.05): the within-shot calibration is not data-split " "dependent."
            if stable
            else "**> 0.05**: the calibration is somewhat "
            "split-dependent and should be reported with that caveat."
        )
        + "\n"
    )

    md.append("## Takeaways\n")
    md.append(
        f"- τ99 ≈ the F1-optimal point (F1 {sweep[-1][4]:.3f}); τ95 trades precision "
        "for recall as a review flag. The within-shot quantile is a precision/recall dial."
    )
    md.append(
        f"- v1.5 is {'well calibrated' if ece < 0.05 else 'mildly miscalibrated'} "
        f"(ECE {ece:.3f}); Platt scaling {verdict} it."
    )
    md.append(
        f"- τ95 is {'stable' if stable else 'split-dependent'} across data splits "
        f"(spread {spread:.4f}).\n"
    )
    (REPO / "reports" / "v1_calibration.md").write_text("\n".join(md))

    # ---- figures ----------------------------------------------------------
    fig, ax = plt.subplots(figsize=(5, 5))
    mp = [r[0] for r in rows]
    fp = [r[1] for r in rows]
    mp_p = [r[0] for r in rows_p]
    fp_p = [r[1] for r in rows_p]
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, label="perfect calibration")
    ax.plot(mp, fp, "o-", label=f"raw v1.5 (ECE {ece:.3f})")
    ax.plot(mp_p, fp_p, "s-", label=f"Platt-scaled (ECE {ece_p:.3f})")
    ax.set(
        xlabel="mean predicted probability",
        ylabel="observed fraction positive",
        title="v1.5 reliability diagram (test)",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "v1_reliability.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    qs = [s[0] for s in sweep]
    ax.plot(qs, [s[2] for s in sweep], "o-", label="precision")
    ax.plot(qs, [s[3] for s in sweep], "s-", label="recall")
    ax.plot(qs, [s[4] for s in sweep], "^-", label="F1")
    ax.set(
        xlabel="within-shot percentile τ",
        ylabel="test metric",
        title="v1.5 operating points by within-shot threshold",
    )
    ax.set_xticks(qs)
    ax.legend()
    ax.grid(ls="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "v1_threshold_sweep.png", dpi=120)
    plt.close(fig)

    print("wrote v1_calibration.md + v1_reliability.png + v1_threshold_sweep.png")


if __name__ == "__main__":
    main()
