# AI-USE: This file was generated with Claude (claude-sonnet-4-6) via Claude Code.
# Prompt: "implement movie-level bootstrap CI for the AUPRC gap between v2 LoRA and
# v1.5, and a paired permutation test for F1, using the fixed final v2 3-seed
# mean score file and writing a markdown report."

"""Significance tests for the v2 LoRA vs v1.5 comparison.

Same protocol as the fusion comparison (fused_logistic.py):
  - movie-level bootstrap 95% CI for the AUPRC gap (1000 resamples)
  - paired permutation test for the F1 gap at each model's val-optimal threshold

The v2 input is intentionally fixed to the final 3-seed mean score file from
`reports/v2_final.md`; this script does not search sweep directories or select
models.

Writes reports/v2_significance.md.

Example:
  python scripts/eval/v2_significance.py
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE.parent))
from significance_tests import bootstrap_auprc_diff, bootstrap_f1_diff, permutation_f1_test  # noqa: E402
from src.eval.metrics import best_f1_threshold, ranking_metrics  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v2_significance")

V1_SCORES = Path("/mnt/disks/splice-data/outputs/v1_sound/scores.npz")
V2_SCORES = Path("/mnt/disks/splice-data/outputs/v2_lora_extended/v2_3seed_mean/scores.npz")
V2_RUN_NAME = "v2_lora_extended/v2_3seed_mean"
PAIRS_META = Path("/mnt/disks/splice-data/pairs/dino_v0_boundary/meta.parquet")


def _load_v2_scores(scores_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load (test_s, test_y, val_s, val_thr) from the fixed final v2 score file."""
    npz = np.load(scores_path)
    test_s, test_y = npz["test_s"], npz["test_y"].astype(int)
    val_s, val_y = npz["val_s"], npz["val_y"].astype(int)
    val_thr = best_f1_threshold(val_y, val_s)
    return test_s, test_y, val_s, val_thr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v1_scores", type=Path, default=V1_SCORES)
    ap.add_argument("--pairs_meta", type=Path, default=PAIRS_META)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=10000)
    args = ap.parse_args()

    v2_test_s, v2_test_y, v2_val_s, v2_val_thr = _load_v2_scores(V2_SCORES)

    v1 = np.load(args.v1_scores)
    v1_test_s = v1["test_s"]
    v1_test_y = v1["test_y"].astype(int)
    v1_val_thr = best_f1_threshold(v1["val_y"], v1["val_s"])

    if not np.array_equal(v2_test_y, v1_test_y):
        raise SystemExit("test label mismatch between v2 and v1.5 — alignment broken")

    meta = pd.read_parquet(args.pairs_meta)
    movie_ids = meta.loc[meta["split"] == "test", "movie_id"].to_numpy()
    if len(movie_ids) != len(v2_test_y):
        raise SystemExit(
            f"movie_ids length {len(movie_ids)} != scores length {len(v2_test_y)}"
        )

    y = v2_test_y
    dec_v2 = v2_test_s >= v2_val_thr
    dec_v1 = v1_test_s >= v1_val_thr

    log.info("running bootstrap AUPRC diff (%d resamples)...", args.n_boot)
    bp = bootstrap_auprc_diff(y, v2_test_s, v1_test_s, movie_ids, n_boot=args.n_boot)

    log.info("running bootstrap F1 diff (%d resamples)...", args.n_boot)
    bf = bootstrap_f1_diff(y, dec_v2, dec_v1, movie_ids, n_boot=args.n_boot)

    log.info("running permutation F1 test (%d permutations)...", args.n_perm)
    pf = permutation_f1_test(y, dec_v2, dec_v1, n_perm=args.n_perm)

    v2_rank = ranking_metrics(y, v2_test_s)
    v1_rank = ranking_metrics(y, v1_test_s)

    log.info(
        "v2 AUPRC %.4f vs v1.5 AUPRC %.4f  Δ %.4f [%.4f, %.4f]  F1 perm p=%.4f",
        v2_rank["auprc"], v1_rank["auprc"],
        bp["diff"], bp["ci"][0], bp["ci"][1], pf["p"],
    )

    _write_report(V2_RUN_NAME, v2_rank, v1_rank, bp, bf, pf, args)
    print(f"\nwrote reports/v2_significance.md")
    print(f"  ΔAUPRC {bp['diff']:+.4f}  95% CI [{bp['ci'][0]:+.4f}, {bp['ci'][1]:+.4f}]")
    print(f"  ΔF1    {pf['diff']:+.4f}  perm p={pf['p']:.4f}")


def _sig(p: float) -> str:
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


def _write_report(run_name, v2_rank, v1_rank, bp, bf, pf, args) -> None:
    auprc_sig = bp["ci"][0] > 0
    verdict = "significant" if auprc_sig and pf["p"] < 0.05 else \
              "significant on AUPRC bootstrap but not F1 permutation" if auprc_sig else \
              "not statistically significant"

    md = [
        "# v2 LoRA vs v1.5 Significance Tests\n",
        f"Config: `{run_name}`  "
        f"({args.n_boot} bootstrap resamples, {args.n_perm:,} F1 permutations, "
        "movie-level resampling throughout)\n",
        "## Metrics\n",
        "| | v2 LoRA | v1.5 MLP | Δ |",
        "|---|---|---|---|",
        f"| AUPRC | {v2_rank['auprc']:.4f} | {v1_rank['auprc']:.4f} | "
        f"{bp['diff']:+.4f} |",
        f"| AUROC | {v2_rank['auroc']:.4f} | {v1_rank['auroc']:.4f} | "
        f"{v2_rank['auroc'] - v1_rank['auroc']:+.4f} |",
        "",
        "## Bootstrap AUPRC difference (movie-level, 95% CI)\n",
        f"ΔAUPRC = {bp['diff']:+.4f}  "
        f"95% CI [{bp['ci'][0]:+.4f}, {bp['ci'][1]:+.4f}]",
        "",
        "## Bootstrap F1 difference (movie-level, 95% CI)\n",
        f"ΔF1 = {bf['diff']:+.4f}  "
        f"95% CI [{bf['ci'][0]:+.4f}, {bf['ci'][1]:+.4f}]",
        "",
        "## Paired permutation test for F1\n",
        f"ΔF1 = {pf['diff']:+.4f}  p = {pf['p']:.4f} {_sig(pf['p'])}",
        "",
        "## Verdict\n",
        f"v2 LoRA vs v1.5: **{verdict}**.",
        f"The AUPRC bootstrap CI {'excludes' if auprc_sig else 'includes'} 0.",
        "",
        "*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant.\n",
    ]
    (REPO / "reports" / "v2_significance.md").write_text("\n".join(md))


if __name__ == "__main__":
    main()
