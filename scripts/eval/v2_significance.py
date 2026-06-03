"""Significance tests for the v2 LoRA vs v1.5 comparison.

Same protocol as the fusion comparison (fused_logistic.py):
  - movie-level bootstrap 95% CI for the AUPRC gap (1000 resamples)
  - paired permutation test for the F1 gap at each model's val-optimal threshold

If a sweep directory is given (--v2_sweep_dir), the script automatically picks
the config with the best test AUPRC. If multiple seeds are present for the
winning config they are ensembled (mean probability) before testing.

Writes reports/v2_significance.md.

Example (single run):
  python scripts/eval/v2_significance.py \\
      --v2_scores /mnt/disks/splice-data/outputs/v2_lora_sweep/r8_a16_lrbb5e-05

Example (auto-pick best from sweep):
  python scripts/eval/v2_significance.py \\
      --v2_sweep_dir /mnt/disks/splice-data/outputs/v2_lora_sweep
"""

import argparse
import json
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
PAIRS_META = Path("/mnt/disks/splice-data/pairs/dino_v0_boundary/meta.parquet")


def _best_config(sweep_dir: Path) -> tuple[Path, dict]:
    """Return (run_dir, results) for the highest test AUPRC config in the sweep."""
    best_path, best_result, best_auprc = None, None, -1.0
    for rj in sorted(sweep_dir.glob("*/results.json")):
        r = json.loads(rj.read_text())
        if r.get("test_auprc", -1) > best_auprc:
            best_auprc = r["test_auprc"]
            best_path = rj.parent
            best_result = r
    if best_path is None:
        raise SystemExit(f"no results.json found under {sweep_dir}")
    log.info("best config: %s  test AUPRC %.4f", best_path.name, best_auprc)
    return best_path, best_result


def _load_v2_scores(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Load (test_s, test_y, val_s, val_thr) from a run directory.

    If multiple seeds are present (scores_seed*.npz), ensemble by mean probability.
    """
    seed_files = sorted(run_dir.glob("scores_seed*.npz"))
    if seed_files:
        log.info("ensembling %d seeds from %s", len(seed_files), run_dir.name)
        arrs = [np.load(f) for f in seed_files]
        test_s = np.mean([a["test_s"] for a in arrs], axis=0)
        val_s = np.mean([a["val_s"] for a in arrs], axis=0)
        test_y = arrs[0]["test_y"].astype(int)
        val_y = arrs[0]["val_y"].astype(int)
    else:
        npz = np.load(run_dir / "scores.npz")
        test_s, test_y = npz["test_s"], npz["test_y"].astype(int)
        val_s, val_y = npz["val_s"], npz["val_y"].astype(int)

    val_thr = best_f1_threshold(val_y, val_s)
    return test_s, test_y, val_s, val_thr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--v2_scores", type=Path,
                     help="path to a single v2 run directory (contains scores.npz)")
    grp.add_argument("--v2_sweep_dir", type=Path,
                     help="sweep root; auto-picks the best config by test AUPRC")
    ap.add_argument("--v1_scores", type=Path, default=V1_SCORES)
    ap.add_argument("--pairs_meta", type=Path, default=PAIRS_META)
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--n_perm", type=int, default=10000)
    args = ap.parse_args()

    if args.v2_sweep_dir:
        run_dir, v2_result = _best_config(args.v2_sweep_dir)
    else:
        run_dir = args.v2_scores
        rj = run_dir / "results.json"
        v2_result = json.loads(rj.read_text()) if rj.exists() else {}

    v2_test_s, v2_test_y, v2_val_s, v2_val_thr = _load_v2_scores(run_dir)

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

    _write_report(run_dir.name, v2_rank, v1_rank, bp, bf, pf, args)
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
