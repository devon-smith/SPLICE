<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->
<!-- NOTE: This checked-in report is the archival single-run r8_a16 significance output. The active script now targets the final v2 3-seed mean score file and should be rerun on the VM to refresh this report. -->

# v2 LoRA vs v1.5 Significance Tests

Config: `r8_a16`  (1000 bootstrap resamples, 10,000 F1 permutations, movie-level resampling throughout)

## Metrics

| | v2 LoRA | v1.5 MLP | Δ |
|---|---|---|---|
| AUPRC | 0.4516 | 0.4045 | +0.0472 |
| AUROC | 0.8788 | 0.8585 | +0.0202 |

## Bootstrap AUPRC difference (movie-level, 95% CI)

ΔAUPRC = +0.0472  95% CI [+0.0357, +0.0599]

## Bootstrap F1 difference (movie-level, 95% CI)

ΔF1 = +0.0371  95% CI [+0.0285, +0.0461]

## Paired permutation test for F1

ΔF1 = +0.0371  p = 0.0000 ***

## Verdict

v2 LoRA vs v1.5: **significant**.
The AUPRC bootstrap CI excludes 0.

*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant.
