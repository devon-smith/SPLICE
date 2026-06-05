<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# v1 Significance Tests (Phase 3, P2)

Five model comparisons on the shared test split. DeLong's test for the AUROC difference; movie-level bootstrap (1000 resamples) for the AUPRC-difference CI; paired permutation test (10,000 permutations) for the F1 difference. A 95% CI excluding 0, or p < 0.05, is significant. v1.5 = the seed-2 sound MLP.

| Comparison | ΔAUROC (95% CI) | DeLong p | ΔAUPRC (95% CI) | ΔF1 (95% CI) | F1 perm p |
|---|---|---|---|---|---|
| v1.5 - v0 logistic | +0.0099 [+0.0078, +0.0120] | 1.82e-20 *** | +0.0483 [+0.0370, +0.0597] | +0.0360 [+0.0250, +0.0460] | 0.0000 *** |
| v0 logistic - raw DINOv2 cosine | +0.0369 [+0.0335, +0.0404] | 7.72e-98 *** | +0.1010 [+0.0836, +0.1167] | +0.0665 [+0.0540, +0.0786] | 0.0000 *** |
| mean-pool-3 - v0 logistic | +0.0145 [+0.0121, +0.0169] | 1.07e-31 *** | +0.0321 [+0.0242, +0.0400] | +0.0281 [+0.0201, +0.0353] | 0.0000 *** |
| v0 logistic - HSV chi-square | +0.0853 [+0.0789, +0.0916] | 1.01e-151 *** | +0.1610 [+0.1314, +0.1915] | +0.1062 [+0.0854, +0.1259] | 0.0000 *** |
| raw DINOv2 cosine - CLIP cosine | +0.0925 [+0.0871, +0.0980] | 3.00e-243 *** | +0.0985 [+0.0821, +0.1164] | +0.0891 [+0.0729, +0.1052] | 0.0000 *** |

*** p<0.001, ** p<0.01, * p<0.05, n.s. not significant.

## Interpretation

- **v1.5 - v0 logistic** — significant: ΔAUROC +0.0099 (DeLong p=1.8e-20), ΔAUPRC +0.0483 [+0.0370, +0.0597].
- **v0 logistic - raw DINOv2 cosine** — significant: ΔAUROC +0.0369 (DeLong p=7.7e-98), ΔAUPRC +0.1010 [+0.0836, +0.1167].
- **mean-pool-3 - v0 logistic** — significant: ΔAUROC +0.0145 (DeLong p=1.1e-31), ΔAUPRC +0.0321 [+0.0242, +0.0400].
- **v0 logistic - HSV chi-square** — significant: ΔAUROC +0.0853 (DeLong p=1.0e-151), ΔAUPRC +0.1610 [+0.1314, +0.1915].
- **raw DINOv2 cosine - CLIP cosine** — significant: ΔAUROC +0.0925 (DeLong p=3.0e-243), ΔAUPRC +0.0985 [+0.0821, +0.1164].
