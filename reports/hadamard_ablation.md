<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# Hadamard pair-feature ablation — does eL ⊙ eR help?

The current pair feature is `[eL | eR | |eL-eR| | cos(eL,eR)]` (2305-d). The
literature on siamese pair classification (verification, identity matching)
consistently shows that *both* a difference term and a product term help.
Extending to `[eL | eR | |eL-eR| | eL⊙eR | cos]` (3073-d, an extra 768
elementwise-product columns) is the cheap test.

Produced by `scripts/eval/compute_macro_ap.py` + `scripts/prep/build_pair_features.py
--include_hadamard`. Per-run JSONs at `reports/macro_ap_v0_hadamard.json`,
`reports/macro_ap_v1_hadamard.json`, `reports/macro_ap_compare_*.json`.

## Headline

| model | feature dim | macro AP | pooled AUPRC | Δ macro AP vs baseline |
|---|--:|--:|--:|--:|
| v0 logistic (baseline) | 2305 | 0.3715 | 0.3568 | — |
| **v0 hadamard** | **3073** | **0.3751** | **0.3616** | **+0.004** |
| v1.5 MLP (3-seed mean, baseline) | 2305 | 0.4179 | 0.4053 | — |
| v1.5 MLP (seed 2, baseline) | 2305 | 0.4195 | 0.4039 | — |
| **v1.5 hadamard (seed 42)** | **3073** | **0.4126** | **0.4017** | **−0.005 vs 3-seed, −0.007 vs seed 2** |

v1.5 hadamard ran a single seed (42). The baseline v1.5 is reported as 3-seed
mean. Comparing single-vs-single (seed 42 vs seed 2) the gap is still
negative, so seed alignment does not change the conclusion.

## Significance — movie-level paired bootstrap (1000 resamples)

| comparison | Δ macro AP | 95% CI | CI excludes 0? |
|---|--:|---|:--:|
| v0 hadamard vs v0 logistic | +0.0036 | [−0.0009, +0.0080] | **no** |
| v1.5 hadamard vs v1.5 3-seed ensemble | −0.0221 | [−0.0305, −0.0143] | yes (significantly **worse**) |

## Verdict

**Hadamard does not help on this task.** The v0 logistic gain is +0.004 macro
AP with a CI that crosses zero — not significant. The v1.5 MLP result is
*worse* than baseline (−0.022 vs the 3-seed ensemble, CI excludes 0), most
likely because the MLP can already learn whatever cross-term structure the
Hadamard expressed (and the extra 768 columns + lack of ensemble averaging
push the single-seed MLP toward the noise floor).

A few interpretations:

1. **Continuity ≠ identity verification.** The Hadamard term is well-motivated
   for verification tasks (where same-identity pairs have correlated
   embeddings and the elementwise product captures axis-aligned agreement),
   but our positive class is scene-cut *discontinuity*. The
   absolute-difference term already captures most of what a continuity head
   needs; the product term adds dimensions the model can't use.
2. **The MLP doesn't need it.** v1.5's `[abs-diff, cos]` features plus
   non-linearities can already approximate any useful cross-term combination.
   Adding 768 explicit product columns just inflates input dimensionality
   without new information.
3. **Single-seed vs 3-seed comparison is unfair to hadamard.** v1.5 hadamard's
   −0.022 deficit vs the 3-seed ensemble is partly an
   ensemble-vs-single-seed gap. But even against seed 2 alone, hadamard
   doesn't win — so the seed mismatch doesn't rescue the result.

## Recommendation: do NOT retrain v2 LoRA with Hadamard.

The v0 result is not significant; the v1.5 result is significantly worse. No
mechanism suggests v2 would behave differently. Hadamard is ruled out.

## Why we still added the flag

The `--include_hadamard` flag is preserved in `build_pair_features.py` and
`src.data.pairs` because (a) the cost is trivial (one parameter), (b) it
provides a defensible "we tried this" data point in the report, and (c) the
default is off, so no existing code is affected.
