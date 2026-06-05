<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# Macro per-movie AP — Literature-Aligned MovieNet Metric

MovieNet scene-segmentation papers (ShotCoL, BaSSL, TranS4mer, MEGA, MASRC,
MHRT, NeighborNet) report **AP as the mean of per-movie AP values**, not pooled
AP over every test pair — MHRT states this explicitly. Until we recompute on
the same metric, our pair-level AUPRC numbers cannot be fairly compared to
those baselines.

This report recomputes the canonical macro per-movie AP for all three trained
heads on the MovieNet test split, with movie-level paired bootstrap CIs (the
correct resampling scheme given that pairs from the same movie are correlated).
Produced by `scripts/eval/compute_macro_ap.py`; per-run JSONs at
`reports/macro_ap_v0.json`, `reports/macro_ap_v1.5_seed{0,1,2}.json`,
`reports/macro_ap_v2.json`, `reports/macro_ap_compare_*.json`.

## Main comparison

| Model | Pooled AUPRC | Macro per-movie AP | n_movies_used |
|---|--:|--:|--:|
| v0 logistic | 0.356 | **0.372** | 64 |
| v1.5 MLP (3-seed mean ± std) | 0.405 ± 0.002 | **0.418 ± 0.003** | 64 |
| v2 LoRA r=8 α=16 | 0.452 | **0.468** | 64 |

Per-seed v1.5 macro AP: seed 0 = 0.4146, seed 1 = 0.4195, seed 2 = 0.4195 (the
3-seed std on macro AP, 0.003, mirrors the 0.002 std on pooled — seed stability
is unchanged by the metric).

## Significance — movie-level paired bootstrap

1000 movie-level resamples with replacement; 95% CI = (2.5, 97.5) percentiles
of the delta distribution. The bootstrap shuffles movies, not pairs — pair-level
resampling underestimates uncertainty here because cuts inside one movie are
correlated.

| Comparison | macro AP Δ | 95% CI | CI excludes 0? |
|---|--:|---|:--:|
| **v2 LoRA vs v1.5 MLP (seed 2)** | **+0.048** | [+0.0367, +0.0587] | **yes** |
| **v1.5 MLP (seed 2) vs v0 logistic** | **+0.048** | [+0.0376, +0.0595] | **yes** |

Both gaps are significant under proper movie-level resampling — the v2 advance
over v1.5 is not an artifact of within-movie correlation. By coincidence the
two gaps are essentially the same magnitude (each ~+0.05 macro AP).

## Per-movie spread (test split, seed 2 for v1.5)

| Model | macro AP | median per-movie | std per-movie |
|---|--:|--:|--:|
| v0 logistic | 0.372 | 0.355 | 0.128 |
| v1.5 MLP (seed 2) | 0.420 | 0.402 | 0.140 |
| v2 LoRA r=8 α=16 | 0.468 | 0.464 | 0.150 |

Per-movie AP varies by ~±0.13–0.15 across the 64 test movies for every model
— consistent with the per-movie AUPRC range (~0.11–0.65) reported in
`v1_final.md` §5. Continuity is genuinely easier to score on some films than
others; the macro AP collapses that spread into a single equally-weighted mean.

## Pooled vs macro — how much does the metric change the numbers?

| Model | Pooled | Macro | Δ (macro − pooled) |
|---|--:|--:|--:|
| v0 logistic | 0.356 | 0.372 | +0.015 |
| v1.5 MLP (seed 2) | 0.405 | 0.420 | +0.015 |
| v2 LoRA r=8 α=16 | 0.452 | 0.468 | +0.016 |

The shift is small and uniform: every model gains ~+0.015 going from pooled to
macro. Smaller test movies happen to score marginally higher on average, so
weighting them equally lifts the headline by a constant offset. **The metric
choice does not reorder the models or change relative gaps materially** — what
it does change is the basis on which our number can be compared to the
literature.

## Position vs published SOTA

Now that v2 (0.468 macro AP, i.e. **46.8 AP** on the 0–100 scale literature
uses) is on the comparable metric:

| Method | MovieNet AP (test) | gap to v2 |
|---|--:|--:|
| **SPLICE v2 LoRA r=8 α=16 (ours)** | **46.8** | — |
| BaSSL | 57.40 | −10.6 |
| TranS4mer | 60.78 | −14.0 |
| NeighborNet | 71.9 | −25.1 |
| MASRC | 73.2 | −26.4 |

v2 is **~10.6 points below BaSSL** — the closest published baseline — and 25-26
below the top of the leaderboard. The gap is real and meaningful, but it is not
catastrophic given v2 is a per-pair logistic-style head on a LoRA-tuned frozen
backbone, while the SOTA methods (TranS4mer, MASRC) use full-sequence
transformer models with multi-shot context. The macro recomputation does not
close the gap to SOTA — it reframes how big the gap actually is in
literature-comparable terms.

## Methodology notes

- Per-movie AP computed via `sklearn.metrics.average_precision_score` on each
  movie's test-cut slice; the macro AP is the unweighted mean of those values.
- A movie is skipped if it has fewer than 5 test cuts, 0 positives, or all
  positives (AP is undefined or degenerate in those cases). **No movie was
  skipped on the MovieNet test split** — all 64 movies have ≥5 cuts and mixed
  classes.
- Score arrays for all three models are row-aligned to the test slice of
  `pairs/dino_v0_boundary/meta.parquet` (which preserves `cuts.parquet` order);
  `compute_macro_ap.py` asserts length match before scoring.
- Bootstrap uses precomputed per-movie APs (deterministic given the movie
  slice), then resamples the AP vector — exactly equivalent to resampling
  movies and re-scoring, ~1000× faster.

All v2 vs v1.5 (and v1.5 vs v0) statistical claims going forward should use
the movie-level paired bootstrap here, not the pair-level resampling used in
`v1_significance.md`.
