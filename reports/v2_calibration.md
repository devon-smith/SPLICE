<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# Calibration — v0 / v1.5 / v2 reliability + ECE + Brier

All three heads were trained with class-weighted BCE on a ~7.5%-positive
distribution (pos_weight ≈ 12 in the loss). The cost: the raw sigmoid outputs
are *not* calibrated probabilities — they over-predict positive everywhere, in
exchange for usable F1 at deployment thresholds. This report quantifies how
miscalibrated each model is, and whether v2's better discrimination comes with
better or worse calibration. Reliability diagram at
`reports/figures/calibration_v0_v15_v2.png`; metrics at
`reports/v2_calibration_metrics.json`.

## Headline

| Model | Brier ↓ | ECE ↓ | mean(pred) | pos rate | gap |
|---|--:|--:|--:|--:|--:|
| **v0 logistic** (val)  | 0.1589 | 0.2487 | 0.327 | 0.079 | +0.249 |
| **v0 logistic** (test) | 0.1599 | 0.2598 | 0.329 | 0.069 | +0.260 |
| **v1.5 MLP s2** (val)  | 0.1402 | 0.2191 | 0.298 | 0.079 | +0.219 |
| **v1.5 MLP s2** (test) | 0.1383 | 0.2246 | 0.294 | 0.069 | +0.225 |
| **v2 LoRA s0** (val)   | 0.1190 | 0.1708 | 0.249 | 0.079 | +0.171 |
| **v2 LoRA s0** (test)  | 0.1161 | 0.1737 | 0.243 | 0.069 | +0.174 |

(`gap` = mean predicted probability − true positive rate. Positive = over-confident.)

ECE/Brier improve monotonically v0 → v1.5 → v2 on both splits — **the
discrimination improvement carries a calibration improvement with it**, against
the common intuition that better-discriminating models tend to be more
over-confident. Mean predicted probability drops 0.33 → 0.29 → 0.24 (still
~3.5× the true base rate, but trending the right way).

## Reading the reliability diagram

`reports/figures/calibration_v0_v15_v2.png` shows per-bin observed vs predicted
on both splits. Common pattern across all three models:

1. **Mid-range bins (0.3–0.8) overshoot the diagonal heavily** — the model
   predicts 0.7 confidence on bins whose actual positive rate is ~0.25 (v0)
   to ~0.35 (v2). This is the dominant ECE contribution.
2. **High-confidence bins (0.9–1.0) sit close to the diagonal** — pred 0.97
   is observed at ~0.85, which is the cleanest part of the curve for all
   three.
3. **Low-confidence bins (0–0.1) also sit close to the diagonal**, near 0
   observed positives. v2 is slightly tighter here.

v2 shifts the middle-bin overshoot down (its 0.7-bin observes ~0.45 instead of
v0's ~0.25), which is what reduces both ECE and Brier. The shape of the
miscalibration is unchanged across models — just the magnitude.

## val_thr drift across v2 runs

The val-optimal-F1 threshold has moved as we've trained v2 multiple times:

| run | seed | epochs | val_thr | val_auprc | test_auprc |
|---|--:|--:|--:|--:|--:|
| original Lily 20-ep | 231 | 20 (T_max=20) | 0.828 | 0.483 | 0.452 |
| Phase 1 extended | 42 | 50 (T_max=50) | 0.852 | 0.483 | 0.447 |
| Phase 2 seed 0 | 0 | 35 (T_max=35) | 0.816 | 0.487 | 0.460 |

Range 0.816–0.852 — ~4 percentage points of drift across runs. The drift is
**not just seed noise** — the original (T_max=20) and Phase 1 (T_max=50) used
different cosine schedules; the higher T_max=50 schedule pushed val_thr up
(model became more confident at boundaries during longer training); T_max=35
landed in between. Practical implication: any downstream deployment using
val_thr should re-derive it from the val split of the actually-shipped seed,
not assume a fixed value.

## What this means for downstream use

1. **Don't use raw v2 scores as probabilities.** A v2 score of 0.5 corresponds
   to a real positive rate of ~0.10 on test, not 0.5. If a downstream consumer
   needs calibrated probabilities (e.g. for ensembling, expected-utility
   thresholding, or risk-aware UI), apply temperature scaling or Platt
   scaling on the val set first.
2. **The val-optimal F1 threshold remains the canonical operating point.**
   It's been derived per-run; F1@val_thr is what we report. The reliability
   gap doesn't break this — it just means "v2 says 0.85" means "this is in
   the high-F1 region" not "this is 85% likely to be inconsistent".
3. **Calibration improves with the model.** v2's lower ECE means temperature
   scaling on v2 will need to push less than on v0 — useful to know if you
   later run a calibration step.

## Read

v2 is the best-calibrated of the three frozen-backbone-style heads, in both
absolute terms (ECE 0.17 vs v0's 0.26) and relative terms (the gap between mean
prediction and pos rate is 0.17 for v2 vs 0.25 for v0 on test). The
miscalibration is large in absolute terms but **trending the right way as the
model improves**. Calibration-aware deployment still needs a post-hoc step
(temperature / Platt), but v2 is a better starting point for that step than
either prior model.
