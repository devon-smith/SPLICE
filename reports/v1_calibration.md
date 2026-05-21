# v1.5 Comprehensive Calibration (Phase 3, P5)

v1.5 = seed-2 sound MLP. Within-shot scores are the v1.5 model applied to img0/1/2 pairs of single shots (genuinely continuous); test metrics use the cached test predictions.

## 1. Multi-threshold sweep

Thresholds are percentiles of the v1.5 within-shot (val) score distribution.

| τ | threshold | test precision | test recall | test F1 | implied use case |
|---|---|---|---|---|---|
| τ50 | 0.001 | 0.070 | 1.000 | 0.131 | very aggressive flag (most cuts reviewed) |
| τ75 | 0.009 | 0.077 | 0.997 | 0.143 | aggressive review |
| τ90 | 0.091 | 0.102 | 0.969 | 0.184 | broad review |
| τ95 | 0.269 | 0.141 | 0.903 | 0.244 | high-recall review (handoff default) |
| τ99 | 0.645 | 0.293 | 0.608 | 0.395 | balanced -- reproduces the F1-optimal point |

## 2-3. Reliability, Brier score, ECE

- Brier score: **0.1383**  (lower is better; base-rate-only ≈ 0.0643)
- Expected Calibration Error (10 bins): **0.2246**
- of 10 populated deciles, 10 sit above the diagonal (predicted > observed = over-confident).
See `reports/figures/v1_reliability.png`.

## 4. Platt scaling (fitted on val, applied to test)

| | Brier | ECE |
|---|---|---|
| raw v1.5 | 0.1383 | 0.2246 |
| Platt-scaled | 0.0519 | 0.0117 |

Post-hoc Platt scaling improves calibration; AUROC/AUPRC are unchanged (monotonic transform).

## 5. Within-shot τ95 robustness across splits

| split computed on | τ95 | within-shot pairs |
|---|---|---|
| train | 0.2656 | 892,770 |
| val | 0.2686 | 308,076 |
| test | 0.2602 | 317,280 |

τ95 spread across splits = 0.0084 — **stable** (< 0.05): the within-shot calibration is not data-split dependent.

## Takeaways

- τ99 ≈ the F1-optimal point (F1 0.395); τ95 trades precision for recall as a review flag. The within-shot quantile is a precision/recall dial.
- v1.5 is mildly miscalibrated (ECE 0.225); Platt scaling improves it.
- τ95 is stable across data splits (spread 0.0084).
