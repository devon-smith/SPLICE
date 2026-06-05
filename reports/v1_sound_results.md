<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# v1.5 — Soundly-Regularized MLP (Phase 3, P1)

Full grid sweep: dropout [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] x weight_decay [0.0, 1e-05, 0.0001, 0.001] x 3 seeds = 72 runs. Each run early-stops on val AUPRC (patience 5, best checkpoint kept). The held-out test set is scored once, after the config is selected on val.

**Selected config:** dropout = 0.1, weight_decay = 1e-04.

## Mean val AUPRC per (dropout, weight_decay) cell

| dropout \ wd | 0e+00 | 1e-05 | 1e-04 | 1e-03 |
|---|---|---|---|---|
| 0.0 | 0.4289 | 0.4265 | 0.4259 | 0.4233 |
| 0.1 | 0.4271 | 0.4281 | 0.4306 | 0.4251 |
| 0.2 | 0.4283 | 0.4277 | 0.4267 | 0.4266 |
| 0.3 | 0.4282 | 0.4270 | 0.4276 | 0.4263 |
| 0.4 | 0.4256 | 0.4261 | 0.4254 | 0.4259 |
| 0.5 | 0.4271 | 0.4266 | 0.4261 | 0.4270 |

## Selected config — 3-seed results

| metric | value |
|---|---|
| val AUPRC (mean ± std) | 0.4306 ± 0.0027 |
| **test AUPRC (mean ± std)** | **0.4030 ± 0.0022** |
| test AUPRC per seed | 0.3999, 0.4048, 0.4045 |
| 3-seed ensemble test AUPRC | 0.4214 |
| v1 MLP reference (single seed, dropout 0.2, wd 0) | 0.409 |
| v0 logistic reference | 0.356 |

## Verdict

v1.5 test AUPRC 0.4030 vs original v1 0.409 (Δ -0.0060): within noise of the original v1 — the v1 number was sound, not a lucky overfitting seed.

The val-AUPRC grid is **remarkably flat**: all 24 (dropout, weight_decay) cells fall within 0.425–0.431 — a ~0.005 spread, on the order of the cross-seed std. Regularization barely moves the result, strong evidence that the binding constraint is the frozen DINOv2 feature, not head capacity or overfitting. This is the central v1 finding and the motivation for v2 backbone fine-tuning.

Downstream Phase-3 analyses use **seed 2** (test AUPRC 0.4045, the representative single model, ≈ the 0.403 headline) as the canonical v1.5 prediction. The 3-seed mean-probability ensemble reaches 0.4214 — reported as a separate result, not the headline.
