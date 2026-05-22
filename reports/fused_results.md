# Fused DINOv2+CLIP Logistic — Results (Action 3)

Logistic regression on the 4610-d fused boundary feature `[DINOv2 2305-d | CLIP 2305-d]`, trained with v0_logistic.py's recipe (StandardScaler + balanced LogisticRegression, C=1.0, lbfgs, max_iter=2000). The v2 architecture experiment. Produced by `scripts/train/fused_logistic.py`.

## MovieNet test

| model | AUPRC | AUROC | F1@val-thr |
|---|--:|--:|--:|
| v0 logistic (DINOv2 2305-d) | 0.356 | 0.849 | 0.388 |
| v1.5 MLP (DINOv2 2305-d) | 0.405 | 0.859 | 0.424 |
| **fused logistic (4610-d)** | **0.377** | **0.856** | **0.407** |

## Significance vs v1.5

Movie-level paired bootstrap (105,095 test cuts, 2-sided 95% CI, 1000 resamples; protocol of `v1_significance.md`):

- AUPRC(fused) − AUPRC(v1.5) = **-0.0274**, 95% CI [-0.0398, -0.0138]
- CI excludes 0 — significant.

## Veo pilot

Fused model scored on the 10 Veo continuous-action pairs (DINOv2 embedded fresh, CLIP from cache).

| pair | bucket | fused score |
|---|---|--:|
| A015 | major | 0.539 |
| A005 | major | 0.155 |
| A011 | drift | 0.115 |
| A001 | drift | 0.114 |
| A013 | clean | 0.087 |
| A004 | drift | 0.058 |
| A003 | clean | 0.050 |
| A014 | drift | 0.042 |
| A012 | drift | 0.027 |
| A002 | drift | 0.008 |

Bucket means: clean **0.069** / drift **0.060** / major **0.347** (not monotonic).
Spearman vs Dispatch buckets: fused **+0.495** vs v1.5 **+0.440**.

## Outcome

**Outcome C.** fused does not beat v1.5 on MovieNet — fusion of two frozen backbones does not clear the frozen-feature ceiling. LoRA is the remaining v2 lever.

## Read

On MovieNet the fused logistic scores 0.377 AUPRC against v1.5's 0.405 (-0.027); the movie-level bootstrap puts the difference at -0.0274 with 95% CI [-0.0398, -0.0138]. Note the comparison is not perfectly controlled — the fused model is a *logistic* head and v1.5 is a 2-layer MLP — so a fair architecture read also weighs fused logistic against v0 logistic (0.356), the same head on DINOv2 alone: adding the CLIP half moves a logistic head from 0.356 to 0.377. On the Veo pilot the fused model correlates +0.495 with Dispatch's buckets vs v1.5's +0.440 (+0.055). Concatenating a second frozen backbone does not beat v1.5 — consistent with the frozen-feature ceiling in v1_final.md. The remaining v2 lever is LoRA fine-tuning, which unfreezes the backbone rather than widening a frozen feature. The v2 architecture decision (fusion vs LoRA vs both) needs this result plus human input — n=10 on the Veo side remains the binding limitation on the AI-gen claim.
