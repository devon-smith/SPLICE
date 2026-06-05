<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# v2 LoRA r=8 α=16 — Extended Single-Seed Run

Extended training of the v2 r=8/α=16 config (seed 42) to determine where val
AUPRC plateaus. Same architecture as the original 20-epoch run: q+v LoRA on
DINOv2-base, MLP head, class-weighted BCE, lr_backbone=5e-5 cosine, lr_head=1e-3,
dropout 0.1, AMP. **Differs in one way: 50-epoch budget with patience 10**,
which changes the cosine schedule (T_max=50 vs T_max=20).

Output: `/mnt/disks/splice-data/outputs/v2_lora_extended/seed42/r8_a16/`.

## Per-epoch trajectory

Early stop fired at **epoch 30** (best val at **epoch 20**, then 10 consecutive
stale epochs). Total wall time 7h13m on one L4 (13.4 min/epoch).

| epoch | train loss | val_auprc | epoch | train loss | val_auprc |
|--:|--:|--:|--:|--:|--:|
| 0 | 1.086 | 0.314 | 16 | 0.701 | 0.478 |
| 1 | 0.956 | 0.324 | 17 | 0.717 | 0.480 |
| 2 | 0.934 | 0.386 | 18 | 0.714 | 0.475 |
| 3 | 0.892 | 0.410 | 19 | 0.682 | 0.476 |
| 4 | 0.870 | 0.427 | **20** | **0.663** | **0.4828 ←** |
| 5 | 0.821 | 0.438 | 21 | 0.663 | 0.479 |
| 6 | 0.809 | 0.445 | 22 | 0.657 | 0.480 |
| 7 | 0.795 | 0.458 | 23 | 0.645 | 0.478 |
| 8 | 0.797 | 0.461 | 24 | 0.648 | 0.478 |
| 9 | 0.773 | 0.471 | 25 | 0.615 | 0.477 |
| 10 | 0.757 | 0.467 | 26 | 0.623 | 0.475 |
| 11 | 0.755 | 0.465 | 27 | 0.585 | 0.474 |
| 12 | 0.763 | 0.474 | 28 | 0.590 | 0.479 |
| 13 | 0.746 | 0.474 | 29 | 0.602 | 0.471 |
| 14 | 0.739 | 0.482 | 30 | 0.574 | 0.466 |
| 15 | 0.734 | 0.477 |  |  |  |

## Where did val plateau?

**Val plateaus around epoch 14**, hits its formal peak at epoch 20, then drifts
down. The window epochs 14–22 all sit in [0.475, 0.483] — within 0.008 of the
peak, and most are within 0.003. Train loss continues to decrease through the
end (0.66 → 0.57 from ep 20 to ep 30), confirming the model is overfitting train
without further val gain past ep ~14. **N\* = 20** is the formal val-selected
epoch.

## Metrics at the val-selected best epoch

| metric | extended (this run) | original 20-epoch | Δ |
|---|--:|--:|--:|
| best epoch | 20 | 19 | +1 |
| val AUPRC | 0.4828 | 0.4826 | +0.0002 |
| **test AUPRC (pooled)** | **0.4470** | **0.4516** | **−0.0046** |
| **test macro per-movie AP** | **0.4616** | **0.4684** | **−0.0068** |
| test AUROC | 0.8754 | 0.8788 | −0.0034 |
| F1 @ val_thr | 0.4565 | 0.4613 | −0.0048 |
| val_thr | 0.852 | 0.828 | +0.024 |

The two runs converge to essentially the same val AUPRC (Δ +0.0002) but the
extended run generalizes slightly *worse* on test (−0.0046 AUPRC, −0.0068 macro
AP). Both runs use the same architecture, optimizer, data, and seed (only the
cosine T\_max differs: 50 vs 20). At any given epoch in the new schedule the
backbone LR is higher than in the 20-epoch schedule (cosine is gentler over 50
epochs), and the longer trajectory finds a val-equivalent minimum that is on
the wrong side of a small val/test divergence. The seed-42 evidence on its own
is consistent with stochastic test-set variation at this magnitude (the v1.5
3-seed std on macro AP was 0.003); the Phase 2 3-seed reporting will say
whether this is a real regression or noise.

## Read

**The model does plateau within a reasonable epoch budget** — val improvement
past epoch 20 is essentially zero, and patience 10 catches the end of the
plateau cleanly at epoch 30. Phase 2's 3-seed run will use the same 50-epoch /
patience-10 budget (LR-schedule comparability across seeds matters more than
matching N\*+5). The original Lily-style 20-epoch budget is *not* leaving major
val AUPRC on the table — at best ~+0.001 — but the extended schedule does
appear to introduce a small generalization tax on this seed, which is what the
3-seed averaging in Phase 2 will quantify.
