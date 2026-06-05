<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# v1 Results — head and pooling ablations

v1 ablates two axes of the frozen-DINOv2 pipeline, holding the backbone fixed:
**(a)** how a shot's keyframes are aggregated, and **(b)** the classifier head.
All numbers are the **test split** (n = 105,095, 6.9% positive). Raw DINOv2
cosine (AUPRC 0.255) is the zero-training reference from `v0_results.md`.

| Model | Head | Features | AUROC | AUPRC | F1@val-thr |
|---|---|---|---|---|---|
| v0 logistic | linear | boundary (img2 ↔ img0) | 0.849 | 0.356 | 0.388 |
| v0 logistic | linear | mean-pool-3 | 0.863 | 0.388 | 0.416 |
| **v1 MLP** | 2305·512·128·1 | boundary (img2 ↔ img0) | 0.862 | **0.409** | **0.427** |

## Mean-pool-3 ablation (Priority 3)

Replacing the single boundary keyframe per shot with the **mean of that shot's
three keyframe embeddings** lifts the logistic model's AUPRC 0.356 → 0.388 (+9%
relative) and AUROC 0.849 → 0.863. Direction and magnitude match ShotCoL's
frame-aggregation ablation. Notably the *raw DINOv2 cosine* baseline is
essentially unchanged under mean-pooling (AUPRC 0.255 → 0.255): the gain is not
in the geometry of the embeddings but in what the **learned head** can do with a
less noisy, more shot-representative vector.

## MLP head (Priority 2)

A 2-layer MLP (2305 → 512 → 128 → 1, ReLU, dropout 0.2, BCE with `pos_weight`
12.3, Adam 1e-3 + cosine schedule) on the **same boundary features** lifts AUPRC
0.356 → 0.409 (+15% relative) — a larger gain than mean-pooling. Non-linear
interactions over the concatenated/­differenced pair feature carry real signal
that a linear head cannot reach.

**The MLP overfits fast.** Val AUPRC peaks at epoch 1 and early-stopping fires by
epoch 6 while train loss keeps falling — with 295K training cuts the 512·128 head
saturates the frozen features within ~2 epochs. More head capacity overfits
rather than extracting more signal. This sharpens the v0 conclusion: the binding
constraint is the **frozen-feature ceiling**, not head capacity. v2's backbone
fine-tuning (LoRA / supervised contrastive) targets exactly that ceiling.

## Takeaways

- Both axes help and are roughly complementary: pooling +0.03 AUPRC, MLP head
  +0.05 AUPRC. A natural v1.x is an MLP on mean-pool-3 features (one MLP was run
  this phase, per scope).
- Best v1 configuration so far: **MLP head, AUPRC 0.409** — a +60% relative
  improvement over the raw DINOv2 cosine baseline (0.255), and +15% over v0.
- Fast MLP overfitting is the headline diagnostic: it motivates v2 (fine-tune the
  backbone) over simply growing the head.

## Reproduce

```bash
# mean-pool-3 ablation
python scripts/prep/build_pair_features.py --mode mean_pool_3 \
    --out /mnt/disks/splice-data/pairs/dino_v0_mean_pool_3
python scripts/train/v0_logistic.py \
    --features /mnt/disks/splice-data/pairs/dino_v0_mean_pool_3 \
    --out /mnt/disks/splice-data/outputs/v0_mean_pool --skip_baselines

# v1 MLP head
python scripts/train/v1_mlp.py --features /mnt/disks/splice-data/pairs/dino_v0_boundary
```

Artifacts: `outputs/v0_mean_pool/`, `outputs/v1_mlp/` (model, scaler, scores,
results).
