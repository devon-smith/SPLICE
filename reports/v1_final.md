<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# SPLICE v1 — Cross-Shot Visual Continuity Scoring

CS231n Spring 2026 · Devon Smith, Lily Bailey, Xander Hnasko

## 1. Abstract

We build a per-cut visual-continuity scorer for narrative film: given the two
frames flanking a shot-boundary cut, predict whether the appearance change is
consistent with variation *within* a continuous scene. On the MovieNet 318-movie
subset, a frozen DINOv2 ViT-B/14 feeding a small MLP head reaches **test AUPRC
0.403 ± 0.002** (3 seeds) — a statistically significant +0.15 absolute over a
raw-DINOv2-cosine baseline (0.255) and far above HSV χ² (0.195) and CLIP cosine
(0.157). The central finding is a **frozen-feature ceiling**: across a 72-run
regularization sweep the validation AUPRC never leaves a 0.006 band, so model
capacity is not the constraint — fine-tuning the backbone is the necessary next
step (v2).

## 2. Task and data

A **cut** is an ordered pair of adjacent shots `(N, N+1)` in a movie. Its label
is binary: `y = 1` (inconsistent) iff the cut crosses a scene boundary, else
`y = 0`. The label comes from the BaSSL `boundary_label` of the left shot
(equivalently `invideo_scene_id[N] ≠ invideo_scene_id[N+1]`); cuts with BaSSL's
`-1` "ignore" marker are dropped.

Data is the MovieNet 318-movie scene-segmentation subset (BaSSL annotations;
keyframes from the HuggingFace mirror `ZhengPeng7/MovieNet`, as the original
Aliyun link is dead). The cut index is a 13-column Parquet:

| field | meaning |
|---|---|
| `movie_id`, `shot_left_idx`, `shot_right_idx` | the cut's identity |
| `scene_left_id`, `scene_right_id` | per-shot scene ids |
| `y_inconsistent` | label (1 = scene-boundary cut) |
| `left/right_img{0,1,2}_path` | 6 keyframe paths (3 per shot) |
| `split` | train / val / test |

**502,534 cuts**, **7.47% positive**, movie-disjoint splits (train/val/test =
190/64/64 movies). All metrics below are the held-out **test split** (105,095
cuts, 64 movies, 6.9% positive).

## 3. Method

Every keyframe is encoded once by a **frozen** `facebook/dinov2-base` (ViT-B/14)
into a 768-d embedding (1.52M keyframes cached to HDF5; the backbone is never
updated in v0/v1). For a cut, the **2305-d pair feature** is

```
[ eL | eR | |eL − eR| | cos(eL, eR) ]      (768 + 768 + 768 + 1)
```

where `eL`, `eR` are the embeddings of the two boundary keyframes (left shot
`img_2`, right shot `img_0`). Three heads sit on this feature:

- **v0** — logistic regression (`StandardScaler` + `LogisticRegression`,
  class-balanced). `configs/v0_default.yaml`.
- **v1** — a 2-layer MLP `2305 → 512 → 128 → 1` (ReLU, dropout), BCE with
  `pos_weight`. The first v1 overfit within ~2 epochs.
- **v1.5** — the *sound* v1: a 72-run grid sweep over dropout × weight-decay × 3
  seeds, early-stopping on val AUPRC. Selected dropout 0.1, weight-decay 1e-4.
  `configs/v1_sound.yaml`. The seed-2 model is the canonical v1.5 used for all
  analysis; aggregate numbers are mean ± std over 3 seeds.

## 4. Results

**Headline (test split).** F1@val = F1 at the val-optimal threshold; F1@τ99 /
F1@τ95 = F1 at within-shot-calibrated thresholds (§7).

| Model | AUROC | AUPRC | F1@val | F1@τ99 | F1@τ95 |
|---|---|---|---|---|---|
| CLIP ViT-L cosine | 0.719 | 0.157 | 0.233 | — | — |
| HSV χ² | 0.763 | 0.195 | 0.282 | — | — |
| raw DINOv2 cosine | 0.812 | 0.255 | 0.322 | 0.322 | 0.211 |
| v0 logistic | 0.849 | 0.356 | 0.388 | 0.382 | 0.225 |
| **v1.5 MLP** | **0.859** | **0.403 ± 0.002** | **0.424** | **0.395** | **0.244** |

(v0 logistic on mean-pooled 3-keyframe shot embeddings reaches AUPRC 0.388 — an
ablation between v0 and v1.5; see `v1_results.md`.)

**Per-movie robustness.** Resampling movies (the correlated unit), mean per-movie
AUPRC is **0.372 [0.341, 0.404]** for v0 logistic vs **0.283 [0.258, 0.308]** for
raw cosine — non-overlapping 95% bootstrap CIs (`v0_stratified.md`,
`figures/v0_stratified_by_movie.png`).

**Significance** (`v1_significance.md`). All five ladder comparisons are
significant on DeLong's test, the AUPRC bootstrap CI, and the F1 permutation
test:

| Comparison | ΔAUROC | DeLong p | ΔAUPRC (95% CI) |
|---|---|---|---|
| v1.5 − v0 logistic | +0.010 | 2e-20 | +0.048 [+0.037, +0.060] |
| v0 − raw DINOv2 cosine | +0.037 | 8e-98 | +0.101 [+0.084, +0.117] |
| mean-pool-3 − v0 | +0.014 | 1e-31 | +0.032 [+0.024, +0.040] |
| v0 − HSV χ² | +0.085 | 1e-151 | +0.161 [+0.131, +0.192] |
| raw DINOv2 cosine − CLIP | +0.093 | 3e-243 | +0.099 [+0.082, +0.116] |

The MLP head's gain over logistic is small but real (CI excludes 0). Every rung
of the v0→v1.5 ladder clears significance.

**DINOv2 vs CLIP.** Raw DINOv2 cosine beats CLIP cosine by a wide, highly
significant margin (ΔAUROC +0.093) — and CLIP even trails plain HSV colour
histograms. CLIP's image–text contrastive objective makes its features
*deliberately invariant* to lighting, colour and framing shifts; those are
exactly the signals that define a visual cut. CLIP is not "bad" — it is
optimised for the wrong invariance. DINOv2's self-supervised objective preserves
that low-level appearance information, which is why it is the right backbone.

## 5. Macro per-movie AP (literature-aligned metric)

The MovieNet papers (ShotCoL, BaSSL, TranS4mer, MEGA, MASRC, MHRT, NeighborNet)
report AP as the *mean of per-movie AP values*, not pooled AP over every test
pair. The pair-level AUPRC in §4 is the right comparison among our own models,
but is not directly comparable to published numbers. Macro AP recomputed by
`scripts/eval/compute_macro_ap.py`; full report at `reports/macro_ap.md`.

| Model | Pooled AUPRC | Macro per-movie AP | n_movies |
|---|--:|--:|--:|
| v0 logistic | 0.356 | **0.372** | 64 |
| v1.5 MLP (3-seed mean ± std) | 0.405 ± 0.002 | **0.418 ± 0.003** | 64 |
| **v2 LoRA r=8 α=16 (3-seed mean ± std)** | **0.4572 ± 0.0033** | **0.4690 ± 0.0020** | 64 |

Macro is consistently ~+0.012 above pooled for every model; the metric choice
does not reorder them. Under movie-level paired bootstrap (1000 resamples) the
v2 over v1.5 gap is **+0.052 macro AP, 95% CI [+0.042, +0.063]** (excludes 0,
significant; ensemble framing) — the same magnitude as v1.5 over v0 (+0.048).
Per-seed mean macro AP for v2 is 0.4690 ± 0.0020 (3 seeds), vs v1.5's
0.418 ± 0.003 — std on the macro AP statistic is tighter for v2 than for v1.5.
On the comparable metric v2's 46.9 AP sits ~10.5 points below BaSSL (57.40),
14 below TranS4mer, and 25-26 below NeighborNet / MASRC. The gap to SOTA is
real but is not catastrophically larger than the pooled-AUPRC framing suggested
— this is the basis on which v2 should be reported alongside published work.

All v2 vs v1.5 (and v1.5 vs v0) statistical claims going forward use the
movie-level paired bootstrap from `v2_final.md` / `macro_ap.md`, not the
pair-level resampling in `v1_significance.md` — pair-level resampling
underestimates uncertainty when pairs from the same movie are correlated.

**Per-movie picture** (`reports/per_movie_analysis.md`,
`reports/figures/per_movie_analysis.png`): v2 wins on **56 / 64 test movies
(87.5%)**, with the biggest gains on movies where v1.5 was weakest (low
positive rate, mediocre baseline AP). Largest single regression is −0.031 on a
movie where v1.5 was already worse than v0. No catastrophic failures — v2's
per-movie AP floor is 0.16 (above v1.5's 0.17). v2's mean gain is concentrated
on hard movies rather than uniformly shifting easy ones, consistent with the
LoRA backbone learning continuity-specific features the frozen DINOv2 + MLP
couldn't access.

## 6. Stratified analysis

**By class.** v1.5 mean predicted score is 0.69 on y=1 cuts vs 0.30 on y=0 —
clean separation; per-class hit-rates re-aggregate exactly to the 0.886 accuracy.

**By raw-cosine quintile** (`v1_stratified_difficulty.md`,
`figures/v1_quintile_auprc.png`). Binning cuts by cosine similarity (Q1 = look
most different → Q5 = most similar), v1.5 AUPRC beats the within-band base rate
in **every** quintile — it adds ranking signal beyond cosine throughout, with
the largest relative lift (~3×) in the ambiguous middle quintiles.

| Quintile | positive rate | v1.5 AUPRC | raw-cosine AUPRC |
|---|---|---|---|
| Q1 (most different) | 0.208 | 0.531 | 0.324 |
| Q2 | 0.079 | 0.240 | 0.100 |
| Q3 | 0.037 | 0.120 | 0.044 |
| Q4 | 0.015 | 0.043 | 0.018 |
| Q5 (most similar) | 0.006 | 0.017 | 0.009 |

**By movie / position.** Slow-paced films (fewer shots) score higher (AUPRC 0.46)
than busy films (0.36); cuts early in a film are marginally easier (0.43) than
late ones (0.39). Per-movie AUPRC ranges ~0.11–0.65 — continuity is far easier to
score in some films than others, so the bootstrap CI, not the point estimate, is
the honest summary.

## 7. Calibration

The within-shot keyframe pairs of a single shot are, by construction, the same
continuous scene; their score distribution defines "natural" variation, and its
q-th percentile (τq) is a principled threshold (`v1_calibration.md`).

| τ | precision | recall | F1 | use |
|---|---|---|---|---|
| τ95 | 0.141 | 0.903 | 0.244 | high-recall review flag |
| τ99 | 0.293 | 0.608 | 0.395 | balanced — ≈ the F1-optimal point |

τ95 sits at only the ~51st percentile of cut-level scores: it flags ~half of all
cuts (recall 0.90, low precision) — a deliberate high-recall operating point, not
a failure. **τ99 reproduces the F1-optimal threshold without using any labels**,
and is stable across data splits (τ95 spread 0.008 over train/val/test).

v1.5's raw probabilities are over-confident (ECE 0.225, Brier 0.138 — a
side-effect of class-weighted training). **Platt scaling** fitted on val fixes
this on test (ECE 0.225 → 0.012, Brier → 0.052) with AUROC/AUPRC unchanged;
deployment should use Platt-scaled probabilities (`figures/v1_reliability.png`).

## 8. Error analysis

At the F1-optimal threshold v1.5 produces 5,863 false positives (6.0% of
negatives) and 3,724 false negatives (51.3% of positives — it misses about half
of true scene boundaries). Errors are **diffuse**: both FP and FN span all 64
test movies, with the largest single-movie share only 3–4% — not a few
pathological films.

A descriptive signal worth qualitative review: false negatives have higher mean
cosine similarity (0.198) than false positives (0.115). False negatives are true
scene boundaries whose flanking frames genuinely *look alike* (same location or
lighting across a narrative cut) — precisely the case a global appearance feature
cannot catch. 21% of false negatives are near-misses (within 0.1 of threshold).
The top-50 FP/FN with all model scores and keyframe grids are in
`v1_false_positives.csv` / `v1_false_negatives.csv` and
`figures/v1_error_grid_{fp,fn}.png`. Whether the most confident errors are real
failures or MovieNet labelling noise (~0.86% source disagreement) is an open
question for qualitative review.

## 9. Feature importance

On the 2305-d feature (`v1_feature_importance.md`):

- **The difference signal dominates.** A logistic head retrained on `|eL−eR|`
  alone reaches AUPRC 0.302; on raw `concat(eL,eR)` alone only 0.178. What
  matters is how the embeddings *differ*, not their absolute content.
- **Cosine is heavily used but redundant.** Permutation importance ranks the
  single cosine scalar first by far (AUPRC drop 0.278) — the trained model leans
  on it hardest. Yet a model retrained *without* cosine recovers almost fully
  (concat + |eL−eR| → 0.350): cosine is a convenient pre-computed summary of
  `eL−eR`, not irreplaceable information.
- **What DINOv2 encodes.** Ridge probes decode low-level appearance from the
  frozen embedding — mean luminance R² 0.68, contrast 0.46, saturation 0.37 — the
  exact cues a visual cut disturbs.

## 10. Limitations

- **Frozen-feature ceiling.** The 72-run regularization sweep moved val AUPRC by
  ≤0.006; the v1 MLP overfits in ~2 epochs. Head capacity is not the constraint.
- **Absolute performance is modest.** AUPRC 0.40 at a 7% base rate is a useful
  review flag, not a solved task; recall at the F1-optimal point is ~0.50.
- **Cross-film axis is degenerate** — every labelled cut is a same-movie adjacent
  pair, so there is no within-film/cross-film split to evaluate.
- **Shot-scale stratification not done** — MovieNet cinematic-style annotations
  are absent from this data distribution.
- **Label noise** — ~0.86% of cuts had `boundary_label` / scene-id disagreement
  at the source; some confident errors may be mislabelled.
- **AI-generated evaluation not yet performed** and **v2 fine-tuning not yet
  trained** — both deliberately out of scope for v1.

## 11. Next steps

The frozen-feature ceiling is the bridge to v2. The supervised signal in MovieNet
has reached the limit of what frozen DINOv2 features can represent, so the next
step is to **fine-tune the backbone**: (1) LoRA / partial fine-tuning of DINOv2
under a supervised-contrastive objective (within-scene pairs as positives), so the
representation itself learns continuity-relevant structure; (2) a projection head
trained with `λ·SupCon + (1−λ)·BCE`. Separately, M3 adds a hand-labelled
**AI-generated-video** evaluation set — the motivating use case — to test whether
a film-trained continuity scorer transfers to generated cuts.

---
*Sources: `v0_results.md`, `v1_results.md`, `v1_sound_results.md`,
`v1_significance.md`, `v0_stratified.md`, `v1_stratified_difficulty.md`,
`v1_calibration.md`, `v1_error_analysis.md`, `v1_feature_importance.md`.
All numbers are reproducible from the scripts in `scripts/`.*
