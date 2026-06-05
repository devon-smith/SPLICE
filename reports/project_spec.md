<!-- AI-USE: This project-spec document was AI-assisted with Claude via Claude Code. -->

# SPLICE — Project Specification

Cross-shot visual consistency detection for narrative film and AI-generated
video. CS231n Spring 2026. This spec restates the task, data, and model; results
live in `v0_results.md` and `v1_results.md`.

## Task

Given the pair of frames at a shot-boundary cut — the last keyframe of one shot
and the first keyframe of the next — output a single **continuity score**
`s ∈ [0, 1]`: how consistent the visual change across the cut is with variation
that plausibly occurs *within* one continuous scene. Higher means more
discontinuous.

The motivating use case is AI video generation: tools like Runway, Sora, Luma
and Pika generate clips independently with no cross-shot continuity, so assembled
multi-shot sequences show visible jumps at cuts (lighting, colour, background).
SPLICE flags those cuts for human review. It is deliberately a *cinematographic
similarity* scorer trained on real film — **not** an AI-gen detector; whether a
clip is AI-generated is outside what the model sees.

## Label definition

For an ordered pair of adjacent shots `(N, N+1)` in a movie, the cut is labelled

> `y = 1` (inconsistent) iff the cut crosses a scene boundary, else `y = 0`.

Operationally the label is BaSSL's per-shot `boundary_label` on the left shot N
(1 iff shot N is the last of its scene) — equivalently `invideo_scene_id[N] ≠
invideo_scene_id[N+1]`. Cuts whose `boundary_label` is `-1` (BaSSL's "ignore"
marker for unannotated transitions) are dropped. Within-scene cuts are the
negatives; scene-boundary cuts are the positives.

## Dataset

MovieNet, the 318-movie scene-segmentation subset (BaSSL annotations). Each shot
has three 240p keyframes (`img_0/1/2`); keyframes are sourced from the
HuggingFace mirror `ZhengPeng7/MovieNet` (the original BaSSL Aliyun link is
dead). The cut index has **502,534 adjacent-shot cuts**, **7.47% positive**, with
movie-disjoint splits (train/val/test = 190/64/64 movies).

## Model (v0)

A frozen `facebook/dinov2-base` (ViT-B/14) encodes each keyframe to a 768-d
embedding. For a cut, the **pair feature** concatenates the two shot embeddings
with their absolute difference and cosine similarity:
`[ eL | eR | |eL − eR| | cos(eL, eR) ]` → 2305-d. A logistic regression head
(class-balanced) predicts `y`. The backbone is never fine-tuned in v0/v1; all
keyframes are embedded once into an HDF5 cache so downstream training is
CPU/GPU-light.

## Version ladder

- **v0** — frozen DINOv2 + logistic head. Establishes the floor: test AUPRC
  0.356, beating raw DINOv2 cosine (0.255), HSV χ² (0.195) and CLIP cosine
  (0.157). Inference threshold calibrated from within-shot variation.
- **v1** — MLP head (2305·512·128·1) lifts AUPRC to 0.409; mean-pool-3 frame
  aggregation lifts the logistic model to 0.388; stratified evaluation
  (by class, by movie with bootstrap CIs) and calibration analysis.
- **v2** (final report) — supervised contrastive loss + projection head, partial
  (LoRA) fine-tuning of DINOv2 to break the frozen-feature ceiling, and a
  hand-labelled AI-generated-video evaluation set.

## Evaluation

The task is class-imbalanced (7.5% positive), so **AUPRC** is the headline metric,
with AUROC alongside. Operating points: the F1-optimal threshold on val, and a
within-shot-calibrated threshold (τ95 high-recall; τ99 ≈ F1-optimal, label-free).
Results are stratified per movie with a movie-level bootstrap CI, since cuts
within a film are correlated. Baselines: raw DINOv2 cosine, HSV colour-histogram
χ², and zero-shot CLIP ViT-L cosine.
