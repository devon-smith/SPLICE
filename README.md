# SPLICE

Cross-shot visual consistency detection for narrative film and AI-generated video.
CS231n Spring 2026 final project.

**Team:** Devon Smith, Lily Bailey, Xander Hnasko

## What we're building

A model that takes the pair of frames at a shot-boundary cut — the last keyframe of
one shot and the first keyframe of the next — and outputs a single **continuity
score**: how consistent the visual change across the cut is with variation that
plausibly occurs within one continuous scene.

Motivating use case: AI video tools (Runway, Sora, Luma, Pika) generate clips
independently with no cross-shot continuity, so assembled multi-shot sequences show
visible jumps at cuts (lighting, colour, background). SPLICE flags those for human
review. It is a cinematographic-similarity scorer trained on real film — *not* an
AI-gen detector; whether a clip is AI-generated is deliberately outside what the
model sees.

See `reports/project_spec.md` for the architecture and `reports/related_work.md`
for the literature review.

## Versions

- **v0** (current milestone) — frozen DINOv2 ViT-B/14 + logistic regression on a
  2305-d pair feature, against three baselines (HSV chi-square, CLIP cosine, raw
  DINOv2 cosine), with an inference threshold calibrated from within-shot
  variation. Results: `reports/v0_results.md`.
- **v1** — MLP head and ablations (boundary vs mean-pooled frames, negative mining).
- **v2** — supervised contrastive loss, partial fine-tuning, AI-generated eval set.

## Data

MovieNet, 318-movie scene-segmentation subset (BaSSL annotations: per-shot
`invideo_scene_id` / `boundary_label`). Keyframes come from the HuggingFace mirror
`ZhengPeng7/MovieNet` — the original BaSSL Aliyun OSS link is dead (HTTP 404).
Large data lives under `/mnt/disks/splice-data` on the GCP VM, not in git.

## Pipeline

```bash
micromamba activate consistency
python scripts/prep/build_cut_index.py        # labelled cut index (Parquet)
python scripts/prep/embed_keyframes.py        # cache DINOv2 embeddings (GPU)
python scripts/prep/build_pair_features.py --mode boundary   # 2305-d pair features
python scripts/train/v0_logistic.py           # train v0 + baselines
python scripts/eval/calibrate_threshold.py    # within-shot threshold calibration
```

`scripts/inspect_movie.py --imdb_id <id>` is a diagnostic for the annotation format.

## Repo structure

- `configs/` — YAML hyperparameter configs (`v0_default.yaml`)
- `scripts/` — runnable entry points (`prep/`, `train/`, `eval/`)
- `src/` — importable modules (`data/`, `models/`, `eval/`, `losses/`)
- `tests/` — unit tests; run with `pytest tests/`
- `reports/` — spec, related work, milestone results

