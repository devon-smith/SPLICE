# SPLICE

Cross-shot visual consistency detection for narrative film and AI-generated video. CS231n Spring 2026 final project.

**Team:** Devon Smith (devon2), Lily Bailey (lbailey1)

## What we're building

A model that takes a pair of frames at a shot boundary and outputs per-attribute consistency scores across 7 attributes: lighting direction, color temperature, color grade, location, time of day, character identity, and weather/environment. Trained on MovieNet via supervised contrastive learning + multi-attribute classification on a fine-tuned DINOv3 backbone. Evaluated on held-out MovieNet, synthetic perturbations, and a human-labeled set of AI-generated video cuts.

See `reports/project_spec.md` for the full technical architecture and `reports/related_work.md` for the literature review.

## Quickstart

```bash
git clone https://github.com/<org>/SPLICE.git
cd SPLICE
micromamba env create -f environment.yaml
micromamba activate consistency
python scripts/test_dinov3.py
```

## Repo structure

- `configs/` — YAML hyperparameter configs
- `data/annotations/` — small annotation files (large data lives in `$HOME` on FarmShare, not here)
- `reports/` — related work, milestones, final report
- `scripts/` — runnable scripts (preprocess, train, evaluate, sbatch)
- `src/` — importable Python modules
- `tests/` — unit tests
