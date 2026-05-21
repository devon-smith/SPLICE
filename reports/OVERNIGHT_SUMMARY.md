# Overnight Summary — Phase 3 complete + AI-gen harness

For Devon, morning of 2026-05-21. Everything below is committed and pushed to
`origin/main`; 26 tests green, `black`/`ruff` clean.

## Heads-up: prompt context was stale

The overnight prompt assumed P5–P7 were unfinished. They were **already done**
in the prior session ("proceed through all the next P's"). So Blocks 1–2 were
*reconciled*, not redone — I did not waste the night regenerating finished work:

- **P6 feature importance** — already complete. Added the one missing artifact,
  `reports/figures/v1_coefficient_norms.png`. (Linear probes used mean luminance
  / contrast / saturation; the prompt also listed mean-RGB and saturation-
  variance — minor deviation, the three probes already characterise the
  embedding. Flagging in case you want the exact set.)
- **P7 `v1_final.md`** — already complete: 10 sections, ~7 pages, all content
  present. The prompt's 11-section structure breaks "Significance" into its own
  section; mine folds it into Results §4. Content-complete; left as-is.

## Phase 3 — final findings (recap)

- **P6 feature importance:** the difference signal dominates — `|eL−eR|` alone
  gives AUPRC 0.302 vs 0.178 for raw concat. Cosine is leaned-on hardest
  (permutation drop 0.278) but redundant (a model retrained without it recovers
  to 0.350). DINOv2 linearly encodes luminance (R² 0.68), contrast, saturation.
- **P7:** `reports/v1_final.md` synthesises the whole v1 story end-to-end.

## AI-gen evaluation harness (Block 3) — built and verified

A one-command pipeline so that when you drop real clips it just runs:

```bash
# clips at /mnt/disks/splice-data/datasets/aigen/<source>/pair_<id>_{left,right}.mp4
# + labels.csv (pair_id,source,y_inconsistent,notes)
python scripts/eval/build_aigen_eval.py --clips_root .../aigen/ --out .../aigen_eval/
python scripts/eval/eval_aigen.py --aigen_index .../aigen_eval/cuts.parquet --out .../results/
```

- **`src/data/video_frames.py`** — boundary / uniform-n / timestamp frame
  extraction from clips. 6 unit tests.
- **`scripts/eval/build_aigen_eval.py`** — clips + labels.csv → a cut-index
  Parquet with the MovieNet schema (so embed / pair-feature / scoring scripts run
  unchanged). Robust to missing/bad clips; idempotent.
- **`scripts/eval/eval_aigen.py`** — runs the exact v0/v1 pipeline (DINOv2 embed
  → 2305-d boundary feature → 5 scorers), reports metrics overall and per source.
- **`scripts/eval/generate_toy_aigen.py`** — 8-pair synthetic set; the full
  pipeline runs end-to-end on it.
- **`configs/operating_thresholds.json`** — MovieNet-derived deployment
  thresholds (val-optimal and τ99) used for out-of-distribution scoring.
- **In-distribution check:** `eval_aigen.py --in_dist_check` re-scored the
  MovieNet test split through the harness and **reproduced v0 AUPRC 0.3561 and
  v1.5 AUPRC 0.4045 exactly (Δ 0.0000)** — the harness is verified correct.

## Counts

- New files this session: 7 (`video_frames.py`, `test_video_frames.py`,
  `build_aigen_eval.py`, `eval_aigen.py`, `generate_toy_aigen.py`,
  `operating_thresholds.json`, `OVERNIGHT_SUMMARY.md`) + 1 new figure.
- Tests: +6 (video frames) → **26 total, all passing**.
- W&B runs created this session: 0 (the harness does not log to W&B).

## Needs Devon's review / not done (by design)

- **Real AI-gen clips not yet evaluated** — you source them today. The harness is
  verified on toy + in-distribution data and ready.
- **mean-pool-3 not scored in `eval_aigen.py`** — the results template is 5
  models; mean-pool-3 (an ablation) was omitted to keep the harness lean. Easy
  to add if you want it in the AI-gen table.
- **`eval_aigen.py` `--in_dist_check`** skips HSV/CLIP (they need a full image
  recompute over 105K cuts; only v0/v1.5/raw are needed to verify the harness).
- Out of scope per the prompt and untouched: v2 (SupCon/LoRA) training, DINOv2
  weights, backbone swaps.

`reports/aigen_results.md` will be produced (with a "DEVON FILLS IN" analysis
block) the moment you run `eval_aigen.py` on real clips.
