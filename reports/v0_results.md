<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# v0 Results — frozen DINOv2 + logistic cut-continuity scorer

**Milestone 2 · May 2026.** v0 establishes the floor: can a pretrained DINOv2
backbone plus a one-layer linear head score cross-shot visual continuity, and
does the learned head beat the obvious zero-training baselines?

## Setup

- **Data.** MovieNet, 318-movie scene-segmentation subset (BaSSL annotations).
  Keyframes from the HF mirror `ZhengPeng7/MovieNet` (the original BaSSL Aliyun
  link is dead). The cut index has **502,534 adjacent-shot cuts** (train/val/test
  = 295,448 / 101,991 / 105,095, movie-disjoint); **7.47% positive** (the cut
  crosses a scene boundary). 4,360 cuts whose BaSSL `boundary_label` is `-1`
  (the dataset's "ignore" marker) were dropped.
- **Encoder.** `facebook/dinov2-base` (ViT-B/14), frozen. 1,521,636 unique
  keyframes embedded once to a float16 HDF5 cache (53.9 min, 471 kf/s on one L4).
- **Feature.** Per cut, `boundary` mode: left shot's last keyframe (`img_2`) and
  right shot's first (`img_0`) → `[concat(eL,eR) | |eL−eR| | cos(eL,eR)]`, 2305-d.
- **v0 model.** `StandardScaler → LogisticRegression` (`class_weight=balanced`).
- **Baselines.** raw DINOv2 cosine · HSV colour-histogram χ² · CLIP ViT-L cosine.
- Operating threshold = F1-optimal on val. Metrics on the held-out test split.

## Results (test split, n = 105,095)

| Model | AUROC | AUPRC | F1@val-thr | F1@τ95 | Precision@val | Recall@val |
|---|---|---|---|---|---|---|
| **logistic (v0)** | **0.849** | **0.356** | **0.388** | 0.226 | 0.308 | 0.525 |
| raw DINOv2 cosine | 0.812 | 0.255 | 0.322 | 0.211 | 0.246 | 0.466 |
| HSV χ² | 0.763 | 0.195 | 0.282 | — | 0.203 | 0.463 |
| CLIP ViT-L cosine | 0.719 | 0.157 | 0.233 | — | 0.176 | 0.343 |

Positive base rate 7.47%. AUPRC is the headline metric (class imbalance).
τ95 is calibrated below; HSV/CLIP have no within-shot calibration (handoff
scopes τ95 to the DINOv2-based scorers).

## Threshold calibration

Within-shot keyframe pairs (`img_0/1/2` of one shot — the same continuous scene)
model "natural" consistent variation. Over 308,076 such pairs from val shots:

| Scorer | within-shot p50 | p95 (**τ95**) | p99 |
|---|---|---|---|
| logistic | 0.013 | **0.251** | 0.715 |
| raw DINOv2 cosine | 0.138 | **0.680** | 0.888 |

The within-shot distribution is concentrated near zero while the cut-level
distribution is far broader (`outputs/calibration/within_shot_vs_cut.png`),
which is exactly the premise the calibration relies on: a cut scoring above τ95
is more discontinuous than 95% of genuinely continuous frame pairs.

## Analysis

**The learned head beats frozen cosine.** Logistic regression on the 2305-d
feature reaches AUPRC 0.356 vs 0.255 for raw DINOv2 cosine (+40% relative) and
AUROC 0.849 vs 0.812. A single cosine collapses the comparison to one number;
letting a linear model weight the concatenated embeddings and their absolute
difference recovers substantially more signal — before any fine-tuning.

**The backbone matters more than its size.** Raw DINOv2-base cosine (0.812
AUROC) clearly beats CLIP ViT-L cosine (0.719), the largest model tried. DINOv2's
self-supervised dense features track scene appearance and lighting; CLIP's
image–text-aligned embedding is comparatively invariant to exactly the low-level
changes (colour grade, lighting, framing) that mark a scene cut — even plain HSV
colour histograms (0.763) outscore CLIP. This validates the choice of DINOv2.

**Where v0 fails.** At the F1-optimal operating point the v0 model catches ~53%
of true scene-boundary cuts at ~31% precision — useful as a review flag, far from
solved. The frozen embedding is a *global appearance* descriptor with no notion
of cinematographic continuity: within-scene shot/reverse-shot, cutaways and
framing changes produce large embedding gaps it cannot distinguish from a true
scene change. F1@τ95 (0.226) sits below F1@val (0.388) because τ95 is a
precision-oriented calibrated cutoff, not an F1-optimal one — a different, more
conservative operating point by design.

**Next (v1/v2).** v1: swap the linear head for a 2-layer MLP to capture
non-linear interactions; ablate boundary frames vs mean-pooled 3-keyframe shot
embeddings, and same-movie vs cross-film negatives. v2: supervised contrastive
loss + projection head, and partial (LoRA) fine-tuning of DINOv2 so the backbone
itself learns continuity-relevant features — the frozen-feature ceiling is the
main limiter here.

## Calibration analysis

The within-shot τ95 threshold gives F1 0.226 versus 0.388 at the F1-optimal val
threshold (logistic). This is not a failure mode — τ95 is a *different operating
point*, and the gap is fully explained.

τ95 means "more discontinuous than 95% of genuinely-continuous (within-shot)
frame pairs." But consecutive keyframes of one shot are extremely similar, so the
within-shot score distribution is far tighter than the cut-level distribution
(`reports/figures/score_distributions.png`). The 95th percentile of that tight
distribution lands *low* in the cut-level distribution:

**logistic (v0)** — test precision/recall/F1 at each operating point:

| operating point | threshold | cut-level percentile | precision | recall | F1 |
|---|---|---|---|---|---|
| τ95 | 0.251 | 51.4 | 0.128 | 0.923 | 0.225 |
| τ99 | 0.715 | 85.9 | 0.288 | 0.565 | 0.382 |
| val-optimal | 0.746 | 87.7 | 0.308 | 0.525 | 0.388 |

**raw DINOv2 cosine:**

| operating point | threshold | cut-level percentile | precision | recall | F1 |
|---|---|---|---|---|---|
| τ95 | 0.680 | 47.4 | 0.119 | 0.909 | 0.211 |
| τ99 | 0.888 | 84.6 | 0.237 | 0.501 | 0.322 |
| val-optimal | 0.897 | 86.1 | 0.246 | 0.466 | 0.322 |

τ95 sits at only the ~51st percentile of cut-level scores — it flags roughly
*half of all cuts*, giving recall 0.92 at precision 0.13. So τ95 is a
**high-recall operating point**, not a conservative one; the low F1 is the
precision cost of near-total recall.

**Key result: τ99 ≈ the val-optimal threshold.** Calibrating at the 99th
percentile of within-shot variation gives 0.715 (logistic) versus the
label-derived F1-optimal 0.746 — F1 0.382 vs 0.388, essentially identical; raw
cosine matches too (0.888 vs 0.897, F1 0.322 both). The within-shot calibration
framework is therefore sound — the quantile is simply a precision/recall dial.

**Recommended operating points.** For the actual use case — flagging cuts for
human review — **τ95** is the right default: 92% recall catches almost every true
inconsistency and a reviewer filters the false positives. For a balanced or
reported number, use **τ99**, which reproduces the F1-optimal threshold *without
ever using labels* — a label-free calibration of the headline operating point.

## Reproduce

```bash
python scripts/prep/build_cut_index.py
python scripts/prep/embed_keyframes.py
python scripts/prep/build_pair_features.py --mode boundary
python scripts/train/v0_logistic.py
python scripts/eval/calibrate_threshold.py
```

Artifacts: `outputs/v0/` (model, scores, results, ROC/PR curves),
`outputs/calibration/` (τ95 JSON + distribution figure).
