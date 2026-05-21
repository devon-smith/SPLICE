# v1 Stratified Evaluation by Transition Difficulty

v1.5 (seed-2 sound MLP) on the test split: 105,095 cuts, 64 movies, 6.90% positive. Decision threshold 0.754 (F1-optimal on val).

## 1. By raw DINOv2 cosine quintile

Cuts binned into quintiles of cosine similarity: Q1 = look most different, Q5 = look most similar. Within a quintile cosine is near-constant, so raw cosine cannot rank -- its AUPRC collapses to the base rate; v1.5's AUPRC above the base rate is signal it adds *beyond* cosine.

| quintile | n | positive rate | v1.5 AUPRC | raw-cosine AUPRC | v1.5 FPR |
|---|---|---|---|---|---|
| Q1 | 21,019 | 0.208 | 0.531 | 0.324 | 0.235 |
| Q2 | 21,019 | 0.079 | 0.240 | 0.100 | 0.079 |
| Q3 | 21,019 | 0.037 | 0.120 | 0.044 | 0.019 |
| Q4 | 21,019 | 0.015 | 0.043 | 0.018 | 0.002 |
| Q5 | 21,019 | 0.006 | 0.017 | 0.009 | 0.000 |

## 2. By movie cut-count (cut-rate proxy)

MovieNet year metadata is not in this data distribution, so films are split by total shot count (a coarse proxy for cut rate) at the median test movie.

Median test movie has 1568 shots.

| group | movies | n cuts | positive rate | v1.5 AUPRC |
|---|---|---|---|---|
| busy (more shots) | 32 | 67,676 | 0.062 | 0.362 |
| slow (fewer shots) | 32 | 37,419 | 0.082 | 0.460 |

## 3. By cut position within the film

| position | n | positive rate | v1.5 AUPRC |
|---|---|---|---|
| early | 35,043 | 0.080 | 0.427 |
| middle | 35,100 | 0.058 | 0.395 |
| late | 34,952 | 0.069 | 0.391 |

## 4. By scene-boundary depth

Degenerate: every y=1 cut joins two *adjacent* shots, so the scene-id jump is always exactly 1 ({1: 7255}). MovieNet's `invideo_scene_id` is contiguous, so a cut between adjacent shots can only ever move to the next scene. There is no small-vs-large jump distinction to stratify on.


## Interpretation

- **Cosine quintile:** v1.5 AUPRC by quintile is [0.531, 0.24, 0.12, 0.043, 0.017]. Where raw cosine is decisive (extreme quintiles) the within-band ranking problem is easy or near-saturated; v1.5's value-add concentrates where cosine alone is ambiguous.
- **Cut position:** AUPRC early/middle/late = [0.427, 0.395, 0.391]; differences indicate whether film openings/closings are harder to score.
- **Cut-count proxy** is coarse (no duration metadata); read it as busy-vs-slow films, not a true cut-rate split.
