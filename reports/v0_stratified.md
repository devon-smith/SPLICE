<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# v0 Stratified Evaluation

Slices of the v0 **test split** predictions (105,095 cuts, 64 movies, 6.90% positive). Compares the v0 logistic model against the raw DINOv2 cosine baseline. No retraining -- cached scores only.

## By class

Per-class count, mean predicted score, and hit-rate at the val-optimal threshold (recall for y=1, specificity for y=0). Hit-rates re-aggregate to the overall accuracy reported in v0_results.

| model | class | n | mean score | hit-rate@thr |
|---|---|---|---|---|
| logistic (v0) | y=1 inconsistent | 7,255 | 0.6912 | 0.5252 |
| logistic (v0) | y=0 consistent | 97,840 | 0.3020 | 0.9124 |
| raw DINOv2 cosine | y=1 inconsistent | 7,255 | 0.8600 | 0.4657 |
| raw DINOv2 cosine | y=0 consistent | 97,840 | 0.6619 | 0.8939 |

## Within-film vs cross-film

Every cut in the index is a pair of **adjacent shots from the same movie** by construction, so `same_movie` is uniformly true: there are 105,095 within-film cuts and 0 cross-film cuts. This axis is therefore degenerate for the current dataset and cannot be stratified. A cross-film evaluation would require synthesising cross-film shot pairs (out of scope for this phase); flagged for the team. All metrics below are within-film.

## By movie

Per-movie AUPRC (movie is the resampling unit, since cuts within a film are correlated). Mean is the macro average over movies; the 95% CI is a movie-level bootstrap (10,000 resamples). Pooled AUPRC is the micro average over all test cuts (the v0_results headline number).

| model | n movies | mean per-movie AUPRC | 95% CI | pooled AUPRC | pooled F1@thr |
|---|---|---|---|---|---|
| logistic (v0) | 64 | 0.3715 | [0.3405, 0.4024] | 0.3561 | 0.3881 |
| raw DINOv2 cosine | 64 | 0.2827 | [0.2576, 0.3075] | 0.2551 | 0.3217 |

## Shot-scale transitions (skipped)

MovieNet cinematic-style annotations (shot scale: long / full / medium / close-up / extreme-close-up) are **not present** in this data distribution -- the BaSSL `anno` files carry scene-boundary fields only (`video_id, shot_id, boundary_label, invideo_scene_id, ...`). The 5x5 scale-transition matrix is therefore skipped rather than approximated. It can be added later if the official MovieNet meta package is fetched.

## Analysis

The logistic model leads the raw-cosine baseline at the movie level too (mean per-movie AUPRC 0.372 vs 0.283), consistent with the pooled result. Per-movie AUPRC varies widely (range ~0.53 across test movies): continuity is far easier to score in some films than others, so the movie-level bootstrap CI -- not the point estimate -- is the honest summary. The within-film/cross-film axis is degenerate here because every labelled cut is an adjacent same-movie shot pair; a cross-film split is future work.
