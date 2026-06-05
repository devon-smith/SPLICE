<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# v1.5 Error Analysis (Phase 3, P4)

v1.5 (seed-2 sound MLP) on the test split, decision threshold 0.754. Of 105,095 test cuts: 5,863 false positives (6.0% of negatives) and 3,724 false negatives (51.3% of positives). The CSVs list the most confident 50 of each with all five model scores and keyframe paths; this section describes the *full* error sets. Description only -- qualitative interpretation is left to the team.

## Error categories

### False positives (y=0 scored above threshold)  (n = 5,863)

- distinct movies: 64 of 64; top contributors: tt1375666 (189), tt0479884 (188), tt0120689 (185)
- single-movie max share: 189/5863 = 3.2%
- v1.5 score: min 0.754, median 0.841, mean 0.849, max 0.998
- mean cosine similarity of the cut: 0.115
- distance above threshold (0.754): median 0.087

### False negatives (y=1 scored below threshold)  (n = 3,724)

- distinct movies: 64 of 64; top contributors: tt1707386 (162), tt0117060 (151), tt0440963 (150)
- single-movie max share: 162/3724 = 4.4%
- v1.5 score: min 0.000, median 0.507, mean 0.469, max 0.754
- mean cosine similarity of the cut: 0.198
- distance below threshold (0.754): median 0.248; 21.2% are within 0.1 of the threshold (near-misses)

## Well-handled, for contrast

- top-20 correct positives: mean v1.5 score 0.998
- top-20 correct negatives: mean v1.5 score 0.000

## Open question for the team

Which of these errors are genuine model failures versus MovieNet labelling noise? The cut index had ~0.86% boundary_label/scene-id disagreement at the source; some confident false positives/negatives may be mislabelled cuts. The grid figures (`v1_error_grid_fp.png`, `v1_error_grid_fn.png`) and the CSVs are for that qualitative review.
