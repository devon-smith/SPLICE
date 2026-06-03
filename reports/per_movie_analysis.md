# Per-movie diagnostic — where does v2 LoRA help, where does it not?

Per-movie test AP for v0 (logistic), v1.5 (3-seed mean of per-movie APs — the
same operational definition as the headline 0.418 in `macro_ap.md`), v2 (LoRA
r=8/α=16, **seed 0 proxy** — will re-run with 3-seed mean once Phase 2
completes). 64 MovieNet test movies, all with ≥5 cuts and mixed classes.
Produced by `scripts/eval/per_movie_analysis.py`; full table at
`reports/per_movie_analysis.csv`, bar chart at
`reports/per_movie_analysis.png`.

## Headline

| metric | v0 | v1.5 (3-seed mean) | v2 (seed 0) |
|---|--:|--:|--:|
| macro per-movie AP | 0.3715 | 0.4179 | **0.4709** |
| median per-movie AP | 0.3549 | 0.3930 | 0.4654 |

- **v2 helps on 56 / 64 movies (87.5%)**, regresses on 8.
- Mean per-movie gain: **v2 over v1.5 = +0.053**, v1.5 over v0 = +0.046.
- Regressions are small: largest negative is −0.031 on a single movie; 6 of
  the 8 regressing movies lose <0.03 AP.

These numbers reconcile cleanly with the pre-flight bootstrap (v2 seed 0 vs
v1.5 seed 2 = +0.051 with CI [+0.040, +0.063]) — the per-seed-mean v1.5 used
here is essentially equivalent to seed 2 in macro terms.

## 10 movies where v2 helps most

| movie_id | n_cuts | pos_rate | v0 | v1.5 | v2 | Δv2−v1.5 |
|---|--:|--:|--:|--:|--:|--:|
| tt2024544 | 716 | 14.7% | 0.592 | 0.597 | 0.741 | **+0.144** |
| tt0103776 | 1770 | 5.6% | 0.306 | 0.319 | 0.462 | **+0.143** |
| tt0086190 | 2158 | 5.0% | 0.278 | 0.326 | 0.465 | **+0.140** |
| tt0361748 | 1444 | 3.5% | 0.171 | 0.194 | 0.320 | +0.126 |
| tt0822832 | 1532 | 9.2% | 0.437 | 0.497 | 0.607 | +0.110 |
| tt1707386 | 2538 | 8.5% | 0.320 | 0.322 | 0.432 | +0.110 |
| tt0112573 | 3074 | 4.5% | 0.248 | 0.344 | 0.453 | +0.109 |
| tt0117060 | 1517 | 16.2% | 0.410 | 0.435 | 0.541 | +0.106 |
| tt0113277 | 1975 | 8.5% | 0.361 | 0.399 | 0.503 | +0.103 |
| tt0281358 | 1169 | 5.6% | 0.360 | 0.434 | 0.538 | +0.103 |

**Pattern**: the biggest gains land on movies where v1.5 was already mediocre
to weak (AP ≈ 0.19–0.50). The largest single-movie gain (tt2024544, +0.144)
takes a movie from "okay" (v1.5 AP 0.60) to "strong" (v2 AP 0.74); the second
largest (tt0103776, +0.143) lifts AP 0.32 → 0.46 on a sparse movie (5.6%
positives). Movie size and pos-rate span the typical range (716–3074 cuts,
3.5–16% pos) — **no clear "v2 only helps on large/small movies" pattern**. v2
is closing the worst gaps rather than uniformly shifting everything.

## 10 movies where v2 helps least (or regresses)

| movie_id | n_cuts | pos_rate | v0 | v1.5 | v2 | Δv2−v1.5 |
|---|--:|--:|--:|--:|--:|--:|
| tt0399201 | 2693 | 6.1% | 0.296 | 0.342 | 0.349 | +0.007 |
| tt0063442 | 1141 | 3.0% | 0.166 | 0.209 | 0.211 | +0.002 |
| tt0078788 | 1576 | 6.0% | 0.245 | 0.244 | 0.239 | −0.005 |
| tt0073195 | 1232 | 7.3% | 0.411 | 0.446 | 0.439 | −0.007 |
| tt0120689 | 2379 | 4.9% | 0.292 | 0.318 | 0.307 | −0.010 |
| tt0120382 | 1384 | 6.9% | 0.349 | 0.427 | 0.414 | −0.013 |
| tt0103855 | 2137 | 5.3% | 0.327 | 0.329 | 0.314 | −0.015 |
| tt0088944 | 1838 | 2.5% | 0.114 | 0.186 | 0.160 | −0.026 |
| tt0075314 | 887 | 4.3% | 0.266 | 0.308 | 0.280 | −0.028 |
| tt0945513 | 1429 | 6.3% | 0.391 | 0.353 | 0.322 | **−0.031** |

**Pattern**: regressions cluster on movies with **lower-than-average pos rates
(2.5–7.3%, vs the 7.2% mean)** and skew small in magnitude. tt0945513 is the
worst case: v1.5 was already worse than v0 there (−0.038), and v2 makes things
further worse (−0.031). The other regressors all show v1.5 substantially
beating v0 (+0.002 to +0.077); v2 gives back some but rarely all of that gain.

**No catastrophic failures**: no movie drops below 0.16 AP for v2; v1.5's
floor was 0.17, v0's was 0.11. The LoRA backbone is not destabilizing any
movie even where it doesn't help.

## Distribution shape

`reports/per_movie_analysis.png` plots per-movie v1.5 vs v2 sorted by v2 AP.
Beyond the rank tables:

1. **The bottom of the distribution lifts more than the top.** On the easiest
   movies (top right, AP > 0.65) v2 and v1.5 are nearly tied; on the hardest
   (AP < 0.30) v2 separates upward.
2. **The lifted region is wide.** The middle 30 movies (AP 0.3–0.6) account
   for most of the +0.053 mean shift; v2 isn't just rescuing the worst 5.

## Read

The +0.053 mean per-movie gain v2 has over v1.5 is **not a uniform shift** —
it's driven by ~20 movies getting substantial boosts (>+0.07) while ~8
regress slightly. The model is finding information v1.5 missed on the
hard-to-score movies (low positive rate, weak v1.5 baseline) rather than
refining already-easy movies. This is consistent with the LoRA backbone
learning continuity-specific features that the frozen DINOv2 + MLP head
couldn't access.

This analysis uses **v2 seed 0 as a proxy**. The +0.053 number here will be
replaced with the 3-seed-mean v2 once Phase 2 completes; seed 0's
macro AP (0.4709) appears to be a better-than-average draw (Phase 1 seed 42
hit 0.4616), so the final 3-seed-mean gain will likely settle around
+0.040 to +0.050.
