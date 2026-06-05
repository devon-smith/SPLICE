<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# AI-Gen Calibration

MovieNet operating thresholds do not transfer to AI-gen footage. This gives domain-specific percentile tables for all six scorers, so a Veo score can be judged against *other Veo continuous-action pairs* rather than against MovieNet. **The Veo side is n=10, single-class, one generator — these baselines are provisional.** Source: `configs/aigen_calibration.json`.

## Veo continuous-action baseline (n=10, all y=0)

The score above which a Veo continuous-action pair is unusual *relative to other Veo continuous-action pairs*.

| model | p50 | p75 | p90 | p95 | p99 |
|---|--:|--:|--:|--:|--:|
| v1.5 | 0.038 | 0.069 | 0.098 | 0.171 | 0.228 |
| v0_logistic | 0.034 | 0.062 | 0.084 | 0.126 | 0.159 |
| mean_pool_3 | 0.015 | 0.083 | 0.111 | 0.115 | 0.119 |
| raw_dino_cosine | 0.259 | 0.288 | 0.323 | 0.428 | 0.513 |
| hsv_chisq | 0.256 | 0.382 | 0.428 | 0.470 | 0.503 |
| clip_cosine | 0.108 | 0.121 | 0.235 | 0.253 | 0.268 |

## MovieNet within-scene (y=0) — reference

| model | p50 | p75 | p90 | p95 | p99 |
|---|--:|--:|--:|--:|--:|
| v1.5 | 0.184 | 0.440 | 0.663 | 0.780 | 0.920 |
| v0_logistic | 0.222 | 0.485 | 0.722 | 0.824 | 0.930 |
| mean_pool_3 | 0.187 | 0.448 | 0.716 | 0.829 | 0.943 |
| raw_dino_cosine | 0.679 | 0.811 | 0.901 | 0.940 | 0.989 |
| hsv_chisq | 0.398 | 0.540 | 0.688 | 0.787 | 0.953 |
| clip_cosine | 0.257 | 0.315 | 0.369 | 0.403 | 0.483 |

## MovieNet scene-boundary (y=1) — reference

| model | p50 | p75 | p90 | p95 | p99 |
|---|--:|--:|--:|--:|--:|
| v1.5 | 0.745 | 0.902 | 0.963 | 0.979 | 0.993 |
| v0_logistic | 0.763 | 0.898 | 0.955 | 0.972 | 0.988 |
| mean_pool_3 | 0.791 | 0.923 | 0.969 | 0.982 | 0.994 |
| raw_dino_cosine | 0.888 | 0.951 | 0.988 | 1.007 | 1.034 |
| hsv_chisq | 0.621 | 0.784 | 0.907 | 0.962 | 0.997 |
| clip_cosine | 0.326 | 0.384 | 0.439 | 0.479 | 0.560 |

## Where do the major-identity pairs land?

Dispatch flagged A005 and A015 as major identity failures. Per model: the pair's score and its percentile rank *within the 10 Veo pairs*.

| model | A005 score | A005 Veo-%ile | A015 score | A015 Veo-%ile |
|---|--:|--:|--:|--:|
| v1.5 | 0.243 | 95 | 0.036 | 45 |
| v0_logistic | 0.066 | 75 | 0.167 | 95 |
| mean_pool_3 | 0.012 | 35 | 0.110 | 85 |
| raw_dino_cosine | 0.534 | 95 | 0.166 | 15 |
| hsv_chisq | 0.342 | 65 | 0.269 | 55 |
| clip_cosine | 0.272 | 95 | 0.231 | 85 |

## Interpretation

With n=10 the Veo percentiles are coarse — p90 is effectively the 9th of 10 scores, p99 the near-maximum — so a 'flag above Veo p90' rule can only ever fire on the top one or two pairs. Within that limit: A005 is extreme for v1.5 (score 0.243, 95th Veo percentile) and for CLIP cosine (95th), but only mid-pack for v0 logistic (75th). A015 is the mirror image: top of the Veo range for v0 logistic (95th) and high for CLIP (85th), but only 45th for v1.5 — v1.5 reads A015 as an ordinary Veo pair.

## Recommended thresholds for AI-gen flagging

No single model's Veo-p90 threshold flags both A005 and A015 — v1.5 catches A005, v0 catches A015, CLIP comes closest to both. A combination is needed. For deployment, two regimes:

- **Same-domain (relative) flag:** flag a Veo continuous-action pair if its score exceeds the `veo_continuous_action` **p90** for the chosen model (v1.5 p90 = 0.098, CLIP p90 = 0.235, v0 p90 = 0.084). This is provisional at n=10 and should be refit once more Veo pairs exist.
- **Cross-domain (rank) flag:** rank each Veo pair against the 98k MovieNet within-scene cuts — a far more stable reference than 10 pilot pairs. This is Action 2; it is the more reliable flag until the Veo baseline has a real sample size.

Bottom line: the MovieNet thresholds (val ~0.75, τ99 ~0.65) must not be used on AI-gen footage; use the `veo_continuous_action` percentiles here for same-domain flagging, and prefer the rank-based flag for anything deployed.