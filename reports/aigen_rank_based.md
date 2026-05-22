# AI-Gen Rank-Based Flagging

Each Veo pilot pair is ranked by percentile against the MovieNet within-scene (y=0) test distribution (n=97,840). A pair is **flagged** when that percentile exceeds **95** for any of v1.5, v0 logistic, or CLIP cosine. This sidesteps absolute calibration. Source: `outputs/aigen_eval/results/rank_based_scores.csv`.

## Per-pair percentiles (vs MovieNet within-scene)

| pair | bucket | v1.5 %ile | v0 %ile | clip %ile | max %ile | flagged? |
|---|---|--:|--:|--:|--:|:--:|
| A005 | major | 56.6 | 22.5 | 56.9 | 56.9 | no |
| A015 | major | 24.0 | 42.4 | 38.7 | 42.4 | no |
| A012 | drift | 35.1 | 7.6 | 3.5 | 35.1 | no |
| A011 | drift | 33.6 | 14.2 | 0.7 | 33.6 | no |
| A001 | drift | 28.6 | 24.9 | 2.9 | 28.6 | no |
| A003 | clean | 25.2 | 11.1 | 4.3 | 25.2 | no |
| A004 | drift | 14.3 | 18.0 | 6.4 | 18.0 | no |
| A002 | drift | 11.3 | 3.0 | 5.1 | 11.3 | no |
| A014 | drift | 9.7 | 4.2 | 6.4 | 9.7 | no |
| A013 | clean | 7.1 | 0.9 | 3.3 | 7.1 | no |

## Flag evaluation at the 95th-percentile threshold

| flag | major caught (/2) | clean cleared (/2) | drift flagged (/6) |
|---|:--:|:--:|:--:|
| flag_v1.5 | 0 | 2 | 0 |
| flag_v0 | 0 | 2 | 0 |
| flag_clip | 0 | 2 | 0 |
| flag_ensemble | 0 | 2 | 0 |

**At the 95th-percentile threshold no flag fires on any pair.** Every Veo pair — including the major-identity failures — ranks below the 95th percentile of MovieNet within-scene cuts: the worst AI-gen identity drift is still visually *more* continuous than 5% of ordinary real-film within-scene cuts. The binary p95 rank flag cannot discriminate because nothing reaches it — the distribution shift (see `v1_distribution_shift.md`) defeats the absolute cutoff, not the ranking idea itself.

## Does the percentile *ordering* still separate the buckets?

The binary p95 flag is empty, but the percentile ranks still order the pairs. The useful question: can a *single threshold* on a percentile column isolate the two major-identity pairs from all eight other pairs (clean + drift)? `non-major max` is the highest-ranked non-major pair — the major pairs must clear it.

| flag model | clean max | major min | non-major max | isolates major? | margin |
|---|--:|--:|--:|:--:|--:|
| v1.5_percentile | 25.2 | 24.0 | 35.1 (A012) | no | -11.1 |
| v0_percentile | 11.1 | 22.5 | 24.9 (A001) | no | -2.4 |
| clip_percentile | 4.3 | 38.7 | 6.4 (A004) | **yes** | +32.3 |
| max_percentile | 25.2 | 42.4 | 35.1 (A012) | **yes** | +7.4 |

## Recommendation

The p95 rank flag fires on nothing, so the four binary flags are tied at useless — the **threshold**, not the ranking idea, is wrong. The usable signal is in the percentile *ordering*, and there it is decisive. **clip_percentile** is the clear winner: a single threshold isolates *both* major-identity pairs from all eight other pairs. The major pairs rank at the 39th percentile and above; every clean and drift pair ranks at or below the 6th (A004) — a 32-point margin. A rank flag at ~23 (anywhere in that window, not 95) catches A005 and A015 and nothing else: 2/2 major, 2/2 clean cleared, 0/6 drift over-flagged. That CLIP — the weakest MovieNet scorer — gives the cleanest AI-gen identity-failure flag is consistent with every prior experiment: CLIP captures the semantic identity drift DINOv2 misses. `max_percentile` also isolates both major pairs but on a tighter 7-point margin. 

**Single recommendation:** flag a Veo continuous-action pair when its **clip** percentile against the MovieNet within-scene distribution exceeds ~23. Rank-against-MovieNet is the right idea; the 95th-percentile cutoff is simply far too high for the AI-gen distribution — the entire Veo set sits below it. Pair this ordering with the same-domain Veo calibration from Action 1 for a deployable flag. Caveat in bold: n=10, single class, one generator — the threshold is provisional and must be refit as Veo data grows.