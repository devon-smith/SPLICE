# Veo pilot diagnostic — investigating v2's lower Spearman

Spearman ρ vs Dispatch buckets (clean=1, drift=2, major=3): v0 +0.66, CLIP +0.61,
v1.5 +0.44, **v2 +0.39**. Before committing the "LoRA suppressed identity
features" narrative to the final report, this diagnostic tests four hypotheses:

1. **Real finding** — v2's LoRA adaptation suppressed identity signal.
2. **Frame selection** — boundary frames don't show drift Dispatch saw.
3. **Label-score mismatch** — models agree among themselves, all disagree with Dispatch.
4. **Pipeline bug** — wrong frames / wrong labels / wrong join.

Produced by `scripts/eval/aigen_diagnostic.py`. Tables and figures at
`reports/aigen_full_ranking_table.csv`, `reports/aigen_diagnostic_metrics.json`,
`reports/figures/a005_frame_inspection.png`.

## Task 3: Pipeline sanity check — VERIFIED, no bug

Spearman recomputed from scratch by re-loading scores from the CSVs and
joining on `pair_id`:

| model | ρ (this script) | ρ (`aigen_v2_pilot.md`) | match |
|---|--:|--:|:--:|
| v0_logistic | +0.6606 | +0.661 | ✓ |
| v1_5_MLP | +0.4404 | +0.440 | ✓ |
| v2_3seed | +0.3853 | +0.385 | ✓ |
| clip_cos | +0.6055 | +0.606 | ✓ |

- **Frame mapping confirmed**: `score_aigen_v2.py:131-132` loads
  `<pid>/left_img2.jpg` (last keyframe of left clip) and `<pid>/right_img0.jpg`
  (first keyframe of right clip). Same boundary protocol as v0/v1.5 use for
  MovieNet pairs (see `cuts.parquet` schema `left_img2_path` / `right_img0_path`).
- **Label mapping documented**: `clean=1, drift=2, major=3`. The other
  direction (clean=3, drift=2, major=1) would invert sign — but all four
  Spearman values are positive, so the direction is correct.
- **Join key correct**: pair_id strings match across v0/v1.5/v2 CSVs (all 10
  pairs present in both files).

**Verdict for Hypothesis 4: ruled out.** No pipeline bug.

## Task 1: A005 frame inspection

A005 Dispatch label: **"Identity drift — A rider lighter-skinned with beard, B
clearly Black and clean-shaven; green backpack in A absent in B; Manhattan
cabs and golden light hold."**

Figure: `reports/figures/a005_frame_inspection.png` — 2×3 grid showing both
clips' three keyframes. The scorer sees `left_img2` and `right_img0` (the
middle column boundary).

**Pixel-RMS analysis** (proxy for "how much does each clip change visually"):

| pair | bucket | L_int (img0→img2) | R_int (img0→img2) | boundary (L2→R0) | boundary / max(L_int,R_int) |
|---|---|--:|--:|--:|--:|
| A001 | drift | 0.408 | 0.313 | 0.324 | 0.79 |
| A002 | drift | 0.288 | 0.397 | 0.305 | 0.77 |
| A003 | clean | 0.162 | 0.261 | 0.232 | 0.89 |
| A004 | drift | 0.156 | 0.287 | 0.352 | 1.23 |
| **A005** | **major** | **0.264** | **0.292** | **0.282** | **0.97** |
| A011 | drift | 0.239 | 0.178 | 0.249 | 1.04 |
| A012 | drift | 0.165 | 0.239 | 0.254 | 1.06 |
| A013 | clean | 0.248 | 0.246 | 0.277 | 1.12 |
| A014 | drift | 0.272 | 0.299 | 0.327 | 1.10 |
| **A015** | **major** | 0.412 | 0.342 | **0.454** | **1.10** |

**Critical observation**: A005's boundary RMS (0.282) is **less than** its
within-clip motion of either clip (0.264, 0.292). The boundary discontinuity
A005 presents to our scorer is *no greater than* the visual change that
happens within either clip on its own. The identity-drift Dispatch labeled
("different rider") is presumably visible *across* the full clips — but is
**not concentrated at the boundary frames** the scorer sees.

By contrast, **A015's boundary RMS (0.454) exceeds either clip's internal
motion** (0.412, 0.342) by a noticeable margin. A015's drift *is* visible at
the boundary. That's why all three models catch A015 (v0 0.167, v1.5 0.036
[the v1.5 miss is its own pair-specific quirk], v2 0.163).

**Verdict for Hypothesis 2: SUPPORTED for A005 specifically.** The boundary
pair is not where A005's identity drift is concentrated. Score-against-Dispatch
correlation conflates "did the model detect a boundary discontinuity?" with
"did the model detect a clip-level identity drift?" — these are different
tasks, and A005's failure mode lives in the gap.

## Task 2: Cross-model ranking agreement

| pair_id | bucket | v0 score | v0 rank | v1.5 score | v1.5 rank | v2 score | v2 rank |
|---|---|--:|:--:|--:|:--:|--:|:--:|
| A015 | major | 0.167 | 1 | 0.036 | 6 | 0.163 | 1 |
| A005 | major | 0.066 | 3 | 0.243 | 1 | 0.041 | 5 |
| A011 | drift | 0.039 | 7 | 0.075 | 4 | 0.042 | 4 |
| A001 | drift | 0.075 | 2 | 0.052 | 5 | 0.038 | 6 |
| A012 | drift | 0.021 | 8 | 0.082 | 3 | 0.011 | 9 |
| A004 | drift | 0.050 | 5 | 0.013 | 8 | 0.027 | 8 |
| A014 | drift | 0.013 | 9 | 0.007 | 10 | 0.030 | 7 |
| A002 | drift | 0.010 | 10 | 0.008 | 9 | 0.126 | 2 |
| A003 | clean | 0.030 | 6 | 0.040 | 7 | 0.111 | 3 |
| A013 | clean | 0.004 | 11 | 0.004 | 11 | 0.001 | 10 |

(Full table: `reports/aigen_full_ranking_table.csv`.)

**Cross-model rank Spearman** (the killing question — do models agree among
themselves?):

| pair | ρ | p |
|---|--:|--:|
| v0 vs v1.5 | +0.55 | 0.10 |
| v0 vs v2 | +0.39 | 0.26 |
| **v1.5 vs v2** | **+0.15** | **0.68** |
| v0 vs CLIP | +0.30 | 0.40 |
| v1.5 vs CLIP | −0.03 | 0.93 |
| v2 vs CLIP | +0.26 | 0.47 |

**Models do not agree with each other.** v1.5 and v2 have essentially zero rank
correlation. Each model picks a different subset of Veo pairs to score high.

**Each major pair is caught by some models, missed by others:**

| | v0 | v1.5 | v2 |
|---|---|---|---|
| A005 (major) | rank 3 — partial catch | **rank 1 — caught** | rank 5 — miss |
| A015 (major) | **rank 1 — caught** | rank 6 — miss | **rank 1 — caught** |

v0 catches both modestly; v1.5 catches A005 and misses A015; v2 catches A015
and misses A005. **No model catches both major pairs strongly**, and each one
picks a different major to flag.

**Verdict for Hypothesis 3 (label-score mismatch): ruled out.** A label-score
mismatch would predict high inter-model agreement plus low correlation with
Dispatch. Here we see the opposite — models disagree with each other roughly
as much as they disagree with Dispatch. The models are not measuring "the same
thing different from what Dispatch measures"; they're each finding different
things.

### Leave-one-out: which pair drives v2's low Spearman?

| dropped pair | v0 ρ | v1.5 ρ | **v2 ρ** | Δ v2 |
|---|--:|--:|--:|--:|
| (full set) | +0.66 | +0.44 | **+0.39** | — |
| drop **A003 (clean)** | +0.76 | +0.60 | **+0.68** | **+0.29** |
| drop A011 (drift) | +0.65 | +0.45 | +0.45 | +0.07 |
| drop A005 (major) | +0.60 | +0.24 | +0.44 | +0.05 |

**v2's biggest single-pair Spearman penalty isn't A005 (the missed major) —
it's A003 (a false-alarm on a clean pair).** Dropping A003 lifts v2's ρ from
0.39 to 0.68, a +0.29 jump that brings v2 essentially into line with v0 (which
also benefits from dropping A003, +0.10). v2 simply scores A003 too high
(0.111, rank #3 out of 10).

Looking at A003's frames: "Strong continuity — same elderly man, tweed jacket,
briefcase, Tower Bridge backdrop; B slightly warmer and lighter dusk grade
than A but otherwise consistent." Lighting + colour-grade change between
clips. v2 plausibly fires on the lighting shift (a real boundary signal that
MovieNet rewards) where Dispatch — judging *identity continuity* — does not.

This is a different failure mode than "v2 missed identity drift in A005". v2
**also has a false-alarm pattern on lighting-only changes** at boundaries —
which is consistent with v2 being a *better* boundary-discontinuity detector,
not a worse one. Boundary discontinuity ≠ identity discontinuity.

### Bucket-level breakdown

| bucket | n | v0 mean | v1.5 mean | v2 mean | v0 above-median | v1.5 above-med | v2 above-med |
|---|--:|--:|--:|--:|---|---|---|
| clean | 2 | 0.017 | 0.022 | 0.056 | 0/2 | 1/2 | 1/2 |
| drift | 6 | 0.034 | 0.040 | 0.046 | 3/6 | 3/6 | 2/6 |
| major | 2 | 0.116 | 0.139 | 0.102 | 2/2 | 1/2 | 2/2 |

v0 and v1.5 are bucket-monotonic; v2's "clean mean" (0.056) > "drift mean"
(0.046) is entirely driven by A003's 0.111. A013 (the other clean pair) gets
v2 = 0.001 — the lowest of all 10 pairs. **v2 doesn't globally flag clean
pairs as discontinuous** — it specifically false-alarms on A003.

**Both major pairs end up above the v2 median.** v2 doesn't *globally* miss
major-bucket discontinuity; the bucket mean dips below v1.5's because A005
specifically lands at score 0.041 while v1.5 puts A005 at 0.243.

## Summary table — verdict per hypothesis

| Hypothesis | Evidence | Verdict |
|---|---|---|
| **#4 Pipeline bug** | Spearman recomputed from scratch matches; frame paths verified; label direction verified | **RULED OUT** |
| **#3 Label-score mismatch** | Cross-model ρ is *low* (v1.5 vs v2 = +0.15); models disagree among themselves | **RULED OUT** |
| **#2 Frame selection** | A005's boundary RMS (0.282) ≤ within-clip motion (0.264, 0.292); A005's identity drift isn't at the boundary | **SUPPORTED for A005** |
| **#1 Real finding (LoRA suppressed identity)** | v2 ranks lowest overall, but only by ~0.05 from v1.5; biggest single-pair penalty is A003 (false-alarm on lighting drift), not A005 | **PARTIALLY supported, but the "LoRA suppressed identity" framing is overreach** |

## What the diagnostic actually shows

1. **n=10 makes Spearman fragile.** A single-pair leave-one-out swings v2's
   ρ by +0.29. The rank ordering across models (v0 > v1.5 > v2) hinges on at
   most 2 of the 10 pairs.

2. **Boundary frames ≠ full-clip evidence.** A005's identity drift is
   distributed across the full clips, not concentrated at the boundary
   keyframes our scorer sees. Dispatch judged identity at the clip level.
   The scorer judged boundary-frame continuity. These are different tasks.
   Pixel-RMS analysis confirms: A005's boundary discontinuity is no greater
   than within-clip motion.

3. **v2 has a different error profile than v0/v1.5, but it isn't globally
   worse.** v2's specific failures are (a) false-alarming on A003 (clean
   continuity with lighting/grade shift) and (b) missing A005 (identity-only
   drift, not visible at the boundary). v2 also gets the second clean pair
   (A013) right and catches A015 strongly.

4. **No single model dominates Veo.** v0 catches A015 cleanly + A005
   partially; v1.5 catches A005 + misses A015; v2 catches A015 + misses A005.
   Different inductive biases produce different misses. With n=10 these
   differences amplify into Spearman differences that look more decisive than
   they are.

## Recommended framing for the final report's AI-gen section

**Replace the "LoRA suppressed identity features" claim** with something like:

> The Veo pilot does not show v2 carrying its MovieNet gain forward into AI-gen
> continuity flagging — v2 has the lowest Spearman correlation with Dispatch
> buckets of the four scorers tested (v0 +0.66, CLIP +0.61, v1.5 +0.44, v2
> +0.39). But the result is fragile and the conclusion should be limited:
>
> - Pipeline correctness verified (re-computed Spearman from scratch; frame
>   and label mappings checked).
> - With n=10, the cross-model Spearman ordering swings by ±0.29 when a single
>   pair is removed. v2's biggest single-pair penalty is a false alarm on a
>   "clean" pair (A003) with a lighting/colour shift at the boundary, not a
>   miss on the major-identity pair (A005).
> - For A005, pixel-RMS analysis confirms that the identity drift is
>   distributed across the full clips, not concentrated at the boundary
>   keyframes our scorer sees. This is a *task mismatch* between
>   boundary-frame scoring and clip-level identity judgement, not necessarily
>   a model deficiency. Earlier work (`aigen_rank_based.md`) already
>   recommended CLIP-percentile as the right Veo identity-drift flag for
>   exactly this reason.
> - **Cross-model ranks have low pairwise agreement (v1.5 vs v2 ρ = +0.15) —
>   no single SPLICE head dominates Veo, and the v0-best ordering on n=10 is
>   not strong evidence that v0 is structurally better for AI-gen than v2.**
>
> For deployment: keep v2 as the MovieNet-style real-film scorer; keep the
> existing CLIP-percentile flag for Veo identity drift. We cannot conclude
> from n=10 that v2 is worse than v0/v1.5 on AI-gen — only that it doesn't
> *win* on this small pilot, and that boundary-frame scoring is the wrong
> abstraction for clip-level identity evaluation.

This framing is more defensible than the original "LoRA suppressed identity
features" story, which is a plausible mechanism but rests on a single pair
(A005) where the underlying evidence (boundary-frame RMS) actually points to
frame selection, not feature suppression.

## Pipeline issues found

None requiring fixes. One process improvement worth recording: for future Veo
pilots, score at multiple boundary granularities (last+first, last-1+first+1,
etc.) and report the max — this would partially compensate for the
boundary-vs-clip-level task mismatch identified here. Out of scope for the
current submission.
