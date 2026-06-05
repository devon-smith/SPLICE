<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# Veo pilot — v2 LoRA 3-seed ensemble scoring (qualitative)

The same 10 Veo continuous-action pairs scored through earlier models, now also
through the v2 LoRA 3-seed ensemble (mean of sigmoid'd logits across seeds 0,
1, 2 best-by-val checkpoints). The point: does v2's MovieNet macro AP gain
carry over to the OOD AI-gen pilot, or does the LoRA-tuned backbone specialize
*away* from the kind of discontinuity Veo produces?

Produced by `scripts/eval/score_aigen_v2.py`; CSV at
`reports/aigen_v2_pilot.csv`, metrics JSON at
`reports/aigen_v2_pilot_metrics.json`.

**Caveat in bold: n=10, all single-class (intended y=0 = continuous-action),
one generator (Veo), Dispatch buckets are qualitative human judgement made
before any scoring — this is signal, not significance.**

## Per-pair scores (sorted by v2 3-seed mean)

| pair | bucket | **v2 (3-seed mean ± std)** | v1.5 | v0 | clip cos |
|---|---|--:|--:|--:|--:|
| A015 | **major** | **0.163 ± 0.085** | 0.036 | 0.167 | 0.231 |
| A002 | drift | 0.126 ± 0.058 | 0.008 | 0.010 | 0.111 |
| A003 | clean | 0.111 ± 0.025 | 0.040 | 0.030 | 0.105 |
| A011 | drift | 0.042 ± 0.013 | 0.075 | 0.039 | 0.049 |
| A005 | **major** | **0.041 ± 0.037** | 0.243 | 0.066 | 0.272 |
| A001 | drift | 0.038 ± 0.012 | 0.052 | 0.075 | 0.091 |
| A014 | drift | 0.030 ± 0.009 | 0.007 | 0.013 | 0.121 |
| A004 | drift | 0.027 ± 0.009 | 0.013 | 0.050 | 0.121 |
| A012 | drift | 0.011 ± 0.005 | 0.082 | 0.021 | 0.097 |
| A013 | clean | 0.001 ± 0.001 | 0.004 | 0.004 | 0.095 |

Per-seed std is small (≤0.085 on the worst pair, ≤0.013 on most), so the
ensemble mean is a stable summary — the noise here is not seed noise.

## Spearman vs Dispatch buckets (clean=0, drift=1, major=2)

| model | Spearman ρ | p-value |
|---|--:|--:|
| **v0 logistic** | **+0.661** | 0.038 |
| **CLIP cosine** | +0.606 | 0.064 |
| **v1.5 MLP** | +0.440 | 0.203 |
| **v2 LoRA (3-seed)** | **+0.385** | **0.272** |

**v2 is the *worst* of the four scorers on this pilot.** v0 (frozen DINOv2 +
logistic) and CLIP cosine each outperform v2 substantially. The ordering reverses
the MovieNet ranking: on MovieNet macro AP, v2 ≫ v1.5 ≫ v0; on Veo, v0 > v1.5 > v2.

## Bucket-level means

| bucket | n | v0 | v1.5 | **v2** |
|---|--:|--:|--:|--:|
| clean | 2 | 0.017 | 0.022 | **0.056** |
| drift | 6 | 0.034 | 0.040 | **0.046** |
| major | 2 | 0.116 | 0.139 | **0.102** |

v0 and v1.5 are monotonic (clean < drift < major). **v2 is not monotonic** at
the bucket level: its "clean" mean (0.056) is higher than its "drift" mean
(0.046). The major-bucket mean is still highest, but only modestly — v2 has
lost the cleanly-ordered structure that v0 and v1.5 had on this pilot.

## What did v2 miss?

The single biggest miss is **A005** ("Identity drift — A rider lighter-skinned
with beard, B clearly Black and clean-shaven; green backpack in A absent in B;
Manhattan cabs and golden light hold"). v0 ranks A005 at #1 (score 0.066);
v1.5 also catches it (0.243); CLIP nails it (cosine 0.272). v2 puts A005 at
**rank #5 (score 0.041)** — well below the median Veo pair.

A005's discontinuity is **identity-semantic**: same composition, same lighting,
same setting — but a different person. The features that flag this are
high-level semantics (face identity, skin tone, build), exactly what CLIP is
trained to encode. DINOv2's pretrained features carry some identity signal
(why v0 still catches A005), but **v2's LoRA fine-tuning on MovieNet
continuity appears to have suppressed that signal** — MovieNet's positive
class is cross-scene cuts where identity differences are correlated with
*lighting and setting* changes, so the LoRA layers learned to weight those
features. When Veo holds setting/lighting but changes identity, v2 has nothing
to fire on.

The opposite miss happens too: v2 puts **A002 (drift)** at rank #2 and **A003
(clean)** at rank #3, both above multiple drift pairs. v2's discriminator is
producing high-variance, low-signal predictions on out-of-distribution video.

## Read

v2 does not carry its MovieNet macro AP gain forward into the Veo pilot —
its Spearman with Dispatch buckets (+0.39) is the lowest of the four scorers.
But before concluding "LoRA suppressed identity features", we ran a full
diagnostic at `reports/aigen_diagnostic.md` which materially changes the
framing. The headline findings of that diagnostic:

1. **Pipeline is clean** — Spearman re-computed from scratch matches
   exactly; frame and label mappings verified.
2. **Cross-model rank Spearman is low** — v1.5 vs v2 ρ = +0.15 (essentially
   zero). The models *do not agree among themselves*. Each picks a different
   subset of Veo pairs to score high. v0 catches A015, v1.5 catches A005, v2
   catches A015. No model catches both major pairs strongly.
3. **Leave-one-out: v2's biggest single-pair penalty is A003, not A005.**
   Dropping A003 (a clean pair v2 scores high) lifts v2's ρ from 0.39 to
   0.68. The "v2 missed A005" framing was misleading — v2's bigger problem on
   this set is a *false alarm* on A003 (a lighting / colour-grade shift at the
   boundary).
4. **A005's identity drift isn't concentrated at the boundary** — pixel-RMS
   between A005's boundary frames (left_img2 + right_img0) is 0.28, *less*
   than the within-clip motion (0.26-0.29). Our scorer sees a boundary that
   isn't visibly more discontinuous than two random adjacent frames within
   either clip. Dispatch presumably judged identity at clip level — a
   different task from frame-pair boundary continuity.

**Revised conclusion**: with n=10, the cross-model Spearman ordering swings by
±0.29 on single-pair removal. The "v2 is structurally worse on AI-gen" claim
is **not supportable from this evidence**. What we can say honestly:

- v2 does not *win* on this small pilot, in contrast to its clean MovieNet win.
- A real **task mismatch** exists: our boundary-frame scorer cannot detect
  identity drift that's distributed across full clips (a known property of the
  Veo failure mode, not a v2-specific deficiency).
- For Veo identity-drift deployment, the existing CLIP-percentile flag (from
  `aigen_rank_based.md`) remains the recommended tool — that recommendation
  was made independently of v2 and is unchanged.

**For the final report, the AI-gen section should say "the MovieNet gain
doesn't transfer to this small pilot, the failure mode looks like
task-mismatch rather than v2 regression, n=10" and leave it there.** The
"LoRA suppressed identity features" story is a plausible mechanism but rests
on a single pair where the underlying evidence (pixel-RMS) points to frame
selection, not feature suppression.
