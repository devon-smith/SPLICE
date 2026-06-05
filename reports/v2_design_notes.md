<!-- AI-USE: This report narrative is AI-generated analysis of run artifacts with Claude via Claude Code. Numeric values come from the referenced scripts/artifacts. -->

# v2 Design Notes — CLIP+DINOv2 Fusion Scaffold

Status: **scaffold built and verified, no head trained.** This note records what
the fusion infrastructure is, the verification result, and an honest read on
whether fusion is worth pursuing as a v2 head architecture. It is not a
commitment to an architecture.

## What was built

A frozen-feature fusion scaffold, parallel to the existing v0/v1 DINOv2 pipeline:

- **`src/data/pairs.py` → `build_fused_pair_features`.** The standard
  `[concat | abs-diff | cos]` pair feature, built once from DINOv2 boundary
  embeddings and once from CLIP boundary embeddings, concatenated:
  `[ DINOv2 2305-d | CLIP 2305-d ] → 4610-d`.
- **`scripts/prep/embed_keyframes_clip.py`.** Caches CLIP image embeddings to an
  HDF5 with the same schema as the DINOv2 cache, so `load_embeddings` reads both
  identically. Idempotent and incremental.
- **`scripts/prep/build_fused_features.py`.** Loads both caches, emits the
  4610-d fused feature per cut, with a cosine-similarity sanity summary.

**Dimension note.** The task spec assumed a 512-d CLIP backbone (1537-d CLIP
feature, 3842-d fused). The project's CLIP baseline is **ViT-L/14**, whose
`.image_embeds` are **768-d** — keeping the fusion consistent with the existing
CLIP cosine baseline and the Experiment-1 ensemble. The CLIP half is therefore
2305-d and the fused feature **4610-d**. A 512-d backbone would shrink this but
would mean two different CLIP models in the codebase; not worth it.

## Verification (100-cut subset)

CLIP-cached 660 keyframes (100 stratified MovieNet test cuts + the 10 Veo
pairs); built fused features for the 100 MovieNet cuts. Pipeline runs
end-to-end, 100×4610, no NaN. Cosine-similarity sanity (within-scene y=0 should
exceed cross-scene y=1):

| half | cos y=0 (within-scene) | cos y=1 (cross-scene) | gap |
|---|--:|--:|--:|
| DINOv2 | 0.348 | 0.154 | **0.194** |
| CLIP   | 0.752 | 0.675 | **0.077** |

Both halves separate the classes in the right direction. Note the DINOv2 half
separates **~2.5× more strongly** than the CLIP half — consistent with the known
MovieNet AUPRC gap (raw DINOv2 cosine 0.255 vs CLIP cosine 0.157). On real film,
CLIP is the weaker signal; its value is elsewhere (see below).

## Is fusion viable as a v2 head architecture?

**The pipeline is viable; the *payoff* is unproven and the prior is mixed.**

Arguments for trying it:

- CLIP carries a genuinely complementary signal. On the Veo pilot (Experiment 1)
  CLIP cosine is the only member that ranked both major-identity-failure pairs
  (A005, A015) in its top-2; it correlates ρ = +0.61 with Dispatch's qualitative
  buckets vs +0.44 for v1.5. CLIP captures semantic identity drift that DINOv2's
  appearance-dominated embedding misses.
- A *trained* head over the 4610-d feature can use CLIP selectively, which a
  post-hoc score ensemble cannot — the Experiment-1 weighted ensemble could only
  re-weight three fixed scores and so could not beat v1.5 on MovieNet (0.389 vs
  0.404). A head sees the full feature.
- It is cheap: no backbone training, the scaffold is ready, the CLIP cache for
  full MovieNet is one (heavy but routine) pass.

Arguments to temper expectations:

- **The frozen-feature ceiling.** `v1_final.md` found the v1 MLP overfits the
  2305-d DINOv2 feature in ~2 epochs and that a 6×4 regularization grid never
  moved validation AUPRC out of a 0.006 band — head capacity is not the
  constraint, the frozen features are. Concatenating a *second* frozen backbone
  does not obviously escape that ceiling; it may just give the head a wider
  feature to overfit. The CLIP half is individually weak on MovieNet (AUPRC
  0.157), so a fusion head may learn to mostly ignore it in-distribution — which
  is exactly what the ensemble LR did (small CLIP weight).
- **The real payoff is on AI-gen, and it cannot be measured yet.** Fusion is
  expected to help where CLIP helps — AI-gen identity drift. But the Veo pilot
  is 10 single-class (all y=0) pairs; AUROC/AUPRC are undefined on it. A fusion
  head's AI-gen benefit cannot be validated until there is labelled AI-gen data
  with **both** classes.
- **Some of the AI-gen gap is calibration, not detection** (Experiment 3): the
  Veo distribution sits an order of magnitude below the MovieNet operating
  thresholds. Fusion addresses detection (better features); it does nothing for
  calibration. The two are orthogonal and calibration is the cheaper fix.

## Recommendation (not a commitment)

1. **Fusion is worth one fast training run** as the cheap v2 experiment, now
   that the scaffold exists — train a small head on the 4610-d feature, compare
   MovieNet AUPRC against v1.5's 0.404. If it clears the ceiling, that is a real
   result; the prior (frozen-feature ceiling) says it probably will not.
2. **It is not a substitute for the LoRA experiment.** The ceiling evidence
   points at the frozen backbones themselves as the bottleneck; unfreezing
   (LoRA) is the more likely lever for a genuine gain. Fusion and LoRA are
   separate v2 candidates, not alternatives — run fusion first because it is
   cheap, but do not let a flat fusion result close out v2.
3. **Before either is judged on AI-gen, source labelled AI-gen data with both
   classes.** Until then, AI-gen evidence is directional (rank-based) only.
4. The full CLIP MovieNet cache (~1.5M keyframes) has **not** been run — only
   the verification subset. Run it before any fusion-head training.
