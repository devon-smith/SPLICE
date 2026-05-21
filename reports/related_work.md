# Related Work

Five papers that frame SPLICE: our backbone, our loss family, the closest prior
art, the use case, and the dataset.

## ShotCoL — Chen et al., CVPR 2021

ShotCoL ("Shot Contrastive Self-Supervised Learning for Scene Boundary
Detection") is the closest prior work. It learns a shot representation with a
contrastive objective on movie shots — nearby shots are treated as positives
(they usually belong to the same scene) and distant or random shots as negatives
— using a ResNet encoder, then trains a scene-boundary detector on top. On
MovieNet scene segmentation it was state of the art at publication, showing that
a shot embedding tuned for "same-scene-ness" transfers well to boundary
detection.

SPLICE is methodologically downstream of ShotCoL but differs on three axes.
**Task:** we score *each individual cut* with a continuity scalar rather than
segmenting a whole movie into scenes — a per-transition output, not a per-movie
one. **Backbone:** we use a frozen DINOv2 ViT-B/14 as a fixed feature extractor
instead of a ResNet trained from scratch, which lets the whole pipeline run on
cached embeddings. **Evaluation:** ShotCoL reports a single aggregate
scene-boundary number on real film; SPLICE adds per-movie stratified evaluation
and, at M3, a held-out set of AI-generated video cuts. We cite ShotCoL as the
anchor and position SPLICE as "ShotCoL's idea, repurposed for per-cut continuity
scoring and AI-gen evaluation."

## DINOv2 — Oquab et al., TMLR 2024

DINOv2 trains Vision Transformers with a self-supervised discriminative
objective (a combination of image-level self-distillation and masked
patch-level prediction) on a large curated image collection, producing
general-purpose visual features that work well *frozen* — without any task
fine-tuning — across classification, segmentation and depth estimation. A key
property is the quality of its dense patch features, which capture local
appearance, lighting and texture rather than only image-level semantics.

DINOv2 is SPLICE's backbone. v0 uses it strictly as a frozen encoder: every
keyframe is embedded once and never updated. The argument for DINOv2 over a
ResNet or a CLIP encoder is that continuity is a *low-level appearance* judgment
— lighting direction, colour grade, framing — and DINOv2's self-supervised
features preserve exactly that information. Our v0 results bear this out: raw
DINOv2 cosine clearly outscores CLIP cosine on the task. In v2 we plan partial
(LoRA) fine-tuning of DINOv2, since the frozen-feature ceiling is v0/v1's main
limiter.

## Supervised Contrastive Learning — Khosla et al., NeurIPS 2020

Supervised Contrastive Learning (SupCon) extends self-supervised contrastive
losses (e.g. SimCLR) to the fully-labelled setting: instead of treating only
augmentations of a single image as positives, it pulls together *all* samples
sharing a class label and pushes apart samples of different classes, in a
normalised embedding space. SupCon outperforms ordinary cross-entropy on
ImageNet and is more robust to hyperparameters and input corruptions, typically
used as a representation-learning stage followed by a lightweight classifier.

SupCon is SPLICE's intended v2 loss family. v0 and v1 use a plain class-balanced
classifier head, but the natural objective for a *continuity metric* is
contrastive: within-scene shot pairs should embed close together and cross-scene
pairs far apart, so that the continuity score is a true distance in a learned
space. v2 adds a projection head and a SupCon-style term alongside the
classification loss; we cite Khosla et al. as the formulation we build on.

## VBench — Huang et al., CVPR 2024

VBench is a comprehensive benchmark suite for video generative models. It
decomposes "video quality" into many fine-grained, individually-measured
dimensions — subject consistency, background consistency, temporal flickering,
motion smoothness, aesthetic and imaging quality, and more — each with an
automatic metric validated against human preference judgments. It has become a
standard yardstick for comparing text-to-video systems.

VBench matters to SPLICE because it formalises *within-clip* consistency:
subject and background consistency measured across the frames of a single
generated clip. SPLICE extends that notion *across a cut* — cross-shot continuity
in an assembled multi-shot sequence — which VBench's single-clip metrics do not
cover. VBench both motivates the use case (assessing AI-generated video) and
gives precedent for treating consistency as a measurable, decomposable quantity;
our M3 AI-gen evaluation is in the same spirit, one level up at the edit.

## MovieNet — Huang et al., ECCV 2020

MovieNet is a large-scale, holistic movie-understanding dataset: 1,100 movies
with trailers, stills, plot synopses, and a rich annotation layer including shot
boundaries, scene boundaries (on a 318-movie subset), character identities and
bounding boxes, cinematic-style tags (shot scale, camera movement), and aligned
subtitles and scripts. It was built to support a broad range of movie-analysis
tasks under one roof.

SPLICE uses the 318-movie scene-segmentation subset as both its training and
evaluation data. Per-cut labels are derived from MovieNet's scene-boundary
annotations (via the BaSSL distribution), yielding 502,534 adjacent-shot cuts;
the three 240p keyframes per shot are the model input. One note for our setting:
MovieNet's cinematic-style annotations (shot scale) were not present in the data
distribution we obtained, which is why the shot-scale-stratified analysis is
left as future work. <!-- TODO(team): fetch the official MovieNet meta package
if a shot-scale breakdown is wanted for the final report. -->
