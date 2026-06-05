<!-- AI-USE: This sourcing protocol was AI-assisted with Claude (claude-sonnet-4-6) via Claude Code. -->
<!-- Scope: drafted workflow and labeling guidance for the AI-gen evaluation set. -->

# AI-Gen Sourcing Protocol

How to generate and label AI-gen clip pairs for the SPLICE evaluation set. Read
this once before you start; it should answer every question without you having to
ask. Prompt templates are in [prompt_library.md](prompt_library.md); the label
file schema is [labels_template.csv](labels_template.csv).

## Goal

SPLICE's scorer is trained on MovieNet (real film). We need to know whether it
**transfers to the AI-generated video distribution** — does a cut-continuity
model trained on film still tell a continuous shot from a scene change when the
footage came from a text-to-video model? To measure that we need a labelled set
of AI-gen clip *pairs*, each pair being two short clips joined at a cut.

## What a pair is

Each pair is two clips — **A** and **B** — that meet at one cut. You generate
A and B from the two prompts in a library template:

- **Prompt A → the left clip**, saved as `pair_<id>_left.mp4`
- **Prompt B → the right clip**, saved as `pair_<id>_right.mp4`

The harness extracts three keyframes per clip and scores the cut between them.

## Labelling rule — intent-based

**Label every pair by what the prompt pair was *designed* to produce, not by what
the AI actually rendered.**

- continuous-action and reverse-shot templates → `intended_label = 0` (consistent)
- cross-scene templates → `intended_label = 1` (inconsistent)

If the AI fails to maintain identity on a continuous-action pair — the character's
face changes, the location drifts — **the label is still `0`**. Detecting that
failure is the model's job; relabelling it `1` would hide exactly the error we
want to measure. Only discard a pair (see below) if the generation is unusable,
not if it merely "looks wrong."

## Quality check — when to discard a generation

Set `quality_check = discard` for a pair (and regenerate it if you can) only when
the output is genuinely unusable:

- the model failed to produce a video, or produced a still image / corrupt file;
- the clip is broken or unwatchable (severe artefacting, garbled frames);
- the prompt was misinterpreted so badly the pair no longer probes its intended
  type — e.g. a cross-scene prompt rendered both clips as the same scene, or a
  continuous-action prompt rendered two unrelated subjects.

Set `quality_check = pass` for everything else, **including pairs where the AI
made a continuity mistake** — those are valid, kept data. Rows marked `discard`
are dropped automatically by `build_aigen_eval.py`.

## Recommended counts

Each teammate generates **≈50 pairs**, mixed roughly:

| Type | Per person | Intended label |
|---|---|---|
| continuous-action | 30 | 0 |
| reverse-shot | 10 | 0 |
| cross-scene | 10 | 1 |

Three people → ≈150 pairs, ~67% consistent / ~33% inconsistent. Draw from and
adapt the 50 library templates — vary the subject, location, and time of day, but
keep each template's pair *structure* intact.

## File layout and naming

Generate with whatever text-to-video model you're using (Veo 3 by default). Set
`source` to the model name in lowercase (`veo3`, `sora`, `runway`); per-source
metrics in the eval are split on this field.

**To avoid `pair_id` collisions, split the ID ranges:** Devon `001–050`, Lily
`051–100`, Xander `101–150`. Use zero-padded 3-digit IDs.

Drop clips here, one folder per `source`:

```
/mnt/disks/splice-data/datasets/aigen/
  labels.csv                       <- one shared file, all pairs from everyone
  veo3/pair_001_left.mp4
  veo3/pair_001_right.mp4
  veo3/pair_002_left.mp4
  ...
```

Copy `scripts/aigen/labels_template.csv` to
`/mnt/disks/splice-data/datasets/aigen/labels.csv` and add one row per pair. The
columns:

| Column | What to put |
|---|---|
| `pair_id` | zero-padded 3-digit id, e.g. `007` |
| `source` | generation model, lowercase, e.g. `veo3` |
| `prompt_A` | exact text used for the left clip |
| `prompt_B` | exact text used for the right clip |
| `intended_label` | `0` (consistent) or `1` (inconsistent) — see the rule above |
| `shot_type` | `continuous-action`, `reverse-shot`, or `cross-scene` |
| `notes` | anything worth recording (e.g. "AI changed the jacket colour") |
| `generated_at` | date you generated it, `YYYY-MM-DD` |
| `quality_check` | `pass` or `discard` |

Example row:

```
007,veo3,"Wide shot of a man in a brown leather jacket ...","Medium close-up of the same man ...",0,continuous-action,identity held well,2026-05-22,pass
```

## Running the eval

Once clips and `labels.csv` are in place, two commands produce the results table.
Use the project env (`micromamba activate consistency`, or the interpreter path
shown below):

```bash
PY=/mnt/disks/splice-data/envs/envs/consistency/bin/python

# 1. build the cut index: extracts keyframes, writes cuts.parquet
$PY scripts/eval/build_aigen_eval.py \
    --clips_root /mnt/disks/splice-data/datasets/aigen/ \
    --out /mnt/disks/splice-data/outputs/aigen_eval/

# 2. score all six models, write the results table + figure
$PY scripts/eval/eval_aigen.py \
    --aigen_index /mnt/disks/splice-data/outputs/aigen_eval/cuts.parquet \
    --out /mnt/disks/splice-data/outputs/aigen_eval/results/
```

Results land in `outputs/aigen_eval/results/`: `aigen_results.md` (the table,
overall and per source, with a MovieNet comparison and an analysis block to fill
in), `aigen_results.json`, and `aigen_auprc_by_source.png`. Check
`outputs/aigen_eval/build_summary.json` first — it reports how many pairs were
processed, how many failed, and any `discard` rows that were skipped.
