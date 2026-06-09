#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

OUT="${1:-splice_code_submission.zip}"
MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT

add_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "missing required file: $path" >&2
    exit 1
  fi
  printf '%s\n' "$path" >> "$MANIFEST"
}

# core: env + shared library + prep + train + the 4 cited eval scripts + the 2 configs
add_file "environment.yaml"
add_file "configs/v1_sound.yaml"
add_file "configs/operating_thresholds.json"

find src -type f -name '*.py' | sort >> "$MANIFEST"

add_file "scripts/prep/build_cut_index.py"
add_file "scripts/prep/embed_keyframes.py"
add_file "scripts/prep/build_pair_features.py"
add_file "scripts/train/v0_logistic.py"
add_file "scripts/train/v1_mlp_sound.py"
add_file "scripts/train/v2_lora.py"
add_file "scripts/eval/compute_macro_ap.py"
add_file "scripts/eval/calibrate_threshold.py"
add_file "scripts/eval/v2_calibration.py"
add_file "scripts/eval/per_movie_analysis.py"

# AI-gen pilot bundle
add_file "configs/aigen_calibration.json"
add_file "scripts/aigen/labels_template.csv"
add_file "scripts/eval/build_aigen_eval.py"
add_file "scripts/eval/eval_aigen.py"
add_file "scripts/eval/score_aigen_v2.py"
add_file "scripts/eval/score_aigen_per_pair.py"
add_file "scripts/eval/aigen_calibration.py"
add_file "scripts/eval/rank_based_flagging.py"
add_file "scripts/eval/aigen_diagnostic.py"

if grep -E '\.md$' "$MANIFEST" >/dev/null; then
  echo "manifest unexpectedly contains a Markdown file" >&2
  exit 1
fi

rm -f "$OUT"
zip -q -@ "$OUT" < "$MANIFEST"

if unzip -Z1 "$OUT" | grep -E '\.md$' >/dev/null; then
  echo "zip unexpectedly contains a Markdown file" >&2
  exit 1
fi

echo "created $OUT"
echo "files included: $(wc -l < "$MANIFEST" | tr -d ' ')"
