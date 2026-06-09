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

add_file "environment.yaml"

find src -type f -name '*.py' | sort >> "$MANIFEST"

add_file "scripts/prep/build_cut_index.py"
add_file "scripts/prep/embed_keyframes.py"
add_file "scripts/prep/build_pair_features.py"
add_file "scripts/train/v0_logistic.py"
add_file "scripts/train/v1_mlp_sound.py"
add_file "scripts/train/v2_lora.py"
add_file "scripts/eval/compute_macro_ap.py"

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
