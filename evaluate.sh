#!/usr/bin/env bash
# Inference / evaluation: decision-native candidate scoring (single forward, no decoding).
#   RUN=<ckpt_dir> bash evaluate.sh                       # IID test
#   RUN=<ckpt_dir> EVAL_JSON=$PWD/data/ood/test.json bash evaluate.sh   # OOD test
# Architecture (encoder/slices/organ-evidence) is read from the checkpoint's run_info.json;
# the flags below override the data/organ-map/cache paths to the release layout.
set -euo pipefail
REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PY="${PY:-python3}"
RUN="${RUN:?usage: RUN=<ckpt_dir> [SPLIT=test] [EVAL_JSON=...] bash evaluate.sh}"
SPLIT="${SPLIT:-test}"
EVAL_JSON="${EVAL_JSON:-$REPO/data/iid/test.json}"
ORGAN_MAP="${ORGAN_MAP:-$REPO/organ_map.tsv}"
VISUAL_ROOT="${VISUAL_ROOT:-$REPO/feature_cache}"
SLICE_ROOT="${SLICE_ROOT:-$REPO/slice_cache}"
ENCODER="${ENCODER:-radar_ft}"
SLICES="${SLICES:-32}"
BS="${BS:-8}"
LIMIT="${LIMIT:-0}"
SLICE_PROC_LRU="${SLICE_PROC_LRU:-8}"
CUDA_DEV="${CUDA_DEV:-0}"
OUT="${OUT:-$RUN}"
echo "[evaluate] RUN=$RUN SPLIT=$SPLIT EVAL_JSON=$EVAL_JSON ORGAN_MAP=$ORGAN_MAP"
exec env CUDA_VISIBLE_DEVICES="$CUDA_DEV" PYTHONFAULTHANDLER=1 "$PY" -u -m radev train eval \
  --run "$RUN" --split "$SPLIT" --out "$OUT" --eval-json "$EVAL_JSON" \
  --organ-map "$ORGAN_MAP" --cache-root "$VISUAL_ROOT" --slice-root "$SLICE_ROOT" \
  --encoder "$ENCODER" --slices "$SLICES" --bs "$BS" --limit "$LIMIT" \
  --slice-proc-lru "$SLICE_PROC_LRU"
