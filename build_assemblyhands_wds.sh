#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash build_assemblyhands_wds.sh

This script exports and repacks AssemblyHands train split only.

Environment overrides:
  PYTHON_BIN         Python interpreter to use. Defaults to .venv/bin/python if present.
  ASSEMBLYHANDS_ROOT AssemblyHands dataset root.
  DEST_ROOT          Final WebDataset output root for AssemblyHands.
  NUM_WORKERS        Worker count passed to preprocess_AssemblyHands_mp.py.
  BBOX_EXPAND_RATIO  Hand bbox expansion ratio. Current production recommendation: 1.0.
  FRAME_GAP          Temporal continuity gap in frame_idx. Default: 2.
  MAX_CLIP_LEN       Maximum frames per exported clip. Default: 256.
  MAX_SIZE_GB        Max size per repacked tar shard.
  MAX_COUNT          Max sample count per repacked tar shard.
  OVERWRITE          When set to 1, remove previous worker tar files and destination tar files first.
  CLEAN_WORKER_TARS  When set to 1, delete temporary worker tar files after successful repack.
  DEBUG_MAX_CLIPS    Optional cap for preprocess_AssemblyHands_mp.py debugging.

Recommended full export:
  PYTHONHASHSEED=0 BBOX_EXPAND_RATIO=1.0 OVERWRITE=1 CLEAN_WORKER_TARS=1 \
    bash build_assemblyhands_wds.sh
EOF
  exit 0
fi

if [[ -x ".venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN=".venv/bin/python"
else
  DEFAULT_PYTHON_BIN="python"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
ASSEMBLYHANDS_ROOT="${ASSEMBLYHANDS_ROOT:-/mnt/qnap/data/datasets/AssemblyHands}"
DEST_ROOT="${DEST_ROOT:-/mnt/qnap/data/datasets/webdatasets2/AssemblyHands}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BBOX_EXPAND_RATIO="${BBOX_EXPAND_RATIO:-1.2}"
FRAME_GAP="${FRAME_GAP:-2}"
MAX_CLIP_LEN="${MAX_CLIP_LEN:-256}"
MAX_SIZE_GB="${MAX_SIZE_GB:-3.0}"
MAX_COUNT="${MAX_COUNT:-1000000}"
OVERWRITE="${OVERWRITE:-0}"
CLEAN_WORKER_TARS="${CLEAN_WORKER_TARS:-0}"

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_preprocess() {
  local -a env_args=(
    "ASSEMBLYHANDS_ROOT=$ASSEMBLYHANDS_ROOT"
    "SPLIT=train"
    "NUM_WORKERS=$NUM_WORKERS"
    "BBOX_EXPAND_RATIO=$BBOX_EXPAND_RATIO"
    "FRAME_GAP=$FRAME_GAP"
    "MAX_CLIP_LEN=$MAX_CLIP_LEN"
  )
  if [[ -n "${DEBUG_MAX_CLIPS:-}" ]]; then
    env_args+=("DEBUG_MAX_CLIPS=$DEBUG_MAX_CLIPS")
  fi

  log "Running AssemblyHands preprocess for split=train"
  env "${env_args[@]}" "$PYTHON_BIN" src/preprocess_AssemblyHands_mp.py
}

run_repack() {
  local src_dir="assemblyhands_train_wds_output"
  local src_pattern="${src_dir}/assemblyhands_train-worker*.tar"
  local dst_dir="${DEST_ROOT}/train"
  local dst_pattern="${dst_dir}/%06d.tar"

  mkdir -p "$dst_dir"
  if [[ "$OVERWRITE" == "1" ]]; then
    log "Removing previous temporary worker tar files"
    rm -f ${src_pattern}
    log "Removing previous destination tar files"
    rm -f "${dst_dir}"/*.tar
  fi

  run_preprocess

  if ! compgen -G "$src_pattern" > /dev/null; then
    echo "No worker tar files matched: $src_pattern" >&2
    return 1
  fi

  log "Repacking AssemblyHands train split to $dst_pattern"
  env \
    "SRC=$src_pattern" \
    "DST=$dst_pattern" \
    "MAX_SIZE_GB=$MAX_SIZE_GB" \
    "MAX_COUNT=$MAX_COUNT" \
    "$PYTHON_BIN" src/data_reorganizer.py

  if [[ "$CLEAN_WORKER_TARS" == "1" ]]; then
    log "Cleaning temporary worker tar files"
    rm -f ${src_pattern}
  fi
}

run_repack

log "AssemblyHands export and repack completed."
