#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  bash build_mtc_wds.sh [train] [test]

If no split is provided, both train and test are processed.

Environment overrides:
  PYTHON_BIN         Python interpreter to use. Defaults to .venv/bin/python if present.
  MTC_ROOT           Monocular Total Capture dataset root.
  DEST_ROOT          Final WebDataset output root for MTC.
  NUM_WORKERS        Worker count passed to preprocess_MTC_mp.py.
  MIN_VISIBLE_RATIO  Minimum ratio of joints that must be inside image and not occluded.
  BBOX_EXPAND_RATIO  Hand bbox expansion ratio.
  MAX_SIZE_GB        Max size per repacked tar shard.
  MAX_COUNT          Max sample count per repacked tar shard.
  OVERWRITE          When set to 1, remove previous worker tar files and destination tar files first.
  CLEAN_WORKER_TARS  When set to 1, delete temporary worker tar files after successful repack.
  DEBUG_MAX_SAMPLES  Optional cap for preprocess_MTC_mp.py debugging.
EOF
  exit 0
fi

if [[ -x ".venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN=".venv/bin/python"
else
  DEFAULT_PYTHON_BIN="python"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
MTC_ROOT="${MTC_ROOT:-/mnt/qnap/data/datasets/mtc/a4_release}"
DEST_ROOT="${DEST_ROOT:-/mnt/qnap/data/datasets/webdatasets2/MTC}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MIN_VISIBLE_RATIO="${MIN_VISIBLE_RATIO:-0.8}"
BBOX_EXPAND_RATIO="${BBOX_EXPAND_RATIO:-1.2}"
MAX_SIZE_GB="${MAX_SIZE_GB:-3.0}"
MAX_COUNT="${MAX_COUNT:-1000000}"
OVERWRITE="${OVERWRITE:-0}"
CLEAN_WORKER_TARS="${CLEAN_WORKER_TARS:-0}"

SELECTED_SPLITS=("$@")
if [[ ${#SELECTED_SPLITS[@]} -eq 0 ]]; then
  SELECTED_SPLITS=(train test)
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_preprocess() {
  local split="$1"
  local -a env_args=(
    "MTC_ROOT=$MTC_ROOT"
    "SPLIT=$split"
    "NUM_WORKERS=$NUM_WORKERS"
    "MIN_VISIBLE_RATIO=$MIN_VISIBLE_RATIO"
    "BBOX_EXPAND_RATIO=$BBOX_EXPAND_RATIO"
  )
  if [[ -n "${DEBUG_MAX_SAMPLES:-}" ]]; then
    env_args+=("DEBUG_MAX_SAMPLES=$DEBUG_MAX_SAMPLES")
  fi

  log "Running MTC preprocess for split=$split"
  env "${env_args[@]}" "$PYTHON_BIN" src/preprocess_MTC_mp.py
}

run_repack() {
  local split="$1"
  local src_dir="mtc_${split}_wds_output"
  local src_pattern="${src_dir}/mtc_${split}-worker*.tar"
  local dst_dir="${DEST_ROOT}/${split}"
  local dst_pattern="${dst_dir}/%06d.tar"

  mkdir -p "$dst_dir"
  if [[ "$OVERWRITE" == "1" ]]; then
    log "Removing previous temporary worker tar files for split=$split"
    rm -f ${src_pattern}
    log "Removing previous destination tar files for split=$split"
    rm -f "${dst_dir}"/*.tar
  fi

  run_preprocess "$split"

  if ! compgen -G "$src_pattern" > /dev/null; then
    echo "No worker tar files matched: $src_pattern" >&2
    return 1
  fi

  log "Repacking MTC split=$split to $dst_pattern"
  env \
    "SRC=$src_pattern" \
    "DST=$dst_pattern" \
    "MAX_SIZE_GB=$MAX_SIZE_GB" \
    "MAX_COUNT=$MAX_COUNT" \
    "$PYTHON_BIN" src/data_reorganizer.py

  if [[ "$CLEAN_WORKER_TARS" == "1" ]]; then
    log "Cleaning temporary worker tar files for split=$split"
    rm -f ${src_pattern}
  fi
}

for split in "${SELECTED_SPLITS[@]}"; do
  case "$split" in
    train|test)
      run_repack "$split"
      ;;
    *)
      echo "Unsupported split: $split" >&2
      exit 1
      ;;
  esac
done

log "MTC export and repack completed."
