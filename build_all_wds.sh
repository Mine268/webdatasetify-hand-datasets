#!/usr/bin/env bash
set -euo pipefail

# Batch export script for all currently supported datasets.
# Flow for each dataset/split:
# 1. run preprocess script to generate worker tar files in the repo
# 2. run data_reorganizer.py to merge them into DEST_ROOT

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage:
  ./build_all_wds.sh [dataset...]

Datasets:
  ih26m
  dexycb
  ho3d
  hot3d
  coco-wb

If no dataset is provided, all supported datasets are processed.

Environment overrides:
  PYTHON_BIN    Python interpreter to use. Defaults to .venv/bin/python if present.
  DEST_ROOT     Final WebDataset output root.
  IH26M_ROOT    InterHand2.6M dataset root.
  DEX_ROOT      DexYCB dataset root.
  HO3D_ROOT     HO3D dataset root.
  HOT3D_ROOT    HOT3D dataset root.
  COCO_ROOT     COCO-WholeBody dataset root.
  NUM_WORKERS   Optional shared worker count passed to all preprocess scripts.

Notes:
  - DexYCB exports only setup s1.
  - HOT3D exports only train because test lacks hand pose annotations.
EOF
  exit 0
fi

if [[ -x ".venv/bin/python" ]]; then
  DEFAULT_PYTHON_BIN=".venv/bin/python"
else
  DEFAULT_PYTHON_BIN="python"
fi

PYTHON_BIN="${PYTHON_BIN:-$DEFAULT_PYTHON_BIN}"
DEST_ROOT="${DEST_ROOT:-/mnt/qnap/data/datasets/webdatasets2}"

IH26M_ROOT="${IH26M_ROOT:-/mnt/qnap/data/datasets/InterHand2.6M_5fps_batch1}"
DEX_ROOT="${DEX_ROOT:-/mnt/qnap/data/datasets/dexycb}"
HO3D_ROOT="${HO3D_ROOT:-/mnt/qnap/data/datasets/ho3d_v3/ho3d_v3}"
HOT3D_ROOT="${HOT3D_ROOT:-/mnt/qnap/data/datasets/hot3d}"
COCO_ROOT="${COCO_ROOT:-/mnt/qnap/data/datasets/coco2017}"

SELECTED_DATASETS=("$@")

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

should_run() {
  local dataset="$1"
  if [[ "${#SELECTED_DATASETS[@]}" -eq 0 ]]; then
    return 0
  fi

  local selected
  for selected in "${SELECTED_DATASETS[@]}"; do
    if [[ "$selected" == "$dataset" ]]; then
      return 0
    fi
  done
  return 1
}

run_preprocess() {
  local script_path="$1"
  shift
  log "Running preprocess: $script_path"
  env "$@" "$PYTHON_BIN" "$script_path"
}

run_repack() {
  local src_pattern="$1"
  local dst_pattern="$2"
  local src_dir
  src_dir="$(dirname "$src_pattern")"
  local dst_dir
  dst_dir="$(dirname "$dst_pattern")"

  if ! compgen -G "$src_pattern" > /dev/null; then
    echo "No tar files matched: $src_pattern" >&2
    return 1
  fi

  log "Repacking $src_pattern -> $dst_pattern"
  env "SRC=$src_pattern" "DST=$dst_pattern" "$PYTHON_BIN" src/data_reorganizer.py

  if ! compgen -G "$dst_dir/*.tar" > /dev/null; then
    echo "Repack finished but no output tar found in: $dst_dir" >&2
    return 1
  fi

  log "Cleaning temporary worker tar files: $src_pattern"
  rm -f $src_pattern

  if [[ -d "$src_dir" ]] && ! compgen -G "$src_dir/*.tar" > /dev/null; then
    rmdir "$src_dir" 2>/dev/null || true
  fi
}

append_num_workers_env() {
  local -n env_ref=$1
  if [[ -n "${NUM_WORKERS:-}" ]]; then
    env_ref+=("NUM_WORKERS=$NUM_WORKERS")
  fi
}

build_ih26m() {
  local split
  for split in train val test; do
    local -a env_args=(
      "IH26M_ROOT=$IH26M_ROOT"
      "SPLIT=$split"
    )
    append_num_workers_env env_args
    run_preprocess src/preprocess_InterHand26M_mp.py "${env_args[@]}"
    run_repack \
      "ih26m_${split}_wds_output/ih26m_${split}-worker*.tar" \
      "$DEST_ROOT/InterHand2.6M/$split/%06d.tar"
  done
}

build_dexycb() {
  local split
  for split in train val test; do
    local -a env_args=(
      "DEX_ROOT=$DEX_ROOT"
      "SETUP=s1"
      "SPLIT=$split"
    )
    append_num_workers_env env_args
    run_preprocess src/preprocess_DexYCB_mp.py "${env_args[@]}"
    run_repack \
      "dexycb_s1_${split}_wds_output/dexycb_s1_${split}-worker*.tar" \
      "$DEST_ROOT/DexYCB/s1/$split/%06d.tar"
  done
}

build_ho3d() {
  local -a train_env=(
    "HO3D_ROOT=$HO3D_ROOT"
    "SPLIT=train"
  )
  append_num_workers_env train_env
  run_preprocess src/preprocess_HO3D_mp.py "${train_env[@]}"
  run_repack \
    "ho3d_train_wds_output/ho3d_train-worker*.tar" \
    "$DEST_ROOT/HO3D_v3/train/%06d.tar"

  local -a eval_env=(
    "HO3D_ROOT=$HO3D_ROOT"
  )
  append_num_workers_env eval_env
  run_preprocess src/preprocess_HO3D_eval.py "${eval_env[@]}"
  run_repack \
    "ho3d_evaluation_wds_output/ho3d_evaluation-worker*.tar" \
    "$DEST_ROOT/HO3D_v3/evaluation/%06d.tar"
}

build_hot3d() {
  local -a env_args=(
    "HOT3D_ROOT=$HOT3D_ROOT"
    "SPLIT=train"
  )
  append_num_workers_env env_args
  run_preprocess src/preprocess_HOT3D.py "${env_args[@]}"
  run_repack \
    "hot3d_train_wds_output/hot3d_train-worker*.tar" \
    "$DEST_ROOT/HOT3D/train/%06d.tar"
}

build_coco_wb() {
  local split
  for split in train val; do
    local -a env_args=(
      "COCO_ROOT=$COCO_ROOT"
      "SPLIT=$split"
    )
    append_num_workers_env env_args
    run_preprocess src/preprocess_COCOWholeBody.py "${env_args[@]}"
    run_repack \
      "coco_wholebody_${split}_wds_output/coco_wholebody_${split}-worker*.tar" \
      "$DEST_ROOT/COCO-WholeBody/$split/%06d.tar"
  done
}

if should_run ih26m; then
  build_ih26m
fi

if should_run dexycb; then
  build_dexycb
fi

if should_run ho3d; then
  build_ho3d
fi

if should_run hot3d; then
  build_hot3d
fi

if should_run coco-wb; then
  build_coco_wb
fi

log "All requested dataset builds completed."
