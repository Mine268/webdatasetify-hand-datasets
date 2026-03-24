#!/usr/bin/env bash
set -euo pipefail

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

DEST_ROOT="${DEST_ROOT:-/mnt/qnap/data/datasets/webdatasets2/RHD}"
MAX_SIZE_GB="${MAX_SIZE_GB:-3.0}"
MAX_COUNT="${MAX_COUNT:-1000000}"
OVERWRITE="${OVERWRITE:-0}"

repack_split() {
  local split="$1"
  local src_pattern="$2"
  local dst_dir="$DEST_ROOT/$split"
  local dst_pattern="$dst_dir/%06d.tar"

  mkdir -p "$dst_dir"
  if [[ "$OVERWRITE" == "1" ]]; then
    rm -f "$dst_dir"/*.tar
  fi

  echo "[repack] split=$split"
  echo "[repack] src=$src_pattern"
  echo "[repack] dst=$dst_pattern"

  SRC="$src_pattern" \
  DST="$dst_pattern" \
  MAX_SIZE_GB="$MAX_SIZE_GB" \
  MAX_COUNT="$MAX_COUNT" \
  "$PYTHON_BIN" src/data_reorganizer.py
}

repack_split train "rhd_train_wds_output/rhd_train-worker*.tar"
repack_split evaluation "rhd_evaluation_wds_output/rhd_evaluation-worker*.tar"
