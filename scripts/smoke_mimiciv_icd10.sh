#!/usr/bin/env bash
set -euo pipefail

# Smoke test: minimal MIMIC-IV ICD10 multi-label run.
# Requires you to provide a notes+labels file (not present in this repo).
# Usage:
#   ./scripts/smoke_mimiciv_icd10.sh /path/to/notes.feather /path/to/splits.feather

NOTES_FILE=${1:-}
SPLITS_FILE=${2:-}
if [[ -z "${NOTES_FILE}" || -z "${SPLITS_FILE}" ]]; then
  echo "Usage: $0 /path/to/notes.feather /path/to/splits.feather" >&2
  exit 1
fi

PY=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-runs/smoke_mimiciv_icd10}

${PY} -m downstream.mimiciv \
  --notes_file "${NOTES_FILE}" \
  --splits_file "${SPLITS_FILE}" \
  --output_dir "${OUT_DIR}" \
  --model_name_or_path distilbert-base-uncased \
  --max_length 256 \
  --num_train_epochs 1 \
  --batch_size 4 \
  --lr 2e-5 \
  --weight_decay 0.01 \
  --seed 13 \
  --max_train_examples 32 \
  --max_eval_examples 32

echo "OK: smoke MIMIC-IV ICD10 finished. Outputs in ${OUT_DIR}"
