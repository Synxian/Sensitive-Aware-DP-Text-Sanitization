#!/usr/bin/env bash
set -euo pipefail

# Smoke test: run a very small i2b2 NER training to validate the pipeline end-to-end.
# Usage:
#   ./scripts/smoke_i2b2_ner.sh /path/to/i2b2_xml_dir

DATA_DIR=${1:-}
if [[ -z "${DATA_DIR}" ]]; then
  echo "Usage: $0 /path/to/i2b2_xml_dir" >&2
  exit 1
fi

PY=${PYTHON:-python}
OUT_DIR=${OUT_DIR:-runs/smoke_i2b2}

${PY} -m downstream.i2b2 \
  --data_dir "${DATA_DIR}" \
  --output_dir "${OUT_DIR}" \
  --model_name_or_path distilbert-base-uncased \
  --max_length 256 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --val_ratio 0.1 \
  --eval_strategy steps \
  --save_strategy no \
  --logging_steps 5 \
  --seed 13

echo "OK: smoke i2b2 finished. Outputs in ${OUT_DIR}"
