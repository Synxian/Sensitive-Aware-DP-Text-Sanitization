#!/usr/bin/env bash
set -euo pipefail

# Train a token-classification NER model on MEDDOCAN XML.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$REPO_ROOT/downstream/meddocan.py" \
  --data_dir "$REPO_ROOT/data/meddocan" \
  --output_dir "$REPO_ROOT/output/downstream/meddocan_ner" \
  --model_name_or_path "dccuchile/bert-base-spanish-wwm-cased" \
  --max_length 1024 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 3 \
  --seed 42
