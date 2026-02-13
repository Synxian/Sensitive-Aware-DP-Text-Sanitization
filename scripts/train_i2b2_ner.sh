#!/usr/bin/env bash
set -euo pipefail

# Train a token-classification NER model on i2b2 XML.
# Adjust paths if needed.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$REPO_ROOT/downstream/i2b2.py" \
  --data_dir "$REPO_ROOT/data/i2b2" \
  --output_dir "$REPO_ROOT/output/downstream/i2b2_ner" \
  --model_name_or_path "distilbert-base-uncased" \
  --max_length 1024 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --learning_rate 5e-5 \
  --num_train_epochs 5 \
  --seed 42
