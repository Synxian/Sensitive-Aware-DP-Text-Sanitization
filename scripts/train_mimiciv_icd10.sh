#!/usr/bin/env bash
set -euo pipefail

# Train an ICD-10 multilabel classifier on a MIMIC-IV dataset (notes + list of ICD10 codes).
# IMPORTANT: this repo currently only contains the splits file. You must provide --notes_file.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

python "$REPO_ROOT/downstream/mimiciv.py" \
  --notes_file "$REPO_ROOT/data/mimic/mimiciv_icd10_split.feather" \
  --splits_file "$REPO_ROOT/data/mimic/splits_val_train.feather" \
  --output_dir "$REPO_ROOT/output/downstream/mimiciv_icd10" \
  --model_name_or_path "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract" \
  --max_length 512 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --seed 42
