#!/usr/bin/env bash
# Run the full SST-2 experiment: santext / Ours (normal) / Ours++ (plus)
# across 5 seeds. Each (seed, method) combination is sanitized, converted
# back to TSVs, and evaluated with run_glue.
#
# Skips combinations whose run_glue eval JSON already exists, so re-running
# the script resumes where it left off.
#
# Run from project root:  bash scripts/run_sst2_experiment.sh

set -u

# ---- Fixed config (edit if needed) ----
SEEDS=(1 21 42 84 132)
METHODS=(santext normal plus)
DISTANCES=(cosine euclidean)
EPSILONS=(2 4 8 16 32)      # s_epsilon is derived as epsilon/2 inside the loop
TASK="SST-2"
DATA_DIR="./datasets/SST-2"
P=0.7
SENSITIVE_FILE="./sensitive_mapping/flair_0.6_sst2.json"
LANG="en"

# ---- Derived ----
GLUE_TASK="sst-2"           # name expected by run_glue.py / Sst2Processor
EVAL_FILE="eval_results_${GLUE_TASK}.json"

for seed in "${SEEDS[@]}"; do
  for method in "${METHODS[@]}"; do
   for DISTANCE in "${DISTANCES[@]}"; do
    for EPSILON in "${EPSILONS[@]}"; do
     S_EPSILON=$((EPSILON / 2))
    echo
    echo "================================================================"
    echo "  seed=${seed}  method=${method}  eps=${EPSILON}  s_eps=${S_EPSILON}  p=${P}  distance=${DISTANCE}"
    echo "================================================================"

    case "$method" in
      santext) REP_DIR="replacements_flair/${seed}/santext/${TASK}/epsilon_${EPSILON}.0" ;;
      normal)  REP_DIR="replacements_flair/${seed}/normal/${TASK}/n_epsilon_${EPSILON}.0_s_epsilon_${S_EPSILON}.0" ;;
      plus)    REP_DIR="replacements_flair/${seed}/plus/${TASK}/p_${P}_n_epsilon_${EPSILON}.0_s_epsilon_${S_EPSILON}.0" ;;
    esac
    TSV_OUT="output_run_sanitizer/${TASK}/${method}_eps${EPSILON}_s${S_EPSILON}_p${P}_seed${seed}_${DISTANCE}"
    GLUE_OUT="./tmp/${TASK}-${method}-eps${EPSILON}-s${S_EPSILON}-p${P}-seed${seed}-${DISTANCE}"

    if [ -f "${GLUE_OUT}/${EVAL_FILE}" ]; then
      echo "  [skip] ${GLUE_OUT}/${EVAL_FILE} already exists"
      continue
    fi

    # ---- 1) Sanitize ----
    SAN_FLAGS=(
      --data_dir "$DATA_DIR"
      --task "$TASK"
      --method "$method"
      --epsilon "$EPSILON"
      --s_epsilon "$S_EPSILON"
      --distance "$DISTANCE"
      --no-redistribute
      --sensitive_words_file_path "$SENSITIVE_FILE"
      --language "$LANG"
      --seed "$seed"
    )
    if [ "$method" = "plus" ]; then
      SAN_FLAGS+=(--p "$P")
    fi
    echo "[1/3] Sanitizing -> $REP_DIR"
    python run_sanitizer.py "${SAN_FLAGS[@]}" || { echo "  sanitize FAILED, skipping"; continue; }

    # ---- 2) Reassemble TSVs ----
    echo "[2/3] Converting   -> $TSV_OUT"
    python convert_replacements_to_tsv.py \
      --replacements_dir "$REP_DIR" \
      --output_dir       "$TSV_OUT" \
      || { echo "  convert FAILED, skipping"; continue; }

    # ---- 3) Fine-tune + eval ----
    echo "[3/3] run_glue     -> $GLUE_OUT"
    python run_glue.py \
      --model_name_or_path bert-base-uncased \
      --task_name "$GLUE_TASK" \
      --do_train --do_eval \
      --data_dir "$TSV_OUT" \
      --max_seq_length 128 \
      --per_device_train_batch_size 128 \
      --per_device_eval_batch_size 128 \
      --learning_rate 2e-5 \
      --num_train_epochs 2.0 \
      --fp16 \
      --save_strategy no \
      --seed "$seed" \
      --output_dir "$GLUE_OUT" \
      --overwrite_output_dir \
      --overwrite_cache \
      || { echo "  run_glue FAILED"; continue; }
    done
   done
  done
done

echo
echo "Done. Eval JSONs are under tmp/${TASK}-*-seed*/${EVAL_FILE}"
