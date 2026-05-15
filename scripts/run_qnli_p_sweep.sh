#!/usr/bin/env bash
# Mirror of run_sst2_p_sweep.sh for QNLI.
# Plus is the one that actually depends on p; normal ignores it, so we run
# normal once per (seed, distance) with placeholder p=0.7 (de-dupes against
# run_qnli_experiment.sh outputs via the skip-on-existing-eval check).
#
# ε is fixed at 8, s_ε at 4. P sweep is {0.5, 0.6, 0.8}; p=0.7 lives in
# run_qnli_experiment.sh so it's omitted here.
#
# Run from project root:
#   bash scripts/run_qnli_p_sweep.sh                     # both distances
#   bash scripts/run_qnli_p_sweep.sh euclidean           # only euclidean
#   bash scripts/run_qnli_p_sweep.sh cosine euclidean    # explicit both

set -u

SEEDS=(1 21 42 84 132)
DISTANCES=(cosine euclidean)
P_VALUES=(0.5 0.6 0.8)
EPSILONS=(2 4 8 16 32)
TASK="QNLI"
DATA_DIR="./datasets/QNLI"
SENSITIVE_FILE="./sensitive_mapping/flair_0.6_qnli.json"
LANG="en"

GLUE_TASK="qnli"
EVAL_FILE="eval_results_${GLUE_TASK}.json"

if [ $# -gt 0 ]; then
  DISTANCES=("$@")
fi

for seed in "${SEEDS[@]}"; do
 for method in normal plus; do
  for DISTANCE in "${DISTANCES[@]}"; do
   for EPSILON in "${EPSILONS[@]}"; do
    S_EPSILON=$((EPSILON / 2))
   if [ "$method" = "plus" ]; then
     ps=("${P_VALUES[@]}")
   else
     ps=(0.7)
   fi
   for P in "${ps[@]}"; do
    echo
    echo "================================================================"
    echo "  seed=${seed}  method=${method}  eps=${EPSILON}  s_eps=${S_EPSILON}  p=${P}  distance=${DISTANCE}"
    echo "================================================================"

    case "$method" in
      normal) REP_DIR="replacements_flair/${seed}/normal/${TASK}/n_epsilon_${EPSILON}.0_s_epsilon_${S_EPSILON}.0" ;;
      plus)   REP_DIR="replacements_flair/${seed}/plus/${TASK}/p_${P}_n_epsilon_${EPSILON}.0_s_epsilon_${S_EPSILON}.0" ;;
    esac
    TSV_OUT="output_run_sanitizer/${TASK}/${method}_eps${EPSILON}_s${S_EPSILON}_p${P}_seed${seed}_${DISTANCE}"
    GLUE_OUT="./tmp/${TASK}-${method}-eps${EPSILON}-s${S_EPSILON}-p${P}-seed${seed}-${DISTANCE}"

    if [ -f "${GLUE_OUT}/${EVAL_FILE}" ]; then
      echo "  [skip] ${GLUE_OUT}/${EVAL_FILE} already exists"
      continue
    fi

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

    echo "[2/3] Converting   -> $TSV_OUT"
    python convert_replacements_to_tsv.py \
      --replacements_dir "$REP_DIR" \
      --output_dir       "$TSV_OUT" \
      || { echo "  convert FAILED, skipping"; continue; }

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
done

echo
echo "Done. Eval JSONs are under tmp/${TASK}-{normal,plus}-eps${EPSILON}-*-seed*/${EVAL_FILE}"
