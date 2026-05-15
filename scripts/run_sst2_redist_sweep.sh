#!/usr/bin/env bash
# Redistribution-version sweep for SST-2.
# Uses redistribution/run_sanitizer.py (Plus uses realized-count redistribution
# via per-doc deterministic flip RNG).
#
# Grid: 5 seeds × 2 methods × 2 distances × 5 epsilons × {p for plus only}
#       = 50 normal + 200 plus = 250 runs.
#
# Output paths get a `-redist` / `_redist` suffix to avoid colliding with the
# non-redistribute eval dirs produced by run_sst2_experiment.sh and
# run_sst2_p_sweep.sh.
#
# Run from project root:
#   bash scripts/run_sst2_redist_sweep.sh                    # both distances
#   bash scripts/run_sst2_redist_sweep.sh euclidean          # only euclidean
#   bash scripts/run_sst2_redist_sweep.sh cosine euclidean   # both (explicit)

set -u

SEEDS=(1 21 42 84 132)
METHODS=(normal plus)
DISTANCES=(cosine euclidean)
EPSILONS=(2 4 8 16 32)
P_VALUES=(0.5 0.6 0.7 0.8)
TASK="SST-2"
DATA_DIR="./datasets/SST-2"
SENSITIVE_FILE="./sensitive_mapping/flair_0.6_sst2.json"
LANG="en"

if [ $# -gt 0 ]; then
  DISTANCES=("$@")
fi

GLUE_TASK="sst-2"
EVAL_FILE="eval_results_${GLUE_TASK}.json"

for seed in "${SEEDS[@]}"; do
 for method in "${METHODS[@]}"; do
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
     echo "  REDIST  seed=${seed}  method=${method}  eps=${EPSILON}  s_eps=${S_EPSILON}  p=${P}  distance=${DISTANCE}"
     echo "================================================================"

     case "$method" in
       normal) REP_DIR="replacements_flair/${seed}/normal/${TASK}/n_epsilon_${EPSILON}.0_s_epsilon_${S_EPSILON}.0" ;;
       plus)   REP_DIR="replacements_flair/${seed}/plus/${TASK}/p_${P}_n_epsilon_${EPSILON}.0_s_epsilon_${S_EPSILON}.0" ;;
     esac
     TSV_OUT="output_run_sanitizer/${TASK}/${method}_eps${EPSILON}_s${S_EPSILON}_p${P}_seed${seed}_${DISTANCE}_redist"
     GLUE_OUT="./tmp/${TASK}-${method}-eps${EPSILON}-s${S_EPSILON}-p${P}-seed${seed}-${DISTANCE}-redist"

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
       --redistribute
       --sensitive_words_file_path "$SENSITIVE_FILE"
       --language "$LANG"
       --seed "$seed"
     )
     if [ "$method" = "plus" ]; then
       SAN_FLAGS+=(--p "$P")
     fi

     echo "[1/3] Sanitizing -> $REP_DIR"
     PYTHONPATH=. python redistribution/run_sanitizer.py "${SAN_FLAGS[@]}" \
       || { echo "  sanitize FAILED, skipping"; continue; }

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
echo "Done. Eval JSONs are under tmp/${TASK}-{normal,plus}-*-redist/${EVAL_FILE}"
