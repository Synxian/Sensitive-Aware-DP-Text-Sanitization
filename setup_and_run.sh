#!/usr/bin/env bash
# ============================================================================
# setup_and_run.sh — Download all data + run full experiment grid from scratch.
#
# Usage:
#     chmod +x setup_and_run.sh
#     ./setup_and_run.sh            # full grid (takes hours on GPU)
#     ./setup_and_run.sh --smoke    # quick smoke test (500 samples, ε=10)
# ============================================================================
set -euo pipefail

CUDA_VISIBLE_DEVICES=1
export CUDA_VISIBLE_DEVICES

ROOT="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$ROOT/data"
GLOVE_PATH="$DATA_DIR/glove.840B.300d.txt"

SMOKE=false
if [[ "${1:-}" == "--smoke" ]]; then
    SMOKE=true
fi

# ──────────────────────────────────────────────────────────────────────────────
# 1. Python dependencies
# ──────────────────────────────────────────────────────────────────────────────
echo "=== [1/5] Installing Python dependencies ==="
pip install -r "$ROOT/requirements.txt"

# ──────────────────────────────────────────────────────────────────────────────
# 2. Download GloVe 840B 300d
# ──────────────────────────────────────────────────────────────────────────────
echo "=== [2/5] GloVe embeddings ==="
if [[ -f "$GLOVE_PATH" ]]; then
    echo "Already exists: $GLOVE_PATH"
else
    mkdir -p "$DATA_DIR"
    echo "Downloading GloVe 840B 300d (~2.2 GB compressed) …"
    wget -q --show-progress -O "$DATA_DIR/glove.840B.300d.zip" \
        "https://nlp.stanford.edu/data/glove.840B.300d.zip"
    echo "Extracting …"
    unzip -o "$DATA_DIR/glove.840B.300d.zip" -d "$DATA_DIR"
    rm -f "$DATA_DIR/glove.840B.300d.zip"
    echo "Done: $GLOVE_PATH"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 3. Download GLUE datasets (SST-2 and QNLI)
# ──────────────────────────────────────────────────────────────────────────────
echo "=== [3/5] GLUE datasets ==="

download_glue_task() {
    local task_name="$1"      # e.g. SST-2, QNLI
    local task_dir="$DATA_DIR/$task_name"

    if [[ -f "$task_dir/train.tsv" && -f "$task_dir/dev.tsv" ]]; then
        echo "$task_name already exists."
        return
    fi

    mkdir -p "$task_dir"
    echo "Downloading $task_name …"
    local base_url="https://dl.fbaipublicfiles.com/glue/data"
    wget -q --show-progress -O "$DATA_DIR/${task_name}.zip" \
        "${base_url}/${task_name}.zip"
    unzip -o "$DATA_DIR/${task_name}.zip" -d "$DATA_DIR"
    rm -f "$DATA_DIR/${task_name}.zip"
    echo "Done: $task_dir"
}

download_glue_task "SST-2"
download_glue_task "QNLIv2"

# QNLI zip extracts as QNLIv2 → rename to QNLI
if [[ -d "$DATA_DIR/QNLIv2" && ! -d "$DATA_DIR/QNLI" ]]; then
    mv "$DATA_DIR/QNLIv2" "$DATA_DIR/QNLI"
elif [[ -d "$DATA_DIR/QNLIv2" && -d "$DATA_DIR/QNLI" ]]; then
    # Already have QNLI, remove extra
    rm -rf "$DATA_DIR/QNLIv2"
fi

# ──────────────────────────────────────────────────────────────────────────────
# 4. Download HuggingFace models (bert-base-uncased + Flair NER)
# ──────────────────────────────────────────────────────────────────────────────
echo "=== [4/5] Pre-downloading models ==="

python3 -c "
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
print('Downloading bert-base-uncased …')
AutoTokenizer.from_pretrained('bert-base-uncased')
AutoConfig.from_pretrained('bert-base-uncased')
AutoModelForSequenceClassification.from_pretrained('bert-base-uncased')
print('bert-base-uncased cached.')
"

python3 -c "
from flair.models import SequenceTagger
print('Downloading flair/ner-english-large …')
SequenceTagger.load('flair/ner-english-large')
print('Flair NER model cached.')
"

echo "Models cached."

# ──────────────────────────────────────────────────────────────────────────────
# 5. Run experiments
# ──────────────────────────────────────────────────────────────────────────────
echo "=== [5/5] Running experiments ==="

if $SMOKE; then
    echo ">>> SMOKE TEST MODE (500 samples, ε=10, sst2 only) <<<"
    python3 "$ROOT/run_experiment.py" \
        --tasks sst2 \
        --methods santext normal plus \
        --epsilons 10 \
        --distance_metric cosine \
        --max_samples 500 \
        --downstream \
        --threads 4
else
    # Full grid: 2 tasks × 3 methods × 4 epsilons × 3 metrics = 72 sanitize + downstream runs
    TASKS="sst2 qnli"
    METHODS="santext normal plus"
    EPSILONS="1 3 10"

    for METRIC in cosine cosine_clipped euclidean_clipped; do
        if [[ "$METRIC" == "cosine" ]]; then
            CLIP_LO=0.0; CLIP_HI=0.81
        elif [[ "$METRIC" == "cosine_clipped" ]]; then
            CLIP_LO=0.0; CLIP_HI=0.81
        elif [[ "$METRIC" == "euclidean_clipped" ]]; then
            CLIP_LO=0.0; CLIP_HI=11.85
        fi

        echo ""
        echo "========================================================"
        echo "  METRIC: $METRIC  clip=[$CLIP_LO, $CLIP_HI]"
        echo "========================================================"

        python3 "$ROOT/run_experiment.py" \
            --tasks $TASKS \
            --methods $METHODS \
            --epsilons $EPSILONS \
            --distance_metric "$METRIC" \
            --clip_lo "$CLIP_LO" \
            --clip_hi "$CLIP_HI" \
            --p_values 0.6 \
            --downstream \
            --threads 4
 # ──────────────────────────────────────────────────────────────────────────────
#            --quality_metrics all \
# ──────────────────────────────────────────────────────────────────────────────
            
    done
fi

echo ""
echo "============================================"
echo "  ALL DONE. Results in: $ROOT/output/"
echo "  Summary CSVs: $ROOT/output/summary_*.csv"
echo "============================================"
