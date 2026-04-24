#!/usr/bin/env bash
# Download GLUE datasets (SST-2, QNLI) and GloVe 840B-300d embeddings.
#
# Usage:
#   bash setup_data.sh             # download everything (~2.2 GB)
#   bash setup_data.sh --no-glove  # skip GloVe (if you already have it)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"
mkdir -p "${DATA_DIR}"
cd "${DATA_DIR}"

SKIP_GLOVE=false
for arg in "$@"; do
  [[ "$arg" == "--no-glove" ]] && SKIP_GLOVE=true
done

echo "=== SST-2 ==="
if [ ! -f "SST-2/train.tsv" ]; then
  curl -LO https://dl.fbaipublicfiles.com/glue/data/SST-2.zip
  unzip -o SST-2.zip && rm SST-2.zip
  echo "  ✓ SST-2 ready"
else
  echo "  (already exists)"
fi

echo "=== QNLI ==="
if [ ! -f "QNLI/train.tsv" ]; then
  curl -LO https://dl.fbaipublicfiles.com/glue/data/QNLIv2.zip
  unzip -o QNLIv2.zip && rm QNLIv2.zip
  [ -d "QNLIv2" ] && mv QNLIv2 QNLI
  echo "  ✓ QNLI ready"
else
  echo "  (already exists)"
fi

if [ "$SKIP_GLOVE" = false ]; then
  echo "=== GloVe 840B-300d (~2 GB) ==="
  if [ ! -f "glove.840B.300d.txt" ]; then
    curl -LO https://nlp.stanford.edu/data/glove.840B.300d.zip
    unzip -o glove.840B.300d.zip && rm glove.840B.300d.zip
    echo "  ✓ GloVe ready"
  else
    echo "  (already exists)"
  fi
fi

echo ""
echo "Data directory:"
find "${DATA_DIR}" -maxdepth 2 -type f | sort
echo "Done → ${DATA_DIR}"
