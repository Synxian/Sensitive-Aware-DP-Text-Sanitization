# Sensitivity-Aware DP Text Sanitization

Differential privacy text sanitization with three methods:

| Method | Description |
|--------|-------------|
| `santext`  | SanText baseline (Yue et al. 2021): uniform ε for all words |
| `normal`   | NADPTextSan: ε_s for sensitive words, ε for normal words |
| `plus`     | NADPTextSan+: mixed sampling with probability p |

Supported tasks: **SST-2**, **QNLI**.

---

## Installation

```bash
# Con uv (recomendado, ~10x más rápido)
make install

# O manualmente:
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Data setup

```bash
bash setup_data.sh              # downloads SST-2, QNLI, GloVe 840B (~2.2 GB)
bash setup_data.sh --no-glove   # skip GloVe if you already have it
```

---

## Usage

### Sanitize a dataset

```bash
python run_sanitize.py \
    --task sst2 \
    --method normal \
    --epsilon 10 \
    --data_dir  ./data/SST-2 \
    --embed_path ./data/glove.840B.300d.txt \
    --output_dir ./output/sst2/normal/eps_10.00_seps_5.00
```

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | — | `sst2`, `qnli` |
| `--method` | — | `santext`, `normal`, `plus` |
| `--epsilon` | 10.0 | Privacy budget for normal words |
| `--s_epsilon` | ε/2 | Privacy budget for sensitive words |
| `--p` | 0.7 | Mixing probability (`plus` only) |
| `--min_freq` | 1 | Min word frequency (use 5 for QNLI to reduce memory) |
| `--sensitive_pct` | 0.5 | Fraction of vocab treated as sensitive (bottom N% by frequency) |
| `--threads` | 4 | Multiprocessing workers |
| `--max_samples` | — | Cap train examples (smoke tests) |

> **QNLI memory note:** Use `--min_freq 5` to reduce the probability matrix
> from ~14 GB to ~3 GB (1.8% token coverage loss).

### Downstream evaluation (BERT fine-tuning)

```bash
python run_downstream.py \
    --task sst2 \
    --data_dir  ./output/sst2/normal/eps_10.00_seps_5.00 \
    --output_dir ./results/sst2/normal/eps_10.00_seps_5.00
```

To get the **baseline** (no sanitization), run downstream directly on the original data:

```bash
python run_downstream.py \
    --task sst2 \
    --data_dir  ./data/SST-2 \
    --output_dir ./results/sst2/baseline
```

Requires `bert-base-uncased` in the HuggingFace cache (`local_files_only=True`).

### Reproduce paper results

To reproduce the full table from the paper (3 methods × 4 epsilons × 2 tasks = 24 configs):

```bash
python run_experiment.py \
    --tasks sst2 qnli \
    --methods santext normal plus \
    --epsilons 1 2 4 8 \
    --p_values 0.7 \
    --min_freq 5 \
    --downstream
```

For the baselines (no sanitization):

```bash
python run_downstream.py --task sst2 --data_dir ./data/SST-2 --output_dir ./results/sst2/baseline
python run_downstream.py --task qnli --data_dir ./data/QNLI --output_dir ./results/qnli/baseline
```

Produces `results/summary.csv`. Skips configs where output already exists (safe to resume).

> **NER caching:** The first run performs Flair NER on the full corpus and saves the
> sensitive words to `data/<TASK>/sensitive_words_ner.json`. Subsequent runs reuse
> this cache automatically. Delete the file to force re-detection.

**Expected results (paper Table 1):**

| Method | ε_n | ε_s | p | SST-2 Acc (%) | QNLI Acc (%) | SST-2 total ε | QNLI total ε |
|--------|-----|-----|---|---------------|---------------|----------------|---------------|
| Original | — | — | — | 92.43 | 90.87 | 0 | 0 |
| SanText | 1 | — | — | 49.66 | 52.48 | 53 | 431 |
| SanText | 2 | — | — | 50.46 | 53.32 | 106 | 862 |
| SanText | 4 | — | — | 49.66 | 52.24 | 212 | 1724 |
| SanText | 8 | — | — | 49.42 | 52.90 | 424 | 3448 |
| NERaseText | 1 | 0.5 | — | 51.48 | 52.50 | 47.5 | 373 |
| NERaseText | 2 | 1 | — | 50.34 | 53.40 | 95 | 746 |
| NERaseText | 4 | 2 | — | 50.34 | 53.50 | 190 | 1492 |
| NERaseText | 8 | 4 | — | 49.54 | 52.30 | 380 | 2984 |
| NERaseText+ | 1 | 0.5 | 0.7 | **51.50** | 52.90 | **43** | 345.5 |
| NERaseText+ | 2 | 1 | 0.7 | 47.13 | 51.90 | 92 | 708 |
| NERaseText+ | 4 | 2 | 0.7 | 51.80 | **53.60** | 180 | **1410** |
| NERaseText+ | 8 | 4 | 0.7 | 48.90 | 53.60 | 360 | 2812 |

---

## Repository structure

```
nadp-text-sanitization/
├── sanitizer.py         # Core DP library: tokenizer, vocab, embeddings, probability, sanitization
├── run_sanitize.py      # CLI: sanitize SST-2 / QNLI datasets
├── run_downstream.py    # CLI: BERT fine-tuning on sanitized (or original) data
├── run_experiment.py    # CLI: sweep orchestrator → results/summary.csv
├── tests/
│   └── test_sanitizer.py
├── data/                # (gitignored) SST-2, QNLI, GloVe
├── requirements.txt
├── Makefile
└── setup_data.sh
```
