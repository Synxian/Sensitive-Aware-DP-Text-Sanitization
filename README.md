# Sensitivity-Aware DP Text Sanitization

Differential privacy text sanitization with three methods:

| Method | Description |
|--------|-------------|
| `santext`  | SanText baseline (Yue et al. 2021): uniform ε for all words |
| `normal`   | NERaseText: ε_s for sensitive words, ε for normal words |
| `plus`     | NERaseText+: mixed sampling with probability p |

Supported tasks: **SST-2**, **QNLI**.

---

## Installation

```bash
# Con uv (recomendado)
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
| `--distance_metric` | cosine | `cosine` or `euclidean` |
| `--sensitive_source` | dataset | `dataset`, `glove`, or `dataset+glove` |
| `--no_ner` | false | Disable NER; use frequency-based detection |
| `--threads` | 4 | Multiprocessing workers |
| `--max_samples` | — | Cap train examples (smoke tests) |

> **Sensitive word sources:** `--sensitive_source` controls where NER runs:
> - `dataset`: NER on task texts only
> - `glove`: NER on GloVe vocabulary words
> - `dataset+glove`: NER on union of both

> **Caching:** Sensitive words are cached automatically in
> `output/sensitive_words/{model}_{source}_{task}_{threshold}thr_{seed}seed.json`.
> Delete the cache file to force re-detection.

### Downstream evaluation (BERT fine-tuning)

```bash
python run_downstream.py \
    --task sst2 \
    --data_dir  ./output/sst2/normal/eps_10.00_seps_5.00 \
    --output_dir ./output/sst2/normal/eps_10.00_seps_5.00
```

Baseline (no sanitization):

```bash
python run_downstream.py \
    --task sst2 \
    --data_dir  ./data/SST-2 \
    --output_dir ./output/baseline/sst2
```

### Quality metrics

```bash
python -m quality_metrics_task.run_quality_metrics \
    --original_dir ./data/SST-2 \
    --sanitized_dir ./output/sst2/normal/eps_10.00_seps_5.00 \
    --task sst2 \
    --metrics all
```

Available metrics: `jaccard`, `bert_score`, `mauve`, `movers_distance`, or `all`.

### Reproduce paper results

```bash
python run_experiment.py \
    --tasks sst2 qnli \
    --methods santext normal plus \
    --epsilons 1 2 4 8 \
    --p_values 0.7 \
    --downstream
```

With quality metrics:

```bash
python run_experiment.py \
    --tasks sst2 \
    --methods santext normal plus \
    --epsilons 1 2 4 8 \
    --downstream \
    --quality_metrics all
```

All output goes to `output/`. Skips configs that already exist (safe to resume).

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
├── sanitizer.py              # Sanitizer class: distance matrices, prob matrices, sanitization
├── sanitize_algorithms.py    # Legacy global-variable pattern (SanText, NADPTextSan, NADPTextSan+)
├── utils.py                  # Tokenizer, vocab, GloVe, build_sensitive_words (NER/freq)
├── run_sanitize.py           # CLI: sanitize SST-2 / QNLI datasets
├── run_downstream.py         # CLI: BERT fine-tuning on sanitized (or original) data
├── run_experiment.py         # CLI: sweep orchestrator → output/summary.csv
├── pydantic_models/
│   └── sanitizerdp.py        # Pydantic data models (SanitizerDPDocument, etc.)
├── quality_metrics_task/
│   ├── run_quality_metrics.py  # Orchestrator: jaccard, BERTScore, MAUVE, movers distance
│   ├── run_bert_score.py
│   ├── run_jaccard.py
│   ├── run_mauve.py
│   └── run_movers_distance.py
├── tests/
│   └── test_sanitizer.py
├── data/                     # (gitignored) SST-2, QNLI, GloVe
├── output/                   # (gitignored) all results: sanitized, downstream, metrics
├── requirements.txt
├── Makefile
└── setup_data.sh
```
