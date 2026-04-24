# Must be set BEFORE importing transformers (via flair)
import os as _os
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"

"""Sanitize a GLUE-style dataset with Sensitivity-Aware DP.

Methods
-------
santext   SanText baseline (Yue et al. 2021): uniform epsilon for all words.
normal    NADPTextSan: s_epsilon for sensitive, epsilon for normal words.
plus      NADPTextSan+: mixed sampling with probability p.

Usage
-----
    python run_sanitize.py \\
        --task sst2 \\
        --method normal \\
        --epsilon 10 \\
        --data_dir  ./data/SST-2 \\
        --embed_path ./data/glove.840B.300d.txt \\
        --output_dir ./output/sst2/normal/eps_10.00_seps_5.00
"""
import argparse
import json
import logging
import os
import random

import numpy as np

from sanitizer import (get_tokenizer, build_vocab, filter_vocab,
                       load_glove, build_sensitive_words_ner,
                       build_sensitive_words, cal_probability,
                       compute_total_epsilon, sanitize_corpus)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task I/O — SST-2 and QNLI (inline, no class hierarchy)
# ---------------------------------------------------------------------------

def read_sst2(path, tokenizer, max_samples=None):
    docs, labels = [], []
    with open(path, encoding="utf-8") as f:
        header = next(f)
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            docs.append([tok.text for tok in tokenizer(parts[0])])
            labels.append(parts[1])
    return docs, labels, header


def write_sst2(path, sanitized, labels, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for text, label in zip(sanitized, labels):
            f.write(f"{text}\t{label}\n")


def read_qnli(path, tokenizer, max_samples=None):
    """Returns 2 docs per example (question + sentence)."""
    docs, labels = [], []
    with open(path, encoding="utf-8") as f:
        header = next(f)
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            docs.append([tok.text for tok in tokenizer(parts[1])])
            docs.append([tok.text for tok in tokenizer(parts[2])])
            labels.append(parts[3])
    return docs, labels, header


def write_qnli(path, sanitized, labels, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for i, label in enumerate(labels):
            f.write(f"{i}\t{sanitized[i*2]}\t{sanitized[i*2+1]}\t{label}\n")


def extract_texts(task, data_dir, max_samples=None):
    """Collect raw text strings across train+dev for vocab building."""
    texts = []
    for split in ("train", "dev"):
        path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(path):
            continue
        cap = max_samples if (max_samples and split == "train") else None
        with open(path, encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if cap is not None and i >= cap:
                    break
                parts = line.strip().split("\t")
                if task == "sst2" and parts:
                    texts.append(parts[0])
                elif task == "qnli" and len(parts) >= 3:
                    texts.append(parts[1])
                    texts.append(parts[2])
    return texts


READERS  = {"sst2": read_sst2,  "qnli": read_qnli}
WRITERS  = {"sst2": write_sst2, "qnli": write_qnli}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     required=True, choices=["sst2", "qnli"])
    p.add_argument("--method",   required=True, choices=["santext", "normal", "plus"])
    p.add_argument("--epsilon",  type=float, default=10.0,
                   help="Privacy budget ε_n for normal words")
    p.add_argument("--s_epsilon", type=float, default=None,
                   help="Privacy budget ε_s for sensitive words (default: epsilon/2)")
    p.add_argument("--p",        type=float, default=0.7,
                   help="Cross-vocab sampling probability for 'plus' (paper default 0.7)")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--embed_path", required=True)
    p.add_argument("--output_dir", required=True)
    # Sensitivity detection
    p.add_argument("--no_ner", action="store_true",
                   help="Disable NER; fall back to frequency-based sensitive_pct")
    p.add_argument("--ner_model", default="flair/ner-english-large",
                   help="Flair NER model tag (default: ner-english-large)")
    p.add_argument("--ner_threshold", type=float, default=0.3,
                   help="Minimum NER entity confidence (paper uses 0.3)")
    p.add_argument("--sensitive_pct", type=float, default=0.5,
                   help="Fallback: fraction of vocab treated as sensitive (--no_ner only)")
    p.add_argument("--sensitive_words_path", default=None,
                   help="Fallback: JSON file {word: ...} with sensitive words (--no_ner only)")
    p.add_argument("--min_freq",  type=int, default=1,
                   help="Min word frequency (use 5 for QNLI to save ~10 GB RAM)")
    p.add_argument("--max_vocab", type=int, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    if args.s_epsilon is None:
        args.s_epsilon = args.epsilon / 2

    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.basicConfig(format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = get_tokenizer()

    # Vocab
    logger.info("Building vocab for %s …", args.task)
    texts = extract_texts(args.task, args.data_dir, max_samples=args.max_samples)
    vocab = build_vocab(texts, tokenizer)
    words = filter_vocab(vocab, min_freq=args.min_freq, max_vocab=args.max_vocab)
    logger.info("Vocab: %d words (min_freq=%d)", len(words), args.min_freq)

    # --- Sensitive word detection (with cache) ---
    ner_cache = os.path.join(args.data_dir, "sensitive_words_ner.json")

    if args.no_ner:
        sensitive = build_sensitive_words(words, args.sensitive_pct,
                                          args.sensitive_words_path)
        logger.info("Sensitive (freq-based): %d / %d words", len(sensitive), len(words))
    elif os.path.exists(ner_cache):
        with open(ner_cache, encoding="utf-8") as f:
            sensitive = set(json.load(f))
        logger.info("Sensitive (NER cached): %d tokens loaded from %s", len(sensitive), ner_cache)
    else:
        sensitive = build_sensitive_words_ner(
            texts, model_name=args.ner_model, threshold=args.ner_threshold)
        with open(ner_cache, "w", encoding="utf-8") as f:
            json.dump(sorted(sensitive), f, ensure_ascii=False, indent=2)
        logger.info("Sensitive (NER): %d tokens → cached to %s", len(sensitive), ner_cache)

    # Embeddings
    word2id, sword2id, nword2id, all_words, all_embed, s_embed, n_embed = \
        load_glove(args.embed_path, set(words), sensitive)
    logger.info("GloVe intersection: %d words", len(word2id))

    # Probability matrices
    logger.info("Building probability matrices (method=%s) …", args.method)
    s_prob = n_prob = None
    if args.method == "santext":
        n_prob = cal_probability(all_embed, all_embed, args.epsilon)
    elif args.method == "normal":
        s_prob = cal_probability(s_embed,   all_embed, args.s_epsilon)
        n_prob = cal_probability(n_embed,   all_embed, args.epsilon)
    elif args.method == "plus":
        s_prob = cal_probability(all_embed, s_embed,   args.s_epsilon)
        n_prob = cal_probability(all_embed, n_embed,   args.epsilon)

    read  = READERS[args.task]
    write = WRITERS[args.task]

    stats = {}
    for split in ("train", "dev"):
        in_path  = os.path.join(args.data_dir,   f"{split}.tsv")
        out_path = os.path.join(args.output_dir, f"{split}.tsv")
        if not os.path.exists(in_path):
            logger.warning("Missing %s — skipping", in_path)
            continue

        max_s = args.max_samples if split == "train" else None
        docs, labels, header = read(in_path, tokenizer, max_samples=max_s)

        sanitized = sanitize_corpus(
            docs, word2id, sword2id, nword2id,
            s_prob, n_prob, all_words,
            method=args.method, p=args.p,
            threads=args.threads, desc=f"  {split}",
        )
        write(out_path, sanitized, labels, header)

        # Total epsilon = max per-document epsilon budget consumed
        if args.method == "santext":
            total_eps = compute_total_epsilon(docs, {}, 0.0, args.epsilon)
        else:
            total_eps = compute_total_epsilon(
                docs, sword2id, args.s_epsilon, args.epsilon)
        stats[split] = {"examples": len(labels), "total_epsilon": total_eps}
        logger.info("Wrote %d examples → %s  (total_ε=%.2f)",
                    len(labels), out_path, total_eps)

    # Save run metadata
    meta = {
        "task": args.task, "method": args.method,
        "epsilon": args.epsilon, "s_epsilon": args.s_epsilon,
        "p": args.p, "use_ner": not args.no_ner,
        "splits": stats,
    }
    with open(os.path.join(args.output_dir, "sanitize_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done → %s", args.output_dir)


if __name__ == "__main__":
    main()
