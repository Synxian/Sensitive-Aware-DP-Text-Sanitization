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

from pydantic_models.sanitizerdp import SanitizerDPDocument
from sanitizer import SanitizerConfig, Sanitizer, compute_per_doc_epsilon, sanitize_corpus
from utils import (get_tokenizer, build_vocab, load_glove,
                   build_sensitive_words, extract_texts, READERS, WRITERS)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",     required=True, choices=["sst2", "qnli"])
    p.add_argument("--method",   required=True, choices=["santext", "normal", "plus"])
    p.add_argument("--distance_metric", default="cosine", choices=["cosine", "euclidean"],
                   help="Distance metric for exponential mechanism (default: cosine)")
    p.add_argument("--epsilon",  type=float, default=10.0,
                   help="Privacy budget epsilon_n for normal words")
    p.add_argument("--s_epsilon", type=float, default=None,
                   help="Privacy budget epsilon_s for sensitive words (default: epsilon/2)")
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
    p.add_argument("--sensitive_source", default="dataset",
                   choices=["dataset", "glove", "dataset+glove"],
                   help="Source texts for NER: dataset, glove, or dataset+glove")
    p.add_argument("--sensitive_pct", type=float, default=0.5,
                   help="Fallback: fraction of vocab treated as sensitive (--no_ner only)")
    p.add_argument("--sensitive_words_path", default=None,
                   help="Fallback: JSON file {word: ...} with sensitive words (--no_ner only)")
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
    words = [w for w, _ in vocab.most_common()]
    logger.info("Vocab: %d words", len(words))

    # Sensitive word detection (with cache)
    sensitive = build_sensitive_words(
        texts=texts,
        vocab_words=words,
        use_ner=not args.no_ner,
        ner_model=args.ner_model,
        ner_threshold=args.ner_threshold,
        sensitive_pct=args.sensitive_pct,
        sensitive_words_path=args.sensitive_words_path,
        source=args.sensitive_source,
        glove_path=args.embed_path,
        task=args.task,
        seed=args.seed,
        output_dir=os.path.join(os.path.dirname(args.output_dir), "sensitive_words"),
    )
    logger.info("Sensitive words: %d", len(sensitive))

    # Embeddings
    vocab_words, embeddings = load_glove(args.embed_path, set(words), sensitive)
    logger.info("GloVe intersection: %d words", len(embeddings.word2id))

    # Build sanitizer and precompute distances
    logger.info("Building distance matrices (method=%s, metric=%s) …",
                args.method, args.distance_metric)
    config = SanitizerConfig(
        epsilon=args.epsilon,
        s_epsilon=args.s_epsilon,
        p=args.p,
        method=args.method,
        distance_metric=args.distance_metric,
        replacements_output_dir=os.path.join(args.output_dir, "replacements"),
    )
    sanitizer = Sanitizer(config=config)
    sanitizer.precompute(vocab_words, embeddings)

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
        docs_tokens, labels, header = read(in_path, tokenizer, max_samples=max_s)

        # Convert to SanitizerDPDocument
        docs = []
        for i, tokens in enumerate(docs_tokens):
            docs.append(SanitizerDPDocument(
                text=" ".join(tokens),
                text_id=f"{args.task}_{split}_{i}",
            ))

        # Compute per-document epsilon redistribution
        if args.method in ("normal", "plus"):
            per_doc_epsilons = compute_per_doc_epsilon(docs, sanitizer)
        else:
            per_doc_epsilons = [None] * len(docs)

        results = sanitize_corpus(
            docs, sanitizer, per_doc_epsilons,
            threads=args.threads, desc=f"  {split}",
        )

        # Extract sanitized texts and epsilons
        sanitized_texts = []
        epsilons = []
        for i, result in enumerate(results):
            # Read sanitized text from replacement file
            repl_path = os.path.join(
                config.replacements_output_dir, f"{result.text_id}.json")
            if os.path.exists(repl_path):
                with open(repl_path, encoding="utf-8") as f:
                    sanitized_texts.append(json.load(f)["sanitized_text"])
            else:
                sanitized_texts.append(" ".join(docs_tokens[i]))
            epsilons.append(result.total_epsilon)

        write(out_path, sanitized_texts, labels, header)
        total_eps = max(epsilons) if epsilons else 0.0
        stats[split] = {"examples": len(labels), "total_epsilon": total_eps}
        logger.info("Wrote %d examples → %s  (max_ε=%.2f)",
                    len(labels), out_path, total_eps)

    # Save run metadata
    meta = {
        "task": args.task, "method": args.method,
        "epsilon": args.epsilon, "s_epsilon": args.s_epsilon,
        "p": args.p, "use_ner": not args.no_ner,
        "distance_metric": args.distance_metric,
        "sensitive_source": args.sensitive_source,
        "splits": stats,
    }
    with open(os.path.join(args.output_dir, "sanitize_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    logger.info("Done → %s", args.output_dir)


if __name__ == "__main__":
    main()
