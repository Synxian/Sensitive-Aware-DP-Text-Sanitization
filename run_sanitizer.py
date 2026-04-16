"""Orchestrator for sensitivity-aware DP text sanitization with budget redistribution.

Two-phase pipeline:
    Phase 1: For each document, compute the redistributed normal epsilon.
             SanText would spend epsilon per in-vocab word, so:
             epsilon_t[i] = epsilon * (ns[i] + nn[i])
             epsilon_n[i] = (epsilon_t[i] - ns[i] * epsilon_s) / nn[i]
    Phase 2: Run Ours/Ours+ using the per-document epsilon_n[i].

Usage:
    python run_sanitizer.py \\
        --data_dir ./datasets/i2b2/ \\
        --method normal \\
        --task i2b2 \\
        --epsilon 16 --s_epsilon 8 \\
        --sensitive_words_file_path ./sensitive_mapping/flair_0.6_i2b2.json \\
        --language en
"""

import argparse
import random
import re
import json
import logging
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from spacy.lang.en import English
from spacy.lang.es import Spanish

from pydantic_models.satsdp import (
    SastdpDocument,
    SastdpMethod,
    SastdpExecutionArgs,
)
from sanitizer import Sanitizer, SanitizerConfig, compute_per_doc_epsilon, init_worker, worker_sanitize
from data_loading import get_vocab, get_word_embeddings_and_mappings

logger = logging.getLogger(__name__)


def _parse_args() -> SastdpExecutionArgs:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./datasets/i2b2/", type=str)
    parser.add_argument("--word_embedding_path", default="embeddings/english/glove.840B.300d.txt", type=str)
    parser.add_argument("--word_embedding_size", default=300, type=int)
    parser.add_argument("--method", default=SastdpMethod.NORMAL)
    parser.add_argument("--task", default="i2b2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=16)
    parser.add_argument("--s_epsilon", type=float, default=8)
    parser.add_argument("--p", type=float, default=0.7)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sensitive_words_file_path", type=str, default="./selective_output/sensitive_mapping/0.6_i2b2.json")
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--redistribute", action=argparse.BooleanOptionalAction, default=True)
    return SastdpExecutionArgs(**parser.parse_args().__dict__)


def _load_docs(args: SastdpExecutionArgs, tokenizer) -> list[SastdpDocument]:
    """Load documents from train.csv, tokenize into space-separated words."""
    data_file = os.path.join(args.data_dir, "train.csv")
    df = pd.read_csv(data_file)
    texts = df["text"].dropna()
    text_ids = df["text_id"].dropna()
    assert len(texts) == len(text_ids)

    docs = []
    for text, text_id in tqdm(zip(texts, text_ids), total=len(texts), desc="Loading docs"):
        text = text.strip().replace("\n", " ").replace("\t", " ")
        text = re.sub(" +", " ", text)
        doc = [token.text for token in tokenizer(text)]
        docs.append(SastdpDocument(text=" ".join(doc), text_id=text_id))
    return docs


def main():
    args = _parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    # ---- Build vocabulary and embeddings ----
    tokenizer = English() if args.language == "en" else Spanish()
    vocab = get_vocab(args, tokenizer, "word")
    words = [key.lower() for key, _ in vocab.most_common()]

    with open(args.sensitive_words_file_path, "r", encoding="utf8") as f:
        sensitive_words = [w.lower() for w in json.load(f).keys()]

    embeddings = get_word_embeddings_and_mappings(args, words, sensitive_words)

    # ---- Load documents ----
    docs = _load_docs(args, tokenizer)
    threads = min(args.threads, cpu_count())

    # ---- Build sanitizer and precompute distances ----
    config = SanitizerConfig(
        epsilon=args.epsilon,
        s_epsilon=args.s_epsilon,
        p=args.p,
        method=SastdpMethod(args.method),
        replacements_output_dir=args.replacements_output_dir,
    )
    sanitizer = Sanitizer(config=config)
    sanitizer.precompute(words, embeddings)

    # ================================================================
    # Compute per-document epsilon_n (only for normal/plus with --redistribute)
    # ================================================================
    if config.method in (SastdpMethod.NORMAL, SastdpMethod.PLUS) and args.redistribute:
        logger.info("Computing per-document redistributed epsilon_n...")
        per_doc_epsilon = compute_per_doc_epsilon(docs, sanitizer)

        epsilon_values = [v for v in per_doc_epsilon.values() if v is not None]
        if epsilon_values:
            logger.info(
                "Redistributed epsilon_n stats: min=%.4f, max=%.4f, mean=%.4f",
                min(epsilon_values), max(epsilon_values), np.mean(epsilon_values),
            )
    else:
        # SanText or --no-redistribute: each word uses its fixed epsilon
        per_doc_epsilon = {doc.text_id: None for doc in docs}
        if config.method != SastdpMethod.SANTEXT:
            logger.info(
                "No redistribution: normal words use epsilon=%.4f, sensitive words use s_epsilon=%.4f",
                config.epsilon, config.s_epsilon,
            )

    # ================================================================
    # Sanitize
    # ================================================================
    logger.info("Running %s...", args.method)

    work_items = [(doc, per_doc_epsilon[doc.text_id]) for doc in docs]

    with Pool(threads, initializer=init_worker, initargs=(sanitizer,)) as pool:
        results = list(tqdm(
            pool.imap(worker_sanitize, work_items, chunksize=32),
            total=len(docs),
            desc=f"Sanitize ({args.method})",
        ))

    # ---- Write corpus statistics ----
    corpus_stats = {}
    for result in results:
        corpus_stats[str(result.text_id)] = result.model_dump()
    corpus_stats["corpus_sensitive_words_count"] = len(embeddings.sword2id)
    corpus_stats["corpus_normal_words_count"] = len(embeddings.nword2id)
    corpus_stats["corpus_total_words_count"] = len(embeddings.word2id)
    if config.method != SastdpMethod.SANTEXT:
        corpus_stats["per_doc_epsilon_n"] = {str(k): v for k, v in per_doc_epsilon.items()}

    stats_path = args.corpus_statistics_path
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(corpus_stats, f, indent=2)

    logger.info("Done. Results in %s", config.replacements_output_dir)


if __name__ == "__main__":
    main()
