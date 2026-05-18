"""Smoke test: compare santext / cosine / cosine_clipped on SST-2 examples.

Uses REAL GloVe if available, synthetic embeddings otherwise.
Prints side-by-side sanitized texts so you can judge quality at a glance.

Usage:
    python smoke_test.py                              # synthetic embeddings
    python smoke_test.py --embed_path ./data/glove.840B.300d.txt
    python smoke_test.py --embed_path ./data/glove.840B.300d.txt --n 20 --epsilons 1,2,4
"""
import argparse
import random
import sys
import textwrap
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Parse args first so we can decide synthetic vs real path
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embed_path", default=None,
                   help="Path to GloVe .txt file. Omit for synthetic mode.")
    p.add_argument("--data_dir", default="./data/SST-2",
                   help="SST-2 data dir (default: ./data/SST-2)")
    p.add_argument("--n", type=int, default=20,
                   help="Number of sentences to sample (default: 20)")
    p.add_argument("--epsilons", default="1,2,4",
                   help="Comma-separated epsilon values (default: 1,2,4)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_lo", type=float, default=0.2)
    p.add_argument("--clip_hi", type=float, default=0.8)
    p.add_argument("--no_ner", action="store_true", default=True,
                   help="Skip NER (default True for speed)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Synthetic GloVe-like embeddings (for offline testing)
# ---------------------------------------------------------------------------

SYNTH_DIM = 300

def make_synthetic_embeddings(words: list[str], seed: int = 42) -> np.ndarray:
    """Random unit vectors. Cosine distances are roughly uniform in [0.2, 0.8]."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((len(words), SYNTH_DIM)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.maximum(norms, 1e-9)


# ---------------------------------------------------------------------------
# Minimal GloVe loader (real file)
# ---------------------------------------------------------------------------

def load_glove_subset(path: str, vocab: set[str]):
    """Load only the words we need from GloVe."""
    words, vecs = [], []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            w = parts[0]
            if w in vocab:
                words.append(w)
                vecs.append(np.array(parts[1:], dtype=np.float32))
    embeddings = np.stack(vecs) if vecs else np.empty((0, SYNTH_DIM), dtype=np.float32)
    return words, embeddings


# ---------------------------------------------------------------------------
# Build Sanitizer from embeddings
# ---------------------------------------------------------------------------

def build_sanitizer(words, embeddings, method_str, distance_metric, epsilon,
                    s_epsilon, clip_lo, clip_hi, sensitive_set):
    """Wire up SanitizerConfig + Sanitizer + precompute."""
    from pydantic_models.sanitizerdp import (
        SanitizerDPEmbeddingAndMappings,
        SanitizerDPMethod,
    )
    from sanitizer import SanitizerConfig, Sanitizer

    method = SanitizerDPMethod(method_str)

    word2id = {w: i for i, w in enumerate(words)}
    swords = [w for w in words if w in sensitive_set]
    nwords = [w for w in words if w not in sensitive_set]

    sword2id = {w: i for i, w in enumerate(swords)}
    nword2id = {w: i for i, w in enumerate(nwords)}

    all_e = embeddings
    s_e = embeddings[[word2id[w] for w in swords]] if swords else np.empty((0, embeddings.shape[1]), dtype=np.float32)
    n_e = embeddings[[word2id[w] for w in nwords]] if nwords else np.empty((0, embeddings.shape[1]), dtype=np.float32)

    emb = SanitizerDPEmbeddingAndMappings(
        word2id=word2id,
        sword2id=sword2id,
        nword2id=nword2id,
        all_word_embed=all_e,
        sensitive_word_embed=s_e,
        normal_word_embed=n_e,
    )

    config = SanitizerConfig(
        epsilon=epsilon,
        s_epsilon=s_epsilon,
        method=method,
        distance_metric=distance_metric,
        clip_lo=clip_lo,
        clip_hi=clip_hi,
        replacements_output_dir="/tmp/smoke_replacements",
    )
    san = Sanitizer(config=config)
    san.precompute(words, emb)
    return san


# ---------------------------------------------------------------------------
# Sanitize a single sentence
# ---------------------------------------------------------------------------

def sanitize_sentence(text: str, sanitizer, method_str: str, epsilon_n=None):
    from pydantic_models.sanitizerdp import SanitizerDPDocument, SanitizerDPMethod
    from sanitizer import compute_per_doc_epsilon

    doc = SanitizerDPDocument(text=text, text_id="smoke_0")
    method = SanitizerDPMethod(method_str)

    if method == SanitizerDPMethod.SANTEXT:
        stat = sanitizer.sanitize_santext(doc)
    elif method == SanitizerDPMethod.NORMAL:
        eps_n = epsilon_n
        if eps_n is None:
            eps_n_list = compute_per_doc_epsilon([doc], sanitizer)
            eps_n = eps_n_list[0]
        stat = sanitizer.sanitize_normal(doc, epsilon_n=eps_n)
    elif method == SanitizerDPMethod.PLUS:
        eps_n = epsilon_n
        if eps_n is None:
            eps_n_list = compute_per_doc_epsilon([doc], sanitizer)
            eps_n = eps_n_list[0]
        stat = sanitizer.sanitize_plus(doc, epsilon_n=eps_n)

    import json, os
    repl = os.path.join("/tmp/smoke_replacements", "smoke_0.json")
    if os.path.exists(repl):
        with open(repl) as f:
            return json.load(f)["sanitized_text"]
    return text


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    epsilons = [float(e) for e in args.epsilons.split(",")]

    # Load SST-2
    sentences = []
    try:
        with open(f"{args.data_dir}/train.tsv", encoding="utf-8") as f:
            next(f)  # header
            for line in f:
                parts = line.strip().split("\t")
                if parts:
                    sentences.append(parts[0].strip().lower())
    except FileNotFoundError:
        print(f"[WARN] Could not read {args.data_dir}/train.tsv — using built-in examples")
        sentences = [
            "hide new secretions from the parental units",
            "contains no wit only labored gags",
            "that loves its characters and communicates something rather beautiful about human nature",
            "remains utterly satisfied to remain the same throughout",
            "on the worst revenge of the year list",
            "a movie that delights in its own weirdness",
        ]

    sampled = random.sample(sentences, min(args.n, len(sentences)))

    # Build vocabulary
    all_words = []
    for s in sampled:
        all_words.extend(s.split())
    vocab_counts = Counter(all_words)
    words = [w for w, _ in vocab_counts.most_common()]

    # Sensitive words = top 15% by frequency (NER-free smoke test)
    n_sensitive = max(1, int(0.15 * len(words)))
    sensitive_set = set(words[:n_sensitive])

    print(f"Vocab: {len(words)} words | Sensitive: {len(sensitive_set)} | Sentences: {len(sampled)}")
    if args.embed_path:
        print(f"Loading GloVe from {args.embed_path} …")
        words, embeddings = load_glove_subset(args.embed_path, set(words))
        if not words:
            print("[WARN] No GloVe overlap — falling back to synthetic")
            words = list(vocab_counts.keys())
            embeddings = make_synthetic_embeddings(words, args.seed)
        else:
            print(f"GloVe intersection: {len(words)} words")
    else:
        print("[synthetic embeddings — pass --embed_path for real results]")
        embeddings = make_synthetic_embeddings(words, args.seed)

    # Update sensitive_set to words that are in final vocab
    sensitive_set = sensitive_set & set(words)

    CONFIGS = [
        # (label,        method,    metric,           eps_scale)
        ("santext/cos",  "santext", "cosine",          1.0),
        ("ours/cos",     "normal",  "cosine",          1.0),
        ("ours/clipped", "normal",  "cosine_clipped",  1.0),
        ("ours+/cos",    "plus",    "cosine",          1.0),
        ("ours+/clip",   "plus",    "cosine_clipped",  1.0),
    ]

    for eps in epsilons:
        s_eps = eps / 2
        print(f"\n{'='*100}")
        print(f"  epsilon={eps}  s_epsilon={s_eps}  clip=[{args.clip_lo}, {args.clip_hi}]  "
              f"sensitivity(clipped)={args.clip_hi - args.clip_lo:.2f}")
        print(f"{'='*100}")

        # Pre-build sanitizers
        sanitizers = {}
        for label, method, metric, _ in CONFIGS:
            try:
                sanitizers[label] = build_sanitizer(
                    words, embeddings, method, metric, eps, s_eps,
                    args.clip_lo, args.clip_hi, sensitive_set
                )
            except Exception as e:
                print(f"  [skip {label}]: {e}")

        # Column headers
        col_w = 32
        header = f"{'ORIGINAL':<{col_w}}"
        for label, _, _, _ in CONFIGS:
            if label in sanitizers:
                header += f"  {label:<{col_w}}"
        print(header)
        print("-" * len(header))

        for i, text in enumerate(sampled[:10]):  # print first 10
            row = f"{text[:col_w-1]:<{col_w}}"
            for label, method, _, _ in CONFIGS:
                if label not in sanitizers:
                    continue
                san = sanitizers[label]
                out = sanitize_sentence(text, san, method)
                row += f"  {out[:col_w-1]:<{col_w}}"
            print(row)

        # Quick utility metric: avg unique tokens per output (higher = more diverse = less useful)
        # and avg word overlap with original (higher = more preserved semantic)
        print()
        print(f"  {'METRIC':<20}", end="")
        for label, _, _, _ in CONFIGS:
            if label in sanitizers:
                print(f"  {label:<{col_w}}", end="")
        print()

        for metric_name, metric_fn in [
            ("orig overlap (%)", lambda orig, out: 100 * len(set(orig.split()) & set(out.split())) / max(len(set(orig.split())), 1)),
            ("len ratio",        lambda orig, out: len(out.split()) / max(len(orig.split()), 1)),
        ]:
            print(f"  {metric_name:<20}", end="")
            for label, method, _, _ in CONFIGS:
                if label not in sanitizers:
                    continue
                san = sanitizers[label]
                scores = []
                for text in sampled:
                    out = sanitize_sentence(text, san, method)
                    scores.append(metric_fn(text, out))
                print(f"  {np.mean(scores):>{col_w}.3f}", end="")
            print()

    print("\nDone.")


if __name__ == "__main__":
    main()
