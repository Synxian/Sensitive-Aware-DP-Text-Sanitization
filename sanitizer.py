"""Class-based sensitivity-aware DP text sanitization.

Replaces the global-variable pattern in sanitize_algorithms.py with a single
Sanitizer object that holds all state. Supports per-document epsilon
redistribution: SanText runs first to establish a budget ceiling per text,
then Ours/Ours+ redistribute the saved sensitive budget to normal words.

Usage:
    sanitizer = Sanitizer(config)
    sanitizer.precompute(words, embeddings)

    # SanText pass (fixed epsilon for all words)
    stats = sanitizer.sanitize_santext(doc)

    # Ours pass (per-document epsilon_n)
    stats = sanitizer.sanitize_normal(doc, epsilon_n=3.5)
"""

import math
import os
import json
import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from dataclasses import dataclass, field

from pydantic_models.sanitizerdp import (
    SanitizerDPDocument,
    SanitizerDPDocumentStatistics,
    SanitizerDPEmbeddingAndMappings,
    SanitizerDPMethod,
)


@dataclass
class SanitizerConfig:
    epsilon: float
    s_epsilon: float
    p: float = 0.7
    method: SanitizerDPMethod = SanitizerDPMethod.NORMAL
    distance_metric: str = "cosine"
    replacements_output_dir: str = "replacements"


@dataclass
class Sanitizer:
    """Holds precomputed distance matrices, mappings, and config.

    The expensive distance computation happens once in precompute().
    Probability matrices are built on demand with the appropriate epsilon.
    """

    config: SanitizerConfig

    # Populated by precompute()
    vocab: list[str] = field(default_factory=list)
    word2id: dict[str, int] = field(default_factory=dict)
    sword2id: dict[str, int] = field(default_factory=dict)
    nword2id: dict[str, int] = field(default_factory=dict)
    id2word: dict[int, str] = field(default_factory=dict)
    id2sword: dict[int, str] = field(default_factory=dict)
    id2nword: dict[int, str] = field(default_factory=dict)

    # Distance matrices (computed once, never change)
    s_distance_matrix: np.ndarray | None = None
    n_distance_matrix: np.ndarray | None = None

    # Sensitivity of the utility function u(x,y) = -distance(x,y).
    sensitivity: float = 1.0

    # Sensitive prob matrix (fixed epsilon_s, precomputed once)
    s_prob_matrix: np.ndarray | None = None

    # Normal prob matrix at the fixed config epsilon.
    n_prob_matrix_fixed: np.ndarray | None = None

    @property
    def mixing_overhead(self) -> float:
        """L = ln(max(p/(1-p), (1-p)/p)) — extra LDP cost per word from mixing."""
        if self.config.method != SanitizerDPMethod.PLUS:
            return 0.0
        p = self.config.p
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1) for method=plus, got p={p}")
        return math.log(max(p / (1 - p), (1 - p) / p))

    def precompute(self, vocab: list[str], embeddings: SanitizerDPEmbeddingAndMappings):
        """Compute distance matrices and fixed sensitive prob matrix.

        Call once after loading embeddings. This is the expensive step.
        """
        self.vocab = vocab
        self.word2id = embeddings.word2id
        self.sword2id = embeddings.sword2id
        self.nword2id = embeddings.nword2id
        self.id2word = {v: k for k, v in self.word2id.items()}
        self.id2sword = {v: k for k, v in self.sword2id.items()}
        self.id2nword = {v: k for k, v in self.nword2id.items()}

        if self.config.method == SanitizerDPMethod.PLUS:
            if not self.sword2id or not self.nword2id:
                raise ValueError("method='plus' requires non-empty sensitive and normal vocabularies.")

        if self.config.distance == "cosine":
            distance_fn = cosine_distances
        elif self.config.distance == "euclidean":
            distance_fn = euclidean_distances
        else:
            raise ValueError(f"Unknown distance: {self.config.distance!r}. Expected 'cosine' or 'euclidean'.")


        if self.config.method == SanitizerDPMethod.NORMAL:
             # s_dist: |V_s| x |V_all|, n_dist: |V_n| x |V_all|
            self.s_distance_matrix = distance_fn(
                embeddings.sensitive_word_embed, embeddings.all_word_embed
            )
            self.n_distance_matrix = distance_fn(
                embeddings.normal_word_embed, embeddings.all_word_embed
            )
        elif self.config.method == SanitizerDPMethod.PLUS:
            # s_dist: |V_all| x |V_s|, n_dist: |V_all| x |V_n|
            self.s_distance_matrix = distance_fn(
                embeddings.all_word_embed, embeddings.sensitive_word_embed
            )
            self.n_distance_matrix = distance_fn(
                embeddings.all_word_embed, embeddings.normal_word_embed
            )
        elif self.config.method == SanitizerDPMethod.SANTEXT:
             # Single square matrix
            self.n_distance_matrix = distance_fn(
                embeddings.all_word_embed, embeddings.all_word_embed
            )

        self.sensitivity = 1.0

        if self.s_distance_matrix is not None:
            L = self.mixing_overhead
            s_mech_eps = self.config.s_epsilon - L
            assert s_mech_eps > 0, (
                f"s_epsilon ({self.config.s_epsilon}) must exceed mixing overhead "
                f"L={L:.4f} for method=plus with p={self.config.p}"
            )
            self.s_prob_matrix = self._build_prob_matrix(
                self.s_distance_matrix, s_mech_eps)

        if self.config.method == SanitizerDPMethod.SANTEXT:
            self.n_prob_matrix_fixed = self._build_n_prob_matrix(self.config.epsilon)

    def _build_prob_matrix(self, distance_matrix: np.ndarray, eps: float) -> np.ndarray:
        """Apply the exponential mechanism: softmax(eps * -distance / (2 * sensitivity))."""
        return softmax(eps * (-distance_matrix) / (2 * self.sensitivity), axis=1)

    def _build_n_prob_matrix(self, epsilon_n: float) -> np.ndarray:
        assert self.n_distance_matrix is not None, "Call precompute() first"
        return self._build_prob_matrix(self.n_distance_matrix, epsilon_n)

    # ------------------------------------------------------------------
    # Word counting
    # ------------------------------------------------------------------

    def count_words(self, doc: SanitizerDPDocument) -> tuple[int, int, int]:
        """Count sensitive, normal, and OOV words. Returns (ns, nn, n_oov)."""
        ns = nn = n_oov = 0
        for raw_word in doc.text.split():
            word = raw_word.lower()
            if word in self.word2id:
                if word in self.sword2id:
                    ns += 1
                else:
                    nn += 1
            else:
                n_oov += 1
        return ns, nn, n_oov

    # ------------------------------------------------------------------
    # Sanitization methods
    # ------------------------------------------------------------------

    def sanitize_santext(self, doc: SanitizerDPDocument) -> SanitizerDPDocumentStatistics:
        """Baseline: all in-vocab words use the same epsilon over full vocab."""
        assert self.n_prob_matrix_fixed is not None, "Call precompute() first"
        n_prob_matrix = self.n_prob_matrix_fixed
        new_doc = []
        total_epsilon = 0.0
        normal_word_count = 0
        out_of_vocab_word_count = 0

        for raw_word in doc.text.split():
            word = raw_word.lower()
            if word in self.word2id:
                normal_word_count += 1
                sampling_prob = n_prob_matrix[self.word2id[word]]
                idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                new_doc.append(self.id2word[idx])
                total_epsilon += self.config.epsilon
            else:
                out_of_vocab_word_count += 1
                idx = np.random.randint(len(self.vocab))
                new_doc.append(self.vocab[idx])

        sanitized_text = " ".join(new_doc)
        self._write_replacements(doc, sanitized_text, total_epsilon)
        return SanitizerDPDocumentStatistics(
            text_id=doc.text_id,
            sensitive_word_count=0,
            normal_word_count=normal_word_count,
            total_word_count=out_of_vocab_word_count,
            total_epsilon=total_epsilon,
        )

    def sanitize_normal(
        self, doc: SanitizerDPDocument, epsilon_n: float | None = None
    ) -> SanitizerDPDocumentStatistics:
        """Ours: sensitive words use epsilon_s, normal words use epsilon_n."""
        assert self.s_prob_matrix is not None, "Call precompute() first"
        eps_n = epsilon_n if epsilon_n is not None else self.config.epsilon
        if epsilon_n is None:
            if self.n_prob_matrix_fixed is None:
                self.n_prob_matrix_fixed = self._build_n_prob_matrix(eps_n)
            n_prob_matrix = self.n_prob_matrix_fixed
        else:
            n_prob_matrix = self._build_n_prob_matrix(eps_n)

        new_doc = []
        total_epsilon = 0.0
        sensitive_word_count = 0
        normal_word_count = 0
        out_of_vocab_word_count = 0

        for raw_word in doc.text.split():
            word = raw_word.lower()
            if word in self.word2id:
                if word in self.sword2id:
                    sensitive_word_count += 1
                    sampling_prob = self.s_prob_matrix[self.sword2id[word]]
                    idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                    new_doc.append(self.id2word[idx])
                    total_epsilon += self.config.s_epsilon
                else:
                    normal_word_count += 1
                    sampling_prob = n_prob_matrix[self.nword2id[word]]
                    idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                    new_doc.append(self.id2word[idx])
                    total_epsilon += eps_n
            else:
                out_of_vocab_word_count += 1
                idx = np.random.randint(len(self.vocab))
                new_doc.append(self.vocab[idx])

        sanitized_text = " ".join(new_doc)
        self._write_replacements(doc, sanitized_text, total_epsilon)
        return SanitizerDPDocumentStatistics(
            text_id=doc.text_id,
            sensitive_word_count=sensitive_word_count,
            normal_word_count=normal_word_count,
            total_word_count=out_of_vocab_word_count,
            total_epsilon=total_epsilon,
        )

    def sanitize_plus(
        self, doc: SanitizerDPDocument, epsilon_n: float | None = None
    ) -> SanitizerDPDocumentStatistics:
        """Ours+: mixed sampling with coin flip probability p."""
        assert self.s_prob_matrix is not None, "Call precompute() first"
        eps_n = epsilon_n if epsilon_n is not None else self.config.epsilon
        L = self.mixing_overhead
        eps_n_mech = eps_n - L
        assert eps_n_mech > 0, (
            f"epsilon_n ({eps_n}) must exceed mixing overhead "
            f"L={L:.4f} for method=plus with p={self.config.p}"
        )
        if epsilon_n is None:
            if self.n_prob_matrix_fixed is None:
                self.n_prob_matrix_fixed = self._build_n_prob_matrix(eps_n_mech)
            n_prob_matrix = self.n_prob_matrix_fixed
        else:
            n_prob_matrix = self._build_n_prob_matrix(eps_n_mech)
        p = self.config.p

        new_doc = []
        total_epsilon = 0.0
        sensitive_word_count = 0
        normal_word_count = 0
        out_of_vocab_word_count = 0

        for raw_word in doc.text.split():
            word = raw_word.lower()
            if word in self.word2id:
                flip = np.random.random()
                index = self.word2id[word]
                if word in self.sword2id:
                    sensitive_word_count += 1
                    if flip <= p:
                        sampling_prob = self.s_prob_matrix[index]
                        idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                        new_doc.append(self.id2sword[idx])
                        total_epsilon += self.config.s_epsilon
                    else:
                        sampling_prob = n_prob_matrix[index]
                        idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                        new_doc.append(self.id2nword[idx])
                        total_epsilon += eps_n
                else:
                    normal_word_count += 1
                    if flip <= p:
                        sampling_prob = n_prob_matrix[index]
                        idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                        new_doc.append(self.id2nword[idx])
                        total_epsilon += eps_n
                    else:
                        sampling_prob = self.s_prob_matrix[index]
                        idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                        new_doc.append(self.id2sword[idx])
                        total_epsilon += self.config.s_epsilon
            else:
                out_of_vocab_word_count += 1
                idx = np.random.randint(len(self.vocab))
                new_doc.append(self.vocab[idx])

        sanitized_text = " ".join(new_doc)
        self._write_replacements(doc, sanitized_text, total_epsilon)
        return SanitizerDPDocumentStatistics(
            text_id=doc.text_id,
            sensitive_word_count=sensitive_word_count,
            normal_word_count=normal_word_count,
            total_word_count=out_of_vocab_word_count,
            total_epsilon=total_epsilon,
        )

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def _write_replacements(
        self, doc: SanitizerDPDocument, sanitized_text: str, total_epsilon: float
    ):
        replacements = {
            "text_id": doc.text_id,
            "original_text": doc.text,
            "sanitized_text": sanitized_text,
            "total_epsilon": total_epsilon,
        }
        path = os.path.join(
            self.config.replacements_output_dir, f"{doc.text_id}.json"
        )
        os.makedirs(self.config.replacements_output_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(replacements, f, ensure_ascii=False, indent=4)


# ----------------------------------------------------------------------
# Per-document epsilon redistribution
# ----------------------------------------------------------------------

def compute_per_doc_epsilon(
    docs: list[SanitizerDPDocument],
    sanitizer: Sanitizer,
) -> list:
    """Compute per-document redistributed epsilon_n.

    SanText spends epsilon per in-vocab word, so for document i:
        epsilon_t[i] = epsilon * (ns[i] + nn[i])
        epsilon_n[i] = (epsilon_t[i] - ns[i] * epsilon_s) / nn[i]
    """
    import logging
    _logger = logging.getLogger(__name__)

    epsilon = sanitizer.config.epsilon
    epsilon_s = sanitizer.config.s_epsilon
    result = []

    for doc in docs:
        ns, nn, _ = sanitizer.count_words(doc)
        if nn == 0:
            result.append(None)
            continue

        epsilon_t = epsilon * (ns + nn)
        epsilon_n = (epsilon_t - ns * epsilon_s) / nn

        if epsilon_n <= 0:
            _logger.warning(
                "Document %s: epsilon_n=%.4f <= 0 (ns=%d, nn=%d). Clamping to 0.01.",
                doc.text_id, epsilon_n, ns, nn,
            )
            epsilon_n = 0.01

        result.append(epsilon_n)

    return result


# ----------------------------------------------------------------------
# Multiprocessing support
# ----------------------------------------------------------------------

_sanitizer: Sanitizer | None = None


def init_worker(sanitizer: Sanitizer):
    """Pool initializer. Stores the Sanitizer instance in the worker."""
    global _sanitizer
    _sanitizer = sanitizer


def worker_sanitize(args: tuple) -> SanitizerDPDocumentStatistics:
    """Worker function. Dispatches to the correct method based on config."""
    assert _sanitizer is not None
    doc, epsilon_n = args
    if _sanitizer.config.method == SanitizerDPMethod.SANTEXT:
        return _sanitizer.sanitize_santext(doc)
    elif _sanitizer.config.method == SanitizerDPMethod.NORMAL:
        return _sanitizer.sanitize_normal(doc, epsilon_n=epsilon_n)
    elif _sanitizer.config.method == SanitizerDPMethod.PLUS:
        return _sanitizer.sanitize_plus(doc, epsilon_n=epsilon_n)
    raise ValueError(f"Unsupported method: {_sanitizer.config.method}")


def sanitize_corpus(docs, sanitizer, per_doc_epsilons, threads=4, desc="Sanitizing"):
    """Sanitize documents in parallel. Returns list of SanitizerDPDocumentStatistics."""
    from multiprocessing import Pool, cpu_count
    from tqdm import tqdm

    work_items = list(zip(docs, per_doc_epsilons))
    threads = min(threads, cpu_count())

    if threads <= 1:
        init_worker(sanitizer)
        return [worker_sanitize(item) for item in tqdm(work_items, desc=desc)]

    with Pool(threads, initializer=init_worker, initargs=(sanitizer,)) as pool:
        return list(tqdm(
            pool.imap(worker_sanitize, work_items, chunksize=32),
            total=len(docs), desc=desc,
        ))
