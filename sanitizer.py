"""Class-based sensitivity-aware DP text sanitization.

Replaces the global-variable pattern in nandptextsan.py with a single
Sanitizer object that holds all state. Supports per-document epsilon
redistribution: SanText runs first to establish a budget ceiling per text,
then Ours/Ours+ redistribute the saved sensitive budget to normal words.

Usage:
    sanitizer = Sanitizer(config)
    sanitizer.precompute(embeddings)

    # SanText pass (fixed epsilon for all words)
    stats = sanitizer.sanitize_santext(doc)

    # Ours pass (per-document epsilon_n)
    stats = sanitizer.sanitize_normal(doc, epsilon_n=3.5)
"""

import os
import json
import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances
from dataclasses import dataclass, field

from pydantic_models.satsdp import (
    SastdpDocument,
    SastdpDocumentStatistics,
    SastdpEmbeddingAndMappings,
    SastdpMethod,
)


@dataclass
class SanitizerConfig:
    epsilon: float
    s_epsilon: float
    p: float = 0.7
    method: SastdpMethod = SastdpMethod.NORMAL
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

    # Sensitivity of the utility function u(x,y) = -d(x,y).
    # Set to max pairwise distance in the embedding space.
    # NOTE: this is computed from the corpus vocabulary, which makes it
    # data-dependent and strictly speaking not DP.
    # TODO: compute d_max over the full public GloVe file (data-independent),
    # and clamp corpus distances to that value: min(d(x,y), d_max_glove).
    # This makes sensitivity a public parameter and preserves DP.
    sensitivity: float = 1.0

    # Sensitive prob matrix (fixed epsilon_s, precomputed once)
    s_prob_matrix: np.ndarray | None = None

    def precompute(self, vocab: list[str], embeddings: SastdpEmbeddingAndMappings):
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

        if self.config.method == SastdpMethod.NORMAL:
            # s_dist: |V_s| x |V_all|, n_dist: |V_n| x |V_all|
            self.s_distance_matrix = euclidean_distances(
                embeddings.sensitive_word_embed, embeddings.all_word_embed
            )
            self.n_distance_matrix = euclidean_distances(
                embeddings.normal_word_embed, embeddings.all_word_embed
            )
        elif self.config.method == SastdpMethod.PLUS:
            # s_dist: |V_all| x |V_s|, n_dist: |V_all| x |V_n|
            self.s_distance_matrix = euclidean_distances(
                embeddings.all_word_embed, embeddings.sensitive_word_embed
            )
            self.n_distance_matrix = euclidean_distances(
                embeddings.all_word_embed, embeddings.normal_word_embed
            )
        elif self.config.method == SastdpMethod.SANTEXT:
            # Single square matrix
            self.n_distance_matrix = euclidean_distances(
                embeddings.all_word_embed, embeddings.all_word_embed
            )

        # Sensitivity = max pairwise distance across all distance matrices
        all_maxes = []
        if self.s_distance_matrix is not None:
            all_maxes.append(self.s_distance_matrix.max())
        if self.n_distance_matrix is not None:
            all_maxes.append(self.n_distance_matrix.max())
        self.sensitivity = max(all_maxes) if all_maxes else 1.0

        # Sensitive prob matrix uses fixed epsilon_s (same for every document)
        if self.s_distance_matrix is not None:
            self.s_prob_matrix = self._build_prob_matrix(
                self.s_distance_matrix, self.config.s_epsilon
            )

    def _build_prob_matrix(self, distance_matrix: np.ndarray, eps: float) -> np.ndarray:
        """Apply the exponential mechanism: softmax(eps * -distance / (2 * sensitivity)).

        The division by sensitivity is required by the exponential mechanism
        to guarantee eps-DP. Without it, the effective epsilon is scaled by
        the max distance in the embedding space.
        """
        return softmax(eps * (-distance_matrix) / (2 * self.sensitivity), axis=1)

    def _build_n_prob_matrix(self, epsilon_n: float) -> np.ndarray:
        """Build normal prob matrix for a specific epsilon_n."""
        assert self.n_distance_matrix is not None, "Call precompute() first"
        return self._build_prob_matrix(self.n_distance_matrix, epsilon_n)

    # ------------------------------------------------------------------
    # Word counting (needed before sanitization to compute epsilon_n)
    # ------------------------------------------------------------------

    def count_words(self, doc: SastdpDocument) -> tuple[int, int, int]:
        """Count sensitive, normal, and OOV words in a document.

        Returns:
            (ns, nn, n_oov)
        """
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

    def sanitize_santext(self, doc: SastdpDocument) -> SastdpDocumentStatistics:
        """Baseline: all in-vocab words use the same epsilon over full vocab."""
        n_prob_matrix = self._build_n_prob_matrix(self.config.epsilon)
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

        self._write_replacements(doc, " ".join(new_doc), total_epsilon)
        return SastdpDocumentStatistics(
            text_id=doc.text_id,
            sensitive_word_count=0,
            normal_word_count=normal_word_count,
            total_word_count=out_of_vocab_word_count,
            total_epsilon=total_epsilon,
        )

    def sanitize_normal(
        self, doc: SastdpDocument, epsilon_n: float | None = None
    ) -> SastdpDocumentStatistics:
        """Ours: sensitive words use epsilon_s, normal words use epsilon_n.

        If epsilon_n is None, uses config.epsilon (no redistribution).
        If epsilon_n is provided, builds a per-document normal prob matrix.
        """
        assert self.s_prob_matrix is not None, "Call precompute() first"
        eps_n = epsilon_n if epsilon_n is not None else self.config.epsilon
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

        self._write_replacements(doc, " ".join(new_doc), total_epsilon)
        return SastdpDocumentStatistics(
            text_id=doc.text_id,
            sensitive_word_count=sensitive_word_count,
            normal_word_count=normal_word_count,
            total_word_count=out_of_vocab_word_count,
            total_epsilon=total_epsilon,
        )

    def sanitize_plus(
        self, doc: SastdpDocument, epsilon_n: float | None = None
    ) -> SastdpDocumentStatistics:
        """Ours+: mixed sampling with coin flip probability p.

        If epsilon_n is None, uses config.epsilon (no redistribution).
        """
        assert self.s_prob_matrix is not None, "Call precompute() first"
        eps_n = epsilon_n if epsilon_n is not None else self.config.epsilon
        n_prob_matrix = self._build_n_prob_matrix(eps_n)
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
                        # Within-class: sensitive -> sensitive
                        sampling_prob = self.s_prob_matrix[index]
                        idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                        new_doc.append(self.id2sword[idx])
                        total_epsilon += self.config.s_epsilon
                    else:
                        # Cross-class: sensitive -> normal
                        sampling_prob = n_prob_matrix[index]
                        idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                        new_doc.append(self.id2nword[idx])
                        total_epsilon += eps_n
                else:
                    normal_word_count += 1
                    if flip <= p:
                        # Within-class: normal -> normal
                        sampling_prob = n_prob_matrix[index]
                        idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                        new_doc.append(self.id2nword[idx])
                        total_epsilon += eps_n
                    else:
                        # Cross-class: normal -> sensitive
                        sampling_prob = self.s_prob_matrix[index]
                        idx = np.random.choice(len(sampling_prob), p=sampling_prob)
                        new_doc.append(self.id2sword[idx])
                        total_epsilon += self.config.s_epsilon
            else:
                out_of_vocab_word_count += 1
                idx = np.random.randint(len(self.vocab))
                new_doc.append(self.vocab[idx])

        self._write_replacements(doc, " ".join(new_doc), total_epsilon)
        return SastdpDocumentStatistics(
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
        self, doc: SastdpDocument, sanitized_text: str, total_epsilon: float
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
    docs: list[SastdpDocument],
    sanitizer: "Sanitizer",
) -> dict:
    """Compute per-document redistributed epsilon_n.

    SanText spends epsilon per in-vocab word, so for document i:
        epsilon_t[i] = epsilon * (ns[i] + nn[i])

    Redistributing the budget saved on sensitive words:
        epsilon_n[i] = (epsilon_t[i] - ns[i] * epsilon_s) / nn[i]
                     = (epsilon * (ns + nn) - ns * epsilon_s) / nn

    Args:
        docs: documents to compute redistribution for.
        sanitizer: a Sanitizer (provides config.epsilon, config.s_epsilon
            and word counting via count_words).

    Returns:
        dict mapping text_id -> epsilon_n (or None if nn == 0).
    """
    import logging
    _logger = logging.getLogger(__name__)

    epsilon = sanitizer.config.epsilon
    epsilon_s = sanitizer.config.s_epsilon
    result: dict = {}

    for doc in docs:
        ns, nn, _ = sanitizer.count_words(doc)

        if nn == 0:
            result[doc.text_id] = None
            continue

        # SanText would spend epsilon per in-vocab word
        epsilon_t = epsilon * (ns + nn)
        epsilon_n = (epsilon_t - ns * epsilon_s) / nn

        if epsilon_n <= 0:
            _logger.warning(
                "Document %s: epsilon_n=%.4f <= 0 (ns=%d, nn=%d). "
                "Clamping to 0.01.",
                doc.text_id, epsilon_n, ns, nn,
            )
            epsilon_n = 0.01

        result[doc.text_id] = epsilon_n

    return result


# ----------------------------------------------------------------------
# Multiprocessing support: one module-level Sanitizer per worker
# ----------------------------------------------------------------------

_sanitizer: Sanitizer | None = None


def init_worker(sanitizer: Sanitizer):
    """Pool initializer. Stores the Sanitizer instance in the worker."""
    global _sanitizer
    _sanitizer = sanitizer


def worker_sanitize_santext(doc: SastdpDocument) -> SastdpDocumentStatistics:
    assert _sanitizer is not None
    return _sanitizer.sanitize_santext(doc)


def worker_sanitize(args: tuple[SastdpDocument, float | None]) -> SastdpDocumentStatistics:
    """Worker function. Dispatches to the correct method based on config.

    For santext, epsilon_n is ignored (all words use the same epsilon).
    For normal/plus, epsilon_n is the per-document redistributed budget.
    """
    assert _sanitizer is not None
    doc, epsilon_n = args
    if _sanitizer.config.method == SastdpMethod.SANTEXT:
        return _sanitizer.sanitize_santext(doc)
    elif _sanitizer.config.method == SastdpMethod.NORMAL:
        return _sanitizer.sanitize_normal(doc, epsilon_n=epsilon_n)
    elif _sanitizer.config.method == SastdpMethod.PLUS:
        return _sanitizer.sanitize_plus(doc, epsilon_n=epsilon_n)
    raise ValueError(f"Unsupported method: {_sanitizer.config.method}")
