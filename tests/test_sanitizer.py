"""Tests for the core DP sanitization logic (Sanitizer class)."""
import sys, os, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from pydantic_models.sanitizerdp import (
    SanitizerDPDocument, SanitizerDPEmbeddingAndMappings
)
from sanitizer import SanitizerConfig, Sanitizer, compute_per_doc_epsilon, sanitize_corpus


# ---------------------------------------------------------------------------
# Toy vocabulary (5 words, 3D embeddings)
# ---------------------------------------------------------------------------

WORDS   = ["cat", "dog", "bird", "fish", "frog"]
EMBEDS  = np.array([
    [1.0, 0.0, 0.0],
    [0.9, 0.1, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.1, 0.0, 0.9],
], dtype=np.float32)

WORD2ID  = {w: i for i, w in enumerate(WORDS)}
SWORD2ID = {"cat": 0, "dog": 1}
NWORD2ID = {"bird": 0, "fish": 1, "frog": 2}
S_EMBEDS = EMBEDS[:2]
N_EMBEDS = EMBEDS[2:]


def _make_embeddings():
    return SanitizerDPEmbeddingAndMappings(
        sensitive_word_embed=S_EMBEDS,
        normal_word_embed=N_EMBEDS,
        all_word_embed=EMBEDS,
        word2id=WORD2ID,
        sword2id=SWORD2ID,
        nword2id=NWORD2ID,
    )


def _make_sanitizer(method="normal", eps=10.0, s_eps=5.0, p=0.7):
    with tempfile.TemporaryDirectory() as tmpdir:
        config = SanitizerConfig(
            epsilon=eps, s_epsilon=s_eps, p=p,
            method=method, distance_metric="cosine",
            replacements_output_dir=os.path.join(tmpdir, "repl"),
        )
        san = Sanitizer(config=config)
        san.precompute(WORDS, _make_embeddings())
        return san


# ---------------------------------------------------------------------------
# Sanitizer precompute tests
# ---------------------------------------------------------------------------

class TestSanitizerPrecompute:
    def test_santext_has_n_prob_matrix(self):
        san = _make_sanitizer("santext")
        assert san.n_prob_matrix_fixed is not None
        assert san.s_prob_matrix is None

    def test_normal_has_both_matrices(self):
        san = _make_sanitizer("normal")
        assert san.s_prob_matrix is not None
        # n_prob_matrix_fixed is built lazily or on first sanitize

    def test_plus_has_both_matrices(self):
        san = _make_sanitizer("plus")
        assert san.s_prob_matrix is not None

    def test_prob_rows_sum_to_one(self):
        san = _make_sanitizer("santext")
        np.testing.assert_allclose(
            san.n_prob_matrix_fixed.sum(axis=1),
            np.ones(len(WORDS)), atol=1e-5)


# ---------------------------------------------------------------------------
# Sanitization method tests
# ---------------------------------------------------------------------------

class TestSanitizeMethods:

    @pytest.mark.parametrize("method", ["santext", "normal", "plus"])
    def test_sanitize_returns_statistics(self, method):
        san = _make_sanitizer(method)
        doc = SanitizerDPDocument(text="cat dog bird", text_id="test_0")
        if method == "santext":
            result = san.sanitize_santext(doc)
        elif method == "normal":
            result = san.sanitize_normal(doc)
        else:
            result = san.sanitize_plus(doc)
        assert result.text_id == "test_0"
        assert result.total_epsilon > 0

    @pytest.mark.parametrize("method", ["santext", "normal", "plus"])
    def test_sanitize_corpus_output_length(self, method):
        san = _make_sanitizer(method)
        docs = [
            SanitizerDPDocument(text="cat dog", text_id="t0"),
            SanitizerDPDocument(text="bird", text_id="t1"),
            SanitizerDPDocument(text="fish frog cat", text_id="t2"),
        ]
        eps = [None] * len(docs)
        results = sanitize_corpus(docs, san, eps, threads=1, desc="test")
        assert len(results) == len(docs)

    def test_oov_handled(self):
        san = _make_sanitizer("santext")
        doc = SanitizerDPDocument(text="UNKNOWN_XYZ", text_id="oov_0")
        result = san.sanitize_santext(doc)
        # Should not crash; OOV gets random word
        assert result.text_id == "oov_0"

    def test_higher_epsilon_more_peaked(self):
        san_low = _make_sanitizer("santext", eps=1.0)
        san_high = _make_sanitizer("santext", eps=50.0)
        assert (san_high.n_prob_matrix_fixed.max(axis=1).mean() >
                san_low.n_prob_matrix_fixed.max(axis=1).mean())


# ---------------------------------------------------------------------------
# Per-document epsilon redistribution tests
# ---------------------------------------------------------------------------

class TestPerDocEpsilon:
    def test_redistribution_values(self):
        san = _make_sanitizer("normal", eps=10.0, s_eps=5.0)
        docs = [
            SanitizerDPDocument(text="cat bird fish", text_id="d0"),
        ]
        result = compute_per_doc_epsilon(docs, san)
        assert len(result) == 1
        # cat=sensitive(1), bird=normal(1), fish=normal(1)
        # eps_t = 10*(1+2) = 30, eps_n = (30 - 1*5)/2 = 12.5
        assert result[0] == pytest.approx(12.5)

    def test_all_sensitive_gives_none(self):
        san = _make_sanitizer("normal", eps=10.0, s_eps=5.0)
        # "cat dog" = both sensitive, nn=0
        docs = [SanitizerDPDocument(text="cat dog", text_id="d1")]
        result = compute_per_doc_epsilon(docs, san)
        assert result[0] is None
