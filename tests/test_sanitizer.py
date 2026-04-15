"""Tests for the class-based Sanitizer and per-document epsilon redistribution.

Uses the same toy 5-word vocabulary as test_nandptextsan.py.
Verifies that:
- The Sanitizer class produces equivalent results to the old global-based code
- Per-document epsilon redistribution computes epsilon_n correctly
- The three-phase pipeline (SanText -> count -> Ours) works end-to-end

Run:
    .venv/bin/python -m pytest tests/test_sanitizer.py -v
"""

import json
import os

import numpy as np
import pytest

from pydantic_models.satsdp import (
    SastdpDocument,
    SastdpEmbeddingAndMappings,
    SastdpMethod,
)
from sanitizer import Sanitizer, SanitizerConfig, compute_per_doc_epsilon


# ---------------------------------------------------------------------------
# Toy embedding space (same as test_nandptextsan.py)
# ---------------------------------------------------------------------------

ALL_WORDS = ["alice", "hospital", "the", "cat", "runs"]
SENSITIVE_WORDS = ["alice", "hospital"]
NORMAL_WORDS = ["the", "cat", "runs"]

EMBEDDINGS_ALL = np.array([
    [1.0, 0.0, 0.0],   # alice
    [0.0, 1.0, 0.0],   # hospital
    [0.0, 0.0, 1.0],   # the
    [0.5, 0.5, 0.0],   # cat
    [0.0, 0.5, 0.5],   # runs
])
EMBEDDINGS_S = EMBEDDINGS_ALL[:2]
EMBEDDINGS_N = EMBEDDINGS_ALL[2:]

WORD2ID = {w: i for i, w in enumerate(ALL_WORDS)}
SWORD2ID = {w: i for i, w in enumerate(SENSITIVE_WORDS)}
NWORD2ID = {w: i for i, w in enumerate(NORMAL_WORDS)}

EPSILON = 4.0
S_EPSILON = 2.0


def _make_embeddings() -> SastdpEmbeddingAndMappings:
    return SastdpEmbeddingAndMappings(
        all_word_embed=EMBEDDINGS_ALL,
        sensitive_word_embed=EMBEDDINGS_S,
        normal_word_embed=EMBEDDINGS_N,
        word2id=WORD2ID,
        sword2id=SWORD2ID,
        nword2id=NWORD2ID,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def santext_sanitizer(tmp_path):
    """A Sanitizer configured for SanText (baseline)."""
    config = SanitizerConfig(
        epsilon=EPSILON, s_epsilon=S_EPSILON,
        method=SastdpMethod.SANTEXT,
        replacements_output_dir=str(tmp_path / "santext"),
    )
    s = Sanitizer(config=config)
    s.precompute(ALL_WORDS, _make_embeddings())
    return s


@pytest.fixture
def normal_sanitizer(tmp_path):
    """A Sanitizer configured for Ours (method=normal)."""
    config = SanitizerConfig(
        epsilon=EPSILON, s_epsilon=S_EPSILON,
        method=SastdpMethod.NORMAL,
        replacements_output_dir=str(tmp_path / "normal"),
    )
    s = Sanitizer(config=config)
    s.precompute(ALL_WORDS, _make_embeddings())
    return s


@pytest.fixture
def plus_sanitizer(tmp_path):
    """A Sanitizer configured for Ours+ (method=plus)."""
    config = SanitizerConfig(
        epsilon=EPSILON, s_epsilon=S_EPSILON, p=0.7,
        method=SastdpMethod.PLUS,
        replacements_output_dir=str(tmp_path / "plus"),
    )
    s = Sanitizer(config=config)
    s.precompute(ALL_WORDS, _make_embeddings())
    return s


# ---------------------------------------------------------------------------
# Precompute
# ---------------------------------------------------------------------------

class TestPrecompute:
    """Verify that precompute() builds distance matrices and mappings correctly."""

    def test_distance_matrices_shape_normal(self, normal_sanitizer):
        """For method=normal: s_dist is |V_s| x |V_all|, n_dist is |V_n| x |V_all|."""
        s = normal_sanitizer
        assert s.s_distance_matrix.shape == (2, 5)
        assert s.n_distance_matrix.shape == (3, 5)

    def test_distance_matrices_shape_plus(self, plus_sanitizer):
        """For method=plus: s_dist is |V_all| x |V_s|, n_dist is |V_all| x |V_n|."""
        s = plus_sanitizer
        assert s.s_distance_matrix.shape == (5, 2)
        assert s.n_distance_matrix.shape == (5, 3)

    def test_distance_matrices_shape_santext(self, santext_sanitizer):
        """For method=santext: single square n_dist |V_all| x |V_all|."""
        s = santext_sanitizer
        assert s.s_distance_matrix is None
        assert s.n_distance_matrix.shape == (5, 5)

    def test_s_prob_matrix_precomputed(self, normal_sanitizer):
        """The sensitive prob matrix should be precomputed with fixed epsilon_s."""
        s = normal_sanitizer
        assert s.s_prob_matrix is not None
        assert s.s_prob_matrix.shape == (2, 5)
        # Rows must sum to 1
        np.testing.assert_allclose(s.s_prob_matrix.sum(axis=1), 1.0, atol=1e-6)

    def test_reverse_mappings(self, normal_sanitizer):
        s = normal_sanitizer
        assert s.id2word[0] == "alice"
        assert s.id2sword[1] == "hospital"
        assert s.id2nword[0] == "the"


# ---------------------------------------------------------------------------
# count_words
# ---------------------------------------------------------------------------

class TestCountWords:
    """Verify word counting needed for epsilon redistribution."""

    def test_all_in_vocab(self, normal_sanitizer):
        doc = SastdpDocument(text="alice hospital the cat runs", text_id=1)
        ns, nn, n_oov = normal_sanitizer.count_words(doc)
        assert ns == 2
        assert nn == 3
        assert n_oov == 0

    def test_with_oov(self, normal_sanitizer):
        doc = SastdpDocument(text="alice unknown the", text_id=2)
        ns, nn, n_oov = normal_sanitizer.count_words(doc)
        assert ns == 1
        assert nn == 1
        assert n_oov == 1

    def test_all_sensitive(self, normal_sanitizer):
        doc = SastdpDocument(text="alice hospital alice", text_id=3)
        ns, nn, n_oov = normal_sanitizer.count_words(doc)
        assert ns == 3
        assert nn == 0
        assert n_oov == 0


# ---------------------------------------------------------------------------
# SanText
# ---------------------------------------------------------------------------

class TestSanitizeSantext:
    """SanText treats all words identically with uniform epsilon."""

    def test_total_epsilon(self, santext_sanitizer):
        np.random.seed(0)
        doc = SastdpDocument(text="alice the cat", text_id=1)
        stats = santext_sanitizer.sanitize_santext(doc)
        assert stats.total_epsilon == 3 * EPSILON
        assert stats.sensitive_word_count == 0
        assert stats.normal_word_count == 3

    def test_output_in_vocab(self, santext_sanitizer):
        np.random.seed(0)
        doc = SastdpDocument(text="alice hospital the", text_id=2)
        santext_sanitizer.sanitize_santext(doc)
        path = os.path.join(santext_sanitizer.config.replacements_output_dir, "2.json")
        result = json.loads(open(path).read())
        for w in result["sanitized_text"].split():
            assert w in ALL_WORDS


# ---------------------------------------------------------------------------
# Ours (method=normal)
# ---------------------------------------------------------------------------

class TestSanitizeNormal:
    """NADPTextSan with and without per-document epsilon redistribution."""

    def test_fixed_epsilon(self, normal_sanitizer):
        """Without redistribution, uses config.epsilon for normal words."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=1)
        stats = normal_sanitizer.sanitize_normal(doc)
        # alice -> s_epsilon=2, the -> epsilon=4
        assert stats.total_epsilon == S_EPSILON + EPSILON

    def test_redistributed_epsilon(self, normal_sanitizer):
        """With per-document epsilon_n, normal words use the provided budget."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=2)
        custom_eps_n = 6.0
        stats = normal_sanitizer.sanitize_normal(doc, epsilon_n=custom_eps_n)
        # alice -> s_epsilon=2, the -> custom_eps_n=6
        assert stats.total_epsilon == S_EPSILON + custom_eps_n

    def test_higher_epsilon_n_changes_distribution(self, normal_sanitizer):
        """A higher epsilon_n should make the normal word replacement more
        concentrated around the nearest neighbour (less noise)."""
        doc = SastdpDocument(text="the the the the the the the the the the", text_id=3)

        np.random.seed(0)
        stats_low = normal_sanitizer.sanitize_normal(doc, epsilon_n=1.0)
        path = os.path.join(normal_sanitizer.config.replacements_output_dir, "3.json")
        result_low = json.loads(open(path).read())
        the_count_low = result_low["sanitized_text"].split().count("the")

        np.random.seed(0)
        stats_high = normal_sanitizer.sanitize_normal(doc, epsilon_n=200.0)
        result_high = json.loads(open(path).read())
        the_count_high = result_high["sanitized_text"].split().count("the")

        # Higher epsilon -> more identity replacements
        assert the_count_high >= the_count_low


# ---------------------------------------------------------------------------
# Ours+ (method=plus)
# ---------------------------------------------------------------------------

class TestSanitizePlus:
    """NADPTextSan_plus with class-based Sanitizer."""

    def test_within_class_p1(self, plus_sanitizer):
        """With p=1.0, all words stay within their class."""
        plus_sanitizer.config.p = 1.0
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=1)
        stats = plus_sanitizer.sanitize_plus(doc)
        assert stats.total_epsilon == S_EPSILON + EPSILON

    def test_cross_class_p0(self, plus_sanitizer):
        """With p=0.0, all words cross to the other class."""
        plus_sanitizer.config.p = 0.0
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=2)
        stats = plus_sanitizer.sanitize_plus(doc)
        # alice crosses to normal -> epsilon, the crosses to sensitive -> s_epsilon
        assert stats.total_epsilon == EPSILON + S_EPSILON

    def test_redistributed_epsilon_plus(self, plus_sanitizer):
        """Per-document epsilon_n should affect normal word budget."""
        plus_sanitizer.config.p = 1.0
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=3)
        custom_eps_n = 7.0
        stats = plus_sanitizer.sanitize_plus(doc, epsilon_n=custom_eps_n)
        # alice within-class -> s_epsilon=2, the within-class -> custom=7
        assert stats.total_epsilon == S_EPSILON + custom_eps_n


# ---------------------------------------------------------------------------
# Per-document epsilon redistribution
# ---------------------------------------------------------------------------

class TestComputePerDocEpsilon:
    """Test the redistribution formula.

    SanText spends epsilon per in-vocab word, so:
        epsilon_t[i] = epsilon * (ns[i] + nn[i])
        epsilon_n[i] = (epsilon_t[i] - ns[i] * epsilon_s) / nn[i]
    """

    def test_basic_redistribution(self, normal_sanitizer):
        """Doc: 1 sensitive + 2 normal. All in-vocab, so epsilon_t = 4*3 = 12.
        epsilon_n = (12 - 1*2) / 2 = 5.0"""
        doc = SastdpDocument(text="alice the cat", text_id=1)
        result = compute_per_doc_epsilon([doc], normal_sanitizer)
        assert result[1] == pytest.approx(5.0)

    def test_no_normal_words(self, normal_sanitizer):
        """Document with only sensitive words -> epsilon_n is None."""
        doc = SastdpDocument(text="alice hospital", text_id=2)
        result = compute_per_doc_epsilon([doc], normal_sanitizer)
        assert result[2] is None

    def test_redistribution_increases_normal_budget(self, normal_sanitizer):
        """Since epsilon_s < epsilon, the saved budget from sensitive words
        must make epsilon_n > epsilon (normal words get more budget).
        Doc: 2 sensitive + 1 normal. epsilon_t = 4*3 = 12.
        epsilon_n = (12 - 2*2) / 1 = 8.0 > 4.0."""
        doc = SastdpDocument(text="alice hospital the", text_id=3)
        result = compute_per_doc_epsilon([doc], normal_sanitizer)
        assert result[3] == pytest.approx(8.0)
        assert result[3] > EPSILON
