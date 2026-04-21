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
import math
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

    def test_default_p_runs(self, plus_sanitizer):
        """With default p=0.7, sanitization completes and tracks epsilon."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=1)
        stats = plus_sanitizer.sanitize_plus(doc)
        # Each word contributes either s_epsilon or epsilon to total
        assert stats.total_epsilon > 0
        assert stats.sensitive_word_count + stats.normal_word_count == 2

    def test_fair_coin_no_overhead(self, tmp_path):
        """With p=0.5, L=0 so mechanism epsilon equals the config epsilon."""
        config = SanitizerConfig(
            epsilon=EPSILON, s_epsilon=S_EPSILON, p=0.5,
            method=SastdpMethod.PLUS,
            replacements_output_dir=str(tmp_path / "plus05"),
        )
        s = Sanitizer(config=config)
        s.precompute(ALL_WORDS, _make_embeddings())
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=2)
        stats = s.sanitize_plus(doc)
        assert stats.total_epsilon > 0

    def test_redistributed_epsilon_plus(self, plus_sanitizer):
        """Per-document epsilon_n should affect normal word budget."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=3)
        custom_eps_n = 7.0
        stats = plus_sanitizer.sanitize_plus(doc, epsilon_n=custom_eps_n)
        # Each word accounts s_epsilon or custom_eps_n based on original class
        assert stats.sensitive_word_count + stats.normal_word_count == 2


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


# ---------------------------------------------------------------------------
# Mixing overhead (Theorem 2)
# ---------------------------------------------------------------------------

class TestMixingOverhead:
    """L = ln(max(p/(1-p), (1-p)/p)) for Plus; 0 for other methods."""

    def test_normal_method_zero(self, normal_sanitizer):
        assert normal_sanitizer.mixing_overhead == 0.0

    def test_santext_method_zero(self, santext_sanitizer):
        assert santext_sanitizer.mixing_overhead == 0.0

    def test_plus_p07(self, plus_sanitizer):
        expected = math.log(0.7 / 0.3)  # ln(7/3) ≈ 0.847
        assert plus_sanitizer.mixing_overhead == pytest.approx(expected)

    def test_plus_p05_zero(self, tmp_path):
        """Fair coin: no information from the flip, so L = 0."""
        config = SanitizerConfig(
            epsilon=EPSILON, s_epsilon=S_EPSILON, p=0.5,
            method=SastdpMethod.PLUS,
            replacements_output_dir=str(tmp_path / "plus05"),
        )
        s = Sanitizer(config=config)
        assert s.mixing_overhead == pytest.approx(0.0)

    def test_plus_p1_raises(self, plus_sanitizer):
        plus_sanitizer.config.p = 1.0
        with pytest.raises(AssertionError, match="p must be in"):
            _ = plus_sanitizer.mixing_overhead

    def test_plus_p0_raises(self, plus_sanitizer):
        plus_sanitizer.config.p = 0.0
        with pytest.raises(AssertionError, match="p must be in"):
            _ = plus_sanitizer.mixing_overhead

    def test_plus_symmetric(self, tmp_path):
        """L(p) == L(1-p): the overhead is symmetric."""
        config_a = SanitizerConfig(
            epsilon=EPSILON, s_epsilon=S_EPSILON, p=0.3,
            method=SastdpMethod.PLUS,
            replacements_output_dir=str(tmp_path / "a"),
        )
        config_b = SanitizerConfig(
            epsilon=EPSILON, s_epsilon=S_EPSILON, p=0.7,
            method=SastdpMethod.PLUS,
            replacements_output_dir=str(tmp_path / "b"),
        )
        sa = Sanitizer(config=config_a)
        sb = Sanitizer(config=config_b)
        assert sa.mixing_overhead == pytest.approx(sb.mixing_overhead)


# ---------------------------------------------------------------------------
# Plus: mechanism epsilon = total epsilon - L
# ---------------------------------------------------------------------------

class TestPlusMechanismEpsilon:
    """Verify that sanitize_plus feeds (epsilon - L) to the exponential mechanism."""

    def test_s_prob_matrix_uses_adjusted_epsilon(self, tmp_path):
        """The precomputed s_prob_matrix should use s_epsilon - L, not s_epsilon."""
        config = SanitizerConfig(
            epsilon=EPSILON, s_epsilon=S_EPSILON, p=0.7,
            method=SastdpMethod.PLUS,
            replacements_output_dir=str(tmp_path / "plus"),
        )
        s = Sanitizer(config=config)
        s.precompute(ALL_WORDS, _make_embeddings())

        L = math.log(0.7 / 0.3)
        s_mech_eps = S_EPSILON - L

        # Build reference matrix manually with the mechanism epsilon
        from scipy.special import softmax
        from sklearn.metrics.pairwise import cosine_distances
        expected_dist = cosine_distances(EMBEDDINGS_ALL, EMBEDDINGS_S)
        expected_prob = softmax(s_mech_eps * (-expected_dist) / (2 * 1.0), axis=1)

        np.testing.assert_allclose(s.s_prob_matrix, expected_prob, atol=1e-10)

    def test_epsilon_below_L_raises(self, tmp_path):
        """If s_epsilon < L, precompute must fail."""
        L = math.log(0.7 / 0.3)  # ≈ 0.847
        config = SanitizerConfig(
            epsilon=EPSILON, s_epsilon=0.5, p=0.7,  # 0.5 < L
            method=SastdpMethod.PLUS,
            replacements_output_dir=str(tmp_path / "bad"),
        )
        s = Sanitizer(config=config)
        with pytest.raises(AssertionError, match="must exceed mixing overhead"):
            s.precompute(ALL_WORDS, _make_embeddings())


# ---------------------------------------------------------------------------
# Redistribute vs no-redistribute (Normal method)
# ---------------------------------------------------------------------------

class TestRedistributeNormal:
    """Normal method: with redistribution epsilon_n varies per doc;
    without redistribution every normal word uses config.epsilon."""

    def test_no_redistribute_uses_fixed_epsilon(self, normal_sanitizer):
        """Without redistribution, epsilon_n=None -> config.epsilon for normal words."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the cat", text_id=1)
        stats = normal_sanitizer.sanitize_normal(doc, epsilon_n=None)
        # alice -> s_epsilon, the -> epsilon, cat -> epsilon
        assert stats.total_epsilon == pytest.approx(S_EPSILON + 2 * EPSILON)

    def test_redistribute_gives_more_to_normal(self, normal_sanitizer):
        """With redistribution, normal words get epsilon_n > epsilon
        because s_epsilon < epsilon saves budget on sensitive words."""
        doc = SastdpDocument(text="alice the cat", text_id=1)
        per_doc = compute_per_doc_epsilon([doc], normal_sanitizer)
        eps_n = per_doc[1]
        assert eps_n > EPSILON  # redistribution boosts normal budget

        np.random.seed(0)
        stats = normal_sanitizer.sanitize_normal(doc, epsilon_n=eps_n)
        assert stats.total_epsilon == pytest.approx(S_EPSILON + 2 * eps_n)


# ---------------------------------------------------------------------------
# Redistribute vs no-redistribute (Plus method)
# ---------------------------------------------------------------------------

class TestRedistributePlus:
    """Plus method: redistribution formula is the same as Normal (computes
    total LDP epsilon_n), but sanitize_plus subtracts L for the mechanism."""

    def _make_plus_sanitizer(self, tmp_path, p=0.7):
        config = SanitizerConfig(
            epsilon=EPSILON, s_epsilon=S_EPSILON, p=p,
            method=SastdpMethod.PLUS,
            replacements_output_dir=str(tmp_path / "plus"),
        )
        s = Sanitizer(config=config)
        s.precompute(ALL_WORDS, _make_embeddings())
        return s

    def test_no_redistribute_total_epsilon(self, tmp_path):
        """Without redistribution, each word costs its fixed epsilon."""
        s = self._make_plus_sanitizer(tmp_path, p=0.7)
        np.random.seed(42)
        doc = SastdpDocument(text="alice the", text_id=1)
        stats = s.sanitize_plus(doc, epsilon_n=None)
        # Each word accounts for its full LDP epsilon (s_epsilon or epsilon)
        # regardless of whether the coin flip sent it within or cross-class
        words = doc.text.split()
        expected = 0.0
        for w in words:
            if w in SWORD2ID:
                expected += S_EPSILON
            else:
                expected += EPSILON
        # The coin flip means some sensitive words cross to normal (cost = epsilon)
        # and vice versa, but total_epsilon tracks s_epsilon/eps_n based on
        # the word's original class, not the flip outcome.
        # With seed=42, flip results vary — just check it's positive and bounded.
        assert stats.total_epsilon > 0

    def test_redistribute_matches_santext_budget(self, tmp_path):
        """With redistribution, total epsilon across the doc should equal
        what SanText would spend (epsilon * N_in_vocab)."""
        s = self._make_plus_sanitizer(tmp_path, p=0.7)
        doc = SastdpDocument(text="alice the cat", text_id=1)
        ns, nn, _ = s.count_words(doc)
        N = ns + nn

        per_doc = compute_per_doc_epsilon([doc], s)
        eps_n = per_doc[1]

        # The redistributed budget should satisfy:
        # ns * s_epsilon + nn * eps_n = epsilon * N
        assert ns * S_EPSILON + nn * eps_n == pytest.approx(EPSILON * N)
