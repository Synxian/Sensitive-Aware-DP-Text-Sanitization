"""Tests for the core DP sanitization logic."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from sanitizer import cal_probability, sanitize_corpus


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
SWORD2ID = {"cat": 0, "dog": 1}   # first 2 = sensitive
NWORD2ID = {"bird": 0, "fish": 1, "frog": 2}   # last 3 = normal
S_EMBEDS = EMBEDS[:2]
N_EMBEDS = EMBEDS[2:]


# ---------------------------------------------------------------------------
# Probability matrix tests
# ---------------------------------------------------------------------------

class TestCalProbability:
    def test_rows_sum_to_one(self):
        prob = cal_probability(EMBEDS, EMBEDS, epsilon=10.0)
        np.testing.assert_allclose(prob.sum(axis=1), np.ones(len(WORDS)),
                                   atol=1e-5)

    def test_identity_has_highest_prob(self):
        """Each word should most likely map to itself (short distance)."""
        prob = cal_probability(EMBEDS, EMBEDS, epsilon=10.0)
        for i in range(len(WORDS)):
            assert prob[i, i] == prob[i].max(), \
                f"Word {WORDS[i]} doesn't have highest self-probability"

    def test_higher_epsilon_peaks_more(self):
        """Higher epsilon → more peaked distribution (less noise)."""
        low  = cal_probability(EMBEDS, EMBEDS, epsilon=1.0)
        high = cal_probability(EMBEDS, EMBEDS, epsilon=50.0)
        assert high.max(axis=1).mean() > low.max(axis=1).mean()

    def test_output_dtype_float32(self):
        prob = cal_probability(EMBEDS, EMBEDS, epsilon=10.0)
        assert prob.dtype == np.float32

    def test_asymmetric_shapes(self):
        """Source and target can have different sizes."""
        prob = cal_probability(S_EMBEDS, EMBEDS, epsilon=5.0)
        assert prob.shape == (len(S_EMBEDS), len(EMBEDS))
        np.testing.assert_allclose(prob.sum(axis=1), np.ones(len(S_EMBEDS)),
                                   atol=1e-5)


# ---------------------------------------------------------------------------
# Sanitizer tests
# ---------------------------------------------------------------------------

class TestSanitizeCorpus:

    def _probs(self, method):
        if method == "santext":
            return cal_probability(EMBEDS, EMBEDS, 10.0), \
                   cal_probability(EMBEDS, EMBEDS, 10.0)
        elif method == "normal":
            return cal_probability(S_EMBEDS, EMBEDS, 5.0), \
                   cal_probability(N_EMBEDS, EMBEDS, 10.0)
        elif method == "plus":
            return cal_probability(EMBEDS, S_EMBEDS, 5.0), \
                   cal_probability(EMBEDS, N_EMBEDS, 10.0)

    @pytest.mark.parametrize("method", ["santext", "normal", "plus"])
    def test_output_length_matches_input(self, method):
        docs  = [["cat", "dog"], ["bird"], ["fish", "frog", "cat"]]
        s_p, n_p = self._probs(method)
        results = sanitize_corpus(
            docs, WORD2ID, SWORD2ID, NWORD2ID,
            s_p, n_p, WORDS, method=method, threads=1)
        assert len(results) == len(docs)

    @pytest.mark.parametrize("method", ["santext", "normal", "plus"])
    def test_output_words_in_vocab(self, method):
        """Every output word must come from the vocabulary."""
        docs  = [["cat", "dog", "bird"]] * 5
        s_p, n_p = self._probs(method)
        results = sanitize_corpus(
            docs, WORD2ID, SWORD2ID, NWORD2ID,
            s_p, n_p, WORDS, method=method, threads=1)
        vocab_set = set(WORDS)
        for r in results:
            for w in r.split():
                assert w in vocab_set, f"OOV word {w!r} in output"

    def test_oov_input_handled(self):
        """Words not in vocab should be replaced with a random vocab word."""
        docs    = [["UNKNOWN_WORD_XYZ"]]
        s_p, n_p = self._probs("santext")
        results = sanitize_corpus(
            docs, WORD2ID, SWORD2ID, NWORD2ID,
            s_p, n_p, WORDS, method="santext", threads=1)
        assert len(results) == 1
        assert results[0].split()[0] in set(WORDS)
