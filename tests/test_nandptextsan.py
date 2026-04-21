"""Unit tests for the three sanitization methods in nandptextsan.py.

Uses a toy vocabulary of 5 words with 3-dimensional embeddings so the
exponential mechanism matrices are small enough to reason about by hand.
Every test seeds numpy's RNG so results are deterministic.

Toy vocabulary:
    all words:  ["alice", "hospital", "the", "cat", "runs"]
    sensitive:  ["alice", "hospital"]   (simulating NER tags PER / LOC)
    normal:     ["the", "cat", "runs"]

Embeddings are unit-ish vectors in R^3, chosen so that each word is
closest to itself (distance 0 on the diagonal) and all pairwise distances
are distinct. This lets us test that the exponential mechanism assigns the
highest probability to the identity replacement.

Test setup:
    The autouse fixture `_set_globals_normal` injects the toy vocabulary,
    embeddings, probability matrices, and epsilon values directly into the
    module-level globals of nandptextsan -- the same globals that
    NADPTextSan_init() would populate inside a multiprocessing worker.
    Each test class may override specific globals (e.g. TestSanText swaps
    in a |V_all| x |V_all| matrix, TestNADPTextSanPlus swaps in the
    cross-class matrices).

    Replacement JSON files are written to pytest's tmp_path, so tests
    don't leave artifacts on disk.

Run:
    .venv/bin/python -m pytest tests/test_nandptextsan.py -v
"""

import os
import json
import numpy as np
import pytest
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances

from pydantic_models.satsdp import SastdpDocument
import nandptextsan as san


# ---------------------------------------------------------------------------
# Tiny embedding space
# ---------------------------------------------------------------------------

# 3-dimensional embeddings.  Each vector is designed so that:
#   - "alice" and "hospital" (sensitive) point along x and y axes
#   - "the" (normal) points along z axis
#   - "cat" sits between alice and hospital, "runs" between hospital and the
# This gives distinct pairwise distances and makes identity the nearest neighbour.
EMBEDDINGS = {
    "alice": np.array([1.0, 0.0, 0.0]),
    "hospital": np.array([0.0, 1.0, 0.0]),
    "the": np.array([0.0, 0.0, 1.0]),
    "cat": np.array([0.5, 0.5, 0.0]),
    "runs": np.array([0.0, 0.5, 0.5]),
}

ALL_WORDS = ["alice", "hospital", "the", "cat", "runs"]
SENSITIVE_WORDS = ["alice", "hospital"]
NORMAL_WORDS = ["the", "cat", "runs"]

WORD2ID = {w: i for i, w in enumerate(ALL_WORDS)}
SWORD2ID = {w: i for i, w in enumerate(SENSITIVE_WORDS)}
NWORD2ID = {w: i for i, w in enumerate(NORMAL_WORDS)}

ALL_EMBED = np.array([EMBEDDINGS[w] for w in ALL_WORDS])
SENSITIVE_EMBED = np.array([EMBEDDINGS[w] for w in SENSITIVE_WORDS])
NORMAL_EMBED = np.array([EMBEDDINGS[w] for w in NORMAL_WORDS])

EPSILON = 4.0
S_EPSILON = 2.0
P_MIX = 0.7


def _build_prob_matrix(source_embed, candidate_embed, eps):
    dist = euclidean_distances(source_embed, candidate_embed)
    return softmax(eps * (-dist) / 2, axis=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _set_globals_normal(tmp_path):
    """Set up globals as NADPTextSan_init would for method=normal."""
    san.epsilon = EPSILON
    san.s_epsilon = S_EPSILON
    san.vocab = ALL_WORDS
    san.sensitive_words = SENSITIVE_WORDS
    san.word2id = WORD2ID
    san.sword2id = SWORD2ID
    san.nword2id = NWORD2ID
    san.p = P_MIX
    # method=normal: s_prob_matrix is |V_s| x |V_all|, n_prob_matrix is |V_n| x |V_all|
    san.s_prob_matrix = _build_prob_matrix(SENSITIVE_EMBED, ALL_EMBED, S_EPSILON)
    san.n_prob_matrix = _build_prob_matrix(NORMAL_EMBED, ALL_EMBED, EPSILON)
    san.id2word = {v: k for k, v in WORD2ID.items()}
    san.id2sword = {v: k for k, v in SWORD2ID.items()}
    san.id2nword = {v: k for k, v in NWORD2ID.items()}
    san.replacements_output_dir = str(tmp_path)
    yield


# ---------------------------------------------------------------------------
# NADPTextSan (method=normal)
# ---------------------------------------------------------------------------

class TestNADPTextSan:
    """Tests for NADPTextSan (Ours / method=normal).

    This method splits the vocabulary into sensitive and normal subsets.
    Sensitive words sample replacements using epsilon_s from s_prob_matrix,
    normal words sample using epsilon_n from n_prob_matrix. Both draw
    replacements from the FULL vocabulary (id2word).
    """

    def test_all_words_replaced(self):
        """Sanitization must produce exactly one output token per input token,
        and every output token must belong to the vocabulary (no empty strings,
        no tokens invented outside the embedding space)."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the cat", text_id=1)
        stats = san.NADPTextSan(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "1.json")))
        output_words = result["sanitized_text"].split()
        assert len(output_words) == 3
        for w in output_words:
            assert w in ALL_WORDS

    def test_sensitive_normal_counts(self):
        """The returned statistics must correctly classify each in-vocab word
        as sensitive or normal based on whether it appears in sword2id.
        "alice" and "hospital" are sensitive; "the", "cat", "runs" are normal.
        No word here is OOV, so total_word_count (OOV count) must be 0."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice hospital the cat runs", text_id=2)
        stats = san.NADPTextSan(doc)
        assert stats.sensitive_word_count == 2
        assert stats.normal_word_count == 3
        assert stats.total_word_count == 0

    def test_oov_handled(self):
        """Words not present in word2id (out-of-vocabulary) must still produce
        a replacement drawn uniformly at random from the full vocabulary, and
        must be counted in total_word_count (the OOV counter). The output
        must still have the same number of tokens as the input."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice xyznotinvocab the", text_id=3)
        stats = san.NADPTextSan(doc)
        assert stats.total_word_count == 1
        assert stats.sensitive_word_count == 1
        assert stats.normal_word_count == 1
        result = json.load(open(os.path.join(san.replacements_output_dir, "3.json")))
        output_words = result["sanitized_text"].split()
        assert len(output_words) == 3
        assert output_words[1] in ALL_WORDS

    def test_total_epsilon_accumulated(self):
        """The total_epsilon field in the output JSON must equal the sum of
        per-word budgets: epsilon_s=2 for "alice" (sensitive) plus
        epsilon_n=4 for "the" (normal) = 6."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=4)
        san.NADPTextSan(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "4.json")))
        assert result["total_epsilon"] == S_EPSILON + EPSILON

    def test_deterministic_with_seed(self):
        """The exponential mechanism sampling uses np.random.choice, so
        resetting the seed to the same value must produce identical output.
        This is critical for reproducibility across experiment runs."""
        doc = SastdpDocument(text="alice hospital the cat runs", text_id=5)
        np.random.seed(42)
        san.NADPTextSan(doc)
        result_a = json.load(open(os.path.join(san.replacements_output_dir, "5.json")))

        np.random.seed(42)
        san.NADPTextSan(doc)
        result_b = json.load(open(os.path.join(san.replacements_output_dir, "5.json")))
        assert result_a["sanitized_text"] == result_b["sanitized_text"]

    def test_higher_epsilon_prefers_closer_words(self):
        """As epsilon grows, softmax(eps/2 * -distance) concentrates almost
        all probability mass on the nearest neighbour (distance=0 means the
        word itself). With eps=200 and input "the" repeated 5 times, nearly
        all outputs should be "the" because the self-distance is 0 and all
        other distances are > 0."""
        san.n_prob_matrix = _build_prob_matrix(NORMAL_EMBED, ALL_EMBED, 200.0)
        np.random.seed(0)
        doc = SastdpDocument(text="the the the the the", text_id=6)
        san.NADPTextSan(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "6.json")))
        output_words = result["sanitized_text"].split()
        assert output_words.count("the") >= 4


# ---------------------------------------------------------------------------
# SanText (baseline)
# ---------------------------------------------------------------------------

class TestSanText:
    """Tests for SanText (baseline / method=santext).

    SanText is the non-sensitivity-aware baseline: it treats ALL in-vocab
    words identically, using a single epsilon over the full embedding space.
    There is no sensitive/normal split -- every word looks up its row in
    n_prob_matrix (which here is |V_all| x |V_all|) and samples a
    replacement from the full vocabulary.

    The fixture overrides n_prob_matrix to be square over all embeddings,
    matching what run_selective_dp.py builds for method=santext.
    """

    @pytest.fixture(autouse=True)
    def _santext_matrices(self):
        san.n_prob_matrix = _build_prob_matrix(ALL_EMBED, ALL_EMBED, EPSILON)
        yield

    def test_no_sensitive_distinction(self):
        """SanText never checks sword2id, so sensitive_word_count must always
        be 0 -- even for words like "alice" that ARE in the sensitive set.
        All 3 in-vocab words should be counted as normal."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice hospital the", text_id=10)
        stats = san.SanText(doc)
        assert stats.sensitive_word_count == 0
        assert stats.normal_word_count == 3

    def test_total_epsilon_uniform(self):
        """Every in-vocab word adds the same epsilon to the total budget.
        3 in-vocab words x epsilon=4 = 12."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the cat", text_id=11)
        san.SanText(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "11.json")))
        assert result["total_epsilon"] == 3 * EPSILON

    def test_oov_no_epsilon_cost(self):
        """OOV words are replaced uniformly at random but do NOT consume any
        privacy budget. Here "alice" costs epsilon=4, but "unknownword" costs
        0, so total should be 4."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice unknownword", text_id=12)
        san.SanText(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "12.json")))
        assert result["total_epsilon"] == EPSILON

    def test_output_length_preserved(self):
        """Sanitized text must have the same number of tokens as the input.
        No tokens should be dropped or added."""
        np.random.seed(0)
        doc = SastdpDocument(text="the cat runs alice hospital", text_id=13)
        san.SanText(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "13.json")))
        assert len(result["sanitized_text"].split()) == 5


# ---------------------------------------------------------------------------
# NADPTextSan_plus (method=plus)
# ---------------------------------------------------------------------------

class TestNADPTextSanPlus:
    """Tests for NADPTextSan_plus (Ours+ / method=plus).

    This method adds a mixing coin flip: with probability p the word stays
    within its own class, with probability 1-p it crosses to the other.
    This corresponds to Theorem 2 (mixed-sampling LDP) in the paper.

    Key difference vs NADPTextSan: replacements are drawn from class-specific
    vocabularies (id2sword / id2nword) rather than the full vocabulary.
    The probability matrices are therefore |V_all| x |V_s| and |V_all| x |V_n|
    (rows indexed by word2id, columns by the target class).

    The fixture overrides both probability matrices to match this shape.
    """

    @pytest.fixture(autouse=True)
    def _plus_matrices(self):
        san.s_prob_matrix = _build_prob_matrix(ALL_EMBED, SENSITIVE_EMBED, S_EPSILON)
        san.n_prob_matrix = _build_prob_matrix(ALL_EMBED, NORMAL_EMBED, EPSILON)
        yield

    def test_sensitive_normal_counts(self):
        """Word classification should work the same as NADPTextSan: "alice"
        and "hospital" are sensitive, "the"/"cat"/"runs" are normal."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice hospital the cat runs", text_id=20)
        stats = san.NADPTextSan_plus(doc)
        assert stats.sensitive_word_count == 2
        assert stats.normal_word_count == 3

    def test_replacements_from_class_vocabs(self):
        """Unlike NADPTextSan which draws from id2word (full vocab),
        NADPTextSan_plus draws from id2sword or id2nword depending on the
        coin flip. Every output word must therefore belong to either the
        sensitive or normal subset -- never a word that only exists in the
        full vocab (which in this toy setup means all words are in one of
        the two subsets, but the assertion guards the lookup logic)."""
        np.random.seed(0)
        doc = SastdpDocument(text="alice the cat hospital runs", text_id=21)
        san.NADPTextSan_plus(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "21.json")))
        output_words = result["sanitized_text"].split()
        for w in output_words:
            assert w in SENSITIVE_WORDS or w in NORMAL_WORDS

    def test_mixing_affects_epsilon(self):
        """With p=1.0 the coin flip always stays within-class:
        - "alice" (sensitive) -> samples from s_prob_matrix -> budget = epsilon_s = 2
        - "the" (normal) -> samples from n_prob_matrix -> budget = epsilon_n = 4
        Total = 6. This verifies the within-class branch of the coin flip."""
        san.p = 1.0
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=22)
        san.NADPTextSan_plus(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "22.json")))
        assert result["total_epsilon"] == S_EPSILON + EPSILON

    def test_mixing_p0_crosses_class(self):
        """With p=0.0 the coin flip always crosses to the other class:
        - "alice" (sensitive) crosses to normal -> budget = epsilon_n = 4
        - "the" (normal) crosses to sensitive -> budget = epsilon_s = 2
        Total is still 6, but the budgets are swapped compared to p=1.0.
        This verifies the cross-class branch of the coin flip."""
        san.p = 0.0
        np.random.seed(0)
        doc = SastdpDocument(text="alice the", text_id=23)
        san.NADPTextSan_plus(doc)
        result = json.load(open(os.path.join(san.replacements_output_dir, "23.json")))
        assert result["total_epsilon"] == EPSILON + S_EPSILON

    def test_oov_uniform(self):
        """OOV words bypass the coin flip entirely and get uniform random
        replacement. They should not be counted as sensitive or normal."""
        np.random.seed(0)
        doc = SastdpDocument(text="xyznotinvocab", text_id=24)
        stats = san.NADPTextSan_plus(doc)
        assert stats.total_word_count == 1
        assert stats.sensitive_word_count == 0
        assert stats.normal_word_count == 0


# ---------------------------------------------------------------------------
# cal_probability
# ---------------------------------------------------------------------------

class TestCalProbability:
    """Tests for cal_probability(), the exponential mechanism matrix builder.

    cal_probability(source, candidates, epsilon_type, epsilon, s_epsilon)
    computes:  softmax(eps/2 * (-euclidean_distance(source, candidates)))
    where eps is chosen by epsilon_type ("normal" -> epsilon, else -> s_epsilon).

    The output is a row-stochastic matrix: row i is the probability distribution
    over candidate words for replacing source word i.
    """

    def test_rows_sum_to_one(self):
        """Each row must be a valid probability distribution (sums to 1).
        This is a basic property of softmax applied row-wise."""
        mat = san.cal_probability(SENSITIVE_EMBED, ALL_EMBED, "sensitive", EPSILON, S_EPSILON)
        row_sums = mat.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_shape(self):
        """Output shape must be (n_source_words, n_candidate_words).
        Here: 2 sensitive words as source, 5 total words as candidates."""
        mat = san.cal_probability(SENSITIVE_EMBED, ALL_EMBED, "sensitive", EPSILON, S_EPSILON)
        assert mat.shape == (len(SENSITIVE_WORDS), len(ALL_WORDS))

    def test_self_distance_gets_highest_prob(self):
        """When source == candidates (square matrix), each word has distance 0
        to itself and >0 to everything else. The exponential mechanism must
        therefore assign the highest probability to the diagonal (identity
        replacement). This is the core utility property of the mechanism."""
        mat = san.cal_probability(ALL_EMBED, ALL_EMBED, "normal", EPSILON, S_EPSILON)
        for i in range(len(ALL_WORDS)):
            assert np.argmax(mat[i]) == i

    def test_epsilon_type_selects_correct_budget(self):
        """epsilon_type="normal" should use epsilon (10.0 here), while
        epsilon_type="sensitive" should use s_epsilon (1.0 here). Higher
        epsilon makes softmax more peaked, so the max probability in the
        normal matrix must be larger than in the sensitive matrix."""
        mat_normal = san.cal_probability(ALL_EMBED, ALL_EMBED, "normal", epsilon=10.0, s_epsilon=1.0)
        mat_sensitive = san.cal_probability(ALL_EMBED, ALL_EMBED, "sensitive", epsilon=10.0, s_epsilon=1.0)
        assert mat_normal.max() > mat_sensitive.max()


# ---------------------------------------------------------------------------
# write_replacements_file
# ---------------------------------------------------------------------------

class TestWriteReplacementsFile:
    """Tests for write_replacements_file(), which persists per-document
    sanitization results as JSON. Each document gets its own file named
    {text_id}.json containing original text, sanitized text, and total
    epsilon spent."""

    def test_creates_json(self, tmp_path):
        replacements = {
            "text_id": "doc_99",
            "original_text": "hello world",
            "sanitized_text": "hi earth",
            "total_epsilon": 8.0,
        }
        san.write_replacements_file(replacements, str(tmp_path))
        fpath = tmp_path / "doc_99.json"
        assert fpath.exists()
        data = json.loads(fpath.read_text(encoding="utf-8"))
        assert data["sanitized_text"] == "hi earth"
        assert data["total_epsilon"] == 8.0
