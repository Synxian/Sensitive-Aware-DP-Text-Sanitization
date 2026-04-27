"""Core DP sanitization library.

Exports
-------
get_tokenizer()
build_vocab(texts, tokenizer)          -> Counter
filter_vocab(vocab, ...)               -> list[str]
load_glove(path, vocab, sensitive)     -> (word2id, sword2id, nword2id, all_words,
                                           all_embed, s_embed, n_embed)
build_sensitive_words_ner(texts, ...)  -> set[str]   (primary — Flair NER)
build_sensitive_words(words, ...)      -> set[str]   (fallback — frequency-based)
cal_probability(src, tgt, eps)         -> np.ndarray float32
compute_total_epsilon(docs, ...)       -> float
sanitize_corpus(docs, ...)             -> list[str]
"""
from __future__ import annotations

import os
import re
import unicodedata
import json
import logging
from collections import Counter
from multiprocessing import Pool, cpu_count

import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

try:
    from spacy.lang.en import English as _SpacyEnglish
    _HAS_SPACY = True
except ImportError:
    _HAS_SPACY = False


class _Token:
    __slots__ = ("text",)
    def __init__(self, t): self.text = t


class _SimpleTokenizer:
    _RE = re.compile(r"[\w]+(?:[']\w+)*|[^\s\w]")
    def __call__(self, text):
        return [_Token(m.group()) for m in self._RE.finditer(text)]


def get_tokenizer():
    """Return spaCy English if available, else a regex fallback."""
    return _SpacyEnglish() if _HAS_SPACY else _SimpleTokenizer()


def word_normalize(text: str) -> str:
    return unicodedata.normalize("NFD", text)


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

def build_vocab(texts: list[str], tokenizer) -> Counter:
    """Count lowercased word frequencies across a list of texts."""
    vocab: Counter = Counter()
    for text in tqdm(texts, desc="Building vocab", leave=False):
        for tok in tokenizer(text):
            vocab[tok.text.lower()] += 1
    return vocab


def filter_vocab(vocab: Counter, min_freq: int = 1,
                 max_vocab: int | None = None) -> list[str]:
    """Return words ordered by frequency (most common first).

    min_freq=5 reduces QNLI from ~74k words to ~33k, cutting the
    probability matrix from ~14 GB to ~3 GB with only 1.8% token loss.
    """
    words = [w for w, c in vocab.most_common() if c >= min_freq]
    if max_vocab is not None:
        words = words[:max_vocab]
    return words


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def load_glove(
    path: str,
    vocab: set[str],
    sensitive_words: set[str],
) -> tuple[dict, dict, dict, list, np.ndarray, np.ndarray, np.ndarray]:
    """Load GloVe and partition into sensitive / normal subsets.

    Returns
    -------
    word2id, sword2id, nword2id, all_words,
    all_embed (n, d), s_embed (n_s, d), n_embed (n_n, d)  — all float32
    """
    word2id: dict[str, int]  = {}
    sword2id: dict[str, int] = {}
    nword2id: dict[str, int] = {}
    a_embeds, s_embeds, n_embeds = [], [], []
    a_cnt = s_cnt = n_cnt = 0

    num_lines = sum(1 for _ in open(path, encoding="utf-8"))
    logger.info("Loading GloVe from %s …", path)

    with open(path, encoding="utf-8") as f:
        first = f.readline().rstrip().split(" ")
        if len(first) != 2:   # no header line
            f.seek(0)

        for row in tqdm(f, total=num_lines - 1, desc="Embeddings"):
            parts = row.rstrip().split(" ")
            w = word_normalize(parts[0]).lower()
            if w not in vocab or w in word2id:
                continue
            emb = [float(x) for x in parts[1:]]
            word2id[w] = a_cnt;  a_cnt += 1;  a_embeds.append(emb)
            if w in sensitive_words:
                sword2id[w] = s_cnt;  s_cnt += 1;  s_embeds.append(emb)
            else:
                nword2id[w] = n_cnt;  n_cnt += 1;  n_embeds.append(emb)

    dim = len(a_embeds[0]) if a_embeds else 300
    all_embed = np.array(a_embeds, dtype=np.float32)
    s_embed   = (np.array(s_embeds, dtype=np.float32)
                 if s_embeds else np.empty((0, dim), dtype=np.float32))
    n_embed   = (np.array(n_embeds, dtype=np.float32)
                 if n_embeds else np.empty((0, dim), dtype=np.float32))

    logger.info(
        "Embeddings — all: %s  sensitive: %s  normal: %s  (%.1f GB f32)",
        all_embed.shape, s_embed.shape, n_embed.shape,
        all_embed.nbytes / 1024 ** 3,
    )
    return word2id, sword2id, nword2id, list(word2id.keys()), \
           all_embed, s_embed, n_embed


def build_sensitive_words_ner(
    texts: list[str],
    model_name: str = "flair/ner-english-large",
    threshold: float = 0.3,
    entity_types: tuple[str, ...] = ("PER", "ORG", "LOC"),
    batch_size: int = 32,
) -> set[str]:
    """Primary method: use Flair NER to identify sensitive entity tokens.

    Runs ``model_name`` on all texts and collects tokens that belong to
    any of ``entity_types`` with confidence ≥ ``threshold``.

    Args:
        texts:        raw text strings (train + dev corpus)
        model_name:   Flair model tag, default "ner-english-large"
        threshold:    minimum entity confidence (paper uses 0.3)
        entity_types: entity categories to treat as sensitive
        batch_size:   Flair mini-batch size for prediction

    Returns:
        set of lowercased sensitive token strings
    """
    try:
        from flair.models import SequenceTagger
        from flair.data import Sentence as FlairSentence
    except ImportError:
        raise ImportError(
            "flair is required for NER-based sensitivity detection.\n"
            "Install it with:  pip install flair\n"
            "Or use --no_ner to fall back to frequency-based detection."
        )

    # Force offline mode so transformers doesn't call huggingface.co
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    logger.info("Loading Flair NER model '%s' …", model_name)
    tagger = SequenceTagger.load(model_name)

    sensitive: set[str] = set()
    sentences = [FlairSentence(t) for t in tqdm(texts, desc="NER prep", leave=False)]

    logger.info("Running NER on %d texts (batch_size=%d) …", len(sentences), batch_size)
    tagger.predict(sentences, mini_batch_size=batch_size, verbose=False)

    for sentence in sentences:
        for span in sentence.get_spans("ner"):
            if span.tag in entity_types and span.score >= threshold:
                for token in span.tokens:
                    sensitive.add(token.text.lower())

    logger.info("NER found %d sensitive tokens (types=%s, thr=%.2f)",
                len(sensitive), entity_types, threshold)
    return sensitive


def build_sensitive_words(words: list[str], sensitive_pct: float = 0.5,
                          sensitive_path: str | None = None) -> set[str]:
    """Fallback: bottom ``sensitive_pct`` fraction of vocab by frequency.

    Or load from a JSON file ``{word: anything}`` if path is given.
    """
    if sensitive_path:
        with open(sensitive_path, encoding="utf-8") as f:
            return {k.lower() for k in json.load(f)}
    n = int(sensitive_pct * len(words))
    return set(words[-n:]) if n else set()


# ---------------------------------------------------------------------------
# Object-Oriented Sanitizer (with per-document epsilon redistribution)
# ---------------------------------------------------------------------------

import math
from dataclasses import dataclass, field
from scipy.special import softmax
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
import multiprocessing
from multiprocessing import Pool, cpu_count

from pydantic_models.satsdp import SastdpDocument

@dataclass
class SanitizerConfig:
    epsilon: float
    s_epsilon: float
    p: float = 0.5
    method: str = "normal"
    distance_metric: str = "cosine"

@dataclass
class Sanitizer:
    config: SanitizerConfig
    
    vocab: list = field(default_factory=list)
    word2id: dict = field(default_factory=dict)
    sword2id: dict = field(default_factory=dict)
    nword2id: dict = field(default_factory=dict)
    
    id2word: dict = field(default_factory=dict)
    id2sword: dict = field(default_factory=dict)
    id2nword: dict = field(default_factory=dict)

    s_distance_matrix: np.ndarray = None
    n_distance_matrix: np.ndarray = None
    sensitivity: float = 1.0

    s_prob_matrix: np.ndarray = None
    n_prob_matrix_fixed: np.ndarray = None

    @property
    def mixing_overhead(self) -> float:
        if self.config.method != "plus":
            return 0.0
        p = self.config.p
        if not (0.0 < p < 1.0):
            raise ValueError(f"p must be in (0, 1) for method=plus, got p={p}")
        return math.log(max(p / (1 - p), (1 - p) / p))

    def precompute(self, word2id, sword2id, nword2id, vocab, all_embed, s_embed, n_embed):
        self.vocab = vocab
        self.word2id = word2id
        self.sword2id = sword2id
        self.nword2id = nword2id
        self.id2word = {v: k for k, v in word2id.items()}
        self.id2sword = {v: k for k, v in sword2id.items()}
        self.id2nword = {v: k for k, v in nword2id.items()}
        
        if self.config.method == "plus":
            # PLUS validation (Copilot fix!)
            if not sword2id or not nword2id:
                raise ValueError("method='plus' requires non-empty sensitive and normal vocabularies.")

        metric_fn = cosine_distances if self.config.distance_metric == "cosine" else euclidean_distances

        if self.config.method == "normal":
            self.s_distance_matrix = metric_fn(s_embed, all_embed)
            self.n_distance_matrix = metric_fn(n_embed, all_embed)
        elif self.config.method == "plus":
            self.s_distance_matrix = metric_fn(all_embed, s_embed)
            self.n_distance_matrix = metric_fn(all_embed, n_embed)
        elif self.config.method == "santext":
            self.n_distance_matrix = metric_fn(all_embed, all_embed)
            
        if self.s_distance_matrix is not None:
            L = self.mixing_overhead
            s_mech_eps = self.config.s_epsilon - L
            assert s_mech_eps > 0, "s_epsilon must exceed mixing overhead L"
            self.s_prob_matrix = self._build_prob_matrix(self.s_distance_matrix, s_mech_eps)
            
        if self.config.method == "santext":
            self.n_prob_matrix_fixed = self._build_n_prob_matrix(self.config.epsilon)

    def _build_prob_matrix(self, distance_matrix, eps):
        return softmax(eps * (-distance_matrix) / (2 * self.sensitivity), axis=1)

    def _build_n_prob_matrix(self, epsilon_n):
        return self._build_prob_matrix(self.n_distance_matrix, epsilon_n)

    def count_words(self, doc: SastdpDocument):
        ns = nn = n_oov = 0
        for raw_word in doc.text.split():
            w = raw_word.lower()
            if w in self.word2id:
                if w in self.sword2id: ns += 1
                else: nn += 1
            else: n_oov += 1
        return ns, nn, n_oov
        
    def sanitize_santext(self, doc: SastdpDocument):
        n_prob_matrix = self.n_prob_matrix_fixed
        out = []
        tot_eps = 0.0
        for raw_word in doc.text.split():
            w = raw_word.lower()
            if w in self.word2id:
                prob = n_prob_matrix[self.word2id[w]]
                out.append(self.id2word[np.random.choice(len(prob), p=prob)])
                tot_eps += self.config.epsilon
            else:
                out.append(self.vocab[np.random.randint(len(self.vocab))])
        return " ".join(out), tot_eps

    def sanitize_normal(self, doc: SastdpDocument, epsilon_n=None):
        eps_n = epsilon_n if epsilon_n is not None else self.config.epsilon
        if epsilon_n is None:
            if self.n_prob_matrix_fixed is None: self.n_prob_matrix_fixed = self._build_n_prob_matrix(eps_n)
            n_prob_matrix = self.n_prob_matrix_fixed
        else:
            n_prob_matrix = self._build_n_prob_matrix(eps_n)
            
        out = []
        tot_eps = 0.0
        for raw_word in doc.text.split():
            w = raw_word.lower()
            if w in self.word2id:
                if w in self.sword2id:
                    prob = self.s_prob_matrix[self.sword2id[w]]
                    out.append(self.id2word[np.random.choice(len(prob), p=prob)])
                    tot_eps += self.config.s_epsilon
                else:
                    prob = n_prob_matrix[self.nword2id[w]]
                    out.append(self.id2word[np.random.choice(len(prob), p=prob)])
                    tot_eps += eps_n
            else:
                out.append(self.vocab[np.random.randint(len(self.vocab))])
        return " ".join(out), tot_eps

    def sanitize_plus(self, doc: SastdpDocument, epsilon_n=None):
        eps_n = epsilon_n if epsilon_n is not None else self.config.epsilon
        L = self.mixing_overhead
        eps_n_mech = eps_n - L
        
        if epsilon_n is None:
            if self.n_prob_matrix_fixed is None: self.n_prob_matrix_fixed = self._build_n_prob_matrix(eps_n_mech)
            n_prob_matrix = self.n_prob_matrix_fixed
        else:
            n_prob_matrix = self._build_n_prob_matrix(eps_n_mech)
            
        out = []
        tot_eps = 0.0
        p = self.config.p
        for raw_word in doc.text.split():
            w = raw_word.lower()
            if w in self.word2id:
                flip = np.random.random()
                idx = self.word2id[w]
                if w in self.sword2id:
                    if flip <= p:
                        prob = self.s_prob_matrix[idx]
                        out.append(self.id2sword[np.random.choice(len(prob), p=prob)])
                        tot_eps += self.config.s_epsilon
                    else:
                        prob = n_prob_matrix[idx]
                        out.append(self.id2nword[np.random.choice(len(prob), p=prob)])
                        tot_eps += eps_n
                else:
                    if flip <= p:
                        prob = n_prob_matrix[idx]
                        out.append(self.id2nword[np.random.choice(len(prob), p=prob)])
                        tot_eps += eps_n
                    else:
                        prob = self.s_prob_matrix[idx]
                        out.append(self.id2sword[np.random.choice(len(prob), p=prob)])
                        tot_eps += self.config.s_epsilon
            else:
                out.append(self.vocab[np.random.randint(len(self.vocab))])
        return " ".join(out), tot_eps

def compute_per_doc_epsilon(docs: list[SastdpDocument], sanitizer: Sanitizer) -> list:
    eps = sanitizer.config.epsilon
    eps_s = sanitizer.config.s_epsilon
    result = []
    for doc in docs:
        ns, nn, _ = sanitizer.count_words(doc)
        if nn == 0:
            result.append(None)
            continue
        eps_t = eps * (ns + nn)
        eps_n = (eps_t - ns * eps_s) / nn
        if eps_n <= 0: eps_n = 0.01
        result.append(eps_n)
    return result

_W_sanitizer = None
def _init_worker(san):
    global _W_sanitizer
    _W_sanitizer = san

def _dispatch(args):
    doc, eps_n = args
    if _W_sanitizer.config.method == "santext": return _W_sanitizer.sanitize_santext(doc)
    if _W_sanitizer.config.method == "normal": return _W_sanitizer.sanitize_normal(doc, eps_n)
    if _W_sanitizer.config.method == "plus": return _W_sanitizer.sanitize_plus(doc, eps_n)

def sanitize_corpus(docs, sanitizer, per_doc_epsilons, threads=4, desc="Sanitizing"):
    work_items = list(zip(docs, per_doc_epsilons))
    threads = min(threads, cpu_count())
    if threads <= 1:
        _init_worker(sanitizer)
        from tqdm import tqdm
        return [_dispatch(item) for item in tqdm(work_items, desc=desc)]
    
    with Pool(threads, initializer=_init_worker, initargs=(sanitizer,)) as pool:
        from tqdm import tqdm
        return list(tqdm(pool.imap(_dispatch, work_items, chunksize=32), total=len(docs), desc=desc))
