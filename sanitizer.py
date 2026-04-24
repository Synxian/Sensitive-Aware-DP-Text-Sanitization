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
# Probability matrix
# ---------------------------------------------------------------------------

def cal_probability(embed_src: np.ndarray, embed_tgt: np.ndarray,
                    epsilon: float) -> np.ndarray:
    """Exponential mechanism using cosine similarity as utility.

    P(w→w') = softmax(ε/2 · cosine_sim(w, w'))

    Cosine similarity ∈ [-1, 1] → sensitivity c = 1, matching the LDP
    guarantee in the NERaseText paper. Uses float32 to halve memory.
    """
    e1 = np.asarray(embed_src, dtype=np.float32)
    e2 = np.asarray(embed_tgt, dtype=np.float32)
    sim    = cosine_similarity(e1, e2).astype(np.float32)
    scores = (epsilon * sim / 2).astype(np.float64)
    return softmax(scores, axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Total epsilon tracking
# ---------------------------------------------------------------------------

def compute_total_epsilon(
    docs: list[list[str]],
    sword2id: dict,
    epsilon_s: float,
    epsilon_n: float,
) -> float:
    """Compute total privacy budget consumed by the corpus.

    Per the paper: total ε = max over all documents of
    (n_sensitive_tokens * ε_s + n_normal_tokens * ε_n).

    Args:
        docs:      tokenized documents
        sword2id:  sensitive word → index mapping (from load_glove)
        epsilon_s: privacy budget for sensitive words
        epsilon_n: privacy budget for normal words

    Returns:
        Maximum total epsilon across all documents.
    """
    max_eps = 0.0
    for doc in docs:
        eps = sum(epsilon_s if w.lower() in sword2id else epsilon_n for w in doc)
        if eps > max_eps:
            max_eps = eps
    return max_eps


# ---------------------------------------------------------------------------
# Sanitization (multiprocessing via module-level _W globals)
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(word2id, sword2id, nword2id,
                 s_prob, n_prob, all_words, p, method):
    _W.update(dict(
        word2id=word2id, sword2id=sword2id, nword2id=nword2id,
        id2word={v: k for k, v in word2id.items()},
        id2sword={v: k for k, v in sword2id.items()},
        id2nword={v: k for k, v in nword2id.items()},
        s_prob=s_prob, n_prob=n_prob,
        all_words=all_words, p=p, method=method,
    ))


def _oov_word() -> str:
    return _W["all_words"][np.random.randint(len(_W["all_words"]))]


def _sample(row: np.ndarray, id2word: dict) -> str:
    return id2word[np.random.choice(len(row), p=row)]


def _sanitize_santext(doc):
    w2i, i2w, prob = _W["word2id"], _W["id2word"], _W["n_prob"]
    out = []
    for tok in doc:
        w = tok.lower()
        out.append(_sample(prob[w2i[w]], i2w) if w in w2i else _oov_word())
    return " ".join(out)


def _sanitize_normal(doc):
    sw2i, nw2i = _W["sword2id"], _W["nword2id"]
    i2w = _W["id2word"]
    s_prob, n_prob = _W["s_prob"], _W["n_prob"]
    out = []
    for tok in doc:
        w = tok.lower()
        if w in sw2i:   out.append(_sample(s_prob[sw2i[w]], i2w))
        elif w in nw2i: out.append(_sample(n_prob[nw2i[w]], i2w))
        else:           out.append(_oov_word())
    return " ".join(out)


def _sanitize_plus(doc):
    w2i, sw2i = _W["word2id"], _W["sword2id"]
    i2sw, i2nw = _W["id2sword"], _W["id2nword"]
    s_prob, n_prob, p = _W["s_prob"], _W["n_prob"], _W["p"]
    out = []
    for tok in doc:
        w = tok.lower()
        if w not in w2i:
            out.append(_oov_word()); continue
        idx  = w2i[w]
        flip = np.random.random()
        if w in sw2i:
            out.append(_sample(s_prob[idx], i2sw) if flip <= p
                       else _sample(n_prob[idx], i2nw))
        else:
            out.append(_sample(n_prob[idx], i2nw) if flip <= p
                       else _sample(s_prob[idx], i2sw))
    return " ".join(out)


def _dispatch(doc):
    m = _W["method"]
    if m == "santext": return _sanitize_santext(doc)
    if m == "normal":  return _sanitize_normal(doc)
    if m == "plus":    return _sanitize_plus(doc)
    raise ValueError(f"Unknown method: {m!r}")


def sanitize_corpus(
    docs:      list[list[str]],
    word2id:   dict,
    sword2id:  dict,
    nword2id:  dict,
    s_prob:    "np.ndarray | None",
    n_prob:    "np.ndarray | None",
    all_words: list[str],
    method:    str,
    p:         float = 0.5,
    threads:   int   = 4,
    desc:      str   = "Sanitizing",
) -> list[str]:
    """Sanitize tokenized documents in parallel. Returns list of strings."""
    threads = min(threads, cpu_count())
    with Pool(threads, initializer=_init_worker,
              initargs=(word2id, sword2id, nword2id,
                        s_prob, n_prob, all_words, p, method)) as pool:
        return list(tqdm(
            pool.imap(_dispatch, docs, chunksize=32),
            total=len(docs), desc=desc,
        ))
