"""Utility functions: tokenizer, vocabulary, embeddings, sensitive word detection."""
from __future__ import annotations

import os
import re
import unicodedata
import json
import logging
from collections import Counter

import numpy as np
from tqdm import tqdm

from pydantic_models.sanitizerdp import SanitizerDPEmbeddingAndMappings

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


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def load_glove(
    path: str,
    vocab: set[str],
    sensitive_words: set[str],
) -> tuple[list[str], SanitizerDPEmbeddingAndMappings]:
    """Load GloVe and partition into sensitive / normal subsets.

    Returns
    -------
    words : list[str]
        All words found in both vocab and GloVe (ordered by appearance).
    embeddings : SanitizerDPEmbeddingAndMappings
        Pydantic model with word2id, sword2id, nword2id and embedding arrays.
    """
    word2id: dict[str, int] = {}
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
    s_embed = (np.array(s_embeds, dtype=np.float32)
               if s_embeds else np.empty((0, dim), dtype=np.float32))
    n_embed = (np.array(n_embeds, dtype=np.float32)
               if n_embeds else np.empty((0, dim), dtype=np.float32))

    logger.info(
        "Embeddings — all: %s  sensitive: %s  normal: %s  (%.1f GB f32)",
        all_embed.shape, s_embed.shape, n_embed.shape,
        all_embed.nbytes / 1024 ** 3,
    )

    words = list(word2id.keys())
    embeddings = SanitizerDPEmbeddingAndMappings(
        sensitive_word_embed=s_embed,
        normal_word_embed=n_embed,
        all_word_embed=all_embed,
        word2id=word2id,
        sword2id=sword2id,
        nword2id=nword2id,
    )
    return words, embeddings


def _load_glove_words(path: str) -> list[str]:
    """Load just the word column from GloVe (no embeddings)."""
    words = []
    with open(path, encoding="utf-8") as f:
        first = f.readline().rstrip().split(" ")
        if len(first) != 2:
            f.seek(0)
        for line in tqdm(f, desc="Loading GloVe words"):
            words.append(line.split(" ", 1)[0].lower())
    return words


# ---------------------------------------------------------------------------
# Sensitive word detection
# ---------------------------------------------------------------------------

def _build_sensitive_words_ner(
    texts: list[str],
    model_name: str = "flair/ner-english-large",
    threshold: float = 0.3,
    entity_types: tuple[str, ...] = ("PER", "ORG", "LOC"),
    batch_size: int = 32,
) -> set[str]:
    """Run Flair NER on texts and collect entity tokens as sensitive words."""
    try:
        from flair.models import SequenceTagger
        from flair.data import Sentence as FlairSentence
    except ImportError:
        raise ImportError(
            "flair is required for NER-based sensitivity detection.\n"
            "Install it with:  pip install flair\n"
            "Or use --no_ner to fall back to frequency-based detection."
        )

    # Models are pre-downloaded by setup_and_run.sh; allow online fallback.

    # Resolve model: if it's a HuggingFace model ID (contains '/'), try the local
    # Flair cache first so it works behind proxies that block huggingface.co.
    _model_arg = model_name
    if "/" in model_name and not os.path.exists(model_name):
        _org, _name = model_name.split("/", 1)
        _snapshot_root = os.path.join(
            os.path.expanduser("~/.flair/models"), _name,
            f"models--{_org}--{_name}", "snapshots",
        )
        if os.path.isdir(_snapshot_root):
            _snapshots = sorted(os.listdir(_snapshot_root))
            if _snapshots:
                _candidate = os.path.join(_snapshot_root, _snapshots[-1], "pytorch_model.bin")
                if os.path.exists(_candidate):
                    _model_arg = _candidate
                    logger.info("Using local Flair cache: %s", _candidate)

    logger.info("Loading Flair NER model '%s' …", model_name)
    tagger = SequenceTagger.load(_model_arg)

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


def build_sensitive_words(
    texts: list[str] | None = None,
    vocab_words: list[str] | None = None,
    use_ner: bool = True,
    ner_model: str = "flair/ner-english-large",
    ner_threshold: float = 0.3,
    sensitive_pct: float = 0.5,
    sensitive_words_path: str | None = None,
    source: str = "dataset",
    glove_path: str | None = None,
    task: str = "unknown",
    seed: int = 42,
    output_dir: str = "output/sensitive_words",
) -> set[str]:
    """Build the set of sensitive words (NER or frequency-based).

    Args:
        texts:          raw text strings from the dataset (for source="dataset")
        vocab_words:    ordered word list (for frequency-based fallback)
        use_ner:        True=Flair NER, False=frequency-based
        ner_model:      Flair model tag
        ner_threshold:  minimum entity confidence
        sensitive_pct:  fraction of vocab treated as sensitive (fallback)
        sensitive_words_path: path to a JSON file with sensitive words (fallback)
        source:         "dataset", "glove", or "dataset+glove"
        glove_path:     path to GloVe file (needed for source="glove"|"dataset+glove")
        task:           task name for cache filename
        seed:           seed for cache filename
        output_dir:     directory for cached sensitive word files

    Returns:
        set of lowercased sensitive word strings

    Cache path example:
        output/sensitive_words/flair_ner-english-large_dataset_sst2_0.3thr_42seed.json
        output/sensitive_words/flair_ner-english-large_glove_sst2_0.6thr_42seed.json
    """
    # Build descriptive cache filename
    if use_ner:
        model_short = ner_model.replace("/", "_")
        cache_name = f"{model_short}_{source}_{task}_{ner_threshold}thr_{seed}seed.json"
    elif sensitive_words_path:
        base = os.path.splitext(os.path.basename(sensitive_words_path))[0]
        cache_name = f"from_file_{base}_{task}_{seed}seed.json"
    else:
        cache_name = f"freq_based_{task}_{sensitive_pct}pct_{seed}seed.json"

    cache_path = os.path.join(output_dir, cache_name)

    # Check cache
    if os.path.exists(cache_path):
        with open(cache_path, encoding="utf-8") as f:
            sensitive = set(json.load(f))
        logger.info("Sensitive words loaded from cache: %s (%d words)",
                     cache_path, len(sensitive))
        return sensitive

    # Build sensitive words
    if use_ner:
        ner_texts = []
        if source in ("dataset", "dataset+glove"):
            if texts is None:
                raise ValueError("texts required for source='dataset' or 'dataset+glove'")
            ner_texts.extend(texts)
        if source in ("glove", "dataset+glove"):
            if glove_path is None:
                raise ValueError("glove_path required for source='glove' or 'dataset+glove'")
            logger.info("Loading GloVe words for NER (source=%s) …", source)
            glove_words = _load_glove_words(glove_path)
            ner_texts.extend(glove_words)

        sensitive = _build_sensitive_words_ner(
            ner_texts, model_name=ner_model, threshold=ner_threshold)
    elif sensitive_words_path:
        with open(sensitive_words_path, encoding="utf-8") as f:
            sensitive = {k.lower() for k in json.load(f)}
    else:
        if vocab_words is None:
            raise ValueError("vocab_words required for frequency-based detection")
        n = int(sensitive_pct * len(vocab_words))
        sensitive = set(vocab_words[-n:]) if n else set()

    # Save cache
    os.makedirs(output_dir, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(sorted(sensitive), f, ensure_ascii=False, indent=2)
    logger.info("Sensitive words saved to: %s (%d words)", cache_path, len(sensitive))
    return sensitive


# ---------------------------------------------------------------------------
# Task I/O — SST-2 and QNLI
# ---------------------------------------------------------------------------

def extract_texts(task: str, data_dir: str, max_samples: int | None = None) -> list[str]:
    """Collect raw text strings across train+dev for vocab building."""
    texts = []
    for split in ("train", "dev"):
        path = os.path.join(data_dir, f"{split}.tsv")
        if not os.path.exists(path):
            continue
        cap = max_samples if (max_samples and split == "train") else None
        with open(path, encoding="utf-8") as f:
            next(f)
            for i, line in enumerate(f):
                if cap is not None and i >= cap:
                    break
                parts = line.strip().split("\t")
                if task == "sst2" and parts:
                    texts.append(parts[0])
                elif task == "qnli" and len(parts) >= 3:
                    texts.append(parts[1])
                    texts.append(parts[2])
    return texts


def read_sst2(path, tokenizer, max_samples=None):
    docs, labels = [], []
    with open(path, encoding="utf-8") as f:
        header = next(f)
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            docs.append([tok.text for tok in tokenizer(parts[0])])
            labels.append(parts[1])
    return docs, labels, header


def write_sst2(path, sanitized, labels, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for text, label in zip(sanitized, labels):
            f.write(f"{text}\t{label}\n")


def read_qnli(path, tokenizer, max_samples=None):
    """Returns 2 docs per example (question + sentence)."""
    docs, labels = [], []
    with open(path, encoding="utf-8") as f:
        header = next(f)
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            parts = line.strip().split("\t")
            if len(parts) < 4:
                continue
            docs.append([tok.text for tok in tokenizer(parts[1])])
            docs.append([tok.text for tok in tokenizer(parts[2])])
            labels.append(parts[3])
    return docs, labels, header


def write_qnli(path, sanitized, labels, header):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(header)
        for i, label in enumerate(labels):
            f.write(f"{i}\t{sanitized[i*2]}\t{sanitized[i*2+1]}\t{label}\n")


READERS = {"sst2": read_sst2, "qnli": read_qnli}
WRITERS = {"sst2": write_sst2, "qnli": write_qnli}
