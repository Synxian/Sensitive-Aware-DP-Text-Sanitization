"""Vocabulary construction and embedding loading utilities."""

import os
import numpy as np
from tqdm import tqdm

from text_utils import get_vocab_SST2, get_vocab_QNLI, get_vocab_text, word_normalize
from pydantic_models.satsdp import SastdpExecutionArgs, SastdpEmbeddingAndMappings


def get_vocab(args: SastdpExecutionArgs, tokenizer, tokenizer_type: str):
    """Build word frequency vocabulary from the dataset."""
    if args.task == "SST-2":
        return get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "QNLI":
        return get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    else:
        return get_vocab_text(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)


def matches_sensitive_subword(word: str, sensitive_words: list[str]) -> bool:
    return word.lower() in sensitive_words


def get_word_embeddings_and_mappings(
    args: SastdpExecutionArgs, vocab: list[str], sensitive_words: list[str]
) -> SastdpEmbeddingAndMappings:
    """Load word embeddings and split into sensitive/normal subsets.

    Reads the embedding file (e.g. GloVe), keeps only words present in vocab,
    and partitions them into sensitive and normal based on the sensitive_words
    list. Results are cached as .npz for subsequent runs.
    """
    if os.path.exists(args.mappings_cache_path):
        print(f"cache hit, loading embeddings and mappings from {args.mappings_cache_path}")
        data = np.load(args.mappings_cache_path, allow_pickle=True)
        return SastdpEmbeddingAndMappings(
            sensitive_word_embed=data["sensitive_word_embed"],
            normal_word_embed=data["normal_word_embed"],
            all_word_embed=data["all_word_embed"],
            word2id=data["word2id"].item(),
            sword2id=data["sword2id"].item(),
            nword2id=data["nword2id"].item(),
        )

    sensitive_word_embed: list[list[float]] = []
    normal_word_embed: list[list[float]] = []
    all_word_embed: list[list[float]] = []
    word2id: dict[str, int] = {}
    sword2id: dict[str, int] = {}
    nword2id: dict[str, int] = {}
    sensitive_count = 0
    normal_count = 0
    all_count = 0

    num_lines = sum(1 for _ in open(args.word_embedding_path, "r", encoding="utf-8"))
    with open(args.word_embedding_path, "r", encoding="utf-8") as f:
        line = f.readline().rstrip().split(" ")
        if len(line) != 2:
            f.seek(0)
        for row in tqdm(f, total=num_lines - 1):
            content = row.rstrip().split(" ")
            cur_word = word_normalize(content[0]).lower()
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = all_count
                all_count += 1
                emb = [float(i) for i in content[1:]]
                all_word_embed.append(emb)
                if matches_sensitive_subword(cur_word, sensitive_words):
                    sword2id[cur_word] = sensitive_count
                    sensitive_count += 1
                    sensitive_word_embed.append(emb)
                else:
                    nword2id[cur_word] = normal_count
                    normal_count += 1
                    normal_word_embed.append(emb)
            assert len(word2id) == len(all_word_embed)
            assert len(sword2id) == len(sensitive_word_embed)
            assert len(nword2id) == len(normal_word_embed)

    all_embed_np = np.array(all_word_embed, dtype="f")
    normal_embed_np = np.array(normal_word_embed, dtype="f")
    sensitive_embed_np = np.array(sensitive_word_embed, dtype="f")

    os.makedirs(os.path.dirname(args.mappings_cache_path), exist_ok=True)
    np.savez_compressed(
        args.mappings_cache_path,
        sensitive_word_embed=sensitive_embed_np,
        normal_word_embed=normal_embed_np,
        all_word_embed=all_embed_np,
        word2id=word2id,
        sword2id=sword2id,
        nword2id=nword2id,
    )
    return SastdpEmbeddingAndMappings(
        sensitive_word_embed=sensitive_embed_np,
        normal_word_embed=normal_embed_np,
        all_word_embed=all_embed_np,
        word2id=word2id,
        sword2id=sword2id,
        nword2id=nword2id,
    )
