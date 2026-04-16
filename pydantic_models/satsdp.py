import os
from pydantic import BaseModel
from enum import StrEnum
from typing import Any
import numpy as np


class SastdpMethod(StrEnum):
    NORMAL = "normal"
    PLUS = "plus"
    SANTEXT = "santext"


class SastdpExecutionArgs(BaseModel):
    data_dir: str
    word_embedding_path: str
    word_embedding_size: int = 300
    method: SastdpMethod = SastdpMethod.NORMAL
    task: str
    seed: int = 42
    epsilon: float
    s_epsilon: float
    p: float = 0.7
    threads: int = 2
    adjusted_epsilon: float | None = None
    sensitive_words_file_path: str | None = None
    language: str
    redistribute: bool = True
    corpus_statistics: dict[str, float] = {}

    @property
    def replacements_output_dir(self) -> str:
        sensitive_words_name = os.path.splitext(os.path.basename(self.sensitive_words_file_path))[0]
        if "finetuned" in sensitive_words_name:
            suffix = "finetuned"
        elif "flair" in sensitive_words_name:
            suffix = "flair"
        prefix = f"replacements_{suffix}/{self.seed}/{self.method}/{self.task}"
        if self.method == SastdpMethod.PLUS:
            return f"{prefix}/p_{self.p}_n_epsilon_{self.epsilon}_s_epsilon_{self.s_epsilon}"
        if self.method == SastdpMethod.NORMAL:
            return f"{prefix}/n_epsilon_{self.epsilon}_s_epsilon_{self.s_epsilon}"
        if self.method == SastdpMethod.SANTEXT:
            return f"{prefix}/epsilon_{self.epsilon}"

    @property
    def mappings_cache_dir(self) -> str:
        sensitive_words_name = os.path.splitext(os.path.basename(self.sensitive_words_file_path))[0]
        return f"embeddings_cache/{self.task}/{sensitive_words_name}"

    @property
    def mappings_cache_path(self) -> str:
        embedding_file = os.path.basename(self.word_embedding_path)
        return f"{self.mappings_cache_dir}/{embedding_file}.npz"

    @property
    def corpus_statistics_path(self) -> str:
        sensitive_words_name = os.path.splitext(os.path.basename(self.sensitive_words_file_path))[0]
        folder = f"corpus_statistics/{self.method}"
        os.makedirs(folder, exist_ok=True)
        return f"{folder}/{self.task}_{sensitive_words_name}.json"


class SastdpEmbeddingAndMappings(BaseModel):
    sensitive_word_embed: Any
    normal_word_embed: Any
    all_word_embed: Any
    word2id: dict[str, int]
    sword2id: dict[str, int]
    nword2id: dict[str, int]


class SastdpDocument(BaseModel):
    text: str
    text_id: int | str


class SastdpInitArgs(BaseModel):
    args: SastdpExecutionArgs
    vocab_init: list[Any]
    sensitive_words_init: list[Any]
    word2id_init: Any
    sword2id_init: Any
    nword2id_init: Any
    s_prob_matrix_init: Any
    n_prob_matrix_init: Any


class SastdpDocumentStatistics(BaseModel):
    text_id: int | str
    sensitive_word_count: int
    normal_word_count: int
    total_word_count: int
    total_epsilon: float = 0.0
