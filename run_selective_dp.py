import argparse
import random
import numpy as np
import logging
import os
import re

from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from utils_custext import get_vocab_SST2, get_vocab_QNLI, get_vocab_text, word_normalize
from spacy.lang.en import English
from spacy.lang.es import Spanish
from nandptextsan import NADPTextSan_init, NADPTextSan, NADPTextSan_plus, cal_probability, SanText
import json
import pandas as pd
from pydantic_models.satsdp import (
    SastdpExecutionArgs,
    SastdpDocument,
    SastdpMethod,
    SastdpEmbeddingAndMappings,
    SastdpInitArgs,
)

logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default="./datasets/i2b2/", type=str, help="The input dir")

    parser.add_argument(
        "--word_embedding_path",
        default="embeddings/english/glove.840B.300d.txt",
        type=str,
        help="The pretrained word embedding path. leave it blank if you are using BERT",
    )

    parser.add_argument(
        "--word_embedding_size",
        default=300,
        type=int,
        help="The pretrained word embedding size. leave it blank if you are using BERT",
    )

    parser.add_argument(
        "--method",
        default=SastdpMethod.NORMAL,
        help="Sanitized method",
    )

    parser.add_argument(
        "--task",
        default="i2b2",
        help="NLP eval tasks",
    )

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    parser.add_argument("--epsilon", type=float, default=16, help="privacy parameter epsilon")
    parser.add_argument("--s_epsilon", type=float, default=8, help="privacy parameter epsilon")
    parser.add_argument(
        "--p",
        type=float,
        default=0.7,
        help="SanText+: probability of non-sensitive words to be sanitized",
    )

    parser.add_argument("--threads", type=int, default=2, help="number of processors")
    parser.add_argument(
        "--sensitive_words_file_path", type=str, default="./selective_output/sensitive_mapping/0.6_i2b2.json"
    )
    parser.add_argument("--language", type=str, default="en")

    return SastdpExecutionArgs(**parser.parse_args().__dict__)


def set_seed(args: SastdpExecutionArgs):
    random.seed(args.seed)
    np.random.seed(args.seed)


def _get_vocab(args: SastdpExecutionArgs, tokenizer, tokenizer_type):
    if args.task == "SST-2":
        vocab = get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "QNLI":
        vocab = get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    else:
        vocab = get_vocab_text(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    return vocab


def matches_sensitive_subword(word: str, sensitive_words: list[str]) -> bool:
    word = word.lower()
    return word in sensitive_words


def _get_word_embeddings_and_mappings(args: SastdpExecutionArgs, vocab, sensitive_words) -> SastdpEmbeddingAndMappings:
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
    sensitive_word_embed = []
    normal_word_embed = []
    all_word_embed = []
    word2id = {}
    sword2id = {}
    nword2id = {}
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
        f.close()
    all_word_embed = np.array(all_word_embed, dtype="f")
    normal_word_embed = np.array(normal_word_embed, dtype="f")
    sensitive_word_embed = np.array(sensitive_word_embed, dtype="f")
    np.savez_compressed(
        args.mappings_cache_path,
        sensitive_word_embed=sensitive_word_embed,
        normal_word_embed=normal_word_embed,
        all_word_embed=all_word_embed,
        word2id=word2id,
        sword2id=sword2id,
        nword2id=nword2id,
    )
    return SastdpEmbeddingAndMappings(
        sensitive_word_embed=sensitive_word_embed,
        normal_word_embed=normal_word_embed,
        all_word_embed=all_word_embed,
        word2id=word2id,
        sword2id=sword2id,
        nword2id=nword2id,
    )


def _get_probability_matrix(args: SastdpExecutionArgs, embedding_and_mappings: SastdpEmbeddingAndMappings):
    if args.method == SastdpMethod.NORMAL:
        s_prob_matrix = cal_probability(
            embedding_and_mappings.sensitive_word_embed,
            embedding_and_mappings.all_word_embed,
            "sensitive",
            args.epsilon,
            args.s_epsilon,
        )
        if len(embedding_and_mappings.normal_word_embed) > len(embedding_and_mappings.sensitive_word_embed):
            rest = (
                (len(embedding_and_mappings.sensitive_word_embed))
                * (args.epsilon - args.s_epsilon)
                / len(embedding_and_mappings.normal_word_embed)
            )
            args.adjusted_epsilon = args.epsilon
            print(f"epsilon: {args.epsilon}, s_epsilon: {args.s_epsilon}, Adjusted epsilon: {args.adjusted_epsilon}")
            print(
                f"normal_word_embed: {len(embedding_and_mappings.normal_word_embed)}, sensitive_word_embed: {len(embedding_and_mappings.sensitive_word_embed)}"
            )
        n_prob_matrix = cal_probability(
            embedding_and_mappings.normal_word_embed,
            embedding_and_mappings.all_word_embed,
            "normal",
            args.adjusted_epsilon,
            args.s_epsilon,
        )
    elif args.method == SastdpMethod.PLUS:
        s_prob_matrix = cal_probability(
            embedding_and_mappings.all_word_embed,
            embedding_and_mappings.sensitive_word_embed,
            "sensitive",
            args.epsilon,
            args.s_epsilon,
        )
        if len(embedding_and_mappings.normal_word_embed) > len(embedding_and_mappings.sensitive_word_embed):
            rest = (
                (len(embedding_and_mappings.normal_word_embed) - len(embedding_and_mappings.sensitive_word_embed))
                * (args.epsilon - args.s_epsilon)
                / len(embedding_and_mappings.normal_word_embed)
            )
            args.adjusted_epsilon = args.epsilon
            print(f"Adjusted epsilon: {args.adjusted_epsilon}")
        n_prob_matrix = cal_probability(
            embedding_and_mappings.all_word_embed,
            embedding_and_mappings.normal_word_embed,
            "normal",
            args.adjusted_epsilon,
            args.s_epsilon,
        )
    elif args.method == SastdpMethod.SANTEXT:
        n_prob_matrix = cal_probability(
            embedding_and_mappings.all_word_embed, embedding_and_mappings.all_word_embed, epsilon=args.epsilon
        )
        s_prob_matrix = None
    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")
    return s_prob_matrix, n_prob_matrix


def _get_algorithm(args: SastdpExecutionArgs):
    if args.method == SastdpMethod.NORMAL:
        return NADPTextSan
    elif args.method == SastdpMethod.PLUS:
        return NADPTextSan_plus
    elif args.method == SastdpMethod.SANTEXT:
        return SanText
    else:
        raise NotImplementedError(f"Method {args.method} not implemented.")


def main():
    args = _parse_args()

    set_seed(args)

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(
        "Running method: %s, task: %s,  epsilon = %s, epsilon_s = %s, random_seed: %d",
        args.method,
        args.task,
        args.epsilon,
        args.s_epsilon,
        args.seed,
    )

    if not os.path.exists(args.replacements_output_dir):
        os.makedirs(args.replacements_output_dir, exist_ok=True)

    if not os.path.exists(args.mappings_cache_dir):
        os.makedirs(args.mappings_cache_dir, exist_ok=True)

    logger.info("Building Vocabulary...")

    tokenizer = English() if args.language == "en" else Spanish()
    tokenizer_type = "word"
    vocab = _get_vocab(args, tokenizer, tokenizer_type)
    words = [key.lower() for key, _ in vocab.most_common()]
    with open(args.sensitive_words_file_path, "r", encoding="utf8") as f:
        sensitive_words = json.load(f)
    sensitive_words = list(sensitive_words.keys())
    sensitive_words = [sensitive_word.lower() for sensitive_word in sensitive_words]

    embedding_and_mappings = _get_word_embeddings_and_mappings(args, words, sensitive_words)
    logger.info("getting prob matrix...")
    s_prob_matrix, n_prob_matrix = _get_probability_matrix(args, embedding_and_mappings)

    threads = min(args.threads, cpu_count())
    for file_name in ["train.csv"]:
        data_file = os.path.join(args.data_dir, file_name)
        logger.info("Processing file: %s.", data_file)
        if args.task in ["SST-2", "QNLI"]:
            num_lines = sum(1 for _ in open(data_file))
            with open(data_file, "r", encoding="utf-8") as rf:
                # header
                _ = next(rf)
                labels = []
                docs = []
                if args.task == "SST-2":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text = content[0]
                        label = int(content[1])
                        doc = [token.text for token in tokenizer(text)]
                        docs.append(doc)
                        labels.append(label)
                elif args.task == "QNLI":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text1 = content[1]
                        text2 = content[2]
                        label = content[-1]
                        doc1 = [token.text for token in tokenizer(text1)]
                        doc2 = [token.text for token in tokenizer(text2)]

                        docs.append(doc1)
                        docs.append(doc2)
                        labels.append(label)

                rf.close()
        else:
            docs: list[SastdpDocument] = []
            df = pd.read_csv(data_file)
            texts = df["text"].dropna()
            text_ids = df["text_id"].dropna()
            assert len(texts) == len(text_ids)
            for text, text_id in tqdm(zip(texts, text_ids), total=len(texts)):
                text = text.strip().replace("\n", " ").replace("\t", " ")
                text = re.sub(" +", " ", text)
                doc = [token.text for token in tokenizer(text)]
                docs.append(SastdpDocument(text=" ".join(doc), text_id=text_id))

        alg = _get_algorithm(args)
        print("using method: ", alg, "args: ", args)
        init_args = SastdpInitArgs(
            args=args,
            vocab_init=words,
            sensitive_words_init=sensitive_words,
            word2id_init=embedding_and_mappings.word2id,
            sword2id_init=embedding_and_mappings.sword2id,
            nword2id_init=embedding_and_mappings.nword2id,
            s_prob_matrix_init=s_prob_matrix,
            n_prob_matrix_init=n_prob_matrix,
        )
        with Pool(
            threads,
            initializer=NADPTextSan_init,
            initargs=(init_args.model_dump(),),
        ) as p:
            annotate_ = partial(
                alg,
            )
            results = list(
                tqdm(
                    p.imap(annotate_, docs, chunksize=32),
                    total=len(docs),
                    desc=f"Sanitize docs using {alg}",
                )
            )
            p.close()

        for result in results:
            args.corpus_statistics[result.text_id] = result.model_dump()
        args.corpus_statistics["corpus_sensitive_words_count"] = len(embedding_and_mappings.sensitive_word_embed)
        args.corpus_statistics["corpus_normal_words_count"] = len(embedding_and_mappings.normal_word_embed)
        args.corpus_statistics["corpus_total_words_count"] = len(embedding_and_mappings.all_word_embed)
        with open(args.corpus_statistics_path, "w", encoding="utf-8") as f:
            json.dump(args.corpus_statistics, f, indent=2)


if __name__ == "__main__":
    main()
