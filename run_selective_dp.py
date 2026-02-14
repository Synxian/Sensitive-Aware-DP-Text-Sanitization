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
from nandptextsan import NADPTextSan_init, NADPTextSan, NADPTextSan_plus, cal_probability
import json
import pandas as pd
from pydantic_models.satsdp import SastdpExecutionArgs, SastdpDocument

logger = logging.getLogger(__name__)


def set_seed(args: SastdpExecutionArgs):
    random.seed(args.seed)
    np.random.seed(args.seed)


def main():
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
        choices=["normal", "plus"],
        default="normal",
        help="Sanitized method",
    )

    parser.add_argument(
        "--task",
        choices=["i2b2", "SST-2", "QNLI"],
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

    args = SastdpExecutionArgs(**parser.parse_args().__dict__)

    set_seed(args)

    logging.basicConfig(
        format="%(asctime)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(
        "Running method: %s, task: %s,  epsilon = %s, random_seed: %d"
        % (args.method, args.task, args.epsilon, args.seed)
    )

    if not os.path.exists(args.replacements_output_dir):
        os.makedirs(args.replacements_output_dir)

    logger.info("Building Vocabulary...")

    tokenizer = English()
    tokenizer_type = "word"
    if args.task == "SST-2":
        vocab = get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "QNLI":
        vocab = get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    else:
        vocab = get_vocab_text(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)

    words = [key for key, _ in vocab.most_common()]
    with open("./selective_output/sensitive_mapping/0.6_i2b2.json", "r", encoding="utf8") as f:
        sensitive_words = json.load(f)
    sensitive_words = list(sensitive_words.keys())

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
            cur_word = word_normalize(content[0])
            if cur_word in vocab and cur_word not in word2id:
                word2id[cur_word] = all_count
                all_count += 1
                emb = [float(i) for i in content[1:]]
                all_word_embed.append(emb)
                if cur_word in sensitive_words:
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
    if args.method == "normal":
        s_prob_matrix = cal_probability(
            sensitive_word_embed,
            all_word_embed,
            "sensitive",
            args.epsilon,
            args.s_epsilon,
        )
        if len(normal_word_embed) > len(sensitive_word_embed):
            rest = (len(sensitive_word_embed)) * (args.epsilon - args.s_epsilon) / len(normal_word_embed)
            args.epsilon = args.epsilon + rest
            print(f"epsilon: {args.epsilon}, s_epsilon: {args.s_epsilon}, Adjusted epsilon: {args.epsilon}")
            print(f"normal_word_embed: {len(normal_word_embed)}, sensitive_word_embed: {len(sensitive_word_embed)}")
        n_prob_matrix = cal_probability(normal_word_embed, all_word_embed, "normal", args.epsilon, args.s_epsilon)
    else:
        s_prob_matrix = cal_probability(
            all_word_embed,
            sensitive_word_embed,
            "sensitive",
            args.epsilon,
            args.s_epsilon,
        )
        if len(normal_word_embed) > len(sensitive_word_embed):
            rest = (
                (len(normal_word_embed) - len(sensitive_word_embed))
                * (args.epsilon - args.s_epsilon)
                / len(normal_word_embed)
            )
            args.epsilon = args.epsilon + rest
            print(f"Adjusted epsilon: {args.epsilon}")
        n_prob_matrix = cal_probability(all_word_embed, normal_word_embed, "normal", args.epsilon, args.s_epsilon)

    threads = min(args.threads, cpu_count())
    for file_name in ["train.csv"]:
        data_file = os.path.join(args.data_dir, file_name)
        logger.info("Processing file: %s." % (data_file))
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
                text = text.strip().replace("\n", "").replace("\t", "")
                text = re.sub(" +", " ", text)
                doc = [token.text for token in tokenizer(text)]
                docs.append(SastdpDocument(text=doc, text_id=text_id))

        alg = NADPTextSan if args.method == "normal" else NADPTextSan_plus
        print("using method: ", alg)
        with Pool(
            threads,
            initializer=NADPTextSan_init,
            initargs=(
                args,
                words,
                sensitive_words,
                word2id,
                sword2id,
                nword2id,
                s_prob_matrix,
                n_prob_matrix,
            ),
        ) as p:
            annotate_ = partial(
                alg,
            )
            _ = list(
                tqdm(
                    p.imap(annotate_, docs, chunksize=32),
                    total=len(docs),
                    desc="Sanitize docs using NandpTextSan",
                )
            )
            p.close()


if __name__ == "__main__":
    main()
