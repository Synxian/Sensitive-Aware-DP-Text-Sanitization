import argparse
import torch
import random
import numpy as np
import logging
import os
import re

logger = logging.getLogger(__name__)
from tqdm import tqdm
from scipy.special import softmax
from functools import partial
from multiprocessing import Pool, cpu_count
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from utils_custext import get_vocab_SST2, get_vocab_QNLI, get_vocab_text, word_normalize
from spacy.lang.en import English
from transformers import BertTokenizer, BertForMaskedLM
from nandptextsan import NADPTextSan_init, NADPTextSan, NADPTextSan_plus
import json
import pandas as pd


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def cal_probability(
    word_embed_1, word_embed_2, epsilon_type="normal", epsilon=None, s_epsilon=None
):
    eps = epsilon if epsilon_type == "normal" else s_epsilon
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(eps * sim_matrix / 2, axis=1)
    return prob_matrix


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir", default="./datasets/i2b2/", type=str, help="The input dir"
    )

    parser.add_argument(
        "--bert_model_path",
        default="bert-base-uncased",
        type=str,
        help="bert model name or path. leave it bank if you are using Glove",
    )

    parser.add_argument(
        "--output_dir",
        default="./output_euclidean/i2b2/",
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

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
        "--embedding_type",
        choices=["glove", "bert"],
        default="glove",
        help="embedding used for sanitization",
    )

    parser.add_argument(
        "--task",
        choices=["i2b2", "SST-2", "QNLI"],
        default="i2b2",
        help="NLP eval tasks",
    )

    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--epsilon", type=float, default=16, help="privacy parameter epsilon"
    )
    parser.add_argument(
        "--s_epsilon", type=float, default=8, help="privacy parameter epsilon"
    )
    parser.add_argument(
        "--p",
        type=float,
        default=0.7,
        help="SanText+: probability of non-sensitive words to be sanitized",
    )

    parser.add_argument("--threads", type=int, default=2, help="number of processors")

    args = parser.parse_args()

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

    args.output_dir = os.path.join(args.output_dir, "eps_%.2f" % args.epsilon)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Building Vocabulary...")

    if args.embedding_type == "glove":
        tokenizer = English()
        tokenizer_type = "word"
    else:
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        tokenizer_type = "subword"
    if args.task == "SST-2":
        vocab = get_vocab_SST2(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    elif args.task == "QNLI":
        vocab = get_vocab_QNLI(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)
    else:
        vocab = get_vocab_text(args.data_dir, tokenizer, tokenizer_type=tokenizer_type)

    words = [key for key, _ in vocab.most_common()]
    with open(
        "./selective_output/sensitive_mapping/0.6_i2b2.json", "r", encoding="utf8"
    ) as f:
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
            rest = (
                (len(sensitive_word_embed))
                * (args.epsilon - args.s_epsilon)
                / len(normal_word_embed)
            )
            args.epsilon = args.epsilon + rest
            print(
                f"epsilon: {args.epsilon}, s_epsilon: {args.s_epsilon}, Adjusted epsilon: {args.epsilon}"
            )
            print(
                f"normal_word_embed: {len(normal_word_embed)}, sensitive_word_embed: {len(sensitive_word_embed)}"
            )
        n_prob_matrix = cal_probability(
            normal_word_embed, all_word_embed, "normal", args.epsilon, args.s_epsilon
        )
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
        n_prob_matrix = cal_probability(
            all_word_embed, normal_word_embed, "normal", args.epsilon, args.s_epsilon
        )

    threads = min(args.threads, cpu_count())
    total_epsilon = 0
    for file_name in ["train.csv"]:
        data_file = os.path.join(args.data_dir, file_name)
        out_file = open(os.path.join(args.output_dir, file_name), "w", encoding="utf-8")
        logger.info(
            "Processing file: %s. Will write to: %s"
            % (data_file, os.path.join(args.output_dir, file_name))
        )
        if args.task in ["SST-2", "QNLI"]:
            num_lines = sum(1 for _ in open(data_file))
            with open(data_file, "r", encoding="utf-8") as rf:
                # header
                header = next(rf)
                out_file.write(header)
                labels = []
                docs = []
                if args.task == "SST-2":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text = content[0]
                        label = int(content[1])
                        if args.embedding_type == "glove":
                            doc = [token.text for token in tokenizer(text)]
                        else:
                            doc = tokenizer.tokenize(text)
                        docs.append(doc)
                        labels.append(label)
                elif args.task == "QNLI":
                    for line in tqdm(rf, total=num_lines - 1):
                        content = line.strip().split("\t")
                        text1 = content[1]
                        text2 = content[2]
                        label = content[-1]
                        if args.embedding_type == "glove":
                            doc1 = [token.text for token in tokenizer(text1)]
                            doc2 = [token.text for token in tokenizer(text2)]
                        else:
                            doc1 = tokenizer.tokenize(text1)
                            doc2 = tokenizer.tokenize(text2)

                        docs.append(doc1)
                        docs.append(doc2)
                        labels.append(label)

                rf.close()
        else:
            docs = []
            df = pd.read_csv(data_file)
            texts = df["text"].dropna()

            for text in tqdm(texts, total=len(texts)):
                text = text.strip().replace("\n", "").replace("\t", "")
                text = re.sub(" +", " ", text)
                if args.embedding_type == "glove":
                    doc = [token.text for token in tokenizer(text)]
                else:
                    doc = tokenizer.tokenize(text)

                docs.append(doc)

        alg = NADPTextSan if args.method == "normal" else NADPTextSan_plus
        print("using method: ", alg)
        with Pool(
            threads,
            initializer=NADPTextSan_init,
            initargs=(
                args.epsilon,
                args.s_epsilon,
                words,
                sensitive_words,
                args.p,
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
            results = list(
                tqdm(
                    p.imap(annotate_, docs, chunksize=32),
                    total=len(docs),
                    desc="Sanitize docs using NandpTextSan",
                )
            )
            p.close()

        logger.info("Saving ...")
        debug_flag = True
        if args.task == "SST-2":
            for i, (predicted_text, epsilon) in enumerate(results):
                if debug_flag:
                    print(predicted_text, epsilon)
                    print(results[i])
                    debug_flag = False
                total_epsilon = max(total_epsilon, epsilon)
                write_content = predicted_text + "\t" + str(labels[i]) + "\n"
                out_file.write(write_content)
        elif args.task == "QNLI":
            assert len(results) / 2 == len(labels)
            for i in range(len(labels)):
                predicted_text1 = results[i * 2][0]
                predicted_text2 = results[i * 2 + 1][0]
                total_epsilon = max(
                    total_epsilon, results[i * 2][1], results[i * 2 + 1][1]
                )
                write_content = (
                    str(i)
                    + "\t"
                    + predicted_text1
                    + "\t"
                    + predicted_text2
                    + "\t"
                    + str(labels[i])
                    + "\n"
                )
                out_file.write(write_content)
        else:
            for i, (predicted_text, epsilon) in enumerate(results):
                if debug_flag:
                    print(predicted_text, epsilon)
                    print(results[i])
                    debug_flag = False
                total_epsilon = max(total_epsilon, epsilon)
                write_content = predicted_text + "\n"
                out_file.write(write_content)

        out_file.close()
        logger.info("Total Epsilon: %s" % str(total_epsilon))
    with open(os.path.join(args.output_dir, "total_epsilon.txt"), "w") as f:
        f.write(str(total_epsilon))


if __name__ == "__main__":
    main()
