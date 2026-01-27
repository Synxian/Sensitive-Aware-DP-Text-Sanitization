import os
import random
import numpy as np
from tqdm import tqdm
from scipy.special import softmax
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)
from functools import partial
from multiprocessing import Pool, cpu_count


def cal_probability(word_embed_1, word_embed_2, epsilon_type="normal"):
    if epsilon_type == "normal":
        epsilon = epsilon
    else:
        epsilon = s_epsilon
    distance = cosine_similarity(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(epsilon * sim_matrix / 2, axis=1)
    return prob_matrix


def NADPTextSan_init(
    epsilon_init,
    s_epsilon_init,
    vocab_init,
    sensitive_words_init,
    p_init,
    word2id_init,
    sword2id_init,
    nword2id_init,
    s_prob_matrix_init,
    n_prob_matrix_init,
):
    global epsilon
    global s_epsilon
    global vocab
    global sensitive_words
    global word2id
    global sword2id
    global nword2id
    global p
    global s_prob_matrix
    global n_prob_matrix
    epsilon = epsilon_init
    s_epsilon = s_epsilon_init if s_epsilon_init is not None else epsilon_init
    vocab = vocab_init
    sensitive_words = sensitive_words_init
    word2id = word2id_init
    sword2id = sword2id_init
    nword2id = nword2id_init
    p = p_init
    s_prob_matrix = s_prob_matrix_init
    n_prob_matrix = n_prob_matrix_init
    global id2word
    id2word = {v: k for k, v in word2id.items()}
    global id2sword
    id2sword = {v: k for k, v in sword2id.items()}
    global id2nword
    id2nword = {v: k for k, v in nword2id.items()}


def NADPTextSan(doc):
    replacements = {}
    new_doc = []
    total_epsilon = 0
    for word in doc:
        if word in word2id:
            # In-vocab
            if word in sword2id:
                index = sword2id[word]
                sampling_prob = s_prob_matrix[index]
                sampling_index = np.random.choice(
                    len(sampling_prob), 1, p=sampling_prob
                )
                total_epsilon += s_epsilon
                new_doc.append(id2word[sampling_index[0]])
                replacements[word] = id2word[sampling_index[0]]
            else:
                index = nword2id[word]
                sampling_prob = n_prob_matrix[index]
                sampling_index = np.random.choice(
                    len(sampling_prob), 1, p=sampling_prob
                )
                total_epsilon += epsilon
                new_doc.append(id2word[sampling_index[0]])
                replacements[word] = id2word[sampling_index[0]]
        else:
            # Out-of-Vocab words
            sampling_prob = (
                1
                / len(vocab)
                * np.ones(
                    len(vocab),
                )
            )
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
            new_doc.append(vocab[sampling_index[0]])
            replacements[word] = vocab[sampling_index[0]]
    new_doc = " ".join(new_doc)
    write_replacements_file(replacements)
    return (new_doc, total_epsilon)


def NADPTextSan_plus(doc):
    replacements = {}
    new_doc = []
    total_epsilon = 0
    for word in doc:
        if word in word2id:
            flip_p = np.random.random()
            # In-vocab
            if word in sword2id:
                index = sword2id[word]
                if flip_p <= p:
                    sampling_prob = s_prob_matrix[index]
                    sampling_index = np.random.choice(
                        len(sampling_prob), 1, p=sampling_prob
                    )
                    total_epsilon += s_epsilon
                    new_doc.append(id2sword[sampling_index[0]])
                    replacements[word] = id2sword[sampling_index[0]]
                else:
                    sampling_prob = n_prob_matrix[index]
                    sampling_index = np.random.choice(
                        len(sampling_prob), 1, p=sampling_prob
                    )
                    total_epsilon += epsilon
                    new_doc.append(id2nword[sampling_index[0]])
                    replacements[word] = id2nword[sampling_index[0]]
            else:
                index = nword2id[word]
                if flip_p <= p:
                    sampling_prob = n_prob_matrix[index]
                    sampling_index = np.random.choice(
                        len(sampling_prob), 1, p=sampling_prob
                    )
                    total_epsilon += epsilon
                    new_doc.append(id2nword[sampling_index[0]])
                    replacements[word] = id2nword[sampling_index[0]]
                else:
                    sampling_prob = s_prob_matrix[index]
                    sampling_index = np.random.choice(
                        len(sampling_prob), 1, p=sampling_prob
                    )
                    total_epsilon += s_epsilon
                    new_doc.append(id2sword[sampling_index[0]])
                    replacements[word] = id2sword[sampling_index[0]]
        else:
            # Out-of-Vocab words
            sampling_prob = (
                1
                / len(vocab)
                * np.ones(
                    len(vocab),
                )
            )
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
            new_doc.append(vocab[sampling_index[0]])
            replacements[word] = vocab[sampling_index[0]]
    new_doc = " ".join(new_doc)
    write_replacements_file(replacements)
    return (new_doc, total_epsilon)


def write_replacements_file(replacements):
    folder_name = "replacements"
    os.makedirs(folder_name, exist_ok=True)
    current_files = os.listdir(folder_name)
    if not current_files:
        file_index = 0
    else:
        file_index = max([int(file.split(".")[0]) for file in current_files]) + 1
    file_name = f"{file_index}.txt"
    file_path = os.path.join(folder_name, file_name)
    with open(file_path, "w") as f:
        for word, replacement in replacements.items():
            f.write(f"{word}\t{replacement}\n")
