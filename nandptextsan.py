import os
import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances
from pydantic_models.satsdp import SastdpDocument, SastdpExecutionArgs
import json


def cal_probability(word_embed_1, word_embed_2, epsilon_type="normal", epsilon=None, s_epsilon=None):
    eps = epsilon if epsilon_type == "normal" else s_epsilon
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(eps * sim_matrix / 2, axis=1)
    return prob_matrix


def NADPTextSan_init(
    args: SastdpExecutionArgs,
    vocab_init,
    sensitive_words_init,
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
    global replacements_output_dir
    replacements_output_dir = args.replacements_output_dir
    epsilon = args.epsilon
    s_epsilon = args.s_epsilon
    vocab = vocab_init
    sensitive_words = sensitive_words_init
    word2id = word2id_init
    sword2id = sword2id_init
    nword2id = nword2id_init
    p = args.p
    s_prob_matrix = s_prob_matrix_init
    n_prob_matrix = n_prob_matrix_init
    global id2word
    id2word = {v: k for k, v in word2id.items()}
    global id2sword
    id2sword = {v: k for k, v in sword2id.items()}
    global id2nword
    id2nword = {v: k for k, v in nword2id.items()}


def NADPTextSan(doc: SastdpDocument):
    replacements = {"original_text": " ".join(doc.text)}
    new_doc = []
    total_epsilon = 0
    for word in doc.text:
        if word in word2id:
            # In-vocab
            if word in sword2id:
                index = sword2id[word]
                sampling_prob = s_prob_matrix[index]
                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                total_epsilon += s_epsilon
                new_doc.append(id2word[sampling_index[0]])
            else:
                index = nword2id[word]
                sampling_prob = n_prob_matrix[index]
                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                total_epsilon += epsilon
                new_doc.append(id2word[sampling_index[0]])
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
    new_doc = " ".join(new_doc)
    replacements["sanitized_text"] = new_doc
    replacements["total_epsilon"] = total_epsilon
    replacements["text_id"] = doc.text_id
    write_replacements_file(replacements, replacements_output_dir)
    return (new_doc, total_epsilon)


def NADPTextSan_plus(doc: SastdpDocument):
    replacements = {"original_text": " ".join(doc.text)}
    new_doc = []
    total_epsilon = 0
    for word in doc.text:
        if word in word2id:
            flip_p = np.random.random()
            # In-vocab
            if word in sword2id:
                index = sword2id[word]
                if flip_p <= p:
                    sampling_prob = s_prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    total_epsilon += s_epsilon
                    new_doc.append(id2sword[sampling_index[0]])

                else:
                    sampling_prob = n_prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    total_epsilon += epsilon
                    new_doc.append(id2nword[sampling_index[0]])

            else:
                index = nword2id[word]
                if flip_p <= p:
                    sampling_prob = n_prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    total_epsilon += epsilon
                    new_doc.append(id2nword[sampling_index[0]])

                else:
                    sampling_prob = s_prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    total_epsilon += s_epsilon
                    new_doc.append(id2sword[sampling_index[0]])

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
    new_doc = " ".join(new_doc)
    replacements["sanitized_text"] = new_doc
    replacements["total_epsilon"] = total_epsilon
    replacements["text_id"] = doc.text_id
    write_replacements_file(replacements, replacements_output_dir)
    return (new_doc, total_epsilon)


def write_replacements_file(replacements, replacements_output_dir):
    file_name = f"{replacements['text_id']}.json"
    file_path = os.path.join(replacements_output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(replacements, f, ensure_ascii=False, indent=4)
