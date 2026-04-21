"""Sensitivity-aware DP text sanitization via the exponential mechanism.

Implements three word-level sanitization methods:

- **SanText** (baseline): every in-vocab word is replaced using the same
  privacy budget epsilon over the full embedding space.
- **NADPTextSan** ("Ours" / method=normal): sensitive words sample from the
  sensitive embedding subspace with budget epsilon_s; normal words sample from
  the normal subspace with budget epsilon_n.
- **NADPTextSan_plus** ("Ours+" / method=plus): mixed sampling. With
  probability p the word stays within its own class (sensitive -> sensitive,
  normal -> normal); with probability 1-p it crosses to the other class.

All three functions are designed to run inside a multiprocessing.Pool.
Global state is initialized once per worker via NADPTextSan_init().

Key data structures (set by NADPTextSan_init):
    - s_prob_matrix: |V_s| x |V_all| (or |V_all| x |V_s| for plus).
      Row i holds the sampling distribution for the i-th sensitive word.
    - n_prob_matrix: |V_n| x |V_all| (or |V_all| x |V_n| for plus).
      Row i holds the sampling distribution for the i-th normal word.
    - word2id / sword2id / nword2id: word -> row index in all / sensitive /
      normal embedding matrices respectively.
    - id2word / id2sword / id2nword: reverse mappings (row index -> word).

NOTE: epsilon redistribution per document is NOT yet implemented. Currently
epsilon_n is fixed corpus-wide. See the discussion about redistributing
unused sensitive budget: epsilon_n[i] = (epsilon_t[i] - ns*epsilon_s) / nn.
"""

import os
import numpy as np
from scipy.special import softmax
from sklearn.metrics.pairwise import euclidean_distances
from pydantic_models.satsdp import SastdpDocument, SastdpInitArgs, SastdpDocumentStatistics
import json


def cal_probability(word_embed_1, word_embed_2, epsilon_type="normal", epsilon=None, s_epsilon=None):
    """Build the exponential mechanism probability matrix.

    For each word w_i in word_embed_1, computes a distribution over all words
    in word_embed_2 using: P(w_i -> w_j) = softmax(eps/2 * (-d(w_i, w_j)))

    Args:
        word_embed_1: source embeddings, shape (n_source, dim). Each row is a
            word that will be sanitized.
        word_embed_2: candidate embeddings, shape (n_candidates, dim). The
            replacement vocabulary.
        epsilon_type: "normal" selects epsilon, "sensitive" selects s_epsilon.
        epsilon: privacy budget for normal words.
        s_epsilon: privacy budget for sensitive words.

    Returns:
        prob_matrix: shape (n_source, n_candidates). Row i is the sampling
        distribution for replacing word i.
    """
    eps = epsilon if epsilon_type == "normal" else s_epsilon
    distance = euclidean_distances(word_embed_1, word_embed_2)
    sim_matrix = -distance
    prob_matrix = softmax(eps * sim_matrix / 2, axis=1)
    return prob_matrix


def NADPTextSan_init(init_args: dict):
    """Multiprocessing worker initializer. Unpacks shared state into globals.

    Called once per worker process via Pool(initializer=NADPTextSan_init).
    The init_args dict is serialized by pydantic so it survives pickling
    across process boundaries.
    """
    init_args = SastdpInitArgs(**init_args)
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
    global id2word
    global id2sword
    global id2nword

    replacements_output_dir = init_args.args.replacements_output_dir
    # adjusted_epsilon is set by the orchestrator if budget redistribution is active
    epsilon = init_args.args.adjusted_epsilon if init_args.args.adjusted_epsilon is not None else init_args.args.epsilon
    s_epsilon = init_args.args.s_epsilon
    vocab = init_args.vocab_init
    sensitive_words = init_args.sensitive_words_init
    word2id = init_args.word2id_init
    sword2id = init_args.sword2id_init
    nword2id = init_args.nword2id_init
    p = init_args.args.p
    s_prob_matrix = init_args.s_prob_matrix_init
    n_prob_matrix = init_args.n_prob_matrix_init
    # Reverse mappings: row index -> word string
    id2word = {v: k for k, v in word2id.items()}
    id2sword = {v: k for k, v in sword2id.items()}
    id2nword = {v: k for k, v in nword2id.items()}




def NADPTextSan(doc: SastdpDocument):
    """Ours (method=normal): sensitivity-aware sanitization without mixing.

    For each word in the document:
    - Sensitive word (in sword2id): sample replacement from s_prob_matrix
      using budget epsilon_s. Replacement drawn from the FULL vocab (id2word).
    - Normal word (in nword2id): sample replacement from n_prob_matrix
      using budget epsilon_n. Replacement drawn from the FULL vocab (id2word).
    - OOV word: uniform random replacement from the full vocabulary list.

    Writes a per-document JSON with original + sanitized text to disk.
    Returns per-document word count statistics.
    """
    replacements = {"original_text": doc.text}
    new_doc = []
    total_epsilon = 0
    sensitive_word_count = 0
    normal_word_count = 0
    out_of_vocab_word_count = 0
    for raw_word in doc.text.split():
        word = raw_word.lower()
        if word in word2id:
            if word in sword2id:
                # Sensitive word: sample from sensitive prob matrix with epsilon_s
                sensitive_word_count += 1
                index = sword2id[word]
                sampling_prob = s_prob_matrix[index]
                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                total_epsilon += s_epsilon
                new_doc.append(id2word[sampling_index[0]])
            else:
                # Normal word: sample from normal prob matrix with epsilon_n
                normal_word_count += 1
                index = nword2id[word]
                sampling_prob = n_prob_matrix[index]
                sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                total_epsilon += epsilon
                new_doc.append(id2word[sampling_index[0]])
        else:
            # OOV: uniform random over entire vocabulary
            out_of_vocab_word_count += 1
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
    return SastdpDocumentStatistics(
        text_id=doc.text_id,
        sensitive_word_count=sensitive_word_count,
        normal_word_count=normal_word_count,
        total_word_count=out_of_vocab_word_count,
    )


def NADPTextSan_plus(doc: SastdpDocument):
    """Ours+ (method=plus): sensitivity-aware sanitization WITH mixing.

    For each word, a coin flip (probability p) decides whether the word
    stays within its own class or crosses to the other class:

    - Sensitive word + flip <= p: sample from sensitive space (epsilon_s)
    - Sensitive word + flip > p:  sample from normal space (epsilon_n)
    - Normal word   + flip <= p: sample from normal space (epsilon_n)
    - Normal word   + flip > p:  sample from sensitive space (epsilon_s)

    The mixing provides plausible deniability: an observer cannot tell
    whether a replacement came from the sensitive or normal subspace.
    This corresponds to Theorem 2 in the paper (mixed-sampling LDP).

    Unlike NADPTextSan, replacements here are drawn from class-specific
    vocabs (id2sword / id2nword) rather than the full vocab (id2word).
    """
    replacements = {"original_text": doc.text}
    new_doc = []
    total_epsilon = 0
    sensitive_word_count = 0
    normal_word_count = 0
    out_of_vocab_word_count = 0
    for raw_word in doc.text.split():
        word = raw_word.lower()
        if word in word2id:
            flip_p = np.random.random()
            index = word2id[word]
            if word in sword2id:
                sensitive_word_count += 1
                if flip_p <= p:
                    # Within-class: sensitive -> sensitive
                    sampling_prob = s_prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    total_epsilon += s_epsilon
                    new_doc.append(id2sword[sampling_index[0]])
                else:
                    # Cross-class: sensitive -> normal
                    sampling_prob = n_prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    total_epsilon += epsilon
                    new_doc.append(id2nword[sampling_index[0]])
            else:
                normal_word_count += 1
                if flip_p <= p:
                    # Within-class: normal -> normal
                    sampling_prob = n_prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    total_epsilon += epsilon
                    new_doc.append(id2nword[sampling_index[0]])
                else:
                    # Cross-class: normal -> sensitive
                    sampling_prob = s_prob_matrix[index]
                    sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
                    total_epsilon += s_epsilon
                    new_doc.append(id2sword[sampling_index[0]])
        else:
            # OOV: uniform random over entire vocabulary
            out_of_vocab_word_count += 1
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
    return SastdpDocumentStatistics(
        text_id=doc.text_id,
        sensitive_word_count=sensitive_word_count,
        normal_word_count=normal_word_count,
        total_word_count=out_of_vocab_word_count,
    )


def SanText(doc: SastdpDocument):
    """Baseline (method=santext): non-sensitivity-aware sanitization.

    Every in-vocab word is replaced using the same epsilon over the full
    embedding space. No distinction between sensitive and normal words.
    OOV words get uniform random replacement.

    The total_epsilon accumulated per document can serve as the budget
    ceiling for Ours/Ours+ when implementing per-document redistribution.
    """
    replacements = {"original_text": doc.text}
    new_doc = []
    total_epsilon = 0
    normal_word_count = 0
    out_of_vocab_word_count = 0
    for raw_token in doc.text.split():
        token = raw_token.lower()
        if token in word2id:
            normal_word_count += 1
            sampling_prob = n_prob_matrix[word2id[token]]
            sampling_index = np.random.choice(len(sampling_prob), 1, p=sampling_prob)
            new_doc.append(id2word[sampling_index[0]])
            total_epsilon += epsilon
        else:
            # OOV: uniform random, no epsilon cost
            out_of_vocab_word_count += 1
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
    return SastdpDocumentStatistics(
        text_id=doc.text_id,
        sensitive_word_count=0,
        normal_word_count=normal_word_count,
        total_word_count=out_of_vocab_word_count,
    )


def write_replacements_file(replacements, replacements_output_dir):
    """Write per-document JSON with original text, sanitized text, and metadata."""
    file_name = f"{replacements['text_id']}.json"
    file_path = os.path.join(replacements_output_dir, file_name)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(replacements, f, ensure_ascii=False, indent=4)
