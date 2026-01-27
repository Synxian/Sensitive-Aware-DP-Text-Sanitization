from flair.models import SequenceTagger
from flair.data import Sentence
from flair import device
import numpy as np
import tqdm

SENSITIVE_TAGS = ["PER", "LOC", "ORG"]

def load_embeddings(embeddings_file_path):
    embeddings = {}
    with open(embeddings_file_path, 'r', encoding='utf8') as f:
        for line in tqdm.tqdm(f):
            parts = line.strip().split()
            word = parts[0]
            try:
                vector = np.array(parts[1:], dtype=np.float32)
                embeddings[word] = vector
            except Exception as e:
                print(f"Error loading embeddings for line: {line}")
    return embeddings

def save_embeddings(word_vecs, output_file):
    with open(output_file, "w", encoding="utf8") as f:
        for word, vec in word_vecs.items():
            vec_str = " ".join(map(str, vec.tolist()))
            f.write(f"{word} {vec_str}\n")

def split_embeddings_with_flair(model_path, embeddings_file_path, sensitive_file_destination, normal_file_destination, threshold=0.15):
    print("Loading Flair NER model")
    tagger = SequenceTagger.load(model_path)

    print("Loading GloVe embeddings")
    glove_embeddings = load_embeddings(embeddings_file_path)

    sensitive_embeddings = {}
    normal_embeddings = {}

    print("Splitting embeddings")
    BATCH_SIZE = 128
    words = list(glove_embeddings.keys())
    for i in tqdm.trange(0, len(words), BATCH_SIZE):
        batch_words = words[i:i+BATCH_SIZE]
        sentences = [Sentence(w) for w in batch_words]
        tagger.predict(sentences, return_probabilities_for_all_classes=True)
        
        for word, sentence in zip(batch_words, sentences):
            for token in sentence:
                if any(
                    ner_tag.score > threshold 
                    and any(tag in ner_tag.value for tag in SENSITIVE_TAGS)
                    for ner_tag in token.tags_proba_dist["ner"]
                ):
                    sensitive_embeddings[word] = glove_embeddings[word]
                    break
            else:
                normal_embeddings[word] = glove_embeddings[word]

    save_embeddings(sensitive_embeddings, sensitive_file_destination)
    save_embeddings(normal_embeddings, normal_file_destination)

if __name__ == "__main__":
    glove_path = "embeddings/english/glove.840B.300d.txt" 
    sensitive_out = "embeddings/english/glove/sensitive_embeddings.txt"
    normal_out = "embeddings/english/glove/normal_embeddings.txt"

    split_embeddings_with_flair('flair/ner-english-large', glove_path, sensitive_out, normal_out)

    print(f"Saved sensitive embeddings to {sensitive_out}")
    print(f"Saved normal embeddings to {normal_out}")