import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
import flair
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

SENSITIVE_TAGS = ["PER", "LOC", "ORG"]


def build_flair_mapping(model_path, dataset, dataset_path, threshold, text_col, out_dir):
    save_path = os.path.join(
        out_dir,
        "sensitive_mapping",
    )
    file_name = f"flair_{threshold}_{dataset}.json"
    file_path = os.path.join(save_path, file_name)
    if os.path.isfile(file_path):
        print("loading existing file")
        with open(file_path, "r", encoding="utf8") as json_file:
            data = json.load(json_file)
        return data
    print(f"loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    flair.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    flair.logger.show_progress_bars = True
    print(f"loading model from {model_path}")
    tagger = SequenceTagger.load(model_path)
    mapping = {}

    def filter_tags(token):
        return token.score > threshold and any(tag in token.value for tag in SENSITIVE_TAGS)

    for sample in tqdm(df[text_col], desc="Tagging dataset", total=len(df[text_col])):
        sentence = Sentence(sample.lower())
        tagger.predict(sentence, return_probabilities_for_all_classes=True)
        for word in sentence:
            candidates = list(filter(filter_tags, word.tags_proba_dist["ner"]))
            if len(candidates) > 0:
                if mapping.get(word.text):
                    continue
                else:
                    mapping[word.text] = [[c.value, float(c.score)] for c in candidates]
    try:
        print(f"saving json to {save_path}")
        os.makedirs(save_path, exist_ok=True)
        with open(file_path, "w", encoding="utf8") as fp:
            json.dump(mapping, fp)
        print(f"saved json to {save_path}")
    except Exception as e:
        print(e)
    return mapping


def build_hf_mapping(model_path, dataset, dataset_path, threshold, text_col, out_dir):
    save_path = os.path.join(out_dir, "sensitive_mapping")
    file_name = f"hf_{threshold}_{dataset}.json"
    file_path = os.path.join(save_path, file_name)

    if os.path.isfile(file_path):
        print("Loading existing file")
        with open(file_path, "r", encoding="utf8") as json_file:
            data = json.load(json_file)
        return data

    print(f"Loading dataset from {dataset_path}")
    df = pd.read_csv(dataset_path)

    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, token="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")
    model = AutoModelForTokenClassification.from_pretrained(model_path, token="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")

    if torch.cuda.is_available():
        model = model.to("cuda")

    SENSITIVE_TAGS = [
        "SEX",
        "GENDER",
        "MIDDLENAME",
        "USERNAME",
        "CITY",
        "FIRSTNAME",
        "LASTNAME",
        "COMPANYNAME",
        "EMAIL",
        "ADDRESS",
    ]
    mapping = {}
    CHUNK_SIZE = 400
    for text_row in tqdm(df[text_col], desc="Tagging dataset", total=len(df[text_col])):
        chunks = len(text_row) // CHUNK_SIZE + 1
        for i in range(chunks):
            sample = text_row[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE]
            if pd.isna(sample):
                continue

            inputs = tokenizer(str(sample).lower(), return_tensors="pt")

            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)

            probs = torch.nn.functional.softmax(outputs.logits[0], dim=-1).cpu()
            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

            for token, token_probs in zip(tokens, probs):
                if token in ["[CLS]", "[SEP]", "[PAD]"]:
                    continue

                word = token.replace("##", "").strip()
                if not word:
                    continue

                for label_id, score in enumerate(token_probs):
                    label = model.config.id2label[label_id]
                    if score > threshold and any(tag in label for tag in SENSITIVE_TAGS):
                        if word not in mapping:
                            mapping[word] = []
                        mapping[word].append([label, float(score)])
    try:
        os.makedirs(save_path, exist_ok=True)
        with open(file_path, "w", encoding="utf8") as fp:
            json.dump(mapping, fp)
        print(f"Saved json to {file_path}")
    except Exception as e:
        print(e)

    return mapping


if __name__ == "__main__":
    dataset = "mimic"
    threshold = 0.6
    dataset_path = "datasets/mimic/train.csv"
    text_col = "text"
    out_dir = "selective_output"

    build_flair_mapping("flair/ner-english-large", dataset, dataset_path, threshold, text_col, out_dir)
    # ab_ai = "ab-ai/pii_model"

    # build_hf_mapping(
    #    "tabularisai/eu-pii-safeguard", dataset, dataset_path, threshold, text_col, out_dir
    # )
