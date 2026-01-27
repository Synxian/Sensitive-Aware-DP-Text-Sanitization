import os
import json
import pandas as pd
import torch
from tqdm import tqdm
from flair.data import Sentence
from flair.models import SequenceTagger
import flair

SENSITIVE_TAGS = ['PER', 'LOC', 'ORG']

def build_flair_mapping(model_path, dataset, dataset_path, threshold, text_col, out_dir):
    save_path = os.path.join(
        out_dir,
        'sensitive_mapping',
    )
    file_name = f'{threshold}_{dataset}.json'
    file_path = os.path.join(save_path, file_name)
    if os.path.isfile(file_path):
        print('loading existing file')
        with open(file_path, 'r', encoding='utf8') as json_file:
            data = json.load(json_file)
        return data
    print(f'loading dataset from {dataset_path}')
    df = pd.read_csv(dataset_path, sep='\t')

    flair.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    flair.logger.show_progress_bars = True
    print(f'loading model from {model_path}')
    tagger = SequenceTagger.load(model_path)
    mapping ={}

    print(f'filtering tags')
    def filter_tags(token):
        return token.score > threshold and any(tag in token.value for tag in SENSITIVE_TAGS)

    for sample in tqdm(df[text_col], desc='Tagging dataset', total=len(df[text_col])):
        sentence = Sentence(sample.lower())
        tagger.predict(sentence, return_probabilities_for_all_classes=True)
        for word in sentence:
            candidates = list(filter(filter_tags, word.tags_proba_dist['ner']))
            if len(candidates) > 0:
                if mapping.get(word.text):
                    continue
                else:
                    mapping[word.text] = [[c.value, float(c.score)] for c in candidates]
    try:
        print(f'saving json to {save_path}')
        os.makedirs(save_path, exist_ok=True)
        with open(file_path, 'w', encoding='utf8') as fp:
            json.dump(mapping, fp)
        print(f'saved json to {save_path}')
    except Exception as e:
        print(e)
    return mapping


if __name__ == "__main__":
    dataset = "sst2" 
    sensitive_out = "embeddings/english/glove/sensitive_embeddings.txt"
    normal_out = "embeddings/english/glove/normal_embeddings.txt"
    threshold = 0.3
    dataset_path = "datasets/SST-2/train_dev_combined.tsv"
    text_col = "sentence"
    out_dir = "selective_output"

    build_flair_mapping('flair/ner-english-large', dataset, dataset_path, threshold, text_col, out_dir)

    print(f"Saved sensitive embeddings to {sensitive_out}")
    print(f"Saved normal embeddings to {normal_out}")

