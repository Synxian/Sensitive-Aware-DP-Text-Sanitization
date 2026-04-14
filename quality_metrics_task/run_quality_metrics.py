import argparse
import json
import os
import re

import torch
from run_bert_score import compute_bert_score
from run_jaccard import compute_jaccard_similarity
from run_movers_distance import compute_movers_distance
from run_mauve import compute_mauve

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_folder_data(folder, limit):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                data.append(file_data)
        if limit and len(data) >= limit:
            break
    return data[:limit] if limit else data

def calculate_metrics(data, lang):
    original = []
    sanitized = []
    results = []
    for item in data:
        original.append(item['original_text'])
        sanitized.append(item['sanitized_text'])
    print('jaccard init')
    jaccard_results = [compute_jaccard_similarity(o, s) for o, s in zip(original, sanitized)]
    jaccard_results = sum(jaccard_results)/len(jaccard_results)
    print('bert_score init')
    bert_score = compute_bert_score(original, sanitized, device=device, lang=lang)
    print('movers_distance init')
    movers_distance = compute_movers_distance(original, sanitized, device=device)
    print('mauve_score init')
    mauve_score = compute_mauve(original, sanitized, device=device)
    print('done')

    results.append({
        'jaccard': jaccard_results,
        'bert_score': bert_score,
        'movers_distance': movers_distance,
        'mauve_score': mauve_score
    })
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_folder", type=str)
    parser.add_argument("--lang", type=str, default='en')
    args = parser.parse_args()
    print(args.dataset_folder)
    for execution_folder in os.listdir(args.dataset_folder):
        data = load_folder_data(os.path.join(args.dataset_folder, execution_folder), limit=None)
        metrics = calculate_metrics(data, lang=args.lang)
        os.makedirs(os.path.join('quality_metrics', args.dataset_folder), exist_ok=True)
        with open(os.path.join('quality_metrics', args.dataset_folder, f"{execution_folder}.json"), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)