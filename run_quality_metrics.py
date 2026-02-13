import json
import os
import re

import torch
from run_bert_score import compute_bert_score
from run_jaccard import compute_jaccard_similarity
from run_movers_distance import compute_movers_distance
from run_mauve import compute_mauve

device='cuda' if torch.cuda.is_available() else 'cpu'

def load_data(folder, limit):
    data = []
    for filename in os.listdir(folder):
        if filename.endswith('.json'):
            with open(os.path.join(folder, filename), 'r', encoding='utf-8') as f:
                file_data = json.load(f)
                file_data['new'] =  re.sub(r'(?<=[A-Za-z])\s(?=[A-Za-z])', '', file_data['new']).replace("   ", " ")
                data.append(file_data)
        if limit and len(data) >= limit:
            break
    return data[:limit] if limit else data

def calculate_metrics(data):
    original = []
    sanitized = []
    results = []
    for item in data:
        original.append(item['original'])
        sanitized.append(item['new'])
    jaccard_results = [compute_jaccard_similarity(o, s) for o, s in zip(original, sanitized)]
    bert_score = compute_bert_score(original, sanitized, device=device)
    movers_distance = compute_movers_distance(original, sanitized, device=device)
    mauve_score = compute_mauve(original, sanitized, device=device)

    results.append({
        'jaccard': jaccard_results,
        'bert_score': bert_score,
        'movers_distance': movers_distance,
        'mauve_score': mauve_score
    })
    print(results)
    return results

if __name__ == "__main__":
    folder = "replacements/plus/s_epsilon_16.0"
    limit = 1
    data = load_data(folder, limit)

    calculate_metrics(data)