from bert_score import score
from transformers import AutoTokenizer
import torch
from typing import List, Dict
from tqdm import tqdm


def compute_bert_score(
    original: List[str],
    sanitized: List[str],
    device: str,
    lang: str
) -> Dict[str, float]:

    MODEL_BY_LANG = {
        "en": "roberta-large",
        "es": "xlm-roberta-large",
        "multi": "xlm-roberta-large",
    }

    model_name = MODEL_BY_LANG.get(lang, "roberta-large")
    batch_size = 16

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def chunk_text(text: str, max_words: int = 300) -> List[str]:
        words = text.split()
        return [
            " ".join(words[i:i + max_words])
            for i in range(0, len(words), max_words)
        ]

    all_o_chunks = []
    all_s_chunks = []
    all_weights = []

    for orig_doc, san_doc in zip(original, sanitized):
        orig_chunks = chunk_text(orig_doc, max_words=300)
        san_chunks = chunk_text(san_doc, max_words=300)

        n = min(len(orig_chunks), len(san_chunks))
        orig_chunks = orig_chunks[:n]
        san_chunks = san_chunks[:n]

        all_o_chunks.extend(orig_chunks)
        all_s_chunks.extend(san_chunks)
        all_weights.extend([len(t.split()) for t in orig_chunks])

    with torch.no_grad():
        P, R, F1 = score(
            all_s_chunks,
            all_o_chunks,
            model_type=model_name,
            device=device,
            batch_size=64,
            verbose=True
        )

    w = torch.tensor(all_weights, dtype=torch.float)
    # score() returns 1D tensors, w is also a 1D tensor
    precision = torch.sum(P * w) / torch.sum(w)
    recall = torch.sum(R * w) / torch.sum(w)
    f1 = torch.sum(F1 * w) / torch.sum(w)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }
