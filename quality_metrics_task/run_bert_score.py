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

    all_precisions = []
    all_recalls = []
    all_f1s = []
    all_weights = []

    for orig_doc, san_doc in tqdm(
        zip(original, sanitized),
        total=len(original),
        desc="BERTScore",
    ):
        orig_chunks = chunk_text(orig_doc, max_words=300)
        san_chunks = chunk_text(san_doc, max_words=300)

        n = min(len(orig_chunks), len(san_chunks))
        orig_chunks = orig_chunks[:n]
        san_chunks = san_chunks[:n]

        for i in range(0, n, batch_size):
            o_batch = orig_chunks[i:i + batch_size]
            s_batch = san_chunks[i:i + batch_size]

            with torch.no_grad():
                P, R, F1 = score(
                    s_batch,
                    o_batch,
                    model_type=model_name,
                    device=device,
                    verbose=False
                )

            weights = [len(t.split()) for t in o_batch]

            all_precisions.extend(P.tolist())
            all_recalls.extend(R.tolist())
            all_f1s.extend(F1.tolist())
            all_weights.extend(weights)

    w = torch.tensor(all_weights, dtype=torch.float)
    precision = torch.sum(torch.tensor(all_precisions) * w) / torch.sum(w)
    recall = torch.sum(torch.tensor(all_recalls) * w) / torch.sum(w)
    f1 = torch.sum(torch.tensor(all_f1s) * w) / torch.sum(w)

    return {
        "precision": precision.item(),
        "recall": recall.item(),
        "f1": f1.item()
    }
