from transformers import AutoTokenizer
import mauve


def compute_mauve(originales, sanitizados, device):
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    max_tokens = 1024

    def chunk_text(text):
        input_ids = tokenizer(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_tokens,
            return_tensors=None
        )["input_ids"]

        chunks = []
        for i in range(0, len(input_ids), max_tokens):
            chunk_ids = input_ids[i:i + max_tokens]
            chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks

    p_text, q_text = [], []

    for o in originales:
        p_text.extend(chunk_text(o))

    for s in sanitizados:
        q_text.extend(chunk_text(s))

    if len(p_text) < 2 or len(q_text) < 2:
        return None

    out = mauve.compute_mauve(
        p_text=p_text,
        q_text=q_text,
        device_id=0 if device == "cuda" else -1,
        max_text_length=1024,
        verbose=False,
    )

    return out.mauve
