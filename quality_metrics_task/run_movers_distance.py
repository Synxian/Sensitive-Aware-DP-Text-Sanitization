import numpy as np
from sentence_transformers import SentenceTransformer


def compute_movers_distance(originales, sanitizados, device):

    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    emb_orig = model.encode(
        originales,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    emb_san = model.encode(
        sanitizados,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    dists = np.linalg.norm(emb_orig - emb_san, axis=1)

    return float(np.mean(dists))
