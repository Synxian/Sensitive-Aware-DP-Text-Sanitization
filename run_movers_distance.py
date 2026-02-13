from sentence_transformers import SentenceTransformer
from scipy.stats import wasserstein_distance
import numpy as np


def compute_movers_distance(originales, sanitizados, device):
    print("\n--- Calculando EMD ---")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

    emb_original = model.encode(originales, show_progress_bar=True)
    emb_sanitizado = model.encode(sanitizados, show_progress_bar=True)


    dists = []
    for i in range(emb_original.shape[1]):
        w_d = wasserstein_distance(emb_original[:, i], emb_sanitizado[:, i])
        dists.append(w_d)

    return np.mean(dists)
