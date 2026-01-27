# requirements:
# pip install torch transformers mauve-score tqdm pandas

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import mauve
import pandas as pd
from tqdm import tqdm
import math

# -----------------------------
# Load TSV files
# -----------------------------
ref_file = "datasets/SST-2/train.tsv"
gen_file = "santext-sst2/eps_1.00/train.tsv"

# Assume the first column contains the texts
references = pd.read_csv(ref_file, sep="\t", header=None)[0].tolist()
generated = pd.read_csv(gen_file, sep="\t", header=None)[0].tolist()

# -----------------------------
# Load a pretrained model for perplexity
# -----------------------------
model_name = "gpt2"  # you can change to another LM
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.eval()

# -----------------------------
# Compute Perplexity
# -----------------------------
def compute_perplexity(texts, model, tokenizer, device):
    perplexities = []
    for text in tqdm(texts, desc="Computing perplexity"):
        encodings = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            loss = outputs.loss
        perplexity = math.exp(loss.item())
        perplexities.append(perplexity)
    return perplexities

perplexities = compute_perplexity(generated, model, tokenizer, device)

# Save perplexities to TSV
pd.DataFrame(perplexities).to_csv("generated_perplexities.tsv", sep="\t", index=False)

# -----------------------------
# Compute MAUVE
# -----------------------------
mauve_result = mauve.compute_mauve(
    p_text=references,
    q_text=generated,
    device_id=0 if device=="cuda" else -1,
    max_text_length=512
)

print(f"MAUVE score: {mauve_result.mauve:.4f}")
