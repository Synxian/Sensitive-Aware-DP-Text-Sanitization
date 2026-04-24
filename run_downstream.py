"""Fine-tune BERT on sanitized GLUE data (SST-2 / QNLI).

Usage
-----
    python run_downstream.py \\
        --task sst2 \\
        --data_dir  ./output/sst2/normal/eps_10.00_seps_5.00 \\
        --output_dir ./results/sst2/normal/eps_10.00_seps_5.00
"""
import argparse
import json
import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

TASK_META = {
    "sst2": {"num_labels": 2},
    "qnli": {"num_labels": 2},
}
QNLI_LABEL_MAP = {"entailment": 0, "not_entailment": 1}


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GlueTSVDataset(Dataset):
    def __init__(self, path, task, tokenizer, max_length=128, max_samples=None):
        self.tokenizer  = tokenizer
        self.max_length = max_length
        self.examples   = []

        with open(path, encoding="utf-8") as f:
            next(f)  # skip header
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                parts = line.strip().split("\t")
                if task == "sst2":
                    if len(parts) < 2:
                        continue
                    self.examples.append({"a": parts[0], "b": None,
                                          "label": int(parts[1])})
                elif task == "qnli":
                    if len(parts) < 4:
                        continue
                    self.examples.append({"a": parts[1], "b": parts[2],
                                          "label": QNLI_LABEL_MAP.get(
                                              parts[3].strip(), 1)})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex  = self.examples[idx]
        enc = self.tokenizer(ex["a"], ex["b"], truncation=True,
                             max_length=self.max_length,
                             padding="max_length", return_tensors="pt")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         torch.tensor(ex["label"], dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _device():
    if torch.backends.mps.is_available():  return torch.device("mps")
    if torch.cuda.is_available():          return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(model, loader, optimizer, device):
    model.train()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss  = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        total += loss.item(); n += 1
    return total / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, truths, total, n = [], [], 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out   = model(**batch)
        total += out.loss.item(); n += 1
        preds.extend(out.logits.argmax(-1).cpu().numpy())
        truths.extend(batch["labels"].cpu().numpy())
    acc = float(np.mean(np.array(preds) == np.array(truths)))
    return {"eval_accuracy": acc, "eval_loss": total / max(n, 1)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",       required=True, choices=["sst2", "qnli"])
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--num_epochs", type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--lr",         type=float, default=2e-5)
    p.add_argument("--max_length", type=int,   default=128)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--max_eval_samples",  type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    logging.basicConfig(format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    device    = _device()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    config    = AutoConfig.from_pretrained(
        args.model_name, num_labels=TASK_META[args.task]["num_labels"],
        local_files_only=True)
    model     = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config, local_files_only=True).to(device)

    train_ds = GlueTSVDataset(os.path.join(args.data_dir, "train.tsv"),
                              args.task, tokenizer, args.max_length,
                              args.max_train_samples)
    eval_ds  = GlueTSVDataset(os.path.join(args.data_dir, "dev.tsv"),
                              args.task, tokenizer, args.max_length,
                              args.max_eval_samples)
    logger.info("Train: %d  Eval: %d  Device: %s", len(train_ds), len(eval_ds), device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    eval_loader  = DataLoader(eval_ds,  batch_size=args.batch_size)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_acc, best = 0.0, {}
    for ep in range(1, args.num_epochs + 1):
        t_loss  = train_epoch(model, train_loader, optimizer, device)
        metrics = evaluate(model, eval_loader, device)
        logger.info("Epoch %d/%d  train_loss=%.4f  eval_acc=%.4f",
                    ep, args.num_epochs, t_loss, metrics["eval_accuracy"])
        if metrics["eval_accuracy"] > best_acc:
            best_acc = metrics["eval_accuracy"]
            best     = {**metrics, "epoch": ep, "train_loss": t_loss}
            model.save_pretrained(os.path.join(args.output_dir, "best_model"))
            tokenizer.save_pretrained(os.path.join(args.output_dir, "best_model"))

    out_path = os.path.join(args.output_dir, "eval_results.json")
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    logger.info("Best acc=%.4f → %s", best_acc, out_path)


if __name__ == "__main__":
    main()
