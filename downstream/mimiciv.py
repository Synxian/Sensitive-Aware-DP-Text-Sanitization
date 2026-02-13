"""Train an ICD-10 multi-label classifier on a MIMIC-IV style dataset.

What we need from the data
--------------------------
This script expects TWO inputs:

1) --notes_file: a Feather file containing at least:
   - '_id': (int/str) admission/note id used to join with splits
   - 'text': clinical note text
   - a target column containing a list of ICD-10 codes, by default 'target'

   Typical schemas used in many MIMIC-IV ICD setups:
   - icd10_diag: list[str]
   - icd10_proc: list[str]
   - target: icd10_diag + icd10_proc

2) --splits_file: a Feather file with columns ['_id','split'] where split in
   {'train','val','test'}. This repo already includes:
   - data/mimic/mimiciv_icd10_split.feather

Model
-----
We train a transformer encoder as a text classifier with:
- Input: note text
- Output: multi-hot vector over ICD-10 labels
- Loss: BCEWithLogitsLoss

Metrics
-------
We report micro precision/recall/F1 using a fixed threshold (default 0.5).

NOTE: ICD coding is a large-scale multi-label problem. For real experiments you
will likely need:
- a long-context model (Longformer/BigBird) or truncation strategy
- label filtering (e.g., only top-K codes)
- threshold tuning (a global decision boundary, or per-label)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _ensure_list(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    # sometimes stored as string repr
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("[") and s.endswith("]"):
            try:
                import ast

                v = ast.literal_eval(s)
                return list(v) if isinstance(v, (list, tuple, set)) else [str(v)]
            except Exception:
                return [s]
        return [s]
    return [str(x)]


def load_mimic_dataset(notes_file: str, splits_file: str, text_col: str, target_col: str) -> pd.DataFrame:
    notes = pd.read_feather(notes_file)
    splits = pd.read_feather(splits_file)
    print(f"Loaded notes: {notes.shape}, splits: {splits.shape}")
    print("Example notes rows:")
    print(notes.head())
    print("Example splits rows:")
    print(splits.head())

    if "_id" not in notes.columns:
        raise ValueError(f"notes_file must contain column '_id'. Columns: {notes.columns.tolist()}")
    if text_col not in notes.columns:
        raise ValueError(f"notes_file must contain column '{text_col}'. Columns: {notes.columns.tolist()}")
    if target_col not in notes.columns:
        raise ValueError(f"notes_file must contain column '{target_col}'. Columns: {notes.columns.tolist()}")
    if "_id" not in splits.columns or "split" not in splits.columns:
        raise ValueError(f"splits_file must contain columns ['_id','split']. Columns: {splits.columns.tolist()}")

    notes = notes.copy()
    notes[target_col] = notes[target_col].apply(_ensure_list)

    df = notes.merge(splits[["_id", "split"]], on="_id", how="inner")
    if df.empty:
        raise ValueError(
            "After merging notes with splits, dataframe is empty. "
            "Check that '_id' values match across --notes_file and --splits_file."
        )

    # Basic cleanup
    df[text_col] = df[text_col].astype(str)
    df = df[df[text_col].str.len() > 0]
    df = df[df[target_col].apply(lambda x: len(x) > 0)]

    return df.reset_index(drop=True)


def build_label_space(train_df: pd.DataFrame, target_col: str, min_freq: int = 1, top_k: Optional[int] = None) -> List[str]:
    from collections import Counter

    c = Counter()
    for codes in train_df[target_col].tolist():
        c.update([str(x) for x in codes])

    labels = [code for code, freq in c.items() if freq >= min_freq]
    labels = sorted(labels, key=lambda x: (-c[x], x))

    if top_k is not None:
        labels = labels[: int(top_k)]

    return labels


def multilabel_metrics(logits: np.ndarray, y_true: np.ndarray, threshold: float) -> Dict[str, float]:
    """Compute micro precision/recall/f1 for multi-label predictions."""

    y_pred = (logits >= threshold).astype(np.int32)

    tp = (y_pred * y_true).sum()
    fp = (y_pred * (1 - y_true)).sum()
    fn = ((1 - y_pred) * y_true).sum()

    precision = float(tp / (tp + fp + 1e-10))
    recall = float(tp / (tp + fn + 1e-10))
    f1 = float(2 * precision * recall / (precision + recall + 1e-10))

    exact_match = float((y_pred == y_true).all(axis=1).mean())

    return {
        "micro_precision": precision,
        "micro_recall": recall,
        "micro_f1": f1,
        "exact_match_ratio": exact_match,
    }


@dataclass
class Batch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class ICDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer,
        label2id: Dict[str, int],
        text_col: str,
        target_col: str,
        max_length: int,
    ):
        self.df = df
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.text_col = text_col
        self.target_col = target_col
        self.max_length = max_length
        self.num_labels = len(label2id)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = row[self.text_col]
        codes = [str(x) for x in row[self.target_col]]

        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        y = torch.zeros(self.num_labels, dtype=torch.float)
        for c in codes:
            if c in self.label2id:
                y[self.label2id[c]] = 1.0

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": y,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MIMIC-IV ICD-10 multi-label classifier")

    parser.add_argument("--notes_file", type=str, required=True, help="Feather file with '_id','text', and targets")
    parser.add_argument("--splits_file", type=str, required=True, help="Feather file with '_id' and 'split'")
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--text_col", type=str, default="text")
    parser.add_argument("--target_col", type=str, default="target")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
    )

    parser.add_argument("--max_length", type=int, default=512)

    parser.add_argument("--min_label_freq", type=int, default=1, help="Drop codes that appear < min_label_freq in train")
    parser.add_argument("--top_k_labels", type=int, default=0, help="If >0, keep only top-K most frequent codes")

    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold over sigmoid(logits)")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    set_seed(args.seed)

    logger.info("Loading notes: %s", args.notes_file)
    logger.info("Loading splits: %s", args.splits_file)

    df = load_mimic_dataset(args.notes_file, args.splits_file, args.text_col, args.target_col)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    logger.info("Rows: train=%d val=%d test=%d", len(train_df), len(val_df), len(test_df))

    top_k = args.top_k_labels if args.top_k_labels and args.top_k_labels > 0 else None
    labels = build_label_space(train_df, args.target_col, min_freq=args.min_label_freq, top_k=top_k)
    if not labels:
        raise ValueError("No labels left after filtering; decrease --min_label_freq or check data.")

    label2id = {c: i for i, c in enumerate(labels)}
    id2label = {i: c for c, i in label2id.items()}

    with open(Path(args.output_dir) / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"labels": labels, "label2id": label2id}, f, indent=2)

    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_scheduler

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    train_ds = ICDDataset(train_df, tokenizer, label2id, args.text_col, args.target_col, args.max_length)
    val_ds = ICDDataset(val_df, tokenizer, label2id, args.text_col, args.target_col, args.max_length) if len(val_df) else None
    test_ds = ICDDataset(test_df, tokenizer, label2id, args.text_col, args.target_col, args.max_length) if len(test_df) else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.per_device_train_batch_size, shuffle=True
    )
    val_loader = (
        torch.utils.data.DataLoader(val_ds, batch_size=args.per_device_eval_batch_size, shuffle=False)
        if val_ds is not None
        else None
    )
    test_loader = (
        torch.utils.data.DataLoader(test_ds, batch_size=args.per_device_eval_batch_size, shuffle=False)
        if test_ds is not None
        else None
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    num_update_steps_per_epoch = max(1, len(train_loader))
    num_training_steps = args.num_train_epochs * num_update_steps_per_epoch
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * num_training_steps),
        num_training_steps=num_training_steps,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    def run_eval(loader, split_name: str) -> Dict[str, float]:
        model.eval()
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels_t = batch["labels"].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                all_logits.append(torch.sigmoid(logits).detach().cpu().numpy())
                all_labels.append(labels_t.detach().cpu().numpy())

        logits = np.concatenate(all_logits, axis=0)
        y_true = np.concatenate(all_labels, axis=0)
        metrics = multilabel_metrics(logits, y_true, threshold=args.threshold)
        logger.info("%s metrics: %s", split_name, metrics)
        return metrics

    best_val_f1 = -1.0
    best_path = Path(args.output_dir) / "best_model"

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels_t = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_t)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            total_loss += float(loss.detach().cpu())

        avg_loss = total_loss / max(1, len(train_loader))
        logger.info("Epoch %d/%d - train_loss=%.4f", epoch, args.num_train_epochs, avg_loss)

        if val_loader is not None and len(val_df) > 0:
            val_metrics = run_eval(val_loader, "val")
            if val_metrics["micro_f1"] > best_val_f1:
                best_val_f1 = val_metrics["micro_f1"]
                best_path.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(best_path)
                tokenizer.save_pretrained(best_path)
                with open(best_path / "val_metrics.json", "w", encoding="utf-8") as f:
                    json.dump(val_metrics, f, indent=2)

    # Save final model
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Evaluate best (if we have it); else evaluate final
    if best_path.exists():
        logger.info("Loading best model from: %s", best_path)
        model = model.from_pretrained(best_path).to(device)

    results = {}
    if test_loader is not None and len(test_df) > 0:
        results["test"] = run_eval(test_loader, "test")

    if val_loader is not None and len(val_df) > 0:
        results["val"] = run_eval(val_loader, "val")

    with open(Path(args.output_dir) / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
