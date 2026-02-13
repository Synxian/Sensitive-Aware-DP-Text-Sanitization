"""Train a NER model on the i2b2 de-identification corpus.

This repo stores i2b2 notes as XML files with the following structure:

<deIdi2b2>
  <TEXT><![CDATA[ ... raw note text ... ]]></TEXT>
  <TAGS>
      <DATE start="..." end="..." text="..." TYPE="DATE" />
      <NAME start="..." end="..." text="..." TYPE="PATIENT" />
      ...
  </TAGS>
</deIdi2b2>

Important details:
- Mentions are annotated in *character offsets* [start, end) over the note text.
- Tag element names vary (DATE, NAME, LOCATION, ID, AGE, etc.).
- The label of interest is the attribute TYPE (e.g., PATIENT, DOCTOR, HOSPITAL...).

This script converts the char-span annotations into token-level BIO tags and trains a
Transformers token-classification model.

Output:
- Trained model + tokenizer under --output_dir
- Label mapping and evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

from datasets import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _normalize_whitespace(text: str) -> str:
    """Keep original char offsets stable.

    For i2b2 we must NOT alter the text at all before span alignment.
    This helper is only for optional display/debug; do not call it in parsing.
    """

    return re.sub(r"\s+", " ", text).strip()


@dataclass(frozen=True)
class Span:
    start: int
    end: int  # exclusive
    label: str


def parse_i2b2_xml(path: Union[str, Path]) -> Tuple[str, List[Span]]:
    """Parse one i2b2 XML file into raw text and labeled spans."""

    import xml.etree.ElementTree as ET

    tree = ET.parse(str(path))
    root = tree.getroot()

    text_node = root.find("TEXT")
    if text_node is None or text_node.text is None:
        raise ValueError(f"Missing TEXT in {path}")
    text = text_node.text

    spans: List[Span] = []
    tags_node = root.find("TAGS")
    if tags_node is not None:
        for child in list(tags_node):
            # Each child corresponds to a PHI mention; the label lives in the TYPE attribute.
            # Offsets are inclusive/exclusive in the i2b2 deid format.
            attrib = child.attrib
            if "start" not in attrib or "end" not in attrib or "TYPE" not in attrib:
                continue
            start = int(attrib["start"])
            end = int(attrib["end"])
            label = attrib["TYPE"].strip()
            if end <= start:
                continue
            # Defensive: clamp to text length.
            start = max(0, min(start, len(text)))
            end = max(0, min(end, len(text)))
            if end <= start:
                continue
            spans.append(Span(start=start, end=end, label=label))

    spans = sorted(spans, key=lambda s: (s.start, s.end))
    return text, spans


def load_i2b2_records(data_dir: str) -> List[Dict]:
    """Load all i2b2 XML files under data_dir into a list of dict records."""

    data_path = Path(data_dir)
    files = sorted(data_path.glob("*.xml"))
    if not files:
        raise FileNotFoundError(f"No .xml files found in {data_dir}")

    records: List[Dict] = []
    for p in files:
        text, spans = parse_i2b2_xml(p)
        records.append({"id": p.stem, "text": text, "spans": [s.__dict__ for s in spans]})
    return records


def train_val_test_split(
    items: List[Dict], train_ratio: float, val_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    assert 0 < train_ratio < 1
    assert 0 <= val_ratio < 1
    assert train_ratio + val_ratio < 1

    rng = random.Random(seed)
    idx = list(range(len(items)))
    rng.shuffle(idx)

    n = len(items)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = [items[i] for i in idx[:n_train]]
    val = [items[i] for i in idx[n_train : n_train + n_val]]
    test = [items[i] for i in idx[n_train + n_val :]]
    return train, val, test


def build_label_list(records: Iterable[Dict]) -> List[str]:
    """Collect unique span labels; produce BIO label list."""

    labels = set()
    for r in records:
        for s in r["spans"]:
            labels.add(s["label"])

    base = sorted(labels)
    # BIO scheme
    label_list = ["O"]
    for lab in base:
        label_list.append(f"B-{lab}")
        label_list.append(f"I-{lab}")

    return label_list


def spans_to_char_tags(text: str, spans: List[Dict]) -> List[str]:
    """Convert span annotations (list of dict with start/end/label) to per-character BIO tags."""

    tags = ["O"] * len(text)

    for s in sorted(spans, key=lambda d: (int(d["start"]), int(d["end"]))):
        start, end, lab = int(s["start"]), int(s["end"]), str(s["label"])
        if end <= start:
            continue
        start = max(0, min(start, len(text)))
        end = max(0, min(end, len(text)))
        if end <= start:
            continue

        # Skip if any char already labeled (simple overlap handling)
        if any(t != "O" for t in tags[start:end]):
            continue

        tags[start] = f"B-{lab}"
        for i in range(start + 1, end):
            tags[i] = f"I-{lab}"

    return tags


def align_labels_with_tokens(examples, tokenizer, label2id: Dict[str, int], max_length: int):
    """Tokenize and align BIO labels from char-level tags to token-level tags.

    Assumptions:
    - Using a fast tokenizer that supports offset_mapping.
    - For each token, we look up the char tag at the first character of its span.
    - Special tokens get label -100.
    """

    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    all_labels = []
    for text, spans, offsets, word_ids in zip(
        examples["text"],
        examples["spans"],
        tokenized["offset_mapping"],
        [tokenized.word_ids(i) for i in range(len(tokenized["input_ids"]))],
    ):
        char_tags = spans_to_char_tags(text, spans)
        labels = []

        # word_ids is used to avoid labeling continuation subword pieces multiple times.
        prev_word_id = None
        for (start, end), wid in zip(offsets, word_ids):
            if wid is None:
                labels.append(-100)
                continue

            # Option A: label only the first sub-token of each original word.
            if wid == prev_word_id:
                labels.append(-100)
                continue
            prev_word_id = wid

            if start >= len(char_tags) or end <= 0 or start == end:
                labels.append(label2id["O"])
                continue

            tag = char_tags[start]
            labels.append(label2id.get(tag, label2id["O"]))

        all_labels.append(labels)

    tokenized.pop("offset_mapping")
    tokenized["labels"] = all_labels
    return tokenized


def compute_metrics_factory(label_list: List[str]):
    """Create a compute_metrics function compatible with HuggingFace Trainer."""

    import evaluate

    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)

        true_predictions = []
        true_labels = []
        for pred, lab in zip(predictions, labels):
            cur_preds = []
            cur_labs = []
            for p_i, l_i in zip(pred, lab):
                if l_i == -100:
                    continue
                cur_preds.append(label_list[p_i])
                cur_labs.append(label_list[l_i])
            true_predictions.append(cur_preds)
            true_labels.append(cur_labs)

        results = seqeval.compute(predictions=true_predictions, references=true_labels)
        # seqeval returns nested dict; normalize key set
        return {
            "precision": results.get("overall_precision", 0.0),
            "recall": results.get("overall_recall", 0.0),
            "f1": results.get("overall_f1", 0.0),
            "accuracy": results.get("overall_accuracy", 0.0),
        }

    return compute_metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NER on i2b2 XML")

    parser.add_argument("--data_dir", type=str, required=True, help="Directory with i2b2 .xml files")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save model + artifacts")

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="distilbert-base-uncased",
        help="Any HF checkpoint suitable for token-classification",
    )

    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])
    parser.add_argument("--save_strategy", type=str, default="epoch", choices=["no", "steps", "epoch"])

    parser.add_argument("--fp16", action="store_true")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )

    set_seed(args.seed)

    logger.info("Loading i2b2 from: %s", args.data_dir)
    records = load_i2b2_records(args.data_dir)
    train_records, val_records, test_records = train_val_test_split(
        records, train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.seed
    )

    logger.info(
        "Split sizes: train=%d val=%d test=%d", len(train_records), len(val_records), len(test_records)
    )

    label_list = build_label_list(train_records)
    label2id = {l: i for i, l in enumerate(label_list)}
    id2label = {i: l for l, i in label2id.items()}

    with open(Path(args.output_dir) / "labels.json", "w", encoding="utf-8") as f:
        json.dump({"label_list": label_list, "label2id": label2id}, f, indent=2, ensure_ascii=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=True)

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)
    test_ds = Dataset.from_list(test_records)

    # Convert spans dict-list to python object (datasets may serialize); keep as list[dict]
    def _ensure_spans(ex):
        ex["spans"] = list(ex["spans"]) if ex.get("spans") is not None else []
        return ex

    train_ds = train_ds.map(_ensure_spans)
    val_ds = val_ds.map(_ensure_spans)
    test_ds = test_ds.map(_ensure_spans)

    tokenized_train = train_ds.map(
        lambda ex: align_labels_with_tokens(ex, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_val = val_ds.map(
        lambda ex: align_labels_with_tokens(ex, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=val_ds.column_names,
    )
    tokenized_test = test_ds.map(
        lambda ex: align_labels_with_tokens(ex, tokenizer, label2id, args.max_length),
        batched=True,
        remove_columns=test_ds.column_names,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # If the user sets val_ratio=0 (or otherwise ends up with an empty val split),
    # we must not enable evaluation/best-model selection.
    has_eval = len(val_records) > 0 and args.eval_strategy != "no"

    # Transformers Trainer requires eval_strategy='no' if eval_dataset is None.
    effective_eval_strategy = args.eval_strategy if has_eval else "no"

    # If we're not evaluating, also avoid checkpointing every epoch/steps unless user explicitly wants it.
    # (This keeps smoke tests fast and avoids confusing best-model logic.)
    effective_save_strategy = args.save_strategy if has_eval else "no"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        eval_strategy=effective_eval_strategy,
        save_strategy=effective_save_strategy,
        logging_steps=args.logging_steps,
        seed=args.seed,
        fp16=args.fp16,
        report_to=[],
        remove_unused_columns=False,
        load_best_model_at_end=True if has_eval else False,
        metric_for_best_model="f1" if has_eval else None,
        greater_is_better=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val if has_eval else None,
        data_collator=data_collator,
        compute_metrics=compute_metrics_factory(label_list) if has_eval else None,
    )

    trainer.train()

    # Save final artifacts
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Only compute metrics if we actually have an evaluation loop configured.
    if has_eval and len(test_records) > 0:
        metrics = trainer.evaluate(tokenized_test)
        with open(Path(args.output_dir) / "test_metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Test metrics: %s", metrics)


if __name__ == "__main__":
    main()
