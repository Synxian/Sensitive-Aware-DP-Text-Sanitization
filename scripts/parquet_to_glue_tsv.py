"""Convert HuggingFace SST-2 / QNLI parquet splits into the GLUE-style TSVs that
santext_sample.py and run_sanitizer.py expect.

HuggingFace ships:
    SST-2:  train-*.parquet, validation-*.parquet   columns: sentence, label, idx
    QNLI:   train-*.parquet, validation-*.parquet   columns: question, sentence, label, idx

Outputs (per santext_sample.py expectations):
    SST-2  train.tsv / dev.tsv  with header `sentence\\tlabel`
    QNLI   train.tsv / dev.tsv  with header `index\\tquestion\\tsentence\\tlabel`

The HuggingFace `validation` split is renamed to `dev` (same data, GLUE's original name).

Usage:
    python scripts/parquet_to_glue_tsv.py --task SST-2 --input_dir datasets/sst2 --output_dir data/SST-2
    python scripts/parquet_to_glue_tsv.py --task QNLI --input_dir datasets/qnli --output_dir data/QNLI
"""

import argparse
import glob
import os

import pandas as pd

SPLIT_MAP = {"train": "train", "validation": "dev"}


def _load_concat(input_dir: str, hf_split: str) -> pd.DataFrame:
    matches = sorted(glob.glob(os.path.join(input_dir, f"{hf_split}-*.parquet")))
    if not matches:
        flat = os.path.join(input_dir, f"{hf_split}.parquet")
        if not os.path.exists(flat):
            raise FileNotFoundError(
                f"No parquet shards found for split '{hf_split}' in {input_dir}"
            )
        matches = [flat]
    return pd.concat([pd.read_parquet(p) for p in matches], ignore_index=True)


def _write_sst2(df: pd.DataFrame, out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("sentence\tlabel\n")
        for sentence, label in zip(df["sentence"], df["label"]):
            sentence = str(sentence).replace("\t", " ").replace("\n", " ").rstrip()
            f.write(f"{sentence}\t{int(label)}\n")


def _write_qnli(df: pd.DataFrame, out_path: str) -> None:
    # QNLI labels in HuggingFace are integers (0=entailment, 1=not_entailment).
    # The GLUE TSV uses the string forms — match that so downstream tooling
    # (and santext_sample.py's `label = content[-1]`) sees the same values.
    label_map = {0: "entailment", 1: "not_entailment"}
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write("index\tquestion\tsentence\tlabel\n")
        for i, (q, s, lbl) in enumerate(
            zip(df["question"], df["sentence"], df["label"])
        ):
            q = str(q).replace("\t", " ").replace("\n", " ").rstrip()
            s = str(s).replace("\t", " ").replace("\n", " ").rstrip()
            label_str = label_map.get(int(lbl), str(lbl))
            f.write(f"{i}\t{q}\t{s}\t{label_str}\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["SST-2", "QNLI"], required=True)
    parser.add_argument("--input_dir", required=True, help="Directory of HF parquet shards")
    parser.add_argument("--output_dir", required=True, help="Where train.tsv / dev.tsv will be written")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for hf_split, glue_split in SPLIT_MAP.items():
        df = _load_concat(args.input_dir, hf_split)
        out_path = os.path.join(args.output_dir, f"{glue_split}.tsv")
        if args.task == "SST-2":
            _write_sst2(df, out_path)
        else:
            _write_qnli(df, out_path)
        print(f"{hf_split} -> {out_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
