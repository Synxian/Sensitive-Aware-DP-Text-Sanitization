"""Reconstruct santext-style TSVs from per-document JSONs produced by run_sanitizer.py.

run_sanitizer.py writes one JSON per SastdpDocument and a sidecar
task_metadata.json describing the original SST-2 / QNLI row structure.
This script stitches them back into the train.tsv / dev.tsv shape that
santext_sample.py emits, so downstream GLUE training can consume them.

Usage:
    python convert_replacements_to_tsv.py \\
        --replacements_dir replacements_flair/42/normal/SST-2/n_epsilon_8.0_s_epsilon_4.0 \\
        --output_dir output_run_sanitizer/SST-2/normal_eps_8.00
"""

import argparse
import json
import os

from tqdm import tqdm


def _load_sanitized(replacements_dir: str, text_id: str) -> str:
    path = os.path.join(replacements_dir, f"{text_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["sanitized_text"]


def _write_sst2(metadata: dict, replacements_dir: str, output_dir: str) -> None:
    for split, info in metadata["splits"].items():
        out_path = os.path.join(output_dir, f"{split}.tsv")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(info["header"] + "\n")
            for row in tqdm(info["rows"], desc=f"SST-2 {split}", unit="row"):
                text = _load_sanitized(replacements_dir, row["text_id"])
                out.write(f"{text}\t{row['label']}\n")
        print(f"Wrote {out_path} ({len(info['rows'])} rows)")


def _write_qnli(metadata: dict, replacements_dir: str, output_dir: str) -> None:
    for split, info in metadata["splits"].items():
        out_path = os.path.join(output_dir, f"{split}.tsv")
        with open(out_path, "w", encoding="utf-8") as out:
            out.write(info["header"] + "\n")
            for i, row in enumerate(tqdm(info["rows"], desc=f"QNLI {split}", unit="row")):
                text_a = _load_sanitized(replacements_dir, row["text_id_a"])
                text_b = _load_sanitized(replacements_dir, row["text_id_b"])
                out.write(f"{i}\t{text_a}\t{text_b}\t{row['label']}\n")
        print(f"Wrote {out_path} ({len(info['rows'])} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replacements_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    metadata_path = os.path.join(args.replacements_dir, "task_metadata.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    os.makedirs(args.output_dir, exist_ok=True)
    task = metadata["task"]
    if task == "SST-2":
        _write_sst2(metadata, args.replacements_dir, args.output_dir)
    elif task == "QNLI":
        _write_qnli(metadata, args.replacements_dir, args.output_dir)
    else:
        raise ValueError(f"Unsupported task in task_metadata.json: {task}")


if __name__ == "__main__":
    main()
