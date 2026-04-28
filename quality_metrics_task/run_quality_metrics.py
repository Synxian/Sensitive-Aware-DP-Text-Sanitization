"""Compute quality metrics between original and sanitized TSV datasets.

Usage
-----
    python -m quality_metrics_task.run_quality_metrics \
        --original_dir ./data/SST-2 \
        --sanitized_dir ./output/sst2/normal/eps_1.00_seps_0.50 \
        --task sst2 \
        --metrics all \
        --output_path ./output/sst2/normal/eps_1.00_seps_0.50/quality_metrics.json

    # Single metric
    python -m quality_metrics_task.run_quality_metrics \
        --original_dir ./data/SST-2 \
        --sanitized_dir ./output/sst2/normal/eps_1.00_seps_0.50 \
        --task sst2 \
        --metrics jaccard
"""
import argparse
import json
import os

import torch


def _load_texts_from_tsv(path, task, col="sanitized"):
    """Load text column from a GLUE-style TSV."""
    texts = []
    with open(path, encoding="utf-8") as f:
        next(f)  # skip header
        for line in f:
            parts = line.strip().split("\t")
            if task == "sst2" and parts:
                texts.append(parts[0])
            elif task == "qnli" and len(parts) >= 3:
                texts.append(parts[1] + " " + parts[2])
    return texts


def run_quality_metrics(
    original_dir: str,
    sanitized_dir: str,
    task: str,
    metrics: list[str],
    output_path: str,
    split: str = "dev",
    lang: str = "en",
    limit: int | None = None,
):
    """Compute selected quality metrics and save results.

    Args:
        original_dir:   path to original data (e.g. ./data/SST-2)
        sanitized_dir:  path to sanitized output
        task:           "sst2" or "qnli"
        metrics:        list of metric names: "jaccard", "bert_score", "mauve", "movers_distance"
                        or ["all"] for everything
        output_path:    path to save JSON results
        split:          which split to evaluate ("dev" or "train")
        lang:           language for BERTScore model selection
        limit:          max number of examples (None = all)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    orig_path = os.path.join(original_dir, f"{split}.tsv")
    san_path = os.path.join(sanitized_dir, f"{split}.tsv")

    if not os.path.exists(orig_path) or not os.path.exists(san_path):
        print(f"Missing {split}.tsv in original or sanitized dir — skipping")
        return

    original = _load_texts_from_tsv(orig_path, task)
    sanitized = _load_texts_from_tsv(san_path, task)

    if limit:
        original = original[:limit]
        sanitized = sanitized[:limit]

    # Ensure same length
    n = min(len(original), len(sanitized))
    original = original[:n]
    sanitized = sanitized[:n]

    if "all" in metrics:
        metrics = ["jaccard", "bert_score", "mauve", "movers_distance"]

    results = {}

    if "jaccard" in metrics:
        from quality_metrics_task.run_jaccard import compute_jaccard_similarity
        print("Computing Jaccard similarity …")
        scores = [compute_jaccard_similarity(o, s) for o, s in zip(original, sanitized)]
        results["jaccard"] = sum(scores) / len(scores) if scores else 0.0

    if "bert_score" in metrics:
        from quality_metrics_task.run_bert_score import compute_bert_score
        print("Computing BERTScore …")
        results["bert_score"] = compute_bert_score(original, sanitized, device=device, lang=lang)

    if "movers_distance" in metrics:
        from quality_metrics_task.run_movers_distance import compute_movers_distance
        print("Computing Movers Distance …")
        results["movers_distance"] = compute_movers_distance(original, sanitized, device=device)

    if "mauve" in metrics:
        from quality_metrics_task.run_mauve import compute_mauve
        print("Computing MAUVE …")
        results["mauve"] = compute_mauve(original, sanitized, device=device)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Quality metrics saved to {output_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_dir", required=True)
    parser.add_argument("--sanitized_dir", required=True)
    parser.add_argument("--task", required=True, choices=["sst2", "qnli"])
    parser.add_argument("--metrics", nargs="+", default=["all"],
                        choices=["all", "jaccard", "bert_score", "mauve", "movers_distance"])
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--split", default="dev")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(args.sanitized_dir, "quality_metrics.json")

    run_quality_metrics(
        original_dir=args.original_dir,
        sanitized_dir=args.sanitized_dir,
        task=args.task,
        metrics=args.metrics,
        output_path=args.output_path,
        split=args.split,
        lang=args.lang,
        limit=args.limit,
    )
