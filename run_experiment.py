"""End-to-end sweep: sanitize + downstream across methods × epsilons × tasks.

Skips configs where output already exists (safe to resume after interruption).
Produces results/summary.csv.

Usage
-----
    # Full sweep
    python run_experiment.py \\
        --tasks sst2 qnli \\
        --methods santext normal plus \\
        --epsilons 1 3 5 10 20 \\
        --min_freq 5 \\
        --downstream

    # Smoke test
    python run_experiment.py \\
        --tasks sst2 --methods santext --epsilons 10 \\
        --max_samples 500 --downstream
"""
import argparse
import csv
import json
import logging
import os
import subprocess
import sys

logger    = logging.getLogger(__name__)
ROOT      = os.path.dirname(os.path.abspath(__file__))
SANITIZE  = os.path.join(ROOT, "run_sanitize.py")
TRAIN     = os.path.join(ROOT, "run_downstream.py")

DATA_DIRS = {
    "sst2": "SST-2",
    "qnli": "QNLI",
}


def output_dir(base, cfg):
    m = cfg["method"]
    if m == "santext":
        sub = f"eps_{cfg['epsilon']:.2f}"
    elif m == "normal":
        sub = f"eps_{cfg['epsilon']:.2f}_seps_{cfg['s_epsilon']:.2f}"
    else:
        sub = f"eps_{cfg['epsilon']:.2f}_seps_{cfg['s_epsilon']:.2f}_p_{cfg['p']:.2f}"
    return os.path.join(base, cfg["task"], m, sub)


def build_configs(args):
    configs = []
    for task in args.tasks:
        for method in args.methods:
            for eps in args.epsilons:
                p_list = args.p_values if method == "plus" else [0.0]
                for p in p_list:
                    configs.append(dict(task=task, method=method,
                                        epsilon=eps, s_epsilon=eps / 2, p=p))
    return configs


def run_cmd(cmd, label):
    logger.info("[%s] %s", label, " ".join(cmd))
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        logger.error("FAILED [%s] rc=%d", label, rc)
    return rc == 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks",    nargs="+", default=["sst2"],
                   choices=["sst2", "qnli"])
    p.add_argument("--methods",  nargs="+", default=["santext", "normal", "plus"],
                   choices=["santext", "normal", "plus"])
    p.add_argument("--epsilons", nargs="+", type=float, default=[1, 3, 5, 10, 20])
    p.add_argument("--distance_metric", default="cosine", choices=["cosine", "euclidean"])
    p.add_argument("--p_values", nargs="+", type=float, default=[0.7])
    p.add_argument("--no_ner", action="store_true",
                   help="Disable NER; fall back to frequency-based detection")
    p.add_argument("--sensitive_pct", type=float, default=0.5)
    p.add_argument("--data_base",    default="./data")
    p.add_argument("--output_base",  default="./output")
    p.add_argument("--results_base", default="./results")
    p.add_argument("--embed_path",   default="./data/glove.840B.300d.txt")
    p.add_argument("--downstream",      action="store_true")
    p.add_argument("--downstream_only", action="store_true")
    p.add_argument("--model_name", default="bert-base-uncased")
    p.add_argument("--num_epochs", type=int,   default=3)
    p.add_argument("--batch_size", type=int,   default=32)
    p.add_argument("--max_train_samples", type=int, default=None)
    p.add_argument("--min_freq",    type=int,  default=1)
    p.add_argument("--max_vocab",   type=int,  default=None)
    p.add_argument("--max_samples", type=int,  default=None)
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()

    logging.basicConfig(format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)

    configs = build_configs(args)
    logger.info("Grid: %d configurations", len(configs))

    all_results = []

    for idx, cfg in enumerate(configs):
        tag = f"[{idx+1}/{len(configs)}] {cfg['task']}/{cfg['method']} eps={cfg['epsilon']}"
        logger.info("=" * 55)
        logger.info("START %s", tag)

        san_dir = output_dir(args.output_base, cfg)
        res_dir = output_dir(args.results_base, cfg)

        # --- Sanitize ---
        if not args.downstream_only:
            san_done = (os.path.exists(os.path.join(san_dir, "train.tsv")) and
                        os.path.exists(os.path.join(san_dir, "dev.tsv")))
            if san_done:
                logger.info("SKIP sanitize (exists)")
            else:
                data_dir = os.path.join(args.data_base,
                                        DATA_DIRS.get(cfg["task"], cfg["task"].upper()))
                cmd = [sys.executable, SANITIZE,
                       "--task",        cfg["task"],
                       "--method",      cfg["method"],
                       "--epsilon",     str(cfg["epsilon"]),
                       "--s_epsilon",   str(cfg["s_epsilon"]),
                       "--p",           str(cfg["p"]),
                       "--data_dir",    data_dir,
                       "--embed_path",  args.embed_path,
                       "--distance_metric", args.distance_metric,
                       "--output_dir",  san_dir,
                       "--min_freq",    str(args.min_freq),
                       "--seed",        str(args.seed),
                       "--threads",     str(args.threads),
                       ]
                if args.no_ner:
                    cmd += ["--no_ner", "--sensitive_pct", str(args.sensitive_pct)]
                if args.max_samples: cmd += ["--max_samples", str(args.max_samples)]
                if args.max_vocab:   cmd += ["--max_vocab",   str(args.max_vocab)]
                if not run_cmd(cmd, "sanitize"):
                    all_results.append({**cfg, "accuracy": "FAIL-sanitize"})
                    continue

        # --- Downstream ---
        if args.downstream or args.downstream_only:
            eval_path = os.path.join(res_dir, "eval_results.json")
            if os.path.exists(eval_path):
                logger.info("SKIP train (exists)")
            else:
                cmd = [sys.executable, TRAIN,
                       "--task",       cfg["task"],
                       "--data_dir",   san_dir,
                       "--output_dir", res_dir,
                       "--num_epochs", str(args.num_epochs),
                       "--batch_size", str(args.batch_size),
                       "--model_name", args.model_name,
                       "--seed",       str(args.seed),
                       ]
                if args.max_train_samples:
                    cmd += ["--max_train_samples", str(args.max_train_samples)]
                if not run_cmd(cmd, "train"):
                    all_results.append({**cfg, "accuracy": "FAIL-train"})
                    continue

            acc = "N/A"
            if os.path.exists(eval_path):
                with open(eval_path) as f:
                    acc = json.load(f).get("eval_accuracy", "N/A")
            all_results.append({**cfg, "accuracy": acc})
            logger.info("DONE %s — accuracy=%s", tag, acc)
        else:
            all_results.append({**cfg, "accuracy": "sanitize-only"})
            logger.info("DONE %s", tag)

    # --- Summary ---
    logger.info("=" * 55)
    logger.info("%-8s %-8s %6s %6s %4s %10s",
                "task", "method", "eps", "s_eps", "p", "accuracy")
    logger.info("-" * 50)
    for r in all_results:
        acc = f"{r['accuracy']:.4f}" if isinstance(r["accuracy"], float) else str(r["accuracy"])
        logger.info("%-8s %-8s %6.1f %6.1f %4.2f %10s",
                    r["task"], r["method"], r["epsilon"], r["s_epsilon"], r["p"], acc)

    os.makedirs(args.results_base, exist_ok=True)
    csv_path = os.path.join(args.results_base, "summary.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task","method","epsilon","s_epsilon","p","accuracy"])
        writer.writeheader()
        writer.writerows(all_results)
    logger.info("Summary → %s", csv_path)


if __name__ == "__main__":
    main()
