"""Aggregate run_glue eval JSONs across seeds.

Walks `tmp/*/eval_results_*.json`, groups by (task, method, eps, s_eps, p,
distance), and reports per-group mean and std of eval_acc / eval_loss across
seeds. Writes a CSV at results/summary.csv and prints a markdown table.

Run:  python scripts/aggregate_results.py
"""

import csv
import glob
import json
import os
import re
import statistics
from collections import defaultdict


# Matches dir names like "SST-2-plus-eps8-s4-p0.7-seed42-euclidean" or the
# older shapes "sst2-santext-eps8" / "sst2-plus-eps8-p0.9-eucl".
RUN_RE = re.compile(
    r"""^
    (?P<task>[A-Za-z0-9-]+?)
    -(?P<method>santext|normal|plus)
    -eps(?P<eps>[0-9.]+)
    (?:-s(?P<s_eps>[0-9.]+))?
    (?:-p(?P<p>[0-9.]+))?
    -seed(?P<seed>\d+)
    -(?P<distance>euclidean|cosine|eucl)
    $""",
    re.VERBOSE,
)


def _parse_run(dir_name):
    m = RUN_RE.match(dir_name)
    if not m:
        return None
    d = m.groupdict()
    d["distance"] = "euclidean" if d["distance"] == "eucl" else d["distance"]
    return d


def _key(parsed):
    return (
        parsed["task"],
        parsed["method"],
        parsed["eps"],
        parsed["s_eps"] or "",
        parsed["p"] or "",
        parsed["distance"],
    )


def main():
    rows = []
    skipped = []
    for path in sorted(glob.glob("tmp/*/eval_results_*.json")):
        run_dir = os.path.basename(os.path.dirname(path))
        parsed = _parse_run(run_dir)
        if parsed is None:
            skipped.append(run_dir)
            continue
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics", {})
        rows.append({
            **parsed,
            "eval_acc": m.get("eval_acc"),
            "eval_loss": m.get("eval_loss"),
            "epoch": m.get("epoch"),
            "json_path": path,
        })

    if skipped:
        print(f"Skipped {len(skipped)} dir(s) that don't match the canonical naming "
              "(<task>-<method>-eps<E>[-s<S>][-p<P>]-seed<N>-<dist>):")
        for d in skipped:
            print(f"  {d}")
        print()

    if not rows:
        print("No matching eval JSONs found under tmp/.")
        return

    groups = defaultdict(list)
    for r in rows:
        groups[_key(r)].append(r)

    summary = []
    for key, group in sorted(groups.items()):
        accs = [r["eval_acc"] for r in group if r["eval_acc"] is not None]
        losses = [r["eval_loss"] for r in group if r["eval_loss"] is not None]
        seeds = sorted({r["seed"] for r in group})
        summary.append({
            "task": key[0],
            "method": key[1],
            "eps": key[2],
            "s_eps": key[3],
            "p": key[4],
            "distance": key[5],
            "n_seeds": len(seeds),
            "seeds": ",".join(seeds),
            "acc_mean": statistics.mean(accs) if accs else None,
            "acc_std": statistics.stdev(accs) if len(accs) > 1 else 0.0,
            "loss_mean": statistics.mean(losses) if losses else None,
            "loss_std": statistics.stdev(losses) if len(losses) > 1 else 0.0,
        })

    os.makedirs("results", exist_ok=True)
    csv_path = "results/summary.csv"
    fieldnames = ["task", "method", "eps", "s_eps", "p", "distance",
                  "n_seeds", "seeds", "acc_mean", "acc_std", "loss_mean", "loss_std"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary:
            w.writerow(row)
    print(f"Wrote {csv_path}")
    print()

    # Markdown table for quick eyeballing
    print("| task | method | eps | s_eps | p | dist | n | acc (mean ± std) | loss (mean ± std) |")
    print("|------|--------|----:|------:|--:|------|--:|-----------------:|------------------:|")
    for r in summary:
        am, asd = r["acc_mean"], r["acc_std"]
        lm, lsd = r["loss_mean"], r["loss_std"]
        acc_str = f"{am:.4f} ± {asd:.4f}" if am is not None else "—"
        loss_str = f"{lm:.4f} ± {lsd:.4f}" if lm is not None else "—"
        print(f"| {r['task']} | {r['method']} | {r['eps']} | {r['s_eps'] or '—'} | "
              f"{r['p'] or '—'} | {r['distance']} | {r['n_seeds']} | {acc_str} | {loss_str} |")


if __name__ == "__main__":
    main()
