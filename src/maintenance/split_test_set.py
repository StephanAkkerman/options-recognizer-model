"""Build a stratified held-out test set from every labeled file.

Samples ~15% of valid annotated tasks from each `data/labeled/*.json` using a
deterministic seed and writes them to `data/test/test_<name>.json`. The training
loader already excludes test task IDs from train+val (and from any augmented
copies sharing those IDs), so the sampled tasks are NOT removed from their
source files — the same task can sit in both folders and the loader dedups by ID.

Sampling per file keeps the test mix matched to the training mix (different
batches have different entity densities and alert formats).

Run:
    python -m src.maintenance.split_test_set            # writes test/, prints summary
    python -m src.maintenance.split_test_set --seed 7   # different deterministic split

After running, re-benchmark every adapter (`python -m src.core.benchmark --all`)
because a new test set has a different hash — old cached results don't apply.
"""

import argparse
import glob
import json
import os
import random
from collections import Counter

from rich.console import Console
from rich.table import Table

from src.core.labels import LABELS

console = Console()

LABELED_FOLDER = "data/labeled"
TEST_FOLDER = "data/test"
DEFAULT_SEED = 42
DEFAULT_TEST_FRACTION = 0.15


def _label_counts(task):
    return Counter(
        r["value"]["labels"][0]
        for r in task.get("annotations", [{}])[0].get("result", [])
        if r.get("type") == "labels"
    )


def clear_existing_test_files(test_folder):
    """Wipe prior test_*.json so stale splits don't accumulate."""
    if not os.path.isdir(test_folder):
        return
    for fp in glob.glob(os.path.join(test_folder, "test_*.json")):
        os.remove(fp)


def stratified_split(labeled_folder, test_folder, test_fraction, seed):
    """For each source file, sample `test_fraction` of tasks by ID."""
    os.makedirs(test_folder, exist_ok=True)
    rng = random.Random(seed)

    summary = []
    for fp in sorted(glob.glob(os.path.join(labeled_folder, "*.json"))):
        name = os.path.basename(fp)
        # Skip backups and any unrelated artifact files.
        if name.endswith(".bak"):
            continue

        with open(fp, "r", encoding="utf-8") as f:
            tasks = json.load(f)
        valid = [
            t
            for t in tasks
            if t.get("annotations")
            and not t["annotations"][0].get("was_cancelled")
            and sum(_label_counts(t).values()) > 0
        ]
        if not valid:
            continue

        ids = sorted({t["id"] for t in valid})
        rng.shuffle(ids)
        n_test = max(1, int(round(len(ids) * test_fraction)))
        test_ids = set(ids[:n_test])
        test_tasks = [t for t in valid if t["id"] in test_ids]

        out_path = os.path.join(test_folder, f"test_{name}")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(test_tasks, f, indent=2, ensure_ascii=False)

        label_counts = Counter()
        for t in test_tasks:
            label_counts.update(_label_counts(t))
        summary.append(
            {
                "file": name,
                "total_tasks": len(valid),
                "test_tasks": len(test_tasks),
                "label_counts": label_counts,
            }
        )
    return summary


def _render_summary(summary, seed, test_fraction):
    table = Table(
        title=f"Stratified test split — {test_fraction:.0%} per file, seed={seed}"
    )
    table.add_column("file")
    table.add_column("source tasks", justify="right")
    table.add_column("test tasks", justify="right")
    for label in LABELS:
        table.add_column(label, justify="right")

    src_total = test_total = 0
    totals = Counter()
    for row in summary:
        src_total += row["total_tasks"]
        test_total += row["test_tasks"]
        totals.update(row["label_counts"])
        table.add_row(
            row["file"],
            str(row["total_tasks"]),
            str(row["test_tasks"]),
            *(str(row["label_counts"][label]) for label in LABELS),
        )
    table.add_section()
    table.add_row(
        "[bold]TOTAL[/bold]",
        f"[bold]{src_total}[/bold]",
        f"[bold]{test_total}[/bold]",
        *(f"[bold]{totals[label]}[/bold]" for label in LABELS),
    )
    console.print(table)


def run(
    seed=DEFAULT_SEED,
    test_fraction=DEFAULT_TEST_FRACTION,
    labeled_folder=LABELED_FOLDER,
    test_folder=TEST_FOLDER,
    quiet=False,
):
    """Programmatic entry point. Same seed → same split, every time.

    Called from `train.py` at startup so the held-out test set always reflects
    the current labeled data state.
    """
    clear_existing_test_files(test_folder)
    summary = stratified_split(labeled_folder, test_folder, test_fraction, seed)
    if not summary:
        raise RuntimeError(f"No labeled files found to split in {labeled_folder}.")
    if not quiet:
        _render_summary(summary, seed, test_fraction)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--test-fraction", type=float, default=DEFAULT_TEST_FRACTION)
    parser.add_argument("--labeled-folder", default=LABELED_FOLDER)
    parser.add_argument("--test-folder", default=TEST_FOLDER)
    args = parser.parse_args()

    try:
        run(
            seed=args.seed,
            test_fraction=args.test_fraction,
            labeled_folder=args.labeled_folder,
            test_folder=args.test_folder,
        )
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise SystemExit(1)

    console.print(
        "\n[green]Next:[/green] run `python -m src.core.benchmark --all` to rescore "
        "all adapters against the new test set."
    )


if __name__ == "__main__":
    main()
