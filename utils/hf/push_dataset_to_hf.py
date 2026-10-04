"""Push the human-labeled NER dataset (data/labeled) to the Hugging Face Hub.

    python -m utils.hf.push_dataset_to_hf --repo-id user/options-ner
    python -m utils.hf.push_dataset_to_hf --repo-id user/options-ner --splits --public

This is the *labeled* stage. The raw, unlabeled tweets are published by
`python -m utils.labeling.clean_data --push ...` (see `hf_utils.py`).

Label Studio metadata is stripped; each row keeps `id`, `text` and a list of
`{start, end, label}` entities. Augmented files are skipped so only human
labels are published.
"""

import argparse
import glob
import json
import os
import random
from collections import Counter
from pathlib import Path

from rich.console import Console

from src.core.labels import LABELS

console = Console()

VALID_LABELS = set(LABELS)


def validate_entity(text, entity):
    """Validate that entity boundaries match the text."""
    start, end = entity["start"], entity["end"]
    if start < 0 or end > len(text) or start >= end:
        return False, f"Invalid boundaries: [{start}:{end}] for text length {len(text)}"
    return True, None


def load_clean_gold_dataset(folder_path, verbose=False):
    """Parse human-labeled Label Studio exports into clean records.

    Skips cancelled/unannotated tasks, augmented files, unknown labels and
    spans that fall outside the text. Returns ``[{id, text, entities}, ...]``.
    """
    files = glob.glob(os.path.join(folder_path, "*.json"))
    original_files = sorted(f for f in files if "augmented_" not in os.path.basename(f))

    clean_records = []
    skipped = Counter()

    for file_path in original_files:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                ls_data = json.load(f)
            except json.JSONDecodeError:
                console.print(
                    f"[yellow]Could not parse {Path(file_path).name}, skipping.[/yellow]"
                )
                continue

        for task in ls_data:
            if not task.get("annotations"):
                skipped["no_annotations"] += 1
                continue
            if task["annotations"][0].get("was_cancelled"):
                skipped["cancelled"] += 1
                continue

            task_id = str(task.get("id", len(clean_records)))
            text = task["data"]["text"]

            entities = []
            for r in task["annotations"][0].get("result", []):
                if r.get("type") != "labels":
                    continue
                val = r["value"]
                label = str(val["labels"][0])
                if label not in VALID_LABELS:
                    skipped["invalid_label"] += 1
                    if verbose:
                        console.print(
                            f"[dim]Invalid label '{label}' in task {task_id}[/dim]"
                        )
                    continue

                entity = {
                    "start": int(val["start"]),
                    "end": int(val["end"]),
                    "label": label,
                }
                is_valid, error = validate_entity(text, entity)
                if not is_valid:
                    skipped["validation_error"] += 1
                    if verbose:
                        console.print(f"[dim]Task {task_id}: {error}[/dim]")
                    continue
                entities.append(entity)

            entities.sort(key=lambda x: x["start"])
            clean_records.append({"id": task_id, "text": text, "entities": entities})

    if verbose and skipped:
        console.print(f"\n[dim]Skipped: {dict(skipped)}[/dim]")

    return clean_records


def split_dataset(records, val_fraction=0.1, test_fraction=0.0, seed=42):
    """Split records into a dict of 'train' / 'validation' / 'test' lists."""
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    total = len(shuffled)
    val_size = int(total * val_fraction)
    test_size = int(total * test_fraction)
    train_size = total - val_size - test_size

    splits = {"train": shuffled[:train_size]}
    if val_size > 0:
        splits["validation"] = shuffled[train_size : train_size + val_size]
    if test_size > 0:
        splits["test"] = shuffled[train_size + val_size :]
    return splits


def records_to_hf_format(records):
    """Convert records to Hugging Face columnar format."""
    return {
        "id": [r["id"] for r in records],
        "text": [r["text"] for r in records],
        "entities": [
            {
                "start": [e["start"] for e in r["entities"]],
                "end": [e["end"] for e in r["entities"]],
                "label": [e["label"] for e in r["entities"]],
            }
            for r in records
        ],
    }


def push_dataset_to_hub(
    records, repo_id, private=True, create_splits=False, val_fraction=0.1
):
    """Convert records to a Hugging Face Dataset with a strict schema and push it."""
    from datasets import Dataset, Features, Sequence, Value

    features = Features(
        {
            "id": Value("string"),
            "text": Value("string"),
            "entities": Sequence(
                {
                    "start": Value("int32"),
                    "end": Value("int32"),
                    "label": Value("string"),
                }
            ),
        }
    )

    if create_splits:
        splits = split_dataset(records, val_fraction=val_fraction)
        for split_name, split_records in splits.items():
            console.print(f"  pushing {split_name} ({len(split_records)} samples)...")
            dataset = Dataset.from_dict(
                records_to_hf_format(split_records), features=features
            )
            dataset.push_to_hub(repo_id, split=split_name, private=private)
    else:
        console.print(f"pushing {len(records)} samples...")
        dataset = Dataset.from_dict(records_to_hf_format(records), features=features)
        dataset.push_to_hub(repo_id, private=private)

    visibility = "private" if private else "public"
    console.print(
        f"[bold green]Published ({visibility}):[/bold green] "
        f"https://huggingface.co/datasets/{repo_id}"
    )


def print_dataset_stats(records):
    """Print statistics about the dataset."""
    total_entities = sum(len(r["entities"]) for r in records)
    label_counts = Counter(e["label"] for r in records for e in r["entities"])
    char_counts = [len(r["text"]) for r in records]

    console.print("\n[bold]Dataset statistics[/bold]")
    console.print(f"  Samples: {len(records)}")
    console.print(
        f"  Entities: {total_entities} ({total_entities / len(records):.1f} per sample avg)"
    )
    for label in LABELS:
        console.print(f"    {label}: {label_counts[label]}")
    console.print(f"  Avg text length: {sum(char_counts) / len(char_counts):.0f} chars")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", default="data/labeled")
    parser.add_argument("--repo-id", required=True, help="e.g. username/options-ner")
    parser.add_argument("--public", action="store_true", help="Default: private.")
    parser.add_argument(
        "--splits", action="store_true", help="Create train/validation splits."
    )
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    console.print(f"[bold]Loading dataset from {args.folder}...[/bold]")
    gold_records = load_clean_gold_dataset(args.folder, verbose=args.verbose)
    if not gold_records:
        console.print(
            "[bold red]No valid human-labeled records found to upload.[/bold red]"
        )
        raise SystemExit(1)

    print_dataset_stats(gold_records)
    push_dataset_to_hub(
        gold_records,
        args.repo_id,
        private=not args.public,
        create_splits=args.splits,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
