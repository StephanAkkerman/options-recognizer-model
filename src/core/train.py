"""Train a GLiNER2 LoRA adapter on the Label Studio exports in data/labeled.

    python -m src.core.train

Refreshes the held-out test split first, trains, writes
``models/options_adapter_vN/{best,final}`` plus ``training_metadata.json``, and
benchmarks the result against the test set.
"""

import glob
import hashlib
import json
import os
import random
import re
import subprocess
from datetime import datetime, timezone

from rich.console import Console

from src.core.labels import ENTITY_DESCRIPTIONS
from src.core.spans import bounded_pattern

console = Console()

SEED = 42
EARLY_STOPPING = True
EPOCHS = 10
VAL_FRACTION = 0.15
ADAPTER_PREFIX = "options_adapter"
# Flow-list tweets are number-dense and tokenize long; 256 truncated the tail
# of the longest ones.
MAX_LEN = 384


def set_seed(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_next_version(models_dir="./models"):
    """Scans the models directory to determine the next version number."""
    os.makedirs(models_dir, exist_ok=True)
    max_v = 0
    for adapter in glob.glob(os.path.join(models_dir, f"{ADAPTER_PREFIX}*")):
        folder_name = os.path.basename(adapter)
        match = re.search(r"_v(\d+)$", folder_name)
        if match:
            max_v = max(max_v, int(match.group(1)))
        elif folder_name == ADAPTER_PREFIX:
            max_v = max(max_v, 1)
    return max_v + 1


def chunk_text_with_overlap(text, chunk_word_size=150, overlap_words=40):
    """Splits text into overlapping chunks without splitting words."""
    words = text.split()
    chunks = []
    if len(words) <= chunk_word_size:
        return [text]
    step_size = chunk_word_size - overlap_words
    for i in range(0, len(words), step_size):
        chunks.append(" ".join(words[i : i + chunk_word_size]))
        if i + chunk_word_size >= len(words):
            break
    return chunks


def entity_in_chunk(entity_text, chunk):
    """Token-aware containment check: matches whole tokens only.

    Numbers and single letters are the norm here ("10", "C"), so a plain
    substring test would count "10" as present in "$100K". Boundaries are
    character-class aware (see `bounded_pattern`), which also lets the "p" of
    a fused "210p" count as present.
    """
    return re.search(bounded_pattern(entity_text), chunk) is not None


def task_to_samples(task):
    """Converts one Label Studio task into one or more chunked training samples."""
    full_text = task["data"]["text"]
    results = task["annotations"][0].get("result", [])

    doc_entities = {}
    for r in results:
        if r.get("type") != "labels":
            continue
        val = r["value"]
        label = val["labels"][0]
        entity_text = full_text[val["start"] : val["end"]]
        doc_entities.setdefault(label, set()).add(entity_text)

    samples = []
    for chunk in chunk_text_with_overlap(full_text):
        chunk_entities = {}
        for label, entity_set in doc_entities.items():
            valid_ents = sorted(e for e in entity_set if entity_in_chunk(e, chunk))
            if valid_ents:
                chunk_entities[label] = valid_ents

        samples.append(
            {
                "input": chunk,
                "output": {
                    "entities": chunk_entities,
                    "entity_descriptions": ENTITY_DESCRIPTIONS,
                    # Keeps empty-entity chunks from being dropped by the trainer
                    # so the model sees real negatives.
                    "classifications": [
                        {
                            "task": "valid",
                            "labels": ["yes"],
                            "true_label": ["yes"],
                        }
                    ],
                },
            }
        )
    return samples


def _load_tasks(folder_path):
    """Yields (task, source_file) for every usable annotation in a folder."""
    if not folder_path or not os.path.isdir(folder_path):
        return
    for fp in glob.glob(os.path.join(folder_path, "*.json")):
        with open(fp, "r", encoding="utf-8") as f:
            ls_data = json.load(f)
        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue
            yield task, fp


def parse_all_labeled_data(
    labeled_folder,
    augmented_folder=None,
    test_folder=None,
    val_fraction=VAL_FRACTION,
    seed=SEED,
):
    """Loads originals from `labeled_folder` and (optionally) augmented variants
    from `augmented_folder`, splitting by source task id so augmented copies of
    validation tasks never leak into training.

    If `test_folder` is given, every task id present there is excluded from
    train and val — including augmented duplicates that carry the same id —
    so the held-out test set never contaminates training.
    """
    test_ids = set()
    if test_folder and os.path.isdir(test_folder):
        for task, _ in _load_tasks(test_folder):
            test_ids.add(task["id"])

    originals = [
        (t, fp) for t, fp in _load_tasks(labeled_folder) if t["id"] not in test_ids
    ]

    rng = random.Random(seed)
    ids = sorted({task["id"] for task, _ in originals})
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_fraction)) if ids else 0
    val_ids = set(ids[:n_val])
    train_ids = set(ids[n_val:])

    train_samples, val_samples = [], []
    for task, _ in originals:
        tid = task["id"]
        if tid in val_ids:
            val_samples.extend(task_to_samples(task))
        elif tid in train_ids:
            train_samples.extend(task_to_samples(task))

    # Augmented: training only, and only for tasks whose source is in train_ids.
    for task, _ in _load_tasks(augmented_folder):
        tid = task["id"]
        if tid not in test_ids and tid in train_ids:
            train_samples.extend(task_to_samples(task))

    return train_samples, val_samples


def _git_commit():
    """Best-effort: return current git HEAD, or None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        )
        return out.decode().strip()
    except Exception:
        return None


def _summarise_labeled_file(fp):
    """Read a Label Studio export, return tasks/entities/sha1 summary."""
    with open(fp, "rb") as f:
        raw = f.read()
    sha1 = hashlib.sha1(raw).hexdigest()[:12]
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    n_tasks = 0
    n_ents = 0
    for t in data if isinstance(data, list) else []:
        if not t.get("annotations") or t["annotations"][0].get("was_cancelled"):
            continue
        n_tasks += 1
        n_ents += sum(
            1
            for r in t["annotations"][0].get("result", [])
            if r.get("type") == "labels"
        )
    return {
        "name": os.path.basename(fp),
        "tasks": n_tasks,
        "entities": n_ents,
        "sha1": sha1,
    }


def _summarise_folder(folder):
    files = []
    if os.path.isdir(folder):
        for fp in sorted(glob.glob(os.path.join(folder, "*.json"))):
            info = _summarise_labeled_file(fp)
            if info:
                files.append(info)
    return files


def gather_training_metadata(
    config,
    train_count,
    val_count,
    effective_batch_size,
    labeled_folder,
    test_folder,
    augmented_folder,
):
    """Snapshot everything needed to reproduce this training run.

    Written next to the adapter as `training_metadata.json` so benchmark.py
    can later merge it into the per-adapter params it persists — keeping
    full training context attached to each version even when re-benchmarked
    months later.
    """
    labeled_files = _summarise_folder(labeled_folder)
    test_files = _summarise_folder(test_folder)
    augmented_files = (
        len(glob.glob(os.path.join(augmented_folder, "*.json")))
        if augmented_folder and os.path.isdir(augmented_folder)
        else 0
    )

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "seed": SEED,
        "config": {
            "num_epochs": config.num_epochs,
            "batch_size": config.batch_size,
            "effective_batch_size": effective_batch_size,
            "encoder_lr": config.encoder_lr,
            "task_lr": config.task_lr,
            "max_grad_norm": config.max_grad_norm,
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,
            "early_stopping": config.early_stopping,
            "early_stopping_patience": getattr(config, "early_stopping_patience", None),
            "val_fraction": VAL_FRACTION,
        },
        "data": {
            "labeled_files": labeled_files,
            "test_files": test_files,
            "test_held_out_tasks": sum(f["tasks"] for f in test_files),
            "train_samples": train_count,
            "val_samples": val_count,
        },
        "augmentation": {"augmented_files": augmented_files},
    }


def main():
    import torch
    from gliner2 import GLiNER2
    from gliner2.training.trainer import GLiNER2Trainer, TrainingConfig

    from src.core.benchmark import (
        BASE_MODEL_ID,
        benchmark_adapter,
        locate_adapter_weights,
    )
    from src.maintenance.split_test_set import run as refresh_test_split

    set_seed(SEED)

    # Refresh the held-out test split before loading training data so newly
    # labeled files get their slice held out. Deterministic via SEED.
    console.print("[bold cyan]Refreshing stratified test split...[/bold cyan]")
    refresh_test_split(seed=SEED)

    next_version = get_next_version()
    adapter_name = f"{ADAPTER_PREFIX}_v{next_version}"
    output_dir = f"./models/{adapter_name}"

    console.print(
        f"[bold cyan]Initializing Training Run for: v{next_version}[/bold cyan]"
    )

    labeled_folder = "data/labeled"
    augmented_folder = "data/augmented"
    test_folder = "data/test"
    train_data, val_data = parse_all_labeled_data(
        labeled_folder, augmented_folder, test_folder=test_folder
    )
    console.print(
        f"[bold green]Train: {len(train_data)} samples | Val: {len(val_data)} samples[/bold green]"
    )

    base_model = GLiNER2.from_pretrained(BASE_MODEL_ID)
    model = torch.compile(base_model)

    batch_size = 4
    effective_batch_size = batch_size * 2
    lora_rank = 32

    config = TrainingConfig(
        output_dir=output_dir,
        experiment_name=f"options_lora_v{next_version}",
        num_epochs=EPOCHS,
        batch_size=batch_size,
        max_len=MAX_LEN,
        gradient_accumulation_steps=effective_batch_size // batch_size,
        encoder_lr=2e-5,
        task_lr=5e-4,
        max_grad_norm=1.0,
        use_lora=True,
        lora_r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.1,
        lora_target_modules=["encoder"],
        save_adapter_only=True,
        fp16=False,
        bf16=True,
        seed=SEED,
        early_stopping=EARLY_STOPPING,
        early_stopping_patience=5,
    )

    trainer = GLiNER2Trainer(model=model, config=config)
    trainer.train(train_data=train_data, eval_data=val_data)

    console.print(
        f"[bold green]v{next_version} Adapter trained and saved to {output_dir}/final/[/bold green]"
    )

    # Fix task_type in adapter_config.json for HuggingFace Hub compliance
    adapter_config_path = os.path.join(output_dir, "final", "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_config = json.load(f)
        adapter_config["task_type"] = "TOKEN_CLS"
        with open(adapter_config_path, "w", encoding="utf-8") as f:
            json.dump(adapter_config, f, indent=2)
        console.print("[cyan]Set task_type to TOKEN_CLS in adapter_config.json[/cyan]")

    metadata = gather_training_metadata(
        config,
        train_count=len(train_data),
        val_count=len(val_data),
        effective_batch_size=effective_batch_size,
        labeled_folder=labeled_folder,
        test_folder=test_folder,
        augmented_folder=augmented_folder,
    )
    metadata_path = os.path.join(output_dir, "training_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    console.print(f"[cyan]Wrote training metadata to {metadata_path}[/cyan]")

    adapter_final = locate_adapter_weights(output_dir)
    if adapter_final is None:
        console.print(
            f"[yellow]No adapter weights found under {output_dir}; "
            "skipping post-train benchmark.[/yellow]"
        )
        raise SystemExit(0)
    adapter_label = f"GLiNER2 Large + Adapter v{next_version}"
    console.print(f"[cyan]Benchmarking {adapter_label}...[/cyan]")
    metrics, test_hash = benchmark_adapter(adapter_label, adapter_final)
    overall = metrics["overall"]
    console.print(
        f"[bold]{adapter_label}[/bold] vs test set [yellow]{test_hash}[/yellow]: "
        f"P={overall['p']:.2%}  R={overall['r']:.2%}  F1={overall['f1']:.2%}"
    )


if __name__ == "__main__":
    main()
