"""Benchmark GLiNER2 adapters on the held-out test set.

    python -m src.core.benchmark              # base model + latest adapter
    python -m src.core.benchmark --no-cache   # ignore cached results

Scoring is **exact-span**: a prediction is a true positive only if its absolute
``(start, end, label)`` matches a gold span. Options tweets repeat short
numbers ("10", "350") for different roles within one post, so the per-document
surface-form dedup used for ticker extraction would hide real errors here.

Results are cached per ``(adapter, test_set_hash)`` in ``models/benchmark_results.json``.
"""

import argparse
import copy
import glob
import json
import os
import re

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.core.labels import ENTITY_DESCRIPTIONS
from src.core.results_store import (
    compute_test_set_hash,
    derive_adapter_params,
    get_cached,
    load_store,
    put_result,
    register_test_set,
    save_store,
)

DEFAULT_TEST_FOLDER = "data/test"
DEFAULT_LABELS = ENTITY_DESCRIPTIONS
ADAPTER_PREFIX = "options_adapter"
BASE_MODEL_ID = "fastino/gliner2-large-v1"

console = Console()


def locate_adapter_weights(adapter_dir):
    """Pick which checkpoint subfolder of an adapter dir to evaluate.

    Prefers ``best/`` (the early-stopping optimal) when present, falling back
    to ``final/`` (the last-step save). Returns ``None`` if neither has a
    safetensors file.
    """
    for sub in ("best", "final"):
        candidate = os.path.join(adapter_dir, sub)
        if os.path.exists(os.path.join(candidate, "adapter_model.safetensors")):
            return candidate
    return None


def get_all_adapters(models_dir="./models"):
    """Scans the models directory and returns a sorted list of all valid adapters."""
    if not os.path.exists(models_dir):
        return []

    valid_adapters = []
    for adapter_dir in glob.glob(os.path.join(models_dir, f"{ADAPTER_PREFIX}*")):
        weights_path = locate_adapter_weights(adapter_dir)
        if weights_path is None:
            continue

        folder_name = os.path.basename(adapter_dir)
        match = re.search(r"_v(\d+)$", folder_name)
        if match:
            v = int(match.group(1))
        elif folder_name == ADAPTER_PREFIX:
            v = 1
        else:
            continue

        valid_adapters.append(
            {
                "version": v,
                "name": f"GLiNER2 Large + Adapter v{v}",
                "path": weights_path,
            }
        )

    valid_adapters.sort(key=lambda x: x["version"])
    return valid_adapters


def parse_all_label_studio_exports(folder_path):
    """Parses all Label Studio JSON exports in a folder into clean GLiNER format."""
    if not os.path.isdir(folder_path):
        console.print(f"[red]Error: Folder {folder_path} not found.[/red]")
        return []

    clean_dataset = []
    files = glob.glob(os.path.join(folder_path, "*.json"))

    if not files:
        console.print(f"[red]No JSON files found in {folder_path}.[/red]")
        return clean_dataset

    for export_path in files:
        with open(export_path, "r", encoding="utf-8") as f:
            ls_data = json.load(f)

        for task in ls_data:
            if not task.get("annotations") or task["annotations"][0].get(
                "was_cancelled"
            ):
                continue

            entities = []
            for r in task["annotations"][0].get("result", []):
                if r.get("type") == "labels":
                    val = r["value"]
                    entities.append(
                        {
                            "start": val["start"],
                            "end": val["end"],
                            "label": val["labels"][0],
                        }
                    )

            clean_dataset.append({"text": task["data"]["text"], "entities": entities})

    return clean_dataset


def calculate_metrics(tp, fp, fn):
    """Helper to safely calculate P, R, F1"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if (precision + recall) > 0
        else 0
    )
    return {"p": precision, "r": recall, "f1": f1}


def chunk_text_for_inference(text, chunk_word_size=150, overlap_words=40):
    """
    Slices text into overlapping chunks, returning the chunk string and its
    absolute character start index so predictions can be mapped back accurately.
    """
    matches = list(re.finditer(r"\S+", text))
    chunks = []
    if not matches:
        return [(text, 0)]

    step_size = max(1, chunk_word_size - overlap_words)

    for i in range(0, len(matches), step_size):
        chunk_matches = matches[i : i + chunk_word_size]
        if not chunk_matches:
            break

        start_char = chunk_matches[0].start()
        end_char = chunk_matches[-1].end()
        chunks.append((text[start_char:end_char], start_char))

        if i + chunk_word_size >= len(matches):
            break

    return chunks


def prepare_eval_inputs(dataset, label_keys):
    """Build the per-chunk and per-document structures used by ``evaluate_model``.

    Factored out so the post-training benchmark in ``train.py`` and the analysis
    scripts share the exact same chunking/gold logic. Gold per document is a set
    of absolute ``(start, end, label)`` spans.
    """
    flat_chunks = []
    doc_chunk_ranges = []
    gold_per_doc = []
    gold_by_label_per_doc = []

    for doc_idx, entry in enumerate(dataset):
        start = len(flat_chunks)
        for chunk_text, offset in chunk_text_for_inference(entry["text"]):
            flat_chunks.append((doc_idx, chunk_text, offset))
        doc_chunk_ranges.append((start, len(flat_chunks)))

        gold = {(e["start"], e["end"], e["label"]) for e in entry["entities"]}
        gold_per_doc.append(gold)
        gold_by_label_per_doc.append(
            {label: {e for e in gold if e[2] == label} for label in label_keys}
        )

    return flat_chunks, doc_chunk_ranges, gold_per_doc, gold_by_label_per_doc


def run_inference(
    model,
    flat_chunks,
    label_descriptions,
    threshold,
    batch_size=32,
    progress_context=None,
):
    """Batched inference over all chunks, returned in the original chunk order.

    Chunks are sorted by length so each batch holds similarly-sized inputs,
    which cuts padding waste when lengths are uneven.
    """
    progress, task_id = progress_context if progress_context else (None, None)
    n = len(flat_chunks)
    order = sorted(range(n), key=lambda i: len(flat_chunks[i][1]))
    sorted_texts = [flat_chunks[i][1] for i in order]
    sorted_outputs = [None] * n
    for i in range(0, n, batch_size):
        batch = sorted_texts[i : i + batch_size]
        outputs = model.batch_extract_entities(
            batch,
            label_descriptions,
            batch_size=batch_size,
            threshold=threshold,
            include_spans=True,
        )
        for j, out in enumerate(outputs):
            sorted_outputs[i + j] = out
        if progress and task_id is not None:
            progress.update(task_id, advance=len(batch))

    all_outputs = [None] * n
    for sorted_idx, original_idx in enumerate(order):
        all_outputs[original_idx] = sorted_outputs[sorted_idx]
    return all_outputs


def iter_chunk_items(raw):
    """Yield chunk-local ``(start, end, label)`` for every entity in one model output."""
    if isinstance(raw, dict) and "entities" in raw:
        for label, items in raw["entities"].items():
            for item in items:
                yield item["start"], item["end"], label
    elif isinstance(raw, list):
        for item in raw:
            yield item["start"], item["end"], item["label"]


def collect_pred_per_doc(all_outputs, flat_chunks, doc_chunk_ranges):
    """Map chunk-local model outputs to a per-document set of absolute spans.

    Overlapping chunks can predict the same span twice; the set collapses them.
    """
    pred_per_doc = []
    for start, end in doc_chunk_ranges:
        pred = set()
        for chunk_idx in range(start, end):
            _, _chunk_text, offset = flat_chunks[chunk_idx]
            for s, e, label in iter_chunk_items(all_outputs[chunk_idx]):
                pred.add((offset + s, offset + e, label))
        pred_per_doc.append(pred)
    return pred_per_doc


def score_predictions(pred_per_doc, gold_per_doc, label_keys):
    """Exact-span TP/FP/FN -> P/R/F1, per label and overall."""
    counts = {k: {"tp": 0, "fp": 0, "fn": 0} for k in [*label_keys, "overall"]}
    for pred, gold in zip(pred_per_doc, gold_per_doc):
        groups = [("overall", pred, gold)]
        groups += [
            (
                label,
                {e for e in pred if e[2] == label},
                {e for e in gold if e[2] == label},
            )
            for label in label_keys
        ]
        for key, p, g in groups:
            counts[key]["tp"] += len(p & g)
            counts[key]["fp"] += len(p - g)
            counts[key]["fn"] += len(g - p)
    return {
        key: calculate_metrics(c["tp"], c["fp"], c["fn"]) for key, c in counts.items()
    }


def evaluate_model(
    model,
    flat_chunks,
    doc_chunk_ranges,
    gold_per_doc,
    gold_by_label_per_doc=None,
    model_name="Model",
    label_descriptions=None,
    batch_size=128,
    progress_context=None,
    threshold=0.75,
):
    """Run inference over the whole dataset and score it against gold spans.

    ``gold_by_label_per_doc`` is accepted for call-site symmetry with
    ``prepare_eval_inputs`` but unused: per-label gold is derived from
    ``gold_per_doc``.
    """
    labels = label_descriptions or DEFAULT_LABELS
    label_keys = list(labels)
    outputs = run_inference(
        model, flat_chunks, labels, threshold, batch_size, progress_context
    )
    preds = collect_pred_per_doc(outputs, flat_chunks, doc_chunk_ranges)
    return {"name": model_name, **score_predictions(preds, gold_per_doc, label_keys)}


def load_base_model(device=None):
    """Load the base GLiNER2 onto GPU (fp16) when available, else CPU."""
    import torch
    from gliner2 import GLiNER2

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GLiNER2.from_pretrained(
        BASE_MODEL_ID,
        map_location=device,
        quantize=(device == "cuda"),
    )
    return model, device


def _free_cuda(device):
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def _metrics_only(scores):
    """Strip the leading 'name' key from ``evaluate_model``'s output so the
    payload we persist matches the cached schema."""
    return {k: v for k, v in scores.items() if k != "name"}


def _load_training_metadata(adapter_path):
    """Look for `training_metadata.json` in the adapter's parent dir.

    Adapter paths from ``locate_adapter_weights`` look like
    ``models/options_adapter_v3/best`` or ``.../final``, and the metadata sits
    one level up. Returns None silently if the file isn't there.
    """
    if not adapter_path:
        return None
    metadata_path = os.path.join(
        os.path.dirname(adapter_path), "training_metadata.json"
    )
    if not os.path.exists(metadata_path):
        return None
    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _flatten_metadata_into_params(metadata):
    """Flatten the nested metadata dict into the flat `params` schema used by
    the benchmark store + display, while preserving the file inventories."""
    if not metadata:
        return {}
    flat = {}
    cfg = metadata.get("config", {}) or {}
    data = metadata.get("data", {}) or {}
    aug = metadata.get("augmentation", {}) or {}

    for k in (
        "num_epochs",
        "batch_size",
        "effective_batch_size",
        "encoder_lr",
        "task_lr",
        "max_grad_norm",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "early_stopping",
        "early_stopping_patience",
        "val_fraction",
    ):
        if cfg.get(k) is not None:
            flat[k] = cfg[k]
    for k in ("train_samples", "val_samples", "test_held_out_tasks"):
        if data.get(k) is not None:
            flat[k] = data[k]
    if aug.get("augmented_files") is not None:
        flat["augmented_files"] = aug["augmented_files"]
    for k in ("seed", "git_commit", "timestamp"):
        if metadata.get(k) is not None:
            flat[k] = metadata[k]
    for k in ("labeled_files", "test_files"):
        if data.get(k):
            flat[k] = data[k]
    return flat


def _build_params(adapter_path, training_params=None):
    """Weight-derived facts + persisted training context for one adapter."""
    params = dict(derive_adapter_params(adapter_path)) if adapter_path else {}
    metadata = _load_training_metadata(adapter_path) if adapter_path else None
    if metadata:
        params.update(_flatten_metadata_into_params(metadata))
    if training_params:  # explicit override wins
        params.update(training_params)
    return params


def benchmark_adapter(
    name,
    adapter_path,
    training_params=None,
    test_folder=DEFAULT_TEST_FOLDER,
    labels=None,
    base_model=None,
    device=None,
    batch_size=32,
):
    """Evaluate one model (``adapter_path=None`` for clean base), persist
    the result, and return ``(metrics, test_hash)``.

    Intended to be called from ``train.py`` directly after training so the
    new adapter's numbers land in the store without a full benchmark sweep.
    """
    labels = labels or DEFAULT_LABELS
    label_keys = list(labels)

    dataset = parse_all_label_studio_exports(test_folder)
    if not dataset:
        raise RuntimeError(f"No annotated tasks found in {test_folder}.")

    test_hash = compute_test_set_hash(dataset)
    store = load_store()
    register_test_set(store, test_hash, dataset, source_folder=test_folder)

    flat_chunks, doc_chunk_ranges, gold_per_doc, gold_by_label_per_doc = (
        prepare_eval_inputs(dataset, label_keys)
    )

    if base_model is None:
        base_model, device = load_base_model(device)
    elif device is None:
        device = next(base_model.parameters()).device.type

    if adapter_path and os.path.exists(adapter_path):
        model = copy.deepcopy(base_model)
        model.load_adapter(adapter_path)
    else:
        model = base_model

    scores = evaluate_model(
        model,
        flat_chunks,
        doc_chunk_ranges,
        gold_per_doc,
        gold_by_label_per_doc,
        model_name=name,
        label_descriptions=labels,
        batch_size=batch_size,
    )
    metrics = _metrics_only(scores)
    put_result(
        store,
        name,
        test_hash,
        metrics,
        params=_build_params(adapter_path, training_params),
    )
    save_store(store)

    if model is not base_model:
        del model
        _free_cuda(device)

    return metrics, test_hash


def _render_table(rows, label_keys=None):
    """Render a list of ``{"name", "metrics", "params"}`` dicts to a Rich table."""
    label_keys = label_keys or list(DEFAULT_LABELS)
    table = Table(title="NER Benchmark Breakdown (exact span)", show_lines=False)
    table.add_column("Model Configuration", style="cyan", width=35)
    table.add_column("Entity Type", style="blue")
    table.add_column("Precision (Noise)", justify="right")
    table.add_column("Recall (Detect)", justify="right")
    table.add_column("F1-Score", style="bold magenta", justify="right")

    for row in rows:
        m = row["metrics"]
        suffix = " [dim](cached)[/dim]" if row.get("cached") else ""
        first = True
        for label in label_keys:
            table.add_row(
                f"[bold]{row['name']}[/bold]{suffix}" if first else "",
                label,
                f"{m[label]['p']:.2%}",
                f"{m[label]['r']:.2%}",
                f"{m[label]['f1']:.2%}",
            )
            first = False
        table.add_row(
            "",
            "[bold white]OVERALL[/bold white]",
            f"[bold white]{m['overall']['p']:.2%}[/bold white]",
            f"[bold white]{m['overall']['r']:.2%}[/bold white]",
            f"[bold white]{m['overall']['f1']:.2%}[/bold white]",
        )
        table.add_section()
    return table


def _render_params(rows):
    """Side table summarising the non-metric facts we track per adapter."""
    table = Table(title="Adapter parameters", show_lines=False)
    table.add_column("Model", style="cyan", width=35)
    table.add_column("lora_b_norm", justify="right")
    table.add_column("size (KB)", justify="right")
    table.add_column("epochs", justify="right")
    table.add_column("train / val", justify="right")

    for row in rows:
        p = row.get("params") or {}
        train_n = p.get("train_samples")
        val_n = p.get("val_samples")
        split = (
            f"{train_n} / {val_n}" if train_n is not None and val_n is not None else "—"
        )
        table.add_row(
            row["name"],
            f"{p.get('lora_b_norm', '—')}",
            f"{p.get('adapter_size_kb', '—')}",
            f"{p.get('num_epochs', '—')}",
            split,
        )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-run inference even if cached results exist. "
        "Use when the model weights or evaluation logic change.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Benchmark every adapter instead of only the latest.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    dataset = parse_all_label_studio_exports(DEFAULT_TEST_FOLDER)
    if not dataset:
        console.print("[red]No valid data to evaluate.[/red]")
        raise SystemExit(1)

    test_hash = compute_test_set_hash(dataset)
    console.print(
        f"Loaded [bold green]{len(dataset)}[/bold green] annotated tasks "
        f"from {DEFAULT_TEST_FOLDER}. test-set hash: [yellow]{test_hash}[/yellow]"
    )

    store = load_store()
    register_test_set(store, test_hash, dataset, source_folder=DEFAULT_TEST_FOLDER)

    label_keys = list(DEFAULT_LABELS)
    flat_chunks, doc_chunk_ranges, gold_per_doc, gold_by_label_per_doc = (
        prepare_eval_inputs(dataset, label_keys)
    )
    console.print(
        f"Prepared [bold green]{len(flat_chunks)}[/bold green] chunks "
        f"across {len(dataset)} documents."
    )

    model_configs = [("Base Model (Clean)", None)]
    available_adapters = get_all_adapters()
    if not args.all:
        available_adapters = available_adapters[-1:]
    model_configs += [(a["name"], a["path"]) for a in available_adapters]

    rows = []
    to_evaluate = []
    for name, adapter_path in model_configs:
        cached = None if args.no_cache else get_cached(store, name, test_hash)
        if cached:
            rows.append(
                {
                    "name": name,
                    "metrics": cached["metrics"],
                    "params": cached.get("params") or {},
                    "cached": True,
                }
            )
        else:
            to_evaluate.append((name, adapter_path))

    if to_evaluate:
        shared_base_model, device = load_base_model()
        console.print(
            f"[cyan]Loaded base GLiNER2 onto [bold]{device}[/bold]. "
            f"Need to evaluate {len(to_evaluate)}/{len(model_configs)} configs.[/cyan]"
        )

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            overall_task = progress.add_task(
                "[bold cyan]Overall Evaluation...", total=len(to_evaluate)
            )
            for name, adapter_path in to_evaluate:
                if adapter_path and os.path.exists(adapter_path):
                    model = copy.deepcopy(shared_base_model)
                    model.load_adapter(adapter_path)
                else:
                    model = shared_base_model

                chunk_task = progress.add_task(
                    f"[green]Testing {name}...", total=len(flat_chunks)
                )
                scores = evaluate_model(
                    model,
                    flat_chunks,
                    doc_chunk_ranges,
                    gold_per_doc,
                    gold_by_label_per_doc,
                    model_name=name,
                    label_descriptions=DEFAULT_LABELS,
                    batch_size=args.batch_size,
                    progress_context=(progress, chunk_task),
                )
                metrics = _metrics_only(scores)
                params = _build_params(adapter_path)
                put_result(store, name, test_hash, metrics, params=params)
                rows.append(
                    {
                        "name": name,
                        "metrics": metrics,
                        "params": params,
                        "cached": False,
                    }
                )

                progress.update(overall_task, advance=1)
                progress.remove_task(chunk_task)
                if model is not shared_base_model:
                    del model
                    _free_cuda(device)

        save_store(store)

    order = [n for n, _ in model_configs]
    rows.sort(key=lambda r: order.index(r["name"]))
    console.print(_render_table(rows, label_keys))
    console.print(_render_params(rows))


if __name__ == "__main__":
    main()
