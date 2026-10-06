"""Sweep the decision threshold across all adapters on the held-out test set.

Run:
    python -m src.analysis.threshold_sweep                       # every base model
    python -m src.analysis.threshold_sweep --base-model gliner2.5-small-v1

Prints one Rich table per adapter showing per-label F1 and overall
precision / recall / F1 at each threshold, with the F1-optimal row marked.
Intended for picking the operating point after `benchmark.py` has selected an
adapter version. A less-overfit adapter has flatter confidence distributions
and is unfairly penalised by the default 0.75 threshold.
"""

import argparse
import os

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from src.core.models import base_label, load_adapted, resolve_bases
from src.core.benchmark import (
    DEFAULT_LABELS,
    DEFAULT_TEST_FOLDER,
    collect_pred_per_doc,
    get_all_adapters,
    load_base_model,
    parse_all_label_studio_exports,
    prepare_eval_inputs,
    run_inference,
    score_predictions,
)

console = Console()

# Inclusive range — 0.30 to 0.90 in 0.05 steps. Below 0.30 noise dominates.
THRESHOLDS = [round(0.30 + 0.05 * i, 2) for i in range(13)]
BATCH_SIZE = 32


def sweep_adapter(
    model, flat_chunks, doc_chunk_ranges, gold_per_doc, thresholds, progress_ctx=None
):
    """Returns list of (threshold, scores_dict) for one model across all thresholds."""
    label_keys = list(DEFAULT_LABELS)
    progress, task_id = progress_ctx if progress_ctx else (None, None)
    rows = []
    for t in thresholds:
        outputs = run_inference(model, flat_chunks, DEFAULT_LABELS, t, BATCH_SIZE)
        preds = collect_pred_per_doc(outputs, flat_chunks, doc_chunk_ranges)
        rows.append((t, score_predictions(preds, gold_per_doc, label_keys)))
        if progress and task_id is not None:
            progress.update(task_id, advance=1)
    return rows


def _render(name, rows):
    table = Table(title=f"{name} — threshold sweep", show_lines=False)
    table.add_column("threshold", justify="right")
    for label in DEFAULT_LABELS:
        table.add_column(f"{label} F1", justify="right")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("overall F1", style="bold magenta", justify="right")

    best_f1 = max(r[1]["overall"]["f1"] for r in rows)
    for t, scores in rows:
        o = scores["overall"]
        marker = " [bold green]*[/bold green]" if abs(o["f1"] - best_f1) < 1e-9 else ""
        table.add_row(
            f"{t:.2f}",
            *(f"{scores[label]['f1']:.2%}" for label in DEFAULT_LABELS),
            f"{o['p']:.2%}",
            f"{o['r']:.2%}",
            f"{o['f1']:.2%}{marker}",
        )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-model",
        default="all",
        help="Slug, Hub id, comma-separated list, or 'all' (default).",
    )
    args = parser.parse_args()

    dataset = parse_all_label_studio_exports(DEFAULT_TEST_FOLDER)
    if not dataset:
        console.print(f"[red]No test data in {DEFAULT_TEST_FOLDER}.[/red]")
        raise SystemExit(1)

    flat_chunks, doc_chunk_ranges, gold_per_doc, _ = prepare_eval_inputs(
        dataset, list(DEFAULT_LABELS)
    )
    console.print(
        f"Test set: [bold green]{len(dataset)}[/bold green] docs, "
        f"[bold green]{len(flat_chunks)}[/bold green] chunks, "
        f"thresholds: {THRESHOLDS[0]}..{THRESHOLDS[-1]} ({len(THRESHOLDS)} steps)"
    )

    bases = resolve_bases(args.base_model)
    adapters = get_all_adapters(bases=bases)

    configs = []  # (base, name, adapter_path)
    for base in bases:
        configs.append((base, base_label(base), None))
        configs += [
            (a["base"], a["name"], a["path"]) for a in adapters if a["base"] == base
        ]

    all_results = []
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    ) as progress:
        overall_task = progress.add_task(
            "[bold cyan]Sweeping...", total=len(configs) * len(THRESHOLDS)
        )
        for base in bases:
            base_model, device = load_base_model(base)
            for _, name, adapter_path in [c for c in configs if c[0] == base]:
                if adapter_path and os.path.exists(adapter_path):
                    model = load_adapted(base_model, adapter_path)
                else:
                    model = base_model

                rows = sweep_adapter(
                    model,
                    flat_chunks,
                    doc_chunk_ranges,
                    gold_per_doc,
                    THRESHOLDS,
                    progress_ctx=(progress, overall_task),
                )
                all_results.append((name, rows))
                if model is not base_model:
                    del model
            del base_model
            if device == "cuda":
                import torch

                torch.cuda.empty_cache()

    for name, rows in all_results:
        console.print(_render(name, rows))

    summary = Table(title="Best-F1 operating point per model", show_lines=False)
    summary.add_column("Model", style="cyan")
    summary.add_column("best threshold", justify="right")
    summary.add_column("P", justify="right")
    summary.add_column("R", justify="right")
    summary.add_column("F1", style="bold magenta", justify="right")
    ranked = []
    for name, rows in all_results:
        best_t, best_scores = max(rows, key=lambda r: r[1]["overall"]["f1"])
        ranked.append((name, best_t, best_scores["overall"]))
    ranked.sort(key=lambda r: r[2]["f1"], reverse=True)
    for name, t, o in ranked:
        summary.add_row(
            name, f"{t:.2f}", f"{o['p']:.2%}", f"{o['r']:.2%}", f"{o['f1']:.2%}"
        )
    console.print(summary)
    console.print(
        "[dim]Use the best threshold per model: it goes into that adapter's "
        "recognizer_config.json (push_model_to_hf --threshold).[/dim]"
    )


if __name__ == "__main__":
    main()
