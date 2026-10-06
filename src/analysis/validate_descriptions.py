"""Compare entity-description variants on the held-out test set without retraining.

GLiNER2 uses entity descriptions at inference time, so we can measure whether
edited descriptions reduce FPs before committing to a full training run. The
model is loaded once; inference runs twice (baseline vs candidate).

The candidate is `ENTITY_DESCRIPTIONS` in `src/core/labels.py`. The baseline is
read from a JSON file of ``{label: description}`` (default
`data/baseline_descriptions.json`); freeze a copy of the descriptions you last
trained with there before experimenting.

Usage:
    python -m src.analysis.validate_descriptions                   # latest adapter
    python -m src.analysis.validate_descriptions --adapter v2
    python -m src.analysis.validate_descriptions --adapter base    # no adapter
    python -m src.analysis.validate_descriptions --base-model gliner2.5-small-v1
    python -m src.analysis.validate_descriptions --baseline my.json
    python -m src.analysis.validate_descriptions --show-deltas --top 30
"""

import argparse
import json
import os
from collections import Counter

from rich.console import Console
from rich.table import Table

from src.analysis.error_analysis import make_context, resolve_adapter
from src.core.benchmark import (
    DEFAULT_TEST_FOLDER,
    collect_pred_per_doc,
    load_base_model,
    parse_all_label_studio_exports,
    prepare_eval_inputs,
    run_inference,
    score_predictions,
)
from src.core.labels import ENTITY_DESCRIPTIONS
from src.core.models import DEFAULT_BASE, load_adapted, resolve_base

console = Console()

DEFAULT_BASELINE = "data/baseline_descriptions.json"


def load_baseline(path):
    """Read the baseline descriptions JSON, or exit with a how-to if missing."""
    if not os.path.exists(path):
        raise SystemExit(
            f"No baseline descriptions at {path}. Save the current ones first:\n"
            '  python -c "import json; from src.core.labels import '
            f"ENTITY_DESCRIPTIONS as d; json.dump(d, open('{path}', 'w'), indent=2)\""
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _delta(old_val, new_val, higher_is_better=True, fmt=".2%"):
    diff = new_val - old_val
    if abs(diff) < (0.5 if fmt == "d" else 1e-5):
        return "[dim]—[/dim]"
    sign = "+" if diff > 0 else ""
    color = "green" if (diff > 0) == higher_is_better else "red"
    return f"[{color}]{sign}{diff:{fmt}}[/{color}]"


def _with_counts(pred_per_doc, gold_per_doc, label_keys):
    """`score_predictions` plus raw tp/fp/fn counts, for the comparison table."""
    metrics = score_predictions(pred_per_doc, gold_per_doc, label_keys)
    for key in metrics:
        sel = (lambda e: True) if key == "overall" else (lambda e, k=key: e[2] == k)
        tp = fp = fn = 0
        for pred, gold in zip(pred_per_doc, gold_per_doc):
            p = {e for e in pred if sel(e)}
            g = {e for e in gold if sel(e)}
            tp += len(p & g)
            fp += len(p - g)
            fn += len(g - p)
        metrics[key].update(tp=tp, fp=fp, fn=fn)
    return metrics


def _render_comparison(old_m, new_m, label_keys):
    table = Table(
        title="Entity description comparison (same model, different prompts)",
        show_lines=False,
    )
    table.add_column("entity", style="blue", width=12)
    table.add_column("metric", width=14)
    table.add_column("baseline", justify="right")
    table.add_column("candidate", justify="right")
    table.add_column("Δ", justify="right", width=10)

    rows = [
        ("P (precision)", "p", True, ".2%"),
        ("R (recall)", "r", True, ".2%"),
        ("F1", "f1", True, ".2%"),
        ("FP count", "fp", False, "d"),
        ("FN count", "fn", False, "d"),
    ]
    for key in [*label_keys, "overall"]:
        o, n = old_m[key], new_m[key]
        label_cell = f"[bold white]{key}[/bold white]" if key == "overall" else key
        for i, (metric_label, field, hib, fmt) in enumerate(rows):
            table.add_row(
                label_cell if i == 0 else "",
                metric_label,
                f"{o[field]:{fmt}}",
                f"{n[field]:{fmt}}",
                _delta(o[field], n[field], higher_is_better=hib, fmt=fmt),
            )
        table.add_section()
    return table


def collect_delta_records(
    old_preds, new_preds, gold_per_doc, dataset, context_chars=40
):
    """Classify spans whose prediction state changed between the two variants.

    Returns four lists of dicts (text, label, doc_idx, context):
      suppressed_fp — baseline hallucinated it, candidate correctly skipped it
      new_fp        — candidate introduced a regression FP
      recovered_fn  — baseline missed it, candidate correctly found it
      lost_tp       — baseline found it, candidate now misses it
    """
    buckets = ([], [], [], [])

    for doc_idx, (old_pred, new_pred, gold) in enumerate(
        zip(old_preds, new_preds, gold_per_doc)
    ):
        text = dataset[doc_idx]["text"]
        changes = (
            old_pred - gold - (new_pred - gold),
            (new_pred - gold) - (old_pred - gold),
            (gold - old_pred) - (gold - new_pred),
            (gold - new_pred) - (gold - old_pred),
        )
        for bucket, spans in zip(buckets, changes):
            for s, e, label in spans:
                bucket.append(
                    {
                        "text": text[s:e],
                        "label": label,
                        "doc_idx": doc_idx,
                        "context": make_context(text, s, e, context_chars),
                    }
                )
    return buckets


def _render_delta_table(title, records, top):
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("text", style="bold")
    table.add_column("label")
    table.add_column("example context")
    counter = Counter()
    examples = {}
    for r in records:
        key = (r["text"], r["label"])
        counter[key] += 1
        examples.setdefault(key, r)
    for (text, label), count in counter.most_common(top):
        table.add_row(str(count), text, label, examples[(text, label)]["context"])
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="latest")
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-folder", default=DEFAULT_TEST_FOLDER)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--show-deltas", action="store_true")
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    baseline = load_baseline(args.baseline)
    base = resolve_base(args.base_model)
    adapter_name, adapter_path = resolve_adapter(args.adapter, base)
    dataset = parse_all_label_studio_exports(args.test_folder)
    if not dataset:
        console.print(f"[red]No annotated tasks found in {args.test_folder}[/red]")
        raise SystemExit(1)

    label_keys = list(ENTITY_DESCRIPTIONS)
    flat_chunks, doc_chunk_ranges, gold_per_doc, _ = prepare_eval_inputs(
        dataset, label_keys
    )
    console.print(
        f"Adapter : [bold cyan]{adapter_name}[/bold cyan]\n"
        f"Test set: {len(dataset)} docs | {sum(len(g) for g in gold_per_doc)} gold "
        f"spans | {len(flat_chunks)} chunks | threshold={args.threshold}\n"
    )

    base_model, device = load_base_model(base)
    if adapter_path:
        model = load_adapted(base_model, adapter_path)
        del base_model
    else:
        model = base_model

    runs = {}
    for variant, descriptions in [
        ("baseline", baseline),
        ("candidate", ENTITY_DESCRIPTIONS),
    ]:
        console.print(
            f"[cyan]Inference with [bold]{variant}[/bold] descriptions...[/cyan]"
        )
        outputs = run_inference(
            model, flat_chunks, descriptions, args.threshold, args.batch_size
        )
        preds = collect_pred_per_doc(outputs, flat_chunks, doc_chunk_ranges)
        runs[variant] = (preds, _with_counts(preds, gold_per_doc, label_keys))

    console.print()
    console.print(
        _render_comparison(runs["baseline"][1], runs["candidate"][1], label_keys)
    )

    if args.show_deltas:
        suppressed_fp, new_fp, recovered_fn, lost_tp = collect_delta_records(
            runs["baseline"][0], runs["candidate"][0], gold_per_doc, dataset
        )
        if not (suppressed_fp or new_fp or recovered_fn or lost_tp):
            console.print("[dim]No prediction differences between variants.[/dim]")
        for title, records in [
            (
                "Suppressed FPs — baseline hallucinated, candidate skipped",
                suppressed_fp,
            ),
            ("New FPs — regressions introduced by candidate", new_fp),
            ("Recovered FNs — candidate found previously-missed spans", recovered_fn),
            ("Lost TPs — regressions: baseline found, candidate misses", lost_tp),
        ]:
            if records:
                console.print(
                    _render_delta_table(f"{title} ({len(records)})", records, args.top)
                )

    if device == "cuda":
        import torch

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
