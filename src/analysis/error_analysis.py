"""Break down where a trained adapter fails on the held-out test set.

Run:
    python -m src.analysis.error_analysis                       # latest adapter
    python -m src.analysis.error_analysis --adapter v4
    python -m src.analysis.error_analysis --adapter base
    python -m src.analysis.error_analysis --base-model gliner2.5-small-v1
    python -m src.analysis.error_analysis --threshold 0.5 --top 30
    python -m src.analysis.error_analysis --save-json out.json  # full dump

Scoring is **exact-span** (see `src/core/benchmark.py`). Errors are bucketed by
how the prediction relates to the nearest gold span:

  1. **Label confusions** — same span, different label (e.g. a strike predicted
     as a price). Means the descriptions don't separate the two labels.
  2. **Boundary errors** — same label, overlapping but not identical span
     (e.g. "$350" vs "350"). Usually an annotation-policy question.
  3. **False positives** — predicted span with no gold counterpart. Highest
     leverage signal for hard-negative mining.
  4. **False negatives** — gold span the model never predicted.

Plus a per-document hotspot table — top docs by error count.
"""

import argparse
import json
from collections import Counter

from rich.console import Console
from rich.table import Table

from src.core.models import DEFAULT_BASE, base_label, load_adapted, resolve_base
from src.core.benchmark import (
    DEFAULT_LABELS,
    DEFAULT_TEST_FOLDER,
    collect_pred_per_doc,
    get_all_adapters,
    load_base_model,
    parse_all_label_studio_exports,
    prepare_eval_inputs,
    run_inference,
)

console = Console()


def resolve_adapter(spec, base=DEFAULT_BASE):
    """Resolve a --adapter spec ('latest', 'base', 'v4', etc.) for `base`.

    Returns ``(name, path)``. `base` returns ``(..., None)`` so the caller knows
    to skip adapter loading.
    """
    if spec == "base":
        return (base_label(base), None)
    adapters = get_all_adapters(bases=[base])
    if not adapters:
        raise SystemExit(f"No adapters found under ./models for {base}.")
    if spec in (None, "latest"):
        a = adapters[-1]
        return (a["name"], a["path"])
    target = int(spec.lstrip("v"))
    for a in adapters:
        if a["version"] == target:
            return (a["name"], a["path"])
    raise SystemExit(
        f"Adapter v{target} not found. Available: {[a['version'] for a in adapters]}"
    )


def make_context(text, start, end, n_chars):
    """Return a window of text around (start, end) with the entity bracketed."""
    left = max(0, start - n_chars)
    right = min(len(text), end + n_chars)
    prefix = "..." if left > 0 else ""
    suffix = "..." if right < len(text) else ""
    snippet = (
        text[left:start] + "[" + text[start:end] + "]" + text[end:right]
    ).replace("\n", " ")
    return f"{prefix}{snippet}{suffix}"


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def categorize_errors(pred_per_doc, gold_per_doc, dataset, context_chars=40):
    """Split per-document span errors into confusion / boundary / FP / FN buckets."""
    out = {"confusion": [], "boundary": [], "pure_fp": [], "pure_fn": [], "per_doc": []}

    for doc_idx, (pred, gold) in enumerate(zip(pred_per_doc, gold_per_doc)):
        text = dataset[doc_idx]["text"]
        fps = set(pred - gold)
        fns = set(gold - pred)
        n_fp, n_fn = len(fps), len(fns)

        def record(span, **extra):
            return {
                "doc_idx": doc_idx,
                "text": text[span[0] : span[1]],
                "context": make_context(text, span[0], span[1], context_chars),
                **extra,
            }

        # 1. Same span, different label.
        for fp in sorted(fps):
            for fn in sorted(fns):
                if fp[:2] == fn[:2]:
                    out["confusion"].append(
                        record(fp, gold_label=fn[2], pred_label=fp[2])
                    )
                    fps.discard(fp)
                    fns.discard(fn)
                    break

        # 2. Same label, overlapping boundaries.
        for fp in sorted(fps):
            match = next(
                (fn for fn in sorted(fns) if fn[2] == fp[2] and _overlaps(fp, fn)),
                None,
            )
            if match:
                out["boundary"].append(
                    record(fp, label=fp[2], gold_text=text[match[0] : match[1]])
                )
                fps.discard(fp)
                fns.discard(match)

        # 3 & 4. Everything left is a pure FP / FN.
        out["pure_fp"] += [record(s, label=s[2]) for s in sorted(fps)]
        out["pure_fn"] += [record(s, label=s[2]) for s in sorted(fns)]
        out["per_doc"].append(
            {
                "doc_idx": doc_idx,
                "n_errors": n_fp + n_fn,
                "n_fp": n_fp,
                "n_fn": n_fn,
                "preview": text[:80].replace("\n", " "),
            }
        )
    return out


def _aggregate(records, key_fields, top):
    """Bucket records by `key_fields`, return [(key_tuple, count, example_record), ...]."""
    counter = Counter()
    examples = {}
    for r in records:
        key = tuple(r[f] for f in key_fields)
        counter[key] += 1
        examples.setdefault(key, r)
    n = top if (top and top > 0) else None
    return [(key, count, examples[key]) for key, count in counter.most_common(n)]


def render_table(title, agg, columns):
    """Render aggregated rows: count, each key field, then an example context."""
    table = Table(title=title, show_lines=False)
    table.add_column("#", justify="right", style="dim", width=4)
    for i, col in enumerate(columns):
        table.add_column(col, style="bold" if i == 0 else None)
    table.add_column("example context")
    for key, count, ex in agg:
        table.add_row(str(count), *key, ex["context"])
    return table


def render_doc_hotspots(per_doc, top):
    n = top if (top and top > 0) else None
    table = Table(title=f"Top {n or 'all'} documents by error count", show_lines=False)
    table.add_column("doc", justify="right", style="dim", width=5)
    table.add_column("errors", justify="right")
    table.add_column("FP", justify="right", style="yellow")
    table.add_column("FN", justify="right", style="red")
    table.add_column("preview")
    for d in sorted(per_doc, key=lambda d: d["n_errors"], reverse=True)[:n]:
        if d["n_errors"] == 0:
            continue
        table.add_row(
            str(d["doc_idx"]),
            str(d["n_errors"]),
            str(d["n_fp"]),
            str(d["n_fn"]),
            d["preview"],
        )
    return table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        default="latest",
        help="'latest' (default), 'base', or a version like 'v4' / '4'.",
    )
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE,
        help="Base model slug or Hub id (default: %(default)s).",
    )
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument(
        "--top",
        type=int,
        default=50,
        help="Max rows per error table (default 50; use 0 to show all).",
    )
    parser.add_argument("--context", type=int, default=40)
    parser.add_argument("--test-folder", default=DEFAULT_TEST_FOLDER)
    parser.add_argument("--save-json", default=None)
    args = parser.parse_args()

    base = resolve_base(args.base_model)
    adapter_name, adapter_path = resolve_adapter(args.adapter, base)
    dataset = parse_all_label_studio_exports(args.test_folder)
    if not dataset:
        console.print(f"[red]No data in {args.test_folder}.[/red]")
        raise SystemExit(1)

    label_keys = list(DEFAULT_LABELS)
    flat_chunks, doc_chunk_ranges, gold_per_doc, _ = prepare_eval_inputs(
        dataset, label_keys
    )
    total_gold = sum(len(g) for g in gold_per_doc)
    console.print(
        f"Adapter: [bold cyan]{adapter_name}[/bold cyan] @ threshold={args.threshold}\n"
        f"Test set: [bold green]{len(dataset)}[/bold green] docs, "
        f"[bold green]{total_gold}[/bold green] gold spans, "
        f"[bold green]{len(flat_chunks)}[/bold green] chunks"
    )

    base_model, device = load_base_model(base)
    if adapter_path:
        model = load_adapted(base_model, adapter_path)
    else:
        model = base_model

    outputs = run_inference(model, flat_chunks, DEFAULT_LABELS, args.threshold)
    pred_per_doc = collect_pred_per_doc(outputs, flat_chunks, doc_chunk_ranges)
    categories = categorize_errors(pred_per_doc, gold_per_doc, dataset, args.context)

    total_pred = sum(len(p) for p in pred_per_doc)
    n_tp = sum(len(p & g) for p, g in zip(pred_per_doc, gold_per_doc))
    p = n_tp / total_pred if total_pred else 0.0
    r = n_tp / total_gold if total_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    console.print(
        f"\n[bold]TP={n_tp}  FP={total_pred - n_tp}  FN={total_gold - n_tp}[/bold]   "
        f"P={p:.2%}  R={r:.2%}  F1={f1:.2%}   [dim](exact span)[/dim]"
    )
    console.print(
        f"  confusion: {len(categories['confusion'])}  |  "
        f"boundary: {len(categories['boundary'])}  |  "
        f"pure FP: {len(categories['pure_fp'])}  |  "
        f"pure FN: {len(categories['pure_fn'])}\n"
    )

    top_label = args.top if args.top and args.top > 0 else "all"
    sections = [
        ("confusion", ("text", "gold_label", "pred_label"), "Label confusions"),
        ("boundary", ("text", "gold_text", "label"), "Boundary errors"),
        ("pure_fp", ("text", "label"), "False positives"),
        ("pure_fn", ("text", "label"), "False negatives"),
    ]
    for bucket, fields, title in sections:
        if categories[bucket]:
            agg = _aggregate(categories[bucket], fields, args.top)
            console.print(render_table(f"Top {top_label} {title}", agg, fields))
    console.print(render_doc_hotspots(categories["per_doc"], args.top))

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "adapter": adapter_name,
                    "threshold": args.threshold,
                    "test_folder": args.test_folder,
                    "summary": {"tp": n_tp, "p": p, "r": r, "f1": f1},
                    "categories": categories,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        console.print(f"[green]Full error data written to {args.save_json}[/green]")

    if device == "cuda" and adapter_path:
        import torch

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
