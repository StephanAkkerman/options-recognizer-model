"""Publish a trained adapter as its own Hugging Face model repo.

One repo is created per base model (``<owner>/options-recognizer-<base>`` (``-v1`` dropped))
so users pick a size by picking a repo. The uploaded folder is self-describing:
adapter weights, ``recognizer_config.json`` (base model, labels, threshold) and
a generated model card with the benchmark numbers.

    python -m utils.hf.push_model_to_hf models/options_adapter_gliner2-large-v1_v1 --owner user
    python -m utils.hf.push_model_to_hf models/options_adapter_gliner2.5-small-v1_v2 --owner user --threshold 0.6
    python -m utils.hf.push_model_to_hf --best --owner user   # best-F1 adapter of every base

Run ``python -m src.core.benchmark`` first so the card has numbers, and
``python -m src.analysis.threshold_sweep`` to pick `--threshold`.
"""

import argparse
import inspect
import json
import os
import shutil
import tempfile

from huggingface_hub import HfApi

from src.core.benchmark import get_all_adapters, locate_adapter_weights
from src.core.labels import ENTITY_DESCRIPTIONS
from src.core.models import (
    BASE_MODELS,
    adapter_label,
    default_repo_id,
    parse_adapter_dir,
)
from src.core.results_store import load_store
from src.core.spans import AlnumBoundarySplitter


DATASET_NAME = "options-ner"
GITHUB_URL = "https://github.com/StephanAkkerman/options-recognizer-model"

# Shown on every model card with the model's real output (see `run_examples`).
EXAMPLE_TEXTS = [
    "$AAPL 350 C 11/06/2026 $1.2M 4.37avg",
    "$NVTS - $162K Call buyer",
    "Bought SPY 600 puts expiring 12/19 @ .85, bearish into CPI",
    "Unusual flow: $TSLA $300 calls 01/16/2027 sweep, $2.5M premium, filled at 12.40",
    "Just a normal tweet about the market today, no trades.",
]


def latest_benchmark(store, name):
    """Most recently evaluated ``(test_hash, result)`` for `name`, or ``(None, None)``."""
    runs = store["results"].get(name, {})
    if not runs:
        return None, None
    test_hash = max(runs, key=lambda h: runs[h]["evaluated_at"])
    return test_hash, runs[test_hash]


def best_results(store, models_dir="models"):
    """``{base: (f1, adapter_folder, result)}`` for the highest overall-F1 adapter."""
    best = {}
    for a in get_all_adapters(models_dir):
        _, result = latest_benchmark(store, a["name"])
        if result is None:
            continue
        f1 = result["metrics"]["overall"]["f1"]
        if a["base"] not in best or f1 > best[a["base"]][0]:
            best[a["base"]] = (f1, os.path.dirname(a["path"]), result)
    return best


def best_adapter_dirs(store, models_dir="models"):
    """Highest overall-F1 adapter folder per base model, from the benchmark store."""
    return {b: path for b, (_, path, _) in best_results(store, models_dir).items()}


def sibling_rows(owner, store, models_dir="models"):
    """One row per base model with its best adapter's score, for the card's size table."""
    best = best_results(store, models_dir)
    rows = []
    for slug, info in BASE_MODELS.items():
        if slug not in best:
            continue
        _, _, result = best[slug]
        rows.append(
            {
                "repo_id": default_repo_id(owner, slug),
                "display": info["display"],
                "params_m": info["params_m"],
                "f1": result["metrics"]["overall"]["f1"],
                "ms": (result["metrics"].get("speed") or {}).get("ms_per_doc"),
            }
        )
    return rows


def run_examples(adapter_dir, slug, texts):
    """Run the staged adapter on `texts`; returns the non-empty entities per text.

    Needs ``gliner2>=2``. Runs the real model so the card shows what users get.
    """
    from src.core.models import load_adapted, load_extractor

    with open(
        os.path.join(adapter_dir, "recognizer_config.json"), "r", encoding="utf-8"
    ) as f:
        cfg = json.load(f)
    model = load_adapted(load_extractor(slug), adapter_dir)
    outputs = []
    for text in texts:
        entities = model.extract_entities(
            text, cfg["entity_descriptions"], threshold=cfg["threshold"]
        )["entities"]
        outputs.append({label: vals for label, vals in entities.items() if vals})
    return outputs


def build_model_card(
    slug,
    version,
    repo_id,
    threshold,
    result=None,
    test_hash=None,
    examples=None,
    siblings=None,
):
    """Render the model card (README.md) for one adapter.

    `examples` is a list of ``(text, entities)`` pairs and `siblings` the rows
    from `sibling_rows`; either may be omitted.
    """
    info = BASE_MODELS[slug]
    name = repo_id.split("/")[-1]
    dataset_id = f"{repo_id.split('/')[0]}/{DATASET_NAME}"
    lines = [
        "---",
        f"base_model: {info['hf_id']}",
        "library_name: gliner2",
        "pipeline_tag: token-classification",
        "language:",
        "- en",
        "license: mit",
        "datasets:",
        f"- {dataset_id}",
        "tags:",
        "- gliner2",
        "- lora",
        "- finance",
        "- options",
        "- options-trading",
        "- unusual-options-flow",
        "- twitter",
        "- fintwit",
        "- stocks",
        "- ner",
        "- information-extraction",
        "---",
        "",
        f"# {name}",
        "",
        f"LoRA adapter (v{version}) for [`{info['hf_id']}`](https://huggingface.co/{info['hf_id']}) "
        f"({info['params_m']}M parameters) that extracts option-contract details from "
        "tweets: ticker, strike, option type, expiry, premium and price.",
        "",
        "Smaller bases are faster, larger ones are more accurate — see the "
        "sibling `options-recognizer-*` repos for the other sizes.",
        "",
        f"Code, data pipeline and benchmark: [StephanAkkerman/options-recognizer-model]({GITHUB_URL}).",
        "",
        "## Intended uses",
        "",
        "Retail traders post option trades and unusual-options-flow alerts in a terse, "
        "slang-heavy format (`$AAPL 350 C 11/06 $1.2M 4.37avg`) that regular NER models "
        "and regexes handle poorly. This model turns such text into structured fields, "
        "which you can use to:",
        "",
        "- build a feed or dashboard of option flow from X/Twitter or Discord alerts,",
        "- collect option trades into a dataset for research or backtesting,",
        "- filter or rank alerts by premium, expiry or ticker,",
        "- pre-label more data for further training.",
        "",
        "It is built for English, informal, tweet-length text. It extracts the "
        "pieces of a contract as separate spans; it does not pair them up when a "
        "text mentions several contracts, and it is not financial advice.",
        "",
        "## Entities",
        "",
        "| label | what it captures |",
        "|---|---|",
    ]
    for label, desc in ENTITY_DESCRIPTIONS.items():
        lines.append(f"| `{label}` | {desc} |")
    lines.append("")

    if examples:
        lines += [
            "## Examples",
            "",
            f"Actual output of this model (threshold {threshold}); empty labels are omitted.",
            "",
        ]
        for text, entities in examples:
            lines += [
                "```text",
                f"Input:  {text}",
                f"Output: {json.dumps(entities, ensure_ascii=False)}",
                "```",
                "",
            ]

    if result:
        m = result["metrics"]
        speed = (m.get("speed") or {}).get("ms_per_doc")
        device = (result.get("params") or {}).get("device", "")
        lines += [
            f"## Results (exact-span match, held-out test set `{test_hash}`)",
            "",
            "| label | precision | recall | F1 |",
            "|---|---|---|---|",
        ]
        for label, v in m.items():
            if label == "speed":
                continue
            lines.append(f"| {label} | {v['p']:.1%} | {v['r']:.1%} | {v['f1']:.1%} |")
        if speed is not None:
            lines += [
                "",
                f"Inference: {speed} ms/doc (batched, {device or 'unknown device'}).",
            ]
        lines.append("")

    if siblings:
        lines += [
            "## Choosing a model",
            "",
            "All sizes are trained on the same data and scored on the same test set.",
            "",
            "| model | base params | overall F1 | ms/doc |",
            "|---|---|---|---|",
        ]
        for row in siblings:
            ms = f"{row['ms']}" if row["ms"] is not None else "n/a"
            label = f"[{row['display']}](https://huggingface.co/{row['repo_id']})"
            if row["repo_id"] == repo_id:
                label = f"**{row['display']} (this repo)**"
            lines.append(f"| {label} | {row['params_m']}M | {row['f1']:.1%} | {ms} |")
        lines.append("")

    lines += [
        "## Usage",
        "",
        'Needs `pip install "gliner2>=2" huggingface_hub`.',
        "",
        "```python",
        "import json",
        "import re",
        "",
        "from gliner2 import AutoExtractor",
        "from huggingface_hub import snapshot_download",
        "",
        "# AlnumBoundarySplitter: copy the class from the section below",
        "",
        f'adapter_dir = snapshot_download("{repo_id}")',
        'cfg = json.load(open(f"{adapter_dir}/recognizer_config.json"))',
        'model = AutoExtractor.from_pretrained(cfg["base_model"])',
        "model.load_adapter(adapter_dir)",
        "model.set_word_splitter(AlnumBoundarySplitter())  # required, see below",
        "",
        "result = model.extract_entities(",
        '    "$AAPL 350 C 11/06/2026 $1.2M 4.37avg",',
        '    cfg["entity_descriptions"],',
        '    threshold=cfg["threshold"],',
        ")",
        "print(result)",
        "```",
        "",
        f"Recommended threshold: **{threshold}**. Lower it to find more entities "
        "(higher recall), raise it for fewer, more certain ones.",
        "",
        "## Required word splitter",
        "",
        "The adapter was trained with a custom word splitter that breaks between "
        "letters and digits, so `1.58avg` and `11/20exp` tokenize cleanly. "
        "Define it before loading the adapter, or accuracy drops sharply:",
        "",
        "```python",
        "import re",
        "",
        inspect.getsource(AlnumBoundarySplitter).rstrip(),
        "```",
        "",
        "## Training data",
        "",
        f"[{dataset_id}](https://huggingface.co/datasets/{dataset_id}): about 1,300 "
        "tweets about stock options, pre-labeled with an LLM and reviewed by hand "
        "in Label Studio, plus the held-out test set of 227 tweets used for the "
        "scores above (the dataset's `test` split). LoRA (rank 32) on the attention and dense layers, with early "
        "stopping. The full pipeline is in the "
        f"[GitHub repository]({GITHUB_URL}).",
        "",
        "## Limitations",
        "",
        "- Scores are for tweets in the style of the training data; other text "
        "(news, filings, non-English) is untested.",
        "- `premium` (total trade size) and `price` (per-contract price) can be "
        "confused, especially in the smaller models.",
        "- The GLiNER2.5 variants score noticeably lower than the GLiNER2 ones.",
        "",
        "## Citation",
        "",
        "```bibtex",
        "@misc{akkerman_options_recognizer,",
        "  author = {Stephan Akkerman},",
        "  title  = {options-recognizer-model},",
        "  year   = {2026},",
        f"  url    = {{{GITHUB_URL}}}",
        "}",
        "```",
        "",
    ]
    return "\n".join(lines)


def push_model_to_hub(
    adapter_dir,
    repo_id=None,
    owner=None,
    threshold=None,
    private=True,
    run_card_examples=True,
):
    """Stage and upload an adapter folder; returns the repo URL."""
    if not os.path.isdir(adapter_dir):
        raise FileNotFoundError(f"Adapter folder not found: {adapter_dir}")
    parsed = parse_adapter_dir(os.path.basename(os.path.normpath(adapter_dir)))
    if parsed is None:
        raise ValueError(
            "Folder name must look like options_adapter_<base-slug>_v<N>, "
            f"got {adapter_dir!r}"
        )
    slug, version = parsed
    weights = locate_adapter_weights(adapter_dir)
    if weights is None:
        raise FileNotFoundError(f"No adapter_model.safetensors under {adapter_dir}")
    if repo_id is None:
        if not owner:
            raise ValueError("Pass --repo-id or --owner.")
        repo_id = default_repo_id(owner, slug)

    store = load_store()
    test_hash, result = latest_benchmark(store, adapter_label(slug, version))
    if result is None:
        print("No benchmark result found; the model card will have no metrics.")

    with tempfile.TemporaryDirectory() as staging:
        shutil.copytree(weights, staging, dirs_exist_ok=True)
        cfg_path = os.path.join(staging, "recognizer_config.json")
        cfg = {}
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        if threshold is not None:
            cfg["threshold"] = threshold
        cfg.setdefault("base_model", BASE_MODELS[slug]["hf_id"])
        # Older training runs saved these empty; the model card's usage needs them.
        if not cfg.get("entity_descriptions"):
            cfg["entity_descriptions"] = ENTITY_DESCRIPTIONS
        if not cfg.get("labels"):
            cfg["labels"] = list(ENTITY_DESCRIPTIONS)
        cfg.setdefault("threshold", 0.75)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        examples = None
        if run_card_examples:
            print("Running the adapter on the card examples...")
            examples = list(
                zip(EXAMPLE_TEXTS, run_examples(staging, slug, EXAMPLE_TEXTS))
            )
        with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                build_model_card(
                    slug,
                    version,
                    repo_id,
                    cfg["threshold"],
                    result,
                    test_hash,
                    examples=examples,
                    siblings=sibling_rows(repo_id.split("/")[0], store),
                )
            )

        api = HfApi()
        print(f"Creating/verifying repo: https://huggingface.co/{repo_id}")
        api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"Uploading {weights} to {repo_id}...")
        commit = api.upload_folder(
            folder_path=staging,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Upload adapter v{version}",
        )
        api.create_tag(
            repo_id,
            tag=f"v{version}",
            repo_type="model",
            revision=commit.oid,
            exist_ok=True,
        )

    print(f"Published and tagged v{version}: https://huggingface.co/{repo_id}")
    return f"https://huggingface.co/{repo_id}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "adapter_dir",
        nargs="?",
        help="Adapter folder, e.g. models/options_adapter_gliner2-large-v1_v1",
    )
    parser.add_argument(
        "--owner", help="HF user/org; repo becomes <owner>/options-recognizer-<base>."
    )
    parser.add_argument("--repo-id", help="Explicit repo id (overrides --owner).")
    parser.add_argument(
        "--threshold",
        type=float,
        help="Decision threshold to ship (from threshold_sweep). Default: keep the trained config's.",
    )
    parser.add_argument(
        "--public", action="store_true", help="Make the repo public (default: private)."
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Instead of one folder, push the best-F1 adapter of every base model.",
    )
    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip running the model for the card's examples (needs gliner2>=2).",
    )
    args = parser.parse_args()

    if args.best:
        if not args.owner:
            parser.error("--best needs --owner.")
        dirs = list(best_adapter_dirs(load_store()).values())
    elif args.adapter_dir:
        dirs = [args.adapter_dir]
    else:
        parser.error("Pass an adapter folder or --best.")

    for adapter_dir in dirs:
        push_model_to_hub(
            adapter_dir,
            repo_id=None if args.best else args.repo_id,
            owner=args.owner,
            threshold=args.threshold,
            private=not args.public,
            run_card_examples=not args.no_examples,
        )


if __name__ == "__main__":
    main()
