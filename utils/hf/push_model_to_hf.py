"""Publish a trained adapter as its own Hugging Face model repo.

One repo is created per base model (``<owner>/options-recognizer-<base-slug>``)
so users pick a size by picking a repo. The uploaded folder is self-describing:
adapter weights, ``recognizer_config.json`` (base model, labels, threshold) and
a generated model card with the benchmark numbers.

    python -m utils.hf.push_model_to_hf models/options_adapter_gliner2-large-v1_v1 --owner user
    python -m utils.hf.push_model_to_hf models/options_adapter_gliner2.5-small-v1_v2 --owner user --threshold 0.6

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

from src.core.benchmark import locate_adapter_weights
from src.core.models import (
    BASE_MODELS,
    adapter_label,
    default_repo_id,
    parse_adapter_dir,
)
from src.core.results_store import load_store
from src.core.spans import AlnumBoundarySplitter


def latest_benchmark(store, name):
    """Most recently evaluated ``(test_hash, result)`` for `name`, or ``(None, None)``."""
    runs = store["results"].get(name, {})
    if not runs:
        return None, None
    test_hash = max(runs, key=lambda h: runs[h]["evaluated_at"])
    return test_hash, runs[test_hash]


def build_model_card(slug, version, repo_id, threshold, result=None, test_hash=None):
    """Render the model card (README.md) for one adapter."""
    info = BASE_MODELS[slug]
    lines = [
        "---",
        f"base_model: {info['hf_id']}",
        "library_name: gliner2",
        "pipeline_tag: token-classification",
        "language:",
        "- en",
        "tags:",
        "- gliner2",
        "- lora",
        "- finance",
        "- options",
        "- ner",
        "---",
        "",
        f"# {repo_id.split('/')[-1]}",
        "",
        f"LoRA adapter (v{version}) for [`{info['hf_id']}`](https://huggingface.co/{info['hf_id']}) "
        f"({info['params_m']}M parameters) that extracts option-contract details from "
        "tweets: ticker, strike, option type, expiry, premium and price.",
        "",
        "Smaller bases are faster, larger ones are more accurate — see the "
        "sibling `options-recognizer-*` repos for the other sizes.",
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
    lines += [
        "## Usage",
        "",
        "```python",
        "import json",
        "from gliner2 import GLiNER2",
        "from huggingface_hub import snapshot_download",
        "",
        f'adapter_dir = snapshot_download("{repo_id}")',
        'cfg = json.load(open(f"{adapter_dir}/recognizer_config.json"))',
        'model = GLiNER2.from_pretrained(cfg["base_model"])',
        "model.set_word_splitter(AlnumBoundarySplitter())  # required, see below",
        "model.load_adapter(adapter_dir)",
        "",
        "result = model.extract_entities(",
        '    "$AAPL 350 C 11/06/2026 $1.2M 4.37avg",',
        '    cfg["entity_descriptions"],',
        '    threshold=cfg["threshold"],',
        ")",
        "```",
        "",
        f"Recommended threshold: **{threshold}**.",
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
    ]
    return "\n".join(lines)


def push_model_to_hub(
    adapter_dir, repo_id=None, owner=None, threshold=None, private=True
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

    test_hash, result = latest_benchmark(load_store(), adapter_label(slug, version))
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
        cfg.setdefault("threshold", 0.75)
        with open(cfg_path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        with open(os.path.join(staging, "README.md"), "w", encoding="utf-8") as f:
            f.write(
                build_model_card(
                    slug, version, repo_id, cfg["threshold"], result, test_hash
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
    args = parser.parse_args()

    push_model_to_hub(
        args.adapter_dir,
        repo_id=args.repo_id,
        owner=args.owner,
        threshold=args.threshold,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
