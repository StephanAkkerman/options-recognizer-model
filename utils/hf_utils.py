"""Helpers for reading/writing JSONL and publishing a dataset to Hugging Face.

Only depends on ``huggingface_hub`` (imported lazily), so the cleaning pipeline
runs without it. Uploaded JSONL files are picked up by
``datasets.load_dataset(repo_id)`` automatically.
"""

import json
import os
import textwrap
from collections import Counter

DEFAULT_PATH_IN_REPO = "data/train.jsonl"


def read_jsonl(path):
    """Read a JSON Lines file into a list of dicts, skipping blank lines."""
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    """Write `records` to `path` as JSON Lines (UTF-8, emoji kept readable)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_dataset_card(records, repo_id, description=None):
    """Render a minimal dataset card (README.md with YAML front matter)."""
    users = Counter(r.get("user") for r in records)
    dates = sorted(r["created_at"] for r in records if r.get("created_at"))
    span = f"{dates[0][:10]} to {dates[-1][:10]}" if dates else "unknown"
    description = description or (
        "Tweets about stock options (unusual options flow, trade alerts, "
        "commentary), scraped from X/Twitter via fintwit-web and cleaned for "
        "token/entity annotation."
    )
    top_users = ", ".join(f"@{u} ({n})" for u, n in users.most_common(5))
    return (
        textwrap.dedent(f"""\
        ---
        language:
        - en
        license: mit
        pretty_name: {repo_id.split("/")[-1]}
        size_categories:
        - {"1K<n<10K" if len(records) > 1000 else "n<1K"}
        task_categories:
        - token-classification
        tags:
        - finance
        - options
        - twitter
        ---

        # {repo_id.split("/")[-1]}

        {description}

        ## Dataset structure

        | field | description |
        |---|---|
        | `id` | Tweet id |
        | `user` | Author handle |
        | `created_at` | ISO timestamp |
        | `text` | Cleaned tweet text (links and quote headers removed) |
        | `tickers` | Tickers detected by `ticker-classifier` |
        | `is_options_tweet` | Whether the options keyword filter marked it as options-related |

        ## Stats

        - Rows: {len(records)}
        - Period: {span}
        - Top authors: {top_users}

        The texts are unlabeled; entity labels are added in a later step.
        """).rstrip()
        + "\n"
    )


def push_dataset(
    path,
    repo_id,
    private=True,
    token=None,
    path_in_repo=DEFAULT_PATH_IN_REPO,
    description=None,
):
    """Upload the JSONL at `path` plus a dataset card to ``repo_id``.

    Creates the dataset repo if needed (private by default). `token` falls back
    to the cached ``huggingface-cli login`` / ``HF_TOKEN``. Returns the repo URL.
    """
    from huggingface_hub import HfApi

    records = read_jsonl(path)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_file(
        path_or_fileobj=path,
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=f"Upload {len(records)} rows",
    )
    api.upload_file(
        path_or_fileobj=build_dataset_card(records, repo_id, description).encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )
    return f"https://huggingface.co/datasets/{repo_id}"
