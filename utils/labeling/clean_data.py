"""Clean the scraped options tweets and (optionally) publish them to Hugging Face.

    python -m utils.labeling.clean_data
    python -m utils.labeling.clean_data --max-per-template 0           # keep every template row
    python -m utils.labeling.clean_data --push user/options-tweets     # private repo by default

Input is the JSONL exported from fintwit-web (data/raw/json/o.jsonl). Output is
a slimmed-down JSONL with only the fields needed for labeling, in
data/cleaned/options.jsonl.

Steps: strip links and quote scaffolding from the text -> drop empty/too short/
too long rows -> drop exact duplicates -> cap near-identical "templates".

The cap matters: ~30% of the scrape is one alert bot's "$XXX - $NNNK Call buyer"
format, which would otherwise dominate training and teach the model that
single pattern.
"""

import argparse
import html
import re
from collections import Counter

from rich.console import Console

from utils.hf.hf_utils import push_dataset, read_jsonl, write_jsonl

console = Console()

KEEP_FIELDS = ("id", "user", "created_at", "text", "tickers", "is_options_tweet")

_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_URL = re.compile(r"https?://\S+")
# Quoted-tweet header produced by the scraper: "> @user:" (after link stripping).
_QUOTE_HEADER = re.compile(r"^>\s*@\w+:?\s*$")
_QUOTE_PREFIX = re.compile(r"^>\s?")


def clean_text(text):
    """Strip markdown links, URLs and quoted-tweet scaffolding from a tweet.

    The quoted tweet's body is kept (it is usually the actual alert) but its
    ``> [@user](url):`` header and ``>`` prefixes are removed. Newlines are
    preserved because flow lists are one contract per line.
    """
    text = html.unescape(text)
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("", text)
    lines = []
    for line in text.splitlines():
        if _QUOTE_HEADER.match(line.strip()):
            continue
        line = _QUOTE_PREFIX.sub("", line.strip())
        lines.append(re.sub(r"[ \t]+", " ", line).strip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def template_key(text):
    """Collapse tickers and numbers so same-format alerts share a key.

    ``$WOLF - $993K Call buyer`` and ``$MSTR - $285K Call buyer`` both become
    ``$T - $NK call buyer``.
    """
    key = re.sub(r"\$[A-Za-z]+", "$T", text.lower())
    key = re.sub(r"\d+(?:[.,]\d+)*", "N", key)
    return re.sub(r"\s+", " ", key).strip()


def clean_records(records, min_chars=15, max_chars=1500, max_per_template=20):
    """Clean, filter and dedupe raw records.

    Returns ``(cleaned, stats)`` where `stats` counts why rows were dropped.
    `max_per_template` of 0/None disables the template cap; otherwise the first
    N rows of each template are kept (input order).
    """
    stats = Counter()
    seen_texts = set()
    per_template = Counter()
    cleaned = []
    for rec in records:
        text = clean_text(rec.get("text") or "")
        if len(text) < min_chars:
            stats["too_short"] += 1
            continue
        if len(text) > max_chars:
            stats["too_long"] += 1
            continue
        norm = re.sub(r"\s+", " ", text.lower())
        if norm in seen_texts:
            stats["duplicate"] += 1
            continue
        seen_texts.add(norm)
        if max_per_template:
            key = template_key(text)
            per_template[key] += 1
            if per_template[key] > max_per_template:
                stats["template_cap"] += 1
                continue
        out = {k: rec.get(k) for k in KEEP_FIELDS}
        out["text"] = text
        out["tickers"] = out["tickers"] or []
        cleaned.append(out)
    stats["kept"] = len(cleaned)
    return cleaned, stats


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", default="data/raw/json/o.jsonl")
    parser.add_argument("--output", default="data/cleaned/options.jsonl")
    parser.add_argument("--min-chars", type=int, default=15)
    parser.add_argument("--max-chars", type=int, default=1500)
    parser.add_argument(
        "--max-per-template",
        type=int,
        default=20,
        help="Max rows per same-format alert template; 0 disables (default: 20).",
    )
    parser.add_argument("--push", metavar="REPO_ID", help="Upload to this HF dataset.")
    parser.add_argument(
        "--public", action="store_true", help="Make the --push repo public."
    )
    args = parser.parse_args()

    raw = read_jsonl(args.input)
    cleaned, stats = clean_records(
        raw, args.min_chars, args.max_chars, args.max_per_template
    )
    write_jsonl(args.output, cleaned)

    console.print(
        f"[green]{len(raw)} raw -> {len(cleaned)} cleaned[/green] ({args.output})"
    )
    for reason, n in stats.items():
        if reason != "kept":
            console.print(f"  dropped {n:>5}  {reason}")

    if args.push:
        url = push_dataset(args.output, args.push, private=not args.public)
        console.print(f"[bold green]Uploaded:[/bold green] {url}")


if __name__ == "__main__":
    main()
