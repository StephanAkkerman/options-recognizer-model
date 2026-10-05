"""Auto-label scraped options tweets via a reusable few-shot prompt + SOTA LLM.

Input is the cleaned JSONL from `utils/clean_data.py` (data/cleaned/options.jsonl).

Workflow (manual mode — recommended for the first batch):
  Single post:
    1) `python -m utils.synthetic.auto_label --post-index 0 --print-prompt > prompt.txt`
    2) Paste prompt.txt into your LLM, save the JSON reply as response.json
    3) `python -m utils.synthetic.auto_label --post-index 0 --response-file response.json`

  Batch (recommended for long-context models like Gemini):
    1) `python -m utils.synthetic.auto_label --posts 0-9 --print-prompt > prompt.txt`
    2) Paste prompt.txt into the LLM, save the JSON reply as response.json
    3) `python -m utils.synthetic.auto_label --posts 0-9 --response-file response.json`

  The batch output shape is `{"results": [{"index": N, "entities": [...]}, ...]}`
  with one entry per input. Re-running with the same --posts overwrites those
  task IDs rather than duplicating them.

  Interactive mode (loop through everything without juggling files):
    `python -m utils.synthetic.auto_label --interactive [--batch-size 10]`
    `python -m utils.synthetic.auto_label --interactive --batch-chars 10000`
    Each round writes the prompt to --prompt-file (default
    data/auto_label/prompt.txt); copy it into your LLM, paste the JSON reply
    back in the terminal, then type END on its own line (or `q` to quit).
    Progress saves after every batch, so quitting mid-run keeps completed work.
    --batch-chars groups posts by total character count instead of post count,
    so a batch of short posts and a batch of long posts consume similar context.

Once the prompt is dialled in (the LLM's outputs match what you'd label by hand
on ~5-10 spot checks), this same module can be called from a thin API wrapper
to scale up — `build_prompt()` and `parse_response_to_task()` are pure functions.

Design choices worth knowing:
  - The LLM returns entity *text* + label, not character offsets. We locate
    offsets ourselves — LLMs are notoriously bad at offsets, and re-finding text
    is robust to whitespace/quoting changes.
  - Options text is full of short, repeated tokens ("C", "10", "350"), so the
    LLM lists ONE entry per occurrence in reading order, and `_locate_entities`
    matches them to the text in that order with token-boundary checks (so "C"
    never matches inside "$COIN" and "350" never matches inside "$350K").
  - Output lands in `data/preds/`, not `data/labeled/`. Pre-labels need review
    before becoming training data, matching the existing pipeline convention.
  - Posts are deduped against existing labeled+test so the LLM never wastes
    effort on something already annotated.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import textwrap
import uuid

from rich.console import Console

from src.core.spans import bounded_pattern
from utils.hf.hf_utils import read_jsonl

console = Console()

# Single source of truth for the label set: the prompt, validation and the
# Label Studio config should all agree with this.
LABELS = ("ticker", "strike", "option_type", "expiry", "premium", "price")
_LABEL_CHOICES = " | ".join(f'"{label}"' for label in LABELS)


# Hand-picked from the scraped tweets to cover the patterns that are easy to get
# wrong:
#   1) one-line alert: every field present
#   2) flow-bot "Call buyer" format: ticker + premium + type only
#   3) multi-line flow list: report date ("10/2") is NOT an expiry
#   4) fused strike+type ("210p") and spelled-out premium ("23 million")
#   5) written-out expiry; underlying "stock price" is NOT labeled
#   6) $-prefixed strike ("$1300 calls") — only the ticker is a cashtag
#   7) shorthand "195c 10/16" mixed into prose; @handles/ETF names not labeled
#   8) no options content at all
#
# Edit / extend this list if the LLM starts missing a specific pattern in practice.
FEW_SHOT_EXAMPLES = [
    {
        "input": "🚨 $MCY 110 CALL 10/16 @ .35",
        "output": {
            "entities": [
                {"text": "$MCY", "label": "ticker"},
                {"text": "110", "label": "strike"},
                {"text": "CALL", "label": "option_type"},
                {"text": "10/16", "label": "expiry"},
                {"text": ".35", "label": "price"},
            ]
        },
    },
    {
        "input": "$WOLF - $993K Call buyer",
        "output": {
            "entities": [
                {"text": "$WOLF", "label": "ticker"},
                {"text": "$993K", "label": "premium"},
                {"text": "Call", "label": "option_type"},
            ]
        },
    },
    {
        "input": "10/2 Notable Flow\n\n"
        "$AAPL 350 C 11/06/2026 $1.2M 4.37avg\n"
        "$XOM 175 P 10/09/2026 $172K .16avg (Buzzer Beater)",
        "output": {
            "entities": [
                {"text": "$AAPL", "label": "ticker"},
                {"text": "350", "label": "strike"},
                {"text": "C", "label": "option_type"},
                {"text": "11/06/2026", "label": "expiry"},
                {"text": "$1.2M", "label": "premium"},
                {"text": "4.37", "label": "price"},
                {"text": "$XOM", "label": "ticker"},
                {"text": "175", "label": "strike"},
                {"text": "P", "label": "option_type"},
                {"text": "10/09/2026", "label": "expiry"},
                {"text": "$172K", "label": "premium"},
                {"text": ".16", "label": "price"},
            ]
        },
    },
    {
        "input": "Somebody bought 23 million worth of $SPCX 210p expiring in 2 days",
        "output": {
            "entities": [
                {"text": "23 million", "label": "premium"},
                {"text": "$SPCX", "label": "ticker"},
                {"text": "210", "label": "strike"},
                {"text": "p", "label": "option_type"},
                {"text": "2 days", "label": "expiry"},
            ]
        },
    },
    {
        "input": "$IREN - $430K Call Buyer - November 20, 2026 Expiry\n\n"
        "Current stock price - $41.28",
        "output": {
            "entities": [
                {"text": "$IREN", "label": "ticker"},
                {"text": "$430K", "label": "premium"},
                {"text": "Call", "label": "option_type"},
                {"text": "November 20, 2026", "label": "expiry"},
            ]
        },
    },
    {
        "input": "Institutions buying $1300 calls on $MU data via @blademapai",
        "output": {
            "entities": [
                {"text": "$1300", "label": "strike"},
                {"text": "calls", "label": "option_type"},
                {"text": "$MU", "label": "ticker"},
            ]
        },
    },
    {
        "input": "Technology remains strong, big flows in SPDR $XLK. "
        "Nearly $2M of the 195c 10/16 opened in the last couple of weeks",
        "output": {
            "entities": [
                {"text": "$XLK", "label": "ticker"},
                {"text": "$2M", "label": "premium"},
                {"text": "195", "label": "strike"},
                {"text": "c", "label": "option_type"},
                {"text": "10/16", "label": "expiry"},
            ]
        },
    },
    {
        "input": "Fed meeting tomorrow, market looks shaky. Staying in cash. NFA",
        "output": {"entities": []},
    },
]


BATCH_INSTRUCTIONS = textwrap.dedent(f"""\
    BATCH MODE: You will receive multiple posts labeled "Input 1:", "Input 2:", etc.
    Return ONE JSON object with this structure (no commentary, no code fence):

    {{"results": [
      {{"index": 1, "entities": [{{"text": "...", "label": {_LABEL_CHOICES}}}]}},
      {{"index": 2, "entities": [...]}},
      ...
    ]}}

    The "index" field MUST match the corresponding "Input N:" number. Include an
    entry for EVERY input, even when entities is empty (use "entities": []).
""")


SYSTEM_INSTRUCTIONS = textwrap.dedent(f"""\
    You are an expert at extracting stock-option contract details from tweets
    about options flow and trading.

    TASK: identify every entity of the types below in the input text.

    LABELS:
    - "ticker": the underlying's symbol. A cashtag ($ directly attached) is always
      a ticker and the $ is part of the text: "$AAPL". A bare ALL-CAPS symbol
      (SPY, QQQ, NVDA) is a ticker when it clearly names the underlying.
      Company names ("Nvidia") are NOT labeled.
    - "strike": the strike price: "350", "162.5", "$1300" (a $ here is part of
      the strike, e.g. "$1300 calls"). Not the stock's current price.
    - "option_type": call or put in any form: "C", "P", "Call", "CALL", "calls",
      "puts", "c", "p". Label only the word/letter itself, not "buyer"/"seller".
    - "expiry": the expiration date, as written: "10/16", "11/06/2026",
      "6/17/27", "November 20, 2026", "December", or relative like "2 days".
    - "premium": the total dollar size of the trade: "$1.2M", "$993K",
      "23 million", "$1.4 million".
    - "price": the per-contract price / average fill: ".15" in "@ .15", "4.37" in
      "4.37avg", "156.28" in "@ 156.28". Label the number only, not "avg".

    FUSED TOKENS: split them. "210p" is "210" (strike) + "p" (option_type);
    "195c" is "195" + "c". Use the exact substrings.

    DO NOT LABEL:
    - The date a report was posted ("10/2 Notable Flow", "8/11 Notable Flow").
    - The underlying's current stock price ("Current stock price - $41.28").
    - Percentages and moneyness ("41% OTM", "up 70%"), "avg", "OI", "IV".
    - @handles, hashtags, company names, ETF descriptions (e.g. "SPDR").
    - Slang and generic words (YOLO, NFA, FOMO, "options") — only label call/put
      words when they refer to specific contracts being bought, sold or discussed.

    ONE ENTRY PER OCCURRENCE, IN READING ORDER:
    - If $AAPL appears 3 times, return 3 "$AAPL" entries, placed where they occur.
    - For a multi-line flow list, go line by line, left to right.
    - Preserve exact casing and punctuation from the source — do not normalize.

    OUTPUT FORMAT — return ONLY a JSON object, no commentary, no code fence:
    {{"entities": [{{"text": "<exact substring from input>", "label": {_LABEL_CHOICES}}}]}}

    Empty entities array (`{{"entities": []}}`) is correct for posts with no entities.
""")


def build_prompt(texts, examples=None):
    """Assemble: instructions + few-shot examples + target input(s).

    `texts` may be a single string or a list of strings. With one input the
    output format is `{"entities": [...]}`. With 2+ inputs we switch to batch
    mode and ask for `{"results": [{"index": N, "entities": [...]}, ...]}`.
    """
    if isinstance(texts, str):
        texts = [texts]
    is_batch = len(texts) > 1

    examples = examples if examples is not None else FEW_SHOT_EXAMPLES
    parts = [SYSTEM_INSTRUCTIONS]
    if is_batch:
        parts.append(BATCH_INSTRUCTIONS)
    parts.append("EXAMPLES (single-input format, showing what to label):")
    for i, ex in enumerate(examples, 1):
        parts.append(f"\nExample {i}")
        parts.append(f"Input: {ex['input']}")
        parts.append(f"Output: {json.dumps(ex['output'], ensure_ascii=False)}")

    if is_batch:
        parts.append(
            f"\nNow label the following {len(texts)} inputs. "
            "Return ONE JSON object with a 'results' array as described above."
        )
        for i, text in enumerate(texts, 1):
            parts.append(f"\nInput {i}: {text}")
    else:
        parts.append("\nNow label this input. Return ONLY the JSON object.")
        parts.append(f"\nInput: {texts[0]}")
    parts.append("\nOutput:")
    return "\n".join(parts)


def _region_id():
    """Generate a short random region id.

    Label Studio needs a stable per-region ``id`` (alongside ``from_name`` /
    ``to_name``) to map a result span onto the labeling config — without it the
    region is silently dropped and nothing renders on import.
    """
    return uuid.uuid4().hex[:10]


def _locate_entities(text, entities):
    """Map LLM ``(text, label)`` entries to non-overlapping ``(start, end, ...)`` spans.

    Entries are matched in order: each takes the first free occurrence at or
    after the previous match, falling back to the first free occurrence anywhere
    when the LLM listed things out of order. Returns ``(spans, dropped)``.
    """
    spans = []
    dropped = []
    cursor = 0
    for ent in entities:
        ent_text = (ent.get("text") or "").strip()
        ent_label = ent.get("label", "")
        if not ent_text or ent_label not in LABELS:
            dropped.append((ent_text, ent_label, "invalid"))
            continue
        found = list(re.finditer(bounded_pattern(ent_text), text))
        free = [
            m
            for m in found
            if not any(m.start() < e and s < m.end() for s, e, _, _ in spans)
        ]
        if not free:
            # "already_used": the text is in the post, but the LLM listed it more
            # times than it occurs (or it overlaps an earlier entity).
            reason = "already_used" if found else "not_found"
            dropped.append((ent_text, ent_label, reason))
            continue
        match = next((m for m in free if m.start() >= cursor), free[0])
        spans.append((match.start(), match.end(), ent_text, ent_label))
        cursor = match.end()
    return sorted(spans, key=lambda s: s[0]), dropped


def parse_response_to_task(text, response_obj, task_id):
    """Convert LLM JSON response to a Label Studio task, finding offsets in `text`.

    Drops entities whose `text` field can't be located in the source — better
    than emitting fabricated offsets.
    """
    spans, dropped = _locate_entities(text, response_obj.get("entities", []))
    annotation_results = [
        {
            "id": _region_id(),
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "value": {
                "start": start,
                "end": end,
                "text": ent_text,
                "labels": [ent_label],
            },
        }
        for start, end, ent_text, ent_label in spans
    ]
    return {
        "id": task_id,
        "data": {"text": text},
        "annotations": [
            {
                "was_cancelled": False,
                "result": annotation_results,
            }
        ],
    }, dropped


def _text_hash(text):
    norm = re.sub(r"\s+", " ", text.strip().lower())[:200]
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _load_known_hashes(folders):
    hashes = set()
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for fp in glob.glob(os.path.join(folder, "*.json")):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(data, list):
                continue
            for task in data:
                text = (
                    task.get("data", {}).get("text") if isinstance(task, dict) else None
                )
                if text:
                    hashes.add(_text_hash(text))
    return hashes


def load_unlabeled_posts(jsonl_path, dedup_text_folders=None):
    """Read cleaned JSONL, dedupe against labeled+test+output corpus."""
    known = _load_known_hashes(
        dedup_text_folders or ["data/labeled", "data/test", "data/preds"]
    )

    posts = []
    for row in read_jsonl(jsonl_path):
        text = (row.get("text") or "").strip()
        if not text or _text_hash(text) in known:
            continue
        posts.append({"tweet_id": str(row.get("id")), "text": text})
    return posts


def _strip_code_fence(s):
    """LLMs often wrap JSON in ```json ... ``` — peel that off before parsing."""
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*\n?", "", s)
        s = re.sub(r"\n?```\s*$", "", s)
    return s


def parse_post_spec(spec):
    """Parse '0-9', '0,3,5', or '7' into a sorted unique list of indices."""
    ids = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            ids.update(range(int(a), int(b) + 1))
        else:
            ids.add(int(part))
    return sorted(ids)


def read_until_sentinel(lines):
    """Collect lines until an ``END`` line (case-insensitive) or EOF.

    `lines` is any iterator of strings (e.g. ``sys.stdin``). Returns a
    ``(kind, text)`` tuple. `kind` is ``"quit"`` when a sole ``q``/``quit`` is
    entered before any content (JSON always starts with ``{``/``[``, so this is
    unambiguous), otherwise ``"submit"`` with the newline-joined collected text.
    """
    collected = []
    for line in lines:
        stripped = line.rstrip("\n")
        flat = stripped.strip()
        if flat.upper() == "END":
            break
        if flat.lower() in ("q", "quit") and not any(c.strip() for c in collected):
            return "quit", ""
        collected.append(stripped)
    return "submit", "\n".join(collected)


def save_tasks(tasks, output):
    """Append `tasks` to the JSON array at `output`, overwriting by task id.

    Creates the parent directory if needed. Tasks sharing an id with an
    existing entry replace it (so re-labeling a post overwrites rather than
    duplicates), matching the one-shot ``--response-file`` behavior.
    """
    existing = []
    if os.path.exists(output):
        with open(output, "r", encoding="utf-8") as f:
            existing = json.load(f)
    new_ids = {t["id"] for t in tasks}
    existing = [t for t in existing if t.get("id") not in new_ids]
    existing.extend(tasks)
    parent = os.path.dirname(output)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def parse_batch_response(
    texts, response_obj, task_id_offset, post_indices, tweet_ids=None
):
    """Convert a batch `{"results": [...]}` response into Label Studio tasks.

    `texts` is the list of input strings (1-indexed by the LLM's "index" field).
    `post_indices` is the corresponding list of original post indices, used
    to assign stable task IDs (so re-running with the same posts overwrites
    rather than duplicates).
    Returns (list_of_tasks, list_of_(input_idx, dropped_entries)).
    """
    results_by_idx = {}
    for r in response_obj.get("results", []):
        idx = r.get("index")
        if isinstance(idx, int):
            results_by_idx[idx] = r

    tasks = []
    all_dropped = []
    missing = []
    for input_idx, text in enumerate(texts, 1):
        result = results_by_idx.get(input_idx)
        if result is None:
            missing.append(input_idx)
            continue
        post_idx = post_indices[input_idx - 1]
        task, dropped = parse_response_to_task(text, result, task_id_offset + post_idx)
        tasks.append(task)
        if dropped:
            ref = tweet_ids[input_idx - 1] if tweet_ids else task["id"]
            all_dropped.append((ref, dropped))
    return tasks, all_dropped, missing


def _print_save_summary(tasks, all_dropped, missing, output):
    """Report what was saved, plus any dropped/missing entities."""
    n_ents = sum(len(t["annotations"][0]["result"]) for t in tasks)
    console.print(
        f"[green]Saved {len(tasks)} task(s) ({n_ents} entity spans total) "
        f"to {output}[/green]"
    )
    if missing:
        console.print(
            f"[yellow]Missing results for input index(es): {missing} "
            f"— LLM didn't return entries for these.[/yellow]"
        )
    if all_dropped:
        console.print("[yellow]Dropped entities:[/yellow]")
        for ref, dropped in all_dropped:
            for txt, lab, reason in dropped:
                console.print(f"  - tweet {ref}: {reason}: {txt!r} ({lab})")


def _response_to_tasks(
    texts, response_obj, task_id_offset, post_indices, tweet_ids=None
):
    """Dispatch to batch or single parsing based on input count."""
    if len(texts) > 1:
        return parse_batch_response(
            texts, response_obj, task_id_offset, post_indices, tweet_ids
        )
    task, dropped = parse_response_to_task(
        texts[0], response_obj, task_id_offset + post_indices[0]
    )
    ref = tweet_ids[0] if tweet_ids else task["id"]
    return [task], ([(ref, dropped)] if dropped else []), []


def _build_char_batches(posts, max_chars):
    """Group posts into contiguous (start, end) slices where total text ≤ max_chars.

    A single post that exceeds max_chars is emitted as its own one-post batch
    rather than being silently skipped.
    """
    batches = []
    start = 0
    while start < len(posts):
        total = 0
        end = start
        while end < len(posts):
            n = len(posts[end]["text"])
            if end > start and total + n > max_chars:
                break
            total += n
            end += 1
        batches.append((start, end))
        start = end
    return batches


def run_interactive(posts, args, line_source=None):
    """Loop over all `posts` in batches, prompting + reading replies in-terminal.

    Each round writes the batch prompt to ``args.prompt_file`` (overwriting it),
    then reads the pasted LLM JSON from `line_source` (default ``sys.stdin``)
    until an ``END`` line / EOF, or ``q`` to quit. Saves after every batch so a
    mid-session quit keeps completed work. Bad JSON re-prompts the same batch.
    """
    line_source = sys.stdin if line_source is None else line_source
    total = len(posts)

    batch_chars = getattr(args, "batch_chars", None)
    if batch_chars:
        batch_slices = _build_char_batches(posts, batch_chars)
    else:
        b = args.batch_size
        batch_slices = [(i, min(i + b, total)) for i in range(0, total, b)]
    n_batches = len(batch_slices)

    batch_idx = 0
    while batch_idx < n_batches:
        start, end = batch_slices[batch_idx]
        post_indices = list(range(start, end))
        texts = [posts[i]["text"] for i in post_indices]
        char_count = sum(len(t) for t in texts)

        prompt = build_prompt(texts)
        parent = os.path.dirname(args.prompt_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(args.prompt_file, "w", encoding="utf-8") as f:
            f.write(prompt)

        console.print(
            f"\n[bold cyan]Batch {batch_idx + 1}/{n_batches} — "
            f"posts {start}–{end - 1} ({len(texts)} posts, {char_count:,} chars)[/bold cyan]"
        )
        console.print(
            f"Prompt written to [bold]{args.prompt_file}[/bold] "
            f"({len(prompt):,} chars). Copy it into your LLM."
        )
        console.print(
            "[dim]Paste the JSON response below, then type END on its own line "
            "(or q to quit):[/dim]"
        )

        kind, raw = read_until_sentinel(line_source)
        if kind == "quit":
            console.print("[yellow]Quit — progress saved.[/yellow]")
            return
        raw = _strip_code_fence(raw)
        if not raw.strip():
            console.print("[yellow]Empty response — stopping. Progress saved.[/yellow]")
            return

        try:
            response_obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Invalid JSON: {exc}[/red]")
            console.print(f"[dim]First 200 chars: {raw[:200]}[/dim]")
            console.print("[yellow]Re-paste the response for this batch.[/yellow]")
            continue  # retry the same batch — batch_idx unchanged

        tasks, all_dropped, missing = _response_to_tasks(
            texts,
            response_obj,
            args.task_id_offset,
            post_indices,
            [posts[i]["tweet_id"] for i in post_indices],
        )
        save_tasks(tasks, args.output)
        _print_save_summary(tasks, all_dropped, missing, args.output)
        batch_idx += 1

    console.print(f"\n[bold green]Done — all {total} posts processed.[/bold green]")


def main():
    parser = argparse.ArgumentParser(
        description="Build / parse auto-label prompts for options tweets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--input",
        default="data/cleaned/options.jsonl",
        help="Cleaned tweets JSONL from clean_data.py (default: %(default)s)",
    )
    parser.add_argument(
        "--post-index", type=int, help="Single-post shortcut, equivalent to --posts N."
    )
    parser.add_argument(
        "--posts",
        default=None,
        help="Posts to label. '0-9' for a range, '0,3,5' for a list, "
        "or a single index. Defaults to '0'.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Write the full prompt to stdout (suitable for pasting into an LLM).",
    )
    parser.add_argument(
        "--response-file",
        help="Path to an LLM JSON response. Parses + appends to --output.",
    )
    parser.add_argument(
        "--task-id-offset",
        type=int,
        default=8_000_000,
        help="Starting ID for auto-labeled tasks (avoids labeled.json collisions).",
    )
    parser.add_argument(
        "--output",
        default="data/preds/auto_labeled.json",
        help="Where to append parsed Label Studio tasks.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Loop over all unlabeled posts: emit a prompt, paste the "
        "LLM reply, repeat. Ignores --posts/--post-index.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Posts per prompt in --interactive mode (default: %(default)s). "
        "Ignored when --batch-chars is set.",
    )
    parser.add_argument(
        "--batch-chars",
        type=int,
        default=None,
        help="Max total post characters per --interactive batch. "
        "Overrides --batch-size. Posts are grouped until adding "
        "the next post would exceed this limit.",
    )
    parser.add_argument(
        "--prompt-file",
        default="data/auto_label/prompt.txt",
        help="Where --interactive writes each round's prompt (default: %(default)s).",
    )
    args = parser.parse_args()

    posts = load_unlabeled_posts(args.input)
    if not posts:
        console.print(f"[red]No unlabeled posts in {args.input} after dedup.[/red]")
        sys.exit(1)

    if args.interactive:
        run_interactive(posts, args)
        return

    # Resolve the post-index spec into a concrete list of post positions.
    if args.posts is not None:
        post_indices = parse_post_spec(args.posts)
    elif args.post_index is not None:
        post_indices = [args.post_index]
    else:
        post_indices = [0]

    for idx in post_indices:
        if idx < 0 or idx >= len(posts):
            console.print(
                f"[red]post index {idx} out of range (0..{len(posts) - 1}).[/red]"
            )
            sys.exit(1)

    targets = [posts[i] for i in post_indices]
    texts = [t["text"] for t in targets]
    is_batch = len(targets) > 1

    if args.response_file:
        with open(args.response_file, "r", encoding="utf-8") as f:
            raw = _strip_code_fence(f.read())
        try:
            response_obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            console.print(f"[red]Response file is not valid JSON: {exc}[/red]")
            console.print(f"[dim]First 200 chars: {raw[:200]}[/dim]")
            sys.exit(1)

        tasks, all_dropped, missing = _response_to_tasks(
            texts,
            response_obj,
            args.task_id_offset,
            post_indices,
            [t["tweet_id"] for t in targets],
        )
        save_tasks(tasks, args.output)
        _print_save_summary(tasks, all_dropped, missing, args.output)
        return

    prompt = build_prompt(texts)

    if args.print_prompt:
        sys.stdout.write(prompt)
        sys.stdout.write("\n")
        return

    range_desc = (
        f"posts {post_indices[0]}-{post_indices[-1]} ({len(targets)} total)"
        if is_batch
        else f"post {post_indices[0]}"
    )
    console.print(
        f"\n[bold cyan]{range_desc}[/bold cyan] "
        f"of {len(posts) - 1} available "
        f"({sum(len(t) for t in texts)} chars total input)"
    )
    if not is_batch:
        console.print("[dim]--- POST PREVIEW ---[/dim]")
        preview = texts[0][:400].replace("\n", " ")
        console.print(preview + ("..." if len(texts[0]) > 400 else ""))
    console.print(
        f"\n[dim]--- PROMPT ({len(prompt)} chars) — "
        f"use --print-prompt to write raw to stdout ---[/dim]\n"
    )
    print(prompt)


if __name__ == "__main__":
    main()
