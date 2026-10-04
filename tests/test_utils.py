import json

from auto_label import FEW_SHOT_EXAMPLES, LABELS, build_prompt, parse_response_to_task
from clean_data import clean_records, clean_text, template_key
from hf_utils import build_dataset_card, read_jsonl, write_jsonl


def _spans(text, entities):
    task, dropped = parse_response_to_task(text, {"entities": entities}, 1)
    spans = [r["value"] for r in task["annotations"][0]["result"]]
    return [
        (s["text"], s["labels"][0], text[s["start"] : s["end"]]) for s in spans
    ], dropped


def test_clean_text_strips_quote_scaffolding():
    raw = (
        "How did they know? $WOLF \n\n> [@FL0WG0D](https://twitter.com/FL0WG0D):\n"
        "> $WOLF - $993K Call buyer https://t.co/abc"
    )
    assert clean_text(raw) == "How did they know? $WOLF\n\n$WOLF - $993K Call buyer"


def test_template_key_collapses_ticker_and_numbers():
    assert template_key("$WOLF - $993K Call buyer") == template_key(
        "$MSTR - $285K call buyer"
    )


def test_clean_records_dedupes_and_caps_templates():
    raw = [
        {"id": str(i), "text": f"${t} - ${i}00K Call buyer", "tickers": [t]}
        for i, t in enumerate(["AAA", "BBB", "CCC", "DDD"])
    ]
    raw.append({"id": "x", "text": "$AAA - $000K Call buyer", "tickers": None})
    cleaned, stats = clean_records(raw, max_per_template=2)
    assert len(cleaned) == 2
    assert stats["duplicate"] == 1
    assert stats["template_cap"] == 2
    assert set(cleaned[0]) == {
        "id",
        "user",
        "created_at",
        "text",
        "tickers",
        "is_options_tweet",
    }


def test_jsonl_roundtrip_and_card(tmp_path):
    rows = [{"text": "$A 1 C 1/1 🚨", "user": "u", "created_at": "2026-01-02T00:00:00"}]
    path = tmp_path / "d.jsonl"
    write_jsonl(str(path), rows)
    assert read_jsonl(str(path)) == rows
    card = build_dataset_card(rows, "me/opts")
    assert "Rows: 1" in card and "token-classification" in card


def test_locator_respects_token_boundaries():
    text = "$COIN 110 C 10/16 $3.5M"
    found, dropped = _spans(
        text,
        [
            {"text": "$COIN", "label": "ticker"},
            {"text": "110", "label": "strike"},
            {"text": "C", "label": "option_type"},
            {"text": "3", "label": "strike"},  # only inside "$3.5M" -> not found
        ],
    )
    assert [f[:2] for f in found] == [
        ("$COIN", "ticker"),
        ("110", "strike"),
        ("C", "option_type"),
    ]
    c_start = text.index(" C ") + 1
    task, _ = parse_response_to_task(
        text, {"entities": [{"text": "C", "label": "option_type"}]}, 1
    )
    assert task["annotations"][0]["result"][0]["value"]["start"] == c_start
    assert dropped == [("3", "strike", "not_found")]


def test_locator_assigns_repeated_text_in_order():
    text = "$A 10 C 10/16\n$B 10 P 10/16"
    task, _ = parse_response_to_task(
        text,
        {
            "entities": [
                {"text": "10", "label": "strike"},
                {"text": "10/16", "label": "expiry"},
                {"text": "10", "label": "strike"},
                {"text": "10/16", "label": "expiry"},
            ]
        },
        1,
    )
    starts = [r["value"]["start"] for r in task["annotations"][0]["result"]]
    assert starts == [3, 8, 17, 22]


def test_fused_strike_and_type_split():
    found, _ = _spans(
        "buying 210p expiring soon",
        [{"text": "210", "label": "strike"}, {"text": "p", "label": "option_type"}],
    )
    assert [f[:2] for f in found] == [("210", "strike"), ("p", "option_type")]


def test_invalid_label_dropped():
    _, dropped = _spans("$A 1 C", [{"text": "$A", "label": "company"}])
    assert dropped == [("$A", "company", "invalid")]


def test_few_shot_examples_are_consistent():
    """Every example's entities must be locatable and use a known label."""
    for ex in FEW_SHOT_EXAMPLES:
        found, dropped = _spans(ex["input"], ex["output"]["entities"])
        assert not dropped, (ex["input"], dropped)
        assert all(label in LABELS for _, label, _ in found)


def test_build_prompt_batch_lists_all_labels():
    prompt = build_prompt(["a", "b"])
    assert "Input 2: b" in prompt
    assert all(f'"{label}"' in prompt for label in LABELS)
    json.dumps(FEW_SHOT_EXAMPLES)
