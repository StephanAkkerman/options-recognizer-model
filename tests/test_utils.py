import json
from argparse import Namespace

from utils.synthetic.auto_label import (
    FEW_SHOT_EXAMPLES,
    LABELS,
    build_prompt,
    extract_entities,
    parse_response_to_task,
    reprocess_raw,
    save_raw,
    _save_response,
)
from utils.labeling.clean_data import clean_records, clean_text, template_key
from utils.hf.hf_utils import build_dataset_card, read_jsonl, write_jsonl


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


def test_plural_fused_type_needs_whole_token():
    text = "Trimming $GLXY 9/18 40Cs at +50%"
    found, dropped = _spans(text, [{"text": "Cs", "label": "option_type"}])
    assert found == [("Cs", "option_type", "Cs")] and not dropped
    # The bare "C" is inside the longer token "Cs", so it can't be located.
    _, dropped = _spans(text, [{"text": "C", "label": "option_type"}])
    assert dropped == [("C", "option_type", "not_found")]


def test_task_stores_tweet_id():
    task, _ = parse_response_to_task("$A 1 C", {"entities": []}, 1, "123")
    assert task["data"] == {"text": "$A 1 C", "tweet_id": "123"}
    task, _ = parse_response_to_task("$A 1 C", {"entities": []}, 1)
    assert task["data"] == {"text": "$A 1 C"}


def test_recased_entity_matches_whole_token_in_source_casing():
    text = "WHOS HOLDING CALLS\n\n$UNH 350 Call 4/24"
    found, dropped = _spans(text, [{"text": "calls", "label": "option_type"}])
    assert found == [("CALLS", "option_type", "CALLS")] and not dropped


def test_exact_case_match_preferred_over_recased():
    text = "CALLS and calls"
    task, _ = parse_response_to_task(
        text, {"entities": [{"text": "calls", "label": "option_type"}]}, 1
    )
    assert task["annotations"][0]["result"][0]["value"]["start"] == 10


def test_recased_entity_never_matches_inside_longer_token():
    for text in ("$COIN 110 puts", "buying CALLS", "$COINS"):
        _, dropped = _spans(text, [{"text": "c", "label": "option_type"}])
        assert dropped == [("c", "option_type", "not_found")], text


def test_extract_entities_single_and_batch():
    ents = [{"text": "C", "label": "option_type"}]
    assert extract_entities(["a"], {"entities": ents}) == [ents]
    batch = {"results": [{"index": 2, "entities": ents}]}
    assert extract_entities(["a", "b"], batch) == [None, ents]


def test_raw_output_saved_and_reprocessed_without_llm(tmp_path):
    args = Namespace(
        raw_output=str(tmp_path / "raw.jsonl"),
        output=str(tmp_path / "tasks.json"),
        task_id_offset=100,
    )
    texts = ["WHOS HOLDING CALLS", "$A 1 C"]
    response = {
        "results": [
            {"index": 1, "entities": [{"text": "calls", "label": "option_type"}]},
            {"index": 2, "entities": [{"text": "zzz", "label": "strike"}]},
        ]
    }
    _save_response(texts, response, args, [7, 9], ["t7", "t9"])

    raw = read_jsonl(args.raw_output)
    assert [(r["tweet_id"], r["row_index"]) for r in raw] == [("t7", 7), ("t9", 9)]
    assert raw[0]["entities"] == [{"text": "calls", "label": "option_type"}]

    # Reprocessing rebuilds the same tasks (ids, tweet ids, spans) from raw alone.
    before = json.load(open(args.output, encoding="utf-8"))
    for t in before:  # region ids are random
        for r in t["annotations"][0]["result"]:
            r.pop("id")
    reprocess_raw(args)
    after = json.load(open(args.output, encoding="utf-8"))
    for t in after:
        for r in t["annotations"][0]["result"]:
            r.pop("id")
    assert after == before
    assert [t["id"] for t in after] == [107, 109]


def test_save_raw_upserts_by_tweet_id(tmp_path):
    path = str(tmp_path / "raw.jsonl")
    rec = {"tweet_id": "1", "row_index": 0, "text": "x", "entities": []}
    save_raw([rec], path)
    save_raw([{**rec, "entities": [{"text": "x", "label": "ticker"}]}], path)
    rows = read_jsonl(path)
    assert len(rows) == 1 and rows[0]["entities"][0]["text"] == "x"
