import json

import pytest

from src.core.spans import (
    CURRENT_SPLITTER,
    AlnumBoundarySplitter,
    SPLITTERS,
    adapter_splitter_name,
    make_splitter,
)


def tokens(text):
    return [t for t, _, _ in AlnumBoundarySplitter()(text)]


@pytest.mark.parametrize(
    "text, expected",
    [
        ("1.58avg", ["1.58", "avg"]),
        ("375C", ["375", "c"]),
        ("11/20exp", ["11/20", "exp"]),
        ("$5.1M", ["$", "5.1", "m"]),
        ("10/16/2026", ["10/16/2026"]),
        ("(9/4)", ["(", "9/4", ")"]),
        ("4,925/5,042", ["4,925", "/", "5,042"]),
        ("@ .30", ["@", ".", "30"]),
        ("@everyone", ["@everyone"]),
    ],
)
def test_splitter_tokens(text, expected):
    assert tokens(text) == expected


def test_splitter_offsets_index_original_text():
    text = "$TSLA 375C 11/20exp 1.58avg"
    for token, start, end in AlnumBoundarySplitter()(text, lower=False):
        assert text[start:end] == token


def test_gold_spans_align_to_token_boundaries():
    # The reason the splitter exists: these spans ended mid-word with the stock
    # splitter, so training silently dropped them.
    text = "$TSLA 375C 11/20exp $1.6M 1.58avg"
    toks = list(AlnumBoundarySplitter()(text))
    starts = {s for _, s, _ in toks}
    ends = {e for _, _, e in toks}
    for span in ("375", "11/20", "1.58", "$1.6M"):
        s = text.index(span)
        assert s in starts and s + len(span) in ends, span


def test_adapter_splitter_name_reads_config_and_defaults_to_stock(tmp_path):
    def write(name, cfg):
        d = tmp_path / name
        d.mkdir()
        if cfg is not None:
            (d / "recognizer_config.json").write_text(json.dumps(cfg))
        return str(d)

    assert adapter_splitter_name(write("none", None)) == "whitespace"
    assert adapter_splitter_name(write("old", {"threshold": 0.75})) == "whitespace"
    assert (
        adapter_splitter_name(write("v2", {"word_splitter": "alnum-v1"})) == "alnum-v1"
    )
    current = {"word_splitter": CURRENT_SPLITTER}
    assert adapter_splitter_name(write("v3", current)) == CURRENT_SPLITTER


def test_every_registered_splitter_is_buildable():
    for name in SPLITTERS:
        assert make_splitter(name)


def test_entity_must_be_whole_tokens_in_chunk():
    from src.core.train import entity_in_chunk

    assert entity_in_chunk("1.58", "$1.6M 1.58avg")
    assert entity_in_chunk("375", "$TSLA 375C")
    assert not entity_in_chunk("65", "$DRAM: Jan 65/70 calls")
    assert not entity_in_chunk("1", "$1.2M")
