"""Tests for the post-labeling pipeline (training data, scoring, splits, analysis).

Everything here is pure Python: torch/gliner2 are only imported lazily by the
code paths that run a model, which these tests never touch.
"""

import json

import pytest

from src.analysis.error_analysis import categorize_errors, make_context
from src.analysis.validate_descriptions import collect_delta_records
from src.core.benchmark import (
    chunk_text_for_inference,
    collect_pred_per_doc,
    get_all_adapters,
    latest_adapters,
    parse_all_label_studio_exports,
    prepare_eval_inputs,
    score_predictions,
)
from src.core.labels import ENTITY_DESCRIPTIONS, LABELS
from src.core.models import (
    BASE_MODELS,
    adapter_dir_name,
    default_repo_id,
    parse_adapter_dir,
    resolve_base,
    resolve_bases,
)
from src.core.results_store import compute_test_set_hash
from src.core.train import (
    entity_in_chunk,
    get_next_version,
    parse_all_labeled_data,
    task_to_samples,
)
from src.maintenance.split_test_set import run as run_split
from utils.hf.push_dataset_to_hf import load_clean_gold_dataset
from utils.hf.push_model_to_hf import build_model_card, latest_benchmark
from utils.synthetic import auto_label


def _task(task_id, text, spans):
    """Label Studio task; `spans` is [(substring, label), ...] located in order."""
    results, cursor = [], 0
    for sub, label in spans:
        start = text.index(sub, cursor)
        cursor = start + len(sub)
        results.append(
            {
                "type": "labels",
                "value": {
                    "start": start,
                    "end": cursor,
                    "text": sub,
                    "labels": [label],
                },
            }
        )
    return {
        "id": task_id,
        "data": {"text": text},
        "annotations": [{"was_cancelled": False, "result": results}],
    }


TEXT = "$AAPL 350 C 11/06/2026 $1.2M 4.37avg"
SPANS = [
    ("$AAPL", "ticker"),
    ("350", "strike"),
    ("C", "option_type"),
    ("11/06/2026", "expiry"),
    ("$1.2M", "premium"),
    ("4.37", "price"),
]


def _write(path, tasks):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(tasks), encoding="utf-8")


def test_labels_are_consistent_everywhere():
    assert LABELS == tuple(ENTITY_DESCRIPTIONS)
    assert LABELS == auto_label.LABELS


def test_entity_in_chunk_is_token_aware():
    assert entity_in_chunk("350", "$AAPL 350 C")
    assert not entity_in_chunk("350", "$3500 C")
    assert not entity_in_chunk("10", "$100K")
    assert entity_in_chunk("p", "buying 210p today")  # fused strike+type
    assert not entity_in_chunk("p", "up 5%")


def test_task_to_samples_groups_entities_by_label():
    (sample,) = task_to_samples(_task(1, TEXT, SPANS))
    assert sample["input"] == TEXT
    ents = sample["output"]["entities"]
    assert ents["strike"] == ["350"]
    assert set(ents) == set(LABELS)
    assert sample["output"]["entity_descriptions"] == ENTITY_DESCRIPTIONS


def test_parse_all_labeled_data_excludes_test_ids(tmp_path):
    labeled, test = tmp_path / "labeled", tmp_path / "test"
    tasks = [_task(i, TEXT, SPANS) for i in range(1, 21)]
    _write(labeled / "a.json", tasks)
    _write(test / "test_a.json", tasks[:5])
    train, val = parse_all_labeled_data(str(labeled), test_folder=str(test))
    assert len(train) + len(val) == 15


def test_get_next_version_and_adapter_discovery(tmp_path):
    models = tmp_path / "models"
    for name in (
        "options_adapter_gliner2-large-v1_v3",
        "options_adapter_gliner2-large-v1_v1",
        "options_adapter_gliner2.5-small-v1_v2",
        "options_adapter_unknown-model_v9",  # not in the registry: ignored
    ):
        weights = models / name / "final"
        weights.mkdir(parents=True)
        (weights / "adapter_model.safetensors").write_bytes(b"x")
    assert get_next_version(str(models), "gliner2-large-v1") == 4
    assert get_next_version(str(models), "gliner2.5-small-v1") == 3
    assert get_next_version(str(models), "gliner2-base-v1") == 1
    found = get_all_adapters(str(models))
    assert [(a["base"], a["version"]) for a in found] == [
        ("gliner2.5-small-v1", 2),
        ("gliner2-large-v1", 1),
        ("gliner2-large-v1", 3),
    ]
    assert [a["version"] for a in latest_adapters(found)] == [2, 3]
    only_large = get_all_adapters(str(models), bases=["gliner2-large-v1"])
    assert {a["base"] for a in only_large} == {"gliner2-large-v1"}


def test_model_registry_roundtrips():
    assert resolve_base("fastino/gliner2.5-base-v1") == "gliner2.5-base-v1"
    assert resolve_bases("all") == list(BASE_MODELS)
    assert resolve_bases("gliner2-base-v1, gliner2-large-v1") == [
        "gliner2-base-v1",
        "gliner2-large-v1",
    ]
    with pytest.raises(ValueError):
        resolve_base("nope")
    for slug in BASE_MODELS:
        assert parse_adapter_dir(adapter_dir_name(slug, 7)) == (slug, 7)
    assert parse_adapter_dir("options_adapter_v1") is None
    assert (
        default_repo_id("me", "gliner2-base-v1")
        == "me/options-recognizer-gliner2-base"
    )


def test_model_card_has_metrics_and_base():
    result = {
        "metrics": {
            **score_predictions(
                [{(0, 1, "ticker")}], [{(0, 1, "ticker")}], list(LABELS)
            ),
            "speed": {"ms_per_doc": 12.5},
        },
        "params": {"device": "cuda"},
    }
    card = build_model_card("gliner2.5-small-v1", 2, "me/repo", 0.6, result, "abc123")
    assert "base_model: fastino/gliner2.5-small-v1" in card
    assert "| ticker | 100.0% | 100.0% | 100.0% |" in card
    assert "12.5 ms/doc" in card and "**0.6**" in card
    assert "speed |" not in card


def test_latest_benchmark_picks_newest():
    store = {
        "results": {
            "m": {
                "old": {"evaluated_at": "2026-01-01T00:00:00Z"},
                "new": {"evaluated_at": "2026-02-01T00:00:00Z"},
            }
        }
    }
    assert latest_benchmark(store, "m")[0] == "new"
    assert latest_benchmark(store, "missing") == (None, None)


def test_chunk_offsets_map_back_to_absolute_positions():
    text = "alpha beta gamma delta epsilon zeta"
    chunks = chunk_text_for_inference(text, chunk_word_size=3, overlap_words=1)
    for chunk, offset in chunks:
        assert text[offset : offset + len(chunk)] == chunk


def test_exact_span_scoring_penalizes_wrong_boundaries_and_labels():
    gold = [{(0, 5, "ticker"), (6, 9, "strike")}]
    pred = [{(0, 5, "ticker"), (6, 9, "price"), (10, 11, "option_type")}]
    scores = score_predictions(pred, gold, list(LABELS))
    assert scores["ticker"]["f1"] == 1.0
    assert scores["strike"]["r"] == 0.0 and scores["price"]["p"] == 0.0
    assert round(scores["overall"]["p"], 3) == round(1 / 3, 3)


def test_collect_pred_per_doc_applies_chunk_offset_and_dedupes():
    flat_chunks = [(0, "a 10", 4), (0, "a 10", 4)]
    out = {"entities": {"strike": [{"start": 2, "end": 4}]}}
    preds = collect_pred_per_doc([out, out], flat_chunks, [(0, 2)])
    assert preds == [{(6, 8, "strike")}]


def test_prepare_eval_inputs_and_hash_are_stable(tmp_path):
    folder = tmp_path / "test"
    _write(folder / "t.json", [_task(1, TEXT, SPANS)])
    dataset = parse_all_label_studio_exports(str(folder))
    _, _, gold, by_label = prepare_eval_inputs(dataset, list(LABELS))
    assert len(gold[0]) == 6 and len(by_label[0]["strike"]) == 1
    assert compute_test_set_hash(dataset) == compute_test_set_hash(list(dataset))


def test_split_test_set_is_deterministic(tmp_path):
    labeled, test = tmp_path / "labeled", tmp_path / "test"
    _write(labeled / "a.json", [_task(i, TEXT, SPANS) for i in range(20)])

    def ids():
        run_split(
            seed=1, labeled_folder=str(labeled), test_folder=str(test), quiet=True
        )
        return sorted(
            t["id"] for t in json.loads((test / "test_a.json").read_text("utf-8"))
        )

    first = ids()
    assert len(first) == 3 and first == ids()


def test_error_analysis_buckets():
    text = "$A 10 C 5 x"
    dataset = [{"text": text}]
    gold = [{(0, 2, "ticker"), (3, 5, "strike"), (6, 7, "option_type")}]
    pred = [
        {
            (0, 2, "ticker"),  # TP
            (3, 5, "price"),  # confusion: same span, wrong label
            (6, 8, "option_type"),  # boundary: overlaps gold (6, 7)
            (9, 10, "price"),  # pure FP
        }
    ]
    cats = categorize_errors(pred, gold, dataset)
    assert [c["pred_label"] for c in cats["confusion"]] == ["price"]
    assert len(cats["boundary"]) == 1 and len(cats["pure_fp"]) == 1
    assert cats["pure_fn"] == []
    assert cats["per_doc"][0]["n_errors"] == 5
    assert "[10]" in make_context(text, 3, 5, 3)


def test_delta_records_track_description_changes():
    dataset = [{"text": "$A 10 C"}]
    gold = [{(0, 2, "ticker")}]
    old = [{(0, 2, "ticker"), (3, 5, "strike")}]  # FP on the strike
    new = [{(0, 2, "ticker")}]  # candidate suppresses it
    suppressed, new_fp, recovered, lost = collect_delta_records(old, new, gold, dataset)
    assert [r["text"] for r in suppressed] == ["10"]
    assert not (new_fp or recovered or lost)


def test_push_dataset_loader_filters_bad_spans(tmp_path):
    good = _task(1, TEXT, SPANS)
    bad = _task(2, TEXT, [("350", "strike")])
    bad["annotations"][0]["result"].append(
        {
            "type": "labels",
            "value": {"start": 0, "end": 999, "text": "x", "labels": ["strike"]},
        }
    )
    unknown = _task(3, TEXT, [("350", "strike")])
    unknown["annotations"][0]["result"][0]["value"]["labels"] = ["company"]
    _write(tmp_path / "a.json", [good, bad, unknown])
    _write(tmp_path / "augmented_a.json", [good])
    records = load_clean_gold_dataset(str(tmp_path))
    assert [len(r["entities"]) for r in records] == [6, 1, 0]
