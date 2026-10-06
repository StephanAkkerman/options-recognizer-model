"""Registry of the GLiNER2 base models we train an adapter for.

One adapter (and one Hub repo) is published per base model, so users can trade
accuracy for speed by picking a smaller base. A slug is the model name without
the org prefix and doubles as the adapter-folder / Hub-repo suffix.
"""

import copy
import re

from src.core.spans import CURRENT_SPLITTER, adapter_splitter_name, make_splitter

ADAPTER_PREFIX = "options_adapter"
HUB_REPO_PREFIX = "options-recognizer"

# params_m is the base-model parameter count in millions (from the Hub
# metadata); it is shown next to accuracy so the size/accuracy trade-off is
# visible in the benchmark table and model cards.
BASE_MODELS = {
    "gliner2.5-small-v1": {
        "hf_id": "fastino/gliner2.5-small-v1",
        "display": "GLiNER2.5 Small v1",
        "params_m": 74,
    },
    "gliner2.5-base-v1": {
        "hf_id": "fastino/gliner2.5-base-v1",
        "display": "GLiNER2.5 Base v1",
        "params_m": 194,
    },
    "gliner2-base-v1": {
        "hf_id": "fastino/gliner2-base-v1",
        "display": "GLiNER2 Base v1",
        "params_m": 208,
    },
    "gliner2-large-v1": {
        "hf_id": "fastino/gliner2-large-v1",
        "display": "GLiNER2 Large v1",
        "params_m": 486,
    },
}
DEFAULT_BASE = "gliner2-large-v1"

# Per-base training overrides, merged over the defaults in `train.py`
# (keys: batch_size, lora_rank, encoder_lr, task_lr, epochs). Empty until a
# base is shown to need different settings; smaller models can often take a
# larger batch size or a higher learning rate.
TRAIN_OVERRIDES = {slug: {} for slug in BASE_MODELS}


def resolve_base(spec):
    """Map a slug (``gliner2-large-v1``) or Hub id (``fastino/...``) to a slug."""
    spec = (spec or DEFAULT_BASE).strip()
    if spec in BASE_MODELS:
        return spec
    for slug, info in BASE_MODELS.items():
        if spec == info["hf_id"]:
            return slug
    raise ValueError(
        f"Unknown base model {spec!r}. Choose from: {', '.join(BASE_MODELS)}"
    )


def resolve_bases(spec):
    """Like `resolve_base` but also accepts ``all`` and comma-separated lists."""
    if spec == "all":
        return list(BASE_MODELS)
    return [resolve_base(s) for s in spec.split(",") if s.strip()]


def adapter_dir_name(slug, version):
    return f"{ADAPTER_PREFIX}_{slug}_v{version}"


def parse_adapter_dir(folder_name):
    """``options_adapter_<slug>_v<N>`` -> ``(slug, N)``; ``None`` if not ours."""
    match = re.fullmatch(rf"{ADAPTER_PREFIX}_(?P<slug>.+)_v(?P<v>\d+)", folder_name)
    if match and match["slug"] in BASE_MODELS:
        return match["slug"], int(match["v"])
    return None


def adapter_label(slug, version):
    """Name used in the benchmark store and tables."""
    return f"{BASE_MODELS[slug]['display']} + Adapter v{version}"


def base_label(slug):
    return f"{BASE_MODELS[slug]['display']} (no adapter)"


def default_repo_id(owner, slug):
    return f"{owner}/{HUB_REPO_PREFIX}-{slug}"


def load_extractor(slug, **kwargs):
    """Load a base model with `AutoExtractor`.

    GLiNER2.5 checkpoints use the boundary architecture and cannot be loaded
    with ``GLiNER2.from_pretrained`` (the legacy span loader); `AutoExtractor`
    reads the checkpoint's ``architecture`` field and dispatches to the right
    class for both generations.
    """
    from gliner2 import AutoExtractor

    model = AutoExtractor.from_pretrained(BASE_MODELS[slug]["hf_id"], **kwargs)
    # Train, benchmark and analysis all load through here, so they share one
    # tokenization; see AlnumBoundarySplitter for why the default is unusable.
    model.set_word_splitter(make_splitter(CURRENT_SPLITTER))
    return model


def load_adapted(base_model, adapter_path):
    """Copy of `base_model` with the adapter loaded and its own splitter set.

    The splitter is part of the adapter: scoring an older adapter with the
    current one silently wrecks its recall.
    """
    model = copy.deepcopy(base_model)
    model.load_adapter(adapter_path)
    model.set_word_splitter(make_splitter(adapter_splitter_name(adapter_path)))
    return model
