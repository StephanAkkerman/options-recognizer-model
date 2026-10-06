"""Token-boundary matching shared by auto-labeling and training."""

import json
import os
import re


def bounded_pattern(ent_text):
    """Regex for `ent_text` that refuses to match inside a longer token.

    Boundaries depend on the character class at each end: a letter-edge must not
    touch another letter (so "C" skips "$COIN" but matches "210C"), a digit-edge
    must not touch another digit or a decimal continuation (so "350" skips
    "$3500" and "1" skips "$1.2M"). Other edges ("$", "/") are unconstrained.
    """
    first, last = ent_text[0], ent_text[-1]
    if first.isalpha():
        left = r"(?<![A-Za-z])"
    elif first.isdigit() or first == ".":
        left = r"(?<!\d)(?<!\d\.)"
    else:
        left = ""
    if last.isalpha():
        right = r"(?![A-Za-z])"
    elif last.isdigit():
        right = r"(?!\d)(?!\.\d)"
    else:
        right = ""
    return left + re.escape(ent_text) + right


class AlnumBoundarySplitter:
    """GLiNER2 word splitter that also breaks between letters and digits.

    The stock splitter keeps whole words together, so in ``1.58avg``, ``11/20exp``
    and ``375C`` the gold span ends mid-word and can neither be trained on nor
    predicted. Splitting at letter/digit boundaries makes those spans
    word-aligned. Dates (``10/16/2026``) and numbers with decimals or thousands
    separators (``5.1``, ``18,140``) stay whole so a fragment such as the ``10``
    of a date can never be proposed as a strike. Must be used identically at
    train and inference time.
    """

    _PATTERN = re.compile(
        r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |[^\W\d_]+(?:[-_][^\W\d_]+)*
        |\d{1,2}/\d{1,2}(?:/\d{2,4})?(?![\d,.]\d)
        |\d+(?:[.,]\d+)*
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    )

    def __call__(self, text, lower=True):
        for m in self._PATTERN.finditer(text):
            token = m.group()
            yield (token.lower() if lower else token), m.start(), m.end()


class AlnumBoundarySplitterV1(AlnumBoundarySplitter):
    """Earlier revision (adapter v2): numbers broke at every ``.``, ``,`` and ``/``.

    Kept only so adapters trained with it can still be scored faithfully.
    """

    _PATTERN = re.compile(
        r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |[^\W\d_]+(?:[-_][^\W\d_]+)*
        |\d+
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    )


# An adapter is only valid with the splitter it was trained under, so training
# records one of these names in recognizer_config.json and loading reads it back.
SPLITTERS = {
    "whitespace": "whitespace",  # GLiNER2's stock splitter (adapter v1)
    "alnum-v1": AlnumBoundarySplitterV1,
    "alnum-v2": AlnumBoundarySplitter,
}
CURRENT_SPLITTER = "alnum-v2"


def make_splitter(name):
    """Splitter spec for `GLiNER2.set_word_splitter` (an instance or built-in name)."""
    spec = SPLITTERS[name]
    return spec if isinstance(spec, str) else spec()


def adapter_splitter_name(adapter_path):
    """Splitter an adapter was trained with, from its ``recognizer_config.json``.

    Adapters without the field predate custom splitters (adapter v1) and used
    GLiNER2's stock one.
    """
    cfg_path = os.path.join(adapter_path, "recognizer_config.json")
    if not os.path.exists(cfg_path):
        return "whitespace"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f).get("word_splitter", "whitespace")
