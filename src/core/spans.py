"""Token-boundary matching shared by auto-labeling and training."""

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
    word-aligned. Must be used identically at train and inference time.
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

    def __call__(self, text, lower=True):
        for m in self._PATTERN.finditer(text):
            token = m.group()
            yield (token.lower() if lower else token), m.start(), m.end()
