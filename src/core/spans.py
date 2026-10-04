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
