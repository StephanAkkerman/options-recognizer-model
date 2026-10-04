"""Entity label set and the natural-language descriptions GLiNER2 conditions on.

GLiNER2 reads these descriptions at both train and inference time, so editing
them changes model behavior; `src/analysis/validate_descriptions.py` compares
variants without retraining. Keep the keys in sync with `data/label_studio.xml`
and `LABELS` in `utils/synthetic/auto_label.py` (a test enforces the latter).
"""

ENTITY_DESCRIPTIONS = {
    "ticker": (
        "The stock or ETF symbol an option contract is written on, usually 1-5 "
        "letters and often preceded by a dollar sign (e.g., $AAPL, SPY, $T). "
        "MUST NOT be a company name, an @handle, or slang (YOLO, NFA, OTM, ITM)."
    ),
    "strike": (
        "The strike price of an option contract, e.g. 350, 162.5, or $1300 in "
        "'$1300 calls'. MUST NOT be the stock's current price, a premium or "
        "per-contract price, a date, or a percentage."
    ),
    "option_type": (
        "Whether the contract is a call or a put, in any written form: C, P, "
        "call, calls, put, puts, c, p. Only the call/put word itself, not words "
        "like buyer or seller."
    ),
    "expiry": (
        "The expiration date of an option contract: 10/16, 11/06/2026, 6/17/27, "
        "November 20, 2026, December, or relative like '2 days'. MUST NOT be the "
        "date a report was posted (e.g. the 10/2 in '10/2 Notable Flow')."
    ),
    "premium": (
        "The total dollar size of the trade, e.g. $1.2M, $993K, 23 million. "
        "MUST NOT be a strike, the per-contract price, or the stock price."
    ),
    "price": (
        "The per-contract price or average fill, e.g. .15 in '@ .15' or 4.37 in "
        "'4.37avg'. Only the number. MUST NOT be a strike, premium, or the "
        "underlying's stock price."
    ),
}

LABELS = tuple(ENTITY_DESCRIPTIONS)
