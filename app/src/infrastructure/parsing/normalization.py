"""Portuguese text normalization for extracted spans.

Functions here operate on a single span's text (no cross-span context)
except [should_rejoin], which decides whether a hyphen at the end of one
line should be merged with the start of the next -- that decision needs to
see both lines, so it stays a pair-based predicate rather than a pipeline
stage.

``NORMALIZATION_VERSION`` feeds the extraction cache key: bump it whenever
any function here changes behavior, so stale cached output is invalidated.
"""

import re
import unicodedata

NORMALIZATION_VERSION = "v1"

_SOFT_HYPHEN = "\xad"
_NBSP = "\xa0"

_QUOTE_REPLACEMENTS = {
    "“": '"',  # left double quotation mark
    "”": '"',  # right double quotation mark
    "„": '"',  # double low-9 quotation mark
    "‟": '"',  # double high-reversed-9 quotation mark
    "«": '"',  # left-pointing double angle quotation mark (guillemet)
    "»": '"',  # right-pointing double angle quotation mark (guillemet)
    "‘": "'",  # left single quotation mark
    "’": "'",  # right single quotation mark
    "‚": "'",  # single low-9 quotation mark
    "‛": "'",  # single high-reversed-9 quotation mark
    "‹": "'",  # single left-pointing angle quotation mark
    "›": "'",  # single right-pointing angle quotation mark
}
_QUOTE_PATTERN = re.compile("|".join(re.escape(char) for char in _QUOTE_REPLACEMENTS))

_LOWERCASE_PT = "a-zàáâãäåçèéêëìíîïñòóôõöùúûüýÿ"
_LINE_END_HYPHEN = re.compile(rf"[{_LOWERCASE_PT}]-\s*$", re.IGNORECASE)
_NEXT_LINE_START_LOWERCASE = re.compile(rf"^\s*[{_LOWERCASE_PT}]")


def apply_nfkc(text: str) -> str:
    """Apply Unicode NFKC normalization, folding ligatures to base letters."""
    return unicodedata.normalize("NFKC", text)


def strip_soft_hyphens(text: str) -> str:
    """Remove soft hyphen characters (U+00AD), unconditionally."""
    return text.replace(_SOFT_HYPHEN, "")


def normalize_nbsp(text: str) -> str:
    """Replace non-breaking spaces (U+00A0) with regular spaces."""
    return text.replace(_NBSP, " ")


def normalize_quotes(text: str) -> str:
    """Normalize curly-quote and guillemet variants to straight quotes."""
    return _QUOTE_PATTERN.sub(lambda match: _QUOTE_REPLACEMENTS[match.group()], text)


def normalize_text(text: str) -> str:
    """Run the per-span pipeline: NFKC, soft hyphens, nbsp, quotes."""
    text = apply_nfkc(text)
    text = strip_soft_hyphens(text)
    text = normalize_nbsp(text)
    text = normalize_quotes(text)
    return text


def should_rejoin(line_end_text: str, next_line_start_text: str) -> bool:
    """Decide whether a line-end hyphen marks a word wrapped at the margin.

    Heuristic: a letter immediately followed by a hyphen at the end of one
    line, continued by a lowercase letter at the very start of the next
    line. This cannot distinguish that from a genuine hyphenated compound
    that happens to break at the same point (e.g. "sócio-" / "cultural") --
    both look identical structurally, and disambiguating would need a
    Portuguese dictionary, which is out of scope here. An uppercase or
    non-letter continuation (e.g. "RCF-" / "V") is left alone, since that is
    far more likely to be a deliberate hyphen than a wrap artifact.
    """
    return bool(_LINE_END_HYPHEN.search(line_end_text)) and bool(
        _NEXT_LINE_START_LOWERCASE.match(next_line_start_text)
    )
