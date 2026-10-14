"""Text cleaning and Unicode normalization utilities for Indic stance data."""

import re
import unicodedata


def clean_indic_text(text: str) -> str:
    """Perform basic Unicode-safe normalization and whitespace cleaning on Indic text.

    Args:
        text: Input string (Hindi, Bengali, or code-mixed text).

    Returns:
        Cleaned, normalized string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Unicode normalization (NFKC decomposes compatibility chars and recomposes standard ones)
    text = unicodedata.normalize("NFKC", text)

    # 2. Standardize whitespace (replace tabs, newlines, zero-width spaces, multiple spaces)
    text = re.sub(r"[\u200B-\u200D\uFEFF]", "", text)  # remove zero-width chars
    text = re.sub(r"\s+", " ", text)

    # 3. Strip leading/trailing whitespace
    return text.strip()
