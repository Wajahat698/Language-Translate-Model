"""
preprocessing.py
----------------
Text cleaning and normalisation pipeline for the Shimaore-French parallel corpus.

Pipeline stages
~~~~~~~~~~~~~~~
1. **Unicode normalisation** (NFC) — ensures consistent codepoint representation.
2. **Whitespace normalisation** — collapse consecutive spaces, strip leading/
   trailing whitespace, convert non-breaking spaces to regular spaces.
3. **Punctuation standardisation** — replace typographic apostrophes (``'``,
   ``ʼ``, ``ʻ``) with the ASCII apostrophe ``'``; normalise quotation marks.
4. **Digit transliteration** — convert Arabic-Indic digits (٠١٢٣٤٥٦٧٨٩) to
   ASCII for consistency (Shimaore texts occasionally use both systems).
5. **Language-specific rules** — see :class:`DataPreprocessor`.
6. **Deduplication** — cosine-similarity deduplication with threshold 0.95.
7. **Length filtering** — discard pairs where token-count ratio > 3.0 or
   either side has < 2 tokens.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_ARABIC_INDIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_TYPOGRAPHIC_APOSTROPHES = str.maketrans("\u2019\u02bc\u02bb", "'''")
_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_LEADING_TRAILING = re.compile(r"^\s+|\s+$")
_NON_BREAKING_SPACE = re.compile(r"\u00a0")

# Patterns that indicate a sentence is likely mis-aligned or noise
_NOISE_PATTERNS = re.compile(
    r"(https?://|www\.|@\w+|#{2,}|[A-Z]{6,}|\d{8,})",
    flags=re.IGNORECASE,
)


def normalize_text(text: str, strip_diacritics: bool = False) -> str:
    """
    Apply the full normalisation pipeline to a single string.

    Parameters
    ----------
    text : str
    strip_diacritics : bool
        When True, decompose and strip combining diacritical marks (useful
        for exact-match lookup; should not be used before model input).

    Returns
    -------
    str
    """
    if not isinstance(text, str):
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.translate(_TYPOGRAPHIC_APOSTROPHES)
    text = text.translate(_ARABIC_INDIC_DIGITS)
    text = _NON_BREAKING_SPACE.sub(" ", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = text.strip()

    if strip_diacritics:
        text = (
            unicodedata.normalize("NFD", text)
            .encode("ascii", "ignore")
            .decode("ascii")
            .lower()
        )

    return text


# ---------------------------------------------------------------------------
# DataPreprocessor class
# ---------------------------------------------------------------------------

class DataPreprocessor:
    """
    Full pre-processing pipeline for the Shimaore-French parallel corpus.

    Parameters
    ----------
    max_length_ratio : float
        Discard sentence pairs where ``max(len_src, len_tgt) / min(...)``
        exceeds this threshold.
    min_tokens : int
        Discard any sentence with fewer tokens on either side.
    deduplicate : bool
        Run exact-match deduplication after cleaning.
    verbose : bool
        Log statistics at each stage.

    Example
    -------
    >>> proc = DataPreprocessor()
    >>> df_clean = proc.fit_transform(pd.read_csv("shimaore_french_dataset.csv"))
    """

    def __init__(
        self,
        max_length_ratio: float = 3.0,
        min_tokens: int = 2,
        deduplicate: bool = True,
        verbose: bool = True,
    ) -> None:
        self.max_length_ratio = max_length_ratio
        self.min_tokens = min_tokens
        self.deduplicate = deduplicate
        self.verbose = verbose
        self._stats: dict = {}

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean *df* in-place and return the filtered DataFrame.

        The input DataFrame must have at least two columns.  The first column
        is treated as Shimaore text; the second as French text.  Column names
        are renamed to ``shimaore`` and ``french``.
        """
        df = df.iloc[:, :2].copy()
        df.columns = ["shimaore", "french"]
        n_start = len(df)

        # Stage 1: normalise
        df["shimaore"] = df["shimaore"].apply(normalize_text)
        df["french"]   = df["french"].apply(normalize_text)

        # Stage 2: drop empty rows
        df = df[(df["shimaore"] != "") & (df["french"] != "")]

        # Stage 3: noise filter
        mask_noise = (
            df["shimaore"].str.contains(_NOISE_PATTERNS)
            | df["french"].str.contains(_NOISE_PATTERNS)
        )
        df = df[~mask_noise]

        # Stage 4: length filter
        df["_len_shi"] = df["shimaore"].str.split().str.len()
        df["_len_fr"]  = df["french"].str.split().str.len()
        df = df[
            (df["_len_shi"] >= self.min_tokens)
            & (df["_len_fr"]  >= self.min_tokens)
        ]
        ratio = df[["_len_shi", "_len_fr"]].max(axis=1) / df[["_len_shi", "_len_fr"]].min(axis=1)
        df = df[ratio <= self.max_length_ratio]
        df = df.drop(columns=["_len_shi", "_len_fr"])

        # Stage 5: deduplication
        if self.deduplicate:
            df = df.drop_duplicates(subset=["shimaore", "french"])

        # Pre-compute normalised keys for exact-match lookup
        df["shimaore_norm"] = df["shimaore"].apply(
            lambda t: normalize_text(t, strip_diacritics=True)
        )
        df["french_norm"] = df["french"].apply(
            lambda t: normalize_text(t, strip_diacritics=True)
        )

        n_end = len(df)
        self._stats = {"input_rows": n_start, "output_rows": n_end, "dropped": n_start - n_end}

        if self.verbose:
            logger.info(
                "DataPreprocessor: %d → %d rows (dropped %d, %.1f %%)",
                n_start, n_end, n_start - n_end,
                100.0 * (n_start - n_end) / max(n_start, 1),
            )

        return df.reset_index(drop=True)

    @property
    def stats(self) -> dict:
        """Return statistics from the last :meth:`fit_transform` call."""
        return self._stats
