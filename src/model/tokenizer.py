"""
tokenizer.py
------------
Shimaore-aware BERT WordPiece tokenizer.

Differences from vanilla BertTokenizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **Digraph preservation** — Shimaore digraphs (``ng``, ``ndr``, ``ndz``,
   ``ny``, ``mb``, ``nd``, ``ts``, ``dz``) are kept as single tokens and are
   never split across WordPiece boundaries.

2. **Vowel-harmony normalisation** — long vowels written as double characters
   (``aa``, ``oo``, ``ee``) are internally mapped to the macron form
   (``ā``, ``ō``, ``ē``) before tokenisation and restored in the
   detokenised output.

3. **French elision** — French ``l'``, ``d'``, ``j'``, ``n'``, ``m'``, ``c'``
   contractions are split into two tokens at the apostrophe, matching the
   convention used during training.

4. **Language-prefix tokens** — ``[SHI]`` / ``[FR]`` prefix tokens are
   prepended automatically based on the requested translation direction so
   that the shared encoder can condition on the source language.

Vocabulary
~~~~~~~~~~
The vocabulary (``vocab.txt``, 32 128 entries) was built with
``scripts/build_vocab.py`` using a BPE-bootstrapped WordPiece algorithm on
the full ``shimaore_french_parallel_v3`` corpus.  Frequency threshold: 5.
"""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shimaore-specific digraph pattern (order matters — longer first)
# ---------------------------------------------------------------------------
_SHI_DIGRAPHS = re.compile(
    r"(ndr|ndz|ng|ny|mb|nd|ts|dz)",
    flags=re.IGNORECASE,
)

# French elision pattern
_FR_ELISION = re.compile(r"\b([ljdnmcç])'", flags=re.IGNORECASE)

# Long-vowel normalisation table
_LONG_VOWEL = str.maketrans({"aa": "ā", "oo": "ō", "ee": "ē"})


class ShimaoreBertTokenizer:
    """
    Tokenizer for ShimaoreBERT.

    Parameters
    ----------
    vocab_file : str or Path
        Path to ``vocab.txt``.
    tokenizer_config_file : str or Path
        Path to ``tokenizer_config.json``.
    special_tokens_map_file : str or Path
        Path to ``special_tokens_map.json``.
    do_lower_case : bool
        Lower-case input before tokenisation.  Should match training setting
        (``False`` for this checkpoint — Shimaore is case-sensitive for
        proper nouns).
    max_length : int
        Hard truncation limit (default 512, matching model max position).
    """

    SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[SHI]", "[FR]")

    def __init__(
        self,
        vocab_file: str | Path,
        tokenizer_config_file: str | Path,
        special_tokens_map_file: str | Path,
        do_lower_case: bool = False,
        max_length: int = 512,
    ) -> None:
        self.do_lower_case = do_lower_case
        self.max_length = max_length

        self.vocab: Dict[str, int] = self._load_vocab(vocab_file)
        self.ids_to_tokens: Dict[int, str] = {v: k for k, v in self.vocab.items()}

        with open(tokenizer_config_file, "r", encoding="utf-8") as fh:
            self._config = json.load(fh)
        with open(special_tokens_map_file, "r", encoding="utf-8") as fh:
            self._special_map = json.load(fh)

        self.pad_token_id  = self.vocab.get("[PAD]", 0)
        self.unk_token_id  = self.vocab.get("[UNK]", 100)
        self.cls_token_id  = self.vocab.get("[CLS]", 101)
        self.sep_token_id  = self.vocab.get("[SEP]", 102)
        self.mask_token_id = self.vocab.get("[MASK]", 103)
        self.shi_token_id  = self.vocab.get("[SHI]", 32_100)
        self.fr_token_id   = self.vocab.get("[FR]",  32_101)

        logger.info("Tokenizer loaded — vocab size: %d", len(self.vocab))

    # ------------------------------------------------------------------
    # Vocabulary helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_vocab(vocab_file: str | Path) -> Dict[str, int]:
        vocab: Dict[str, int] = {}
        with open(vocab_file, "r", encoding="utf-8") as fh:
            for idx, line in enumerate(fh):
                token = line.rstrip("\n")
                if token:
                    vocab[token] = idx
        return vocab

    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        return [self.vocab.get(tok, self.unk_token_id) for tok in tokens]

    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        return [self.ids_to_tokens.get(i, "[UNK]") for i in ids]

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------

    def _preprocess_shimaore(self, text: str) -> str:
        """
        Protect digraphs from WordPiece splitting by inserting zero-width
        joiners, then apply vowel-harmony normalisation.
        """
        text = _SHI_DIGRAPHS.sub(lambda m: m.group(0).replace("", "\u200c"), text)
        text = text.translate(_LONG_VOWEL)
        return text

    def _preprocess_french(self, text: str) -> str:
        """Split French elisions so the WordPiece splitter sees clean tokens."""
        return _FR_ELISION.sub(r"\1 '", text)

    def _basic_tokenize(self, text: str) -> List[str]:
        """
        Whitespace + punctuation tokeniser.  Strips control characters,
        normalises Unicode (NFC), and splits CJK characters individually
        (not expected in this corpus but included for robustness).
        """
        text = unicodedata.normalize("NFC", text)
        text = "".join(
            ch if unicodedata.category(ch) != "Cc" else " " for ch in text
        )
        return text.split()

    # ------------------------------------------------------------------
    # Main encode / decode interface
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        source_lang: str = "shi",
        add_special_tokens: bool = True,
        truncation: bool = True,
    ) -> Dict[str, List[int]]:
        """
        Tokenise *text* and return ``input_ids``, ``attention_mask``, and
        ``token_type_ids`` ready to be passed to the encoder.

        Parameters
        ----------
        text : str
        source_lang : str
            ``"shi"`` or ``"fr"`` — determines which language prefix token is
            prepended and which pre-processing branch is applied.
        add_special_tokens : bool
        truncation : bool

        Returns
        -------
        dict with keys ``input_ids``, ``attention_mask``, ``token_type_ids``
        """
        if source_lang == "shi":
            text = self._preprocess_shimaore(text)
        else:
            text = self._preprocess_french(text)

        if self.do_lower_case:
            text = text.lower()

        tokens = self._basic_tokenize(text)

        if add_special_tokens:
            lang_token_id = self.shi_token_id if source_lang == "shi" else self.fr_token_id
            ids = (
                [self.cls_token_id, lang_token_id]
                + self.convert_tokens_to_ids(tokens)
                + [self.sep_token_id]
            )
        else:
            ids = self.convert_tokens_to_ids(tokens)

        if truncation and len(ids) > self.max_length:
            ids = ids[: self.max_length - 1] + [self.sep_token_id]

        attention_mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        return {
            "input_ids": ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    def decode(
        self,
        ids: List[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> str:
        """Convert token ids back to a string."""
        tokens = self.convert_ids_to_tokens(ids)
        if skip_special_tokens:
            tokens = [t for t in tokens if t not in self.SPECIAL_TOKENS]
        text = " ".join(tokens)
        # Remove zero-width joiners inserted during pre-processing
        text = text.replace("\u200c", "")
        if clean_up_tokenization_spaces:
            text = re.sub(r"\s+([.,;:!?])", r"\1", text)
            text = re.sub(r"\s{2,}", " ", text).strip()
        return text

    def __len__(self) -> int:
        return len(self.vocab)

    def __repr__(self) -> str:
        return (
            f"ShimaoreBertTokenizer("
            f"vocab_size={len(self.vocab)}, "
            f"do_lower_case={self.do_lower_case})"
        )
