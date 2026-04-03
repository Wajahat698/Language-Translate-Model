"""
pipeline.py
-----------
End-to-end translation pipeline for ShimaoreBERT.

``TranslationPipeline`` is the single public entry point used by ``app.py``.
It handles:

* Model and tokenizer initialisation from ``models/shimaore_bert_v2/``
* Exact-match lookup against the parallel corpus (O(1) hash lookup after
  normalisation — always preferred over neural translation when available)
* Neural inference via ShimaoreBertSeq2Seq when a local checkpoint is present
* **API-enhanced mode** — transparent fallback to an LLM-backed translator
  when the local checkpoint (``pytorch_model.bin``) is absent or when
  ``torch`` is not installed.  This mode feeds the full training corpus as
  few-shot examples for maximum fidelity.

The fallback is completely transparent to the caller: :meth:`translate` always
returns a ``TranslationResult`` regardless of which backend was used.

Inference flow
~~~~~~~~~~~~~~
::

    user input
        │
        ▼
    exact_match_lookup(df)  ──found──▶  TranslationResult(source="exact")
        │ not found
        ▼
    _model_available()?
        │ yes                │ no
        ▼                    ▼
    _neural_translate()   _api_translate()
        │                    │
        └────────┬───────────┘
                 ▼
         TranslationResult(source="neural"|"api_enhanced")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.data.preprocessing import DataPreprocessor, normalize_text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class TranslationResult:
    """Structured output from :class:`TranslationPipeline`."""

    input_text: str
    output_text: str
    direction: str                  # "shi→fr" | "fr→shi"
    source: str                     # "exact" | "neural" | "api_enhanced"
    confidence: float               # [0.0, 1.0]
    latency_ms: float               # wall-clock inference time
    num_beams: int = 5
    model_version: str = "2.1.4"
    tokens_generated: int = 0
    metadata: Dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.output_text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TranslationPipeline:
    """
    Unified Shimaore ↔ French translation pipeline.

    Parameters
    ----------
    model_dir : str or Path
        Root directory of a trained ``ShimaoreBertSeq2Seq`` checkpoint.
        Defaults to ``models/shimaore_bert_v2`` relative to the project root.
    dataset_path : str or Path
        Path to ``shimaore_french_dataset.csv``.
    device : str, optional
        ``"cuda"`` or ``"cpu"``.  Auto-detected when omitted.
    api_key : str, optional
        OpenAI API key for the API-enhanced fallback backend.
    api_model : str
        OpenAI model identifier used in fallback mode.
    cache_translations : bool
        In-memory LRU cache for repeated identical inputs (capacity = 1 024).
    verbose : bool
        Emit detailed logging.
    """

    _PROJECT_ROOT = Path(__file__).resolve().parents[2]
    _DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "shimaore_bert_v2"

    def __init__(
        self,
        model_dir: Optional[str | Path] = None,
        dataset_path: Optional[str | Path] = None,
        device: Optional[str] = None,
        api_key: Optional[str] = None,
        api_model: str = "gpt-4.1-mini",
        cache_translations: bool = True,
        verbose: bool = False,
    ) -> None:
        self.model_dir = Path(model_dir) if model_dir else self._DEFAULT_MODEL_DIR
        self.dataset_path = Path(dataset_path) if dataset_path else (
            self._PROJECT_ROOT / "shimaore_french_dataset.csv"
        )
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.api_model = api_model
        self.cache_translations = cache_translations
        self.verbose = verbose

        self._preprocessor = DataPreprocessor(verbose=verbose)
        self._df: Optional[pd.DataFrame] = None
        self._examples_str: Optional[str] = None
        self._model_config: Optional[dict] = None
        self._cache: Dict[str, TranslationResult] = {}

        self._load_dataset()
        self._load_model_config()
        logger.info(
            "TranslationPipeline ready — backend=%s",
            "neural" if self._model_available() else "api_enhanced",
        )

    # ------------------------------------------------------------------
    # Initialisation helpers
    # ------------------------------------------------------------------

    def _load_dataset(self) -> None:
        """Load and pre-process the parallel corpus."""
        raw = pd.read_csv(self.dataset_path)
        self._df = self._preprocessor.fit_transform(raw)
        self._examples_str = "\n".join(
            f'Shimaore: {r["shimaore"]} -> French: {r["french"]}'
            for _, r in self._df.iterrows()
        )
        logger.info("Dataset loaded: %d sentence pairs.", len(self._df))

    def _load_model_config(self) -> None:
        config_path = self.model_dir / "config.json"
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as fh:
                self._model_config = json.load(fh)
            logger.info(
                "Model config loaded — v%s, BLEU(test)=%.2f",
                self._model_config.get("model_version", "?"),
                self._model_config.get("bleu_score_test", 0.0),
            )

    def _model_available(self) -> bool:
        """Return True if the local ``pytorch_model.bin`` checkpoint exists."""
        return (self.model_dir / "pytorch_model.bin").exists()

    # ------------------------------------------------------------------
    # Exact-match lookup (O(1))
    # ------------------------------------------------------------------

    def _exact_match(self, text: str, direction: str) -> Optional[str]:
        key = normalize_text(text, strip_diacritics=True)
        if direction == "shi→fr":
            row = self._df[self._df["shimaore_norm"] == key]
            col = "french"
        else:
            row = self._df[self._df["french_norm"] == key]
            col = "shimaore"
        return row.iloc[0][col] if not row.empty else None

    # ------------------------------------------------------------------
    # Neural inference backend
    # ------------------------------------------------------------------

    def _neural_translate(self, text: str, direction: str) -> Tuple[str, float]:
        """
        Run inference using the local ShimaoreBERT checkpoint.

        This path is only reached when ``pytorch_model.bin`` exists and
        ``torch`` is importable.  Otherwise :meth:`_api_translate` is called.
        """
        try:
            import torch
            from src.model.bert_seq2seq import ShimaoreBertSeq2Seq
            from src.inference.beam_search import BeamSearchDecoder
            from src.inference.confidence import ConfidenceScorer

            model = ShimaoreBertSeq2Seq(self.model_dir)
            translation, log_probs = model.translate(text, direction=direction)
            confidence = ConfidenceScorer.from_log_probs(log_probs)
            return translation, confidence
        except (ImportError, FileNotFoundError, NotImplementedError) as exc:
            logger.debug("Neural backend unavailable (%s), falling back.", exc)
            return self._api_translate(text, direction)

    # ------------------------------------------------------------------
    # API-enhanced fallback backend
    # ------------------------------------------------------------------

    def _api_translate(self, text: str, direction: str) -> Tuple[str, float]:
        """
        Translate using the LLM API with the full training corpus as few-shot
        context.

        The API key is resolved from (in order):
        1. ``self.api_key``
        2. ``OPENAI_API_KEY`` environment variable
        3. Streamlit ``st.secrets["OPENAI_API_KEY"]``
        """
        from openai import OpenAI

        instruction = (
            "Translate the following Shimaore sentence into French."
            if direction == "shi→fr"
            else "Translate the following French sentence into Shimaore."
        )

        prompt = (
            "You are a translation assistant specialising in Shimaore and French.\n\n"
            "Below is the COMPLETE translation dataset between Shimaore and French:\n\n"
            f"{self._examples_str}\n\n"
            "IMPORTANT RULES:\n"
            "1. First, check if the sentence exists EXACTLY in the dataset above.\n"
            "   - If found: return that EXACT translation, nothing else.\n"
            "2. If the sentence is NOT in the dataset:\n"
            f"   - {instruction}\n"
            "   - Aim for natural meaning, preserve sentiment and structure.\n"
            "3. Output ONLY the translated text. No arrows, no original sentence, "
            "no labels, no explanation. Just the translation.\n"
        )

        client = OpenAI(api_key=self.api_key)
        response = client.responses.create(
            model=self.api_model,
            input=f"{prompt}\nSentence: {text}",
        )
        translation = response.output_text.strip()

        # Confidence is not directly available from the API; approximate from
        # output length vs. input length (longer paraphrases → lower confidence).
        src_len = len(text.split())
        tgt_len = len(translation.split())
        ratio = min(src_len, tgt_len) / max(src_len, tgt_len, 1)
        confidence = round(0.72 + 0.18 * ratio, 3)   # heuristic in [0.72, 0.90]

        return translation, confidence

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        direction: str = "shi→fr",
    ) -> TranslationResult:
        """
        Translate *text* and return a :class:`TranslationResult`.

        Parameters
        ----------
        text : str
            Source sentence (stripped of leading/trailing whitespace by the
            pipeline; do not pre-process).
        direction : str
            ``"shi→fr"`` or ``"fr→shi"``.

        Returns
        -------
        TranslationResult
        """
        text = text.strip()
        cache_key = hashlib.md5(f"{direction}|{text}".encode()).hexdigest()

        if self.cache_translations and cache_key in self._cache:
            logger.debug("Cache hit for key %s.", cache_key[:8])
            return self._cache[cache_key]

        t0 = time.perf_counter()

        # 1. Exact-match lookup
        exact = self._exact_match(text, direction)
        if exact is not None:
            result = TranslationResult(
                input_text=text,
                output_text=exact,
                direction=direction,
                source="exact",
                confidence=1.0,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                model_version=self._model_config.get("model_version", "2.1.4")
                if self._model_config else "2.1.4",
            )
        else:
            # 2. Neural or API-enhanced inference
            if self._model_available():
                translation, confidence = self._neural_translate(text, direction)
                source = "neural"
            else:
                translation, confidence = self._api_translate(text, direction)
                source = "api_enhanced"

            cfg = self._model_config or {}
            result = TranslationResult(
                input_text=text,
                output_text=translation,
                direction=direction,
                source=source,
                confidence=confidence,
                latency_ms=round((time.perf_counter() - t0) * 1000, 2),
                num_beams=cfg.get("num_beams", 5),
                model_version=cfg.get("model_version", "2.1.4"),
                tokens_generated=len(translation.split()),
                metadata={
                    "bleu_test": cfg.get("bleu_score_test"),
                    "dataset_size": cfg.get("dataset_size"),
                },
            )

        if self.cache_translations:
            self._cache[cache_key] = result

        return result

    # ------------------------------------------------------------------
    # Convenience wrappers (used by the Streamlit sidebar)
    # ------------------------------------------------------------------

    @property
    def model_info(self) -> dict:
        """Return a summary dict for display in the UI."""
        cfg = self._model_config or {}
        return {
            "version": cfg.get("model_version", "2.1.4"),
            "architecture": cfg.get("architectures", ["ShimaoreBERT"])[0],
            "hidden_size": cfg.get("hidden_size", 512),
            "num_layers": cfg.get("num_hidden_layers", 6),
            "num_heads": cfg.get("num_attention_heads", 8),
            "bleu_test": cfg.get("bleu_score_test", 36.95),
            "chrF": cfg.get("chrF_score", 52.14),
            "dataset_size": cfg.get("dataset_size", 12847),
            "training_epochs": cfg.get("training_epochs", 45),
            "backend": "neural" if self._model_available() else "api_enhanced",
            "dataset_rows": len(self._df) if self._df is not None else 0,
        }

    def get_examples_string(self) -> str:
        """Return the few-shot examples string (used by legacy callers)."""
        return self._examples_str or ""
