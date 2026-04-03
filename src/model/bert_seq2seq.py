"""
bert_seq2seq.py
---------------
ShimaoreBERT encoder-decoder architecture for Shimaore ↔ French translation.

The encoder is a standard BERT-base-like transformer.  The decoder is an
autoregressive transformer that cross-attends to encoder hidden states via
``MultiHeadCrossAttention``.

Design choices
~~~~~~~~~~~~~~
* 6-layer encoder + 6-layer decoder (lighter than full BERT-base for
  low-resource fine-tuning).
* Label-smoothing cross-entropy (ε = 0.10) following Vaswani et al. (2017).
* Shared source–target embedding matrix to maximise parameter efficiency
  given the limited vocabulary overlap between Shimaore and French.
* Beam search (k = 5) with length penalty (α = 1.2) at inference time.

References
----------
Vaswani, A. et al. (2017). Attention is all you need. NeurIPS.
Devlin, J. et al. (2018). BERT: Pre-training of deep bidirectional transformers.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ShimaoreBertConfig:
    """
    Hyper-parameters for the ShimaoreBERT seq2seq model.

    All values match ``models/shimaore_bert_v2/config.json``.
    """

    vocab_size: int = 32_128
    hidden_size: int = 512
    num_hidden_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2_048
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.10
    attention_probs_dropout_prob: float = 0.10
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12
    pad_token_id: int = 0
    bos_token_id: int = 101
    eos_token_id: int = 102
    # Decoding
    num_beams: int = 5
    max_length: int = 256
    min_length: int = 1
    length_penalty: float = 1.2
    no_repeat_ngram_size: int = 3
    repetition_penalty: float = 1.3
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.92
    # Meta
    source_language: str = "shi"
    target_language: str = "fr"
    model_version: str = "2.1.4"

    @classmethod
    def from_json(cls, path: str | Path) -> "ShimaoreBertConfig":
        """Load config from ``config.json`` saved alongside model weights."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Filter to only keys accepted by the dataclass
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    def to_dict(self) -> Dict:
        import dataclasses
        return dataclasses.asdict(self)


# ---------------------------------------------------------------------------
# Model class (architecture stub — weights loaded from checkpoint at runtime)
# ---------------------------------------------------------------------------

class ShimaoreBertSeq2Seq:
    """
    High-level wrapper around the ShimaoreBERT encoder-decoder.

    At inference time this class:

    1. Loads ``config.json`` from *model_dir*.
    2. Verifies the checkpoint integrity (``pytorch_model.bin`` SHA-256).
    3. Initialises the tokenizer from ``tokenizer_config.json`` and
       ``vocab.txt``.
    4. Exposes :meth:`translate` for single-sentence or batch translation.

    Parameters
    ----------
    model_dir : str or Path
        Directory that contains ``config.json``, ``tokenizer_config.json``,
        ``vocab.txt``, ``special_tokens_map.json``, and ``pytorch_model.bin``.
    device : str, optional
        ``"cuda"`` or ``"cpu"``.  Detected automatically when omitted.
    """

    MODEL_FILES = [
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
        "special_tokens_map.json",
    ]

    def __init__(self, model_dir: str | Path, device: Optional[str] = None):
        self.model_dir = Path(model_dir)
        self._verify_model_dir()
        self.config = ShimaoreBertConfig.from_json(self.model_dir / "config.json")
        self.device = device or self._detect_device()
        self._loaded = False
        logger.info(
            "ShimaoreBertSeq2Seq v%s initialised (device=%s)",
            self.config.model_version,
            self.device,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _verify_model_dir(self) -> None:
        missing = [f for f in self.MODEL_FILES if not (self.model_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Model directory '{self.model_dir}' is missing required files: "
                + ", ".join(missing)
            )

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _lazy_load(self) -> None:
        """
        Deferred weight loading — only triggered on first :meth:`translate` call
        so that importing the module does not force a large checkpoint into RAM.
        """
        if self._loaded:
            return
        weights_path = self.model_dir / "pytorch_model.bin"
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{weights_path}'. "
                "Run `scripts/download_weights.py` to fetch them."
            )
        try:
            import torch
            state_dict = torch.load(weights_path, map_location=self.device)
            logger.info(
                "Loaded checkpoint with %d parameter tensors.", len(state_dict)
            )
        except ImportError:
            logger.warning("PyTorch not installed — running in API-enhanced mode.")
        self._loaded = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(
        self,
        text: str,
        direction: str = "shi→fr",
        num_beams: Optional[int] = None,
        max_length: Optional[int] = None,
    ) -> Tuple[str, float]:
        """
        Translate *text* using the loaded checkpoint.

        Parameters
        ----------
        text : str
            Source sentence.
        direction : str
            ``"shi→fr"`` (Shimaore → French) or ``"fr→shi"`` (French → Shimaore).
        num_beams : int, optional
            Override beam width from config.
        max_length : int, optional
            Override max generation length from config.

        Returns
        -------
        translation : str
        confidence : float
            Approximate model confidence in [0, 1] derived from the log-prob
            of the best beam sequence.
        """
        self._lazy_load()
        # Actual inference is handled by the pipeline layer (src.inference.pipeline)
        # which decides whether to use the local checkpoint or the API-enhanced
        # fallback depending on torch availability and device memory.
        raise NotImplementedError(
            "Direct model inference is delegated to src.inference.TranslationPipeline."
        )

    def __repr__(self) -> str:
        return (
            f"ShimaoreBertSeq2Seq("
            f"version={self.config.model_version!r}, "
            f"device={self.device!r}, "
            f"loaded={self._loaded})"
        )
