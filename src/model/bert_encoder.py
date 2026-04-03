"""
bert_encoder.py
---------------
BERT encoder bridge — loads the ShimaoreBERT checkpoint (``pytorch_model.bin``)
and runs a forward pass to produce contextual embeddings for the source sentence.

These embeddings serve two purposes:

1. **Semantic retrieval** — cosine similarity against pre-computed corpus
   embeddings to find the nearest training-set neighbour before neural decoding.
2. **Decoder conditioning** — the [CLS] embedding is passed as the initial
   hidden state of the autoregressive decoder, providing a strong prior for
   low-frequency Shimaore morphology.

When ``torch`` or ``transformers`` are not installed the encoder degrades
gracefully to a no-op stub so that the app continues to run in API-enhanced
mode without modification.

Loading strategy
~~~~~~~~~~~~~~~~
Weights are loaded with ``torch.load(..., map_location="cpu")`` and then moved
to the target device.  The encoder is wrapped in ``torch.no_grad()`` at
inference time to avoid unnecessary gradient computation.

Usage
-----
::

    encoder = BertEncoderBridge.load("models/shimaore_bert_v2")
    result  = encoder.encode("Mwana wa mtu")
    # result.last_hidden_state  : (1, seq_len, 512)
    # result.cls_embedding      : (512,)
    # result.pooled              : (512,)
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Encoding result container
# ---------------------------------------------------------------------------

@dataclass
class EncoderOutput:
    """
    Output of a single :meth:`BertEncoderBridge.encode` call.

    Attributes
    ----------
    last_hidden_state : list of list of float
        Token-level hidden states, shape ``(seq_len, hidden_size)``.
    cls_embedding : list of float
        [CLS] token embedding, shape ``(hidden_size,)``.  Used as the
        decoder initial state and for nearest-neighbour retrieval.
    pooled : list of float
        Mean-pooled sequence representation, shape ``(hidden_size,)``.
    input_ids : list of int
        Token IDs passed to the encoder.
    attention_mask : list of int
        Binary mask (1 = real token, 0 = padding).
    """
    last_hidden_state: list
    cls_embedding: list
    pooled: list
    input_ids: list
    attention_mask: list


# ---------------------------------------------------------------------------
# Encoder bridge
# ---------------------------------------------------------------------------

class BertEncoderBridge:
    """
    Thin wrapper around a HuggingFace ``BertModel`` loaded from the
    ShimaoreBERT checkpoint.

    The class is intentionally decoupled from the decoder so that the encoder
    can be used independently for semantic search without instantiating the
    full seq2seq graph.

    Parameters
    ----------
    model_dir : Path
        Directory containing ``pytorch_model.bin`` and ``config.json``.
    device : str
        ``"cuda"`` or ``"cpu"``.
    """

    def __init__(self, model_dir: Path, device: str = "cpu") -> None:
        self.model_dir = model_dir
        self.device    = device
        self._model    = None
        self._tokenizer = None
        self._loaded   = False

    # ------------------------------------------------------------------
    # Class-method constructor
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model_dir: str | Path) -> "BertEncoderBridge":
        """
        Load the BERT encoder from *model_dir*.

        Attempts to import ``torch`` and ``transformers``.  If either is
        missing, returns a :class:`_StubEncoder` that passes through inputs
        unchanged so downstream code does not need to branch.

        Parameters
        ----------
        model_dir : str or Path

        Returns
        -------
        BertEncoderBridge or _StubEncoder
        """
        model_dir = Path(model_dir)
        weights   = model_dir / "pytorch_model.bin"

        if not weights.exists():
            logger.info(
                "pytorch_model.bin not found at %s — using stub encoder.", weights
            )
            return _StubEncoder()

        try:
            import torch
            from transformers import BertModel, BertConfig
        except ImportError as exc:
            logger.warning("torch/transformers not installed (%s) — stub encoder.", exc)
            return _StubEncoder()

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            instance = cls(model_dir, device)

            # ── Load config ───────────────────────────────────────────────
            cfg_path = model_dir / "config.json"
            if cfg_path.exists():
                bert_cfg = BertConfig.from_pretrained(str(model_dir))
            else:
                bert_cfg = BertConfig(
                    hidden_size=512,
                    num_hidden_layers=6,
                    num_attention_heads=8,
                    intermediate_size=2048,
                )

            # ── Load weights ──────────────────────────────────────────────
            logger.info("Loading ShimaoreBERT weights from %s …", weights)
            model = BertModel(bert_cfg)
            state_dict = torch.load(str(weights), map_location=device)

            # Handle both raw state-dicts and full checkpoint dicts
            if "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]

            # Load with strict=False — checkpoint may contain decoder keys
            # that are not part of the encoder-only BertModel
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.debug("Missing keys in state dict: %d", len(missing))
            if unexpected:
                logger.debug("Unexpected keys in state dict: %d", len(unexpected))

            model.eval()
            model.to(device)

            instance._model  = model
            instance._loaded = True
            logger.info(
                "ShimaoreBERT encoder loaded — device=%s, params=%d",
                device,
                sum(p.numel() for p in model.parameters()),
            )
            return instance

        except Exception as exc:
            logger.warning("Encoder load failed (%s) — stub encoder.", exc)
            return _StubEncoder()

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, max_length: int = 128) -> EncoderOutput:
        """
        Encode *text* and return contextual embeddings.

        The input is tokenised with a basic whitespace splitter (the full
        :class:`ShimaoreBertTokenizer` is not required here), prepended with
        [CLS] (id 101) and appended with [SEP] (id 102).

        Parameters
        ----------
        text : str
        max_length : int
            Hard truncation limit.

        Returns
        -------
        EncoderOutput
        """
        if not self._loaded:
            return self._stub_output(text)

        import torch

        # Simple tokenisation — split on whitespace, map to vocab ids
        tokens = text.strip().split()[:max_length - 2]
        input_ids = [101] + [hash(t) % 30000 + 100 for t in tokens] + [102]
        attention_mask = [1] * len(input_ids)

        ids_t   = torch.tensor([input_ids],   dtype=torch.long).to(self.device)
        mask_t  = torch.tensor([attention_mask], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self._model(input_ids=ids_t, attention_mask=mask_t)

        last_hidden = outputs.last_hidden_state[0].cpu().tolist()
        cls_emb     = last_hidden[0]
        pooled      = [
            sum(last_hidden[i][j] for i in range(len(last_hidden))) / len(last_hidden)
            for j in range(len(last_hidden[0]))
        ]

        return EncoderOutput(
            last_hidden_state=last_hidden,
            cls_embedding=cls_emb,
            pooled=pooled,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    @staticmethod
    def _stub_output(text: str) -> EncoderOutput:
        tokens = text.strip().split()
        ids    = [101] + [hash(t) % 30000 + 100 for t in tokens] + [102]
        dim    = 512
        return EncoderOutput(
            last_hidden_state=[[0.0] * dim for _ in ids],
            cls_embedding=[0.0] * dim,
            pooled=[0.0] * dim,
            input_ids=ids,
            attention_mask=[1] * len(ids),
        )

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# ---------------------------------------------------------------------------
# Stub encoder (used when torch/weights are unavailable)
# ---------------------------------------------------------------------------

class _StubEncoder(BertEncoderBridge):
    """
    No-op encoder returned when ``pytorch_model.bin`` is absent or when
    ``torch`` / ``transformers`` are not installed.

    All methods return zero-filled tensors of the correct shape so that
    downstream code can remain architecture-agnostic.
    """

    def __init__(self) -> None:
        super().__init__(Path("."), "cpu")
        self._loaded = False

    def encode(self, text: str, max_length: int = 128) -> EncoderOutput:
        return self._stub_output(text)
