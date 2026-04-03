"""
src.model
---------
Neural model components for the ShimaoreBERT seq2seq architecture.

Exports
-------
ShimaoreBertConfig        : Dataclass holding all model hyper-parameters.
ShimaoreBertTokenizer     : Tokenizer with Shimaore-specific subword rules.
MultiHeadCrossAttention   : Cross-attention module used in the decoder.
BeamSearchDecoder         : Beam-search wrapper around the autoregressive decoder.
"""

from .bert_seq2seq import ShimaoreBertConfig, ShimaoreBertSeq2Seq
from .tokenizer import ShimaoreBertTokenizer
from .attention import MultiHeadCrossAttention

__all__ = [
    "ShimaoreBertConfig",
    "ShimaoreBertSeq2Seq",
    "ShimaoreBertTokenizer",
    "MultiHeadCrossAttention",
]
