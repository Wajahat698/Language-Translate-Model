"""
src.inference
-------------
High-level translation pipeline and decoding utilities.

Exports
-------
TranslationPipeline   : End-to-end pipeline wrapping tokeniser + model + beam search.
BeamSearchDecoder     : Pure-Python beam-search implementation (used as fallback).
ConfidenceScorer      : Derives per-sentence confidence from decoder log-probs.
"""

from .pipeline import TranslationPipeline
from .beam_search import BeamSearchDecoder
from .confidence import ConfidenceScorer

__all__ = ["TranslationPipeline", "BeamSearchDecoder", "ConfidenceScorer"]
