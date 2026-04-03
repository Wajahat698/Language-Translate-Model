"""
src.data
--------
Dataset loading, pre-processing and augmentation utilities.

Exports
-------
ShimaoReDataset       : torch.utils.data.Dataset wrapper for the parallel corpus.
DataPreprocessor      : Cleaning, normalisation and alignment pipeline.
BackTranslationAugmenter : Data augmentation via round-trip translation.
"""

from .preprocessing import DataPreprocessor, normalize_text
from .augmentation import BackTranslationAugmenter

__all__ = ["DataPreprocessor", "normalize_text", "BackTranslationAugmenter"]
