"""
src — Shimaore-French Translation System
=========================================
Top-level package exposing the public API used by the Streamlit front-end.

Sub-packages
------------
model      : ShimaoreBERT architecture, attention modules, tokenizer wrapper
data       : Dataset loaders, pre/post-processing pipelines, augmentation
inference  : Translation pipeline, beam-search decoder, confidence scoring
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("shimaore-translator")
except PackageNotFoundError:
    __version__ = "2.1.4-dev"

__author__ = "Maore Language Project"
__license__ = "MIT"
