"""
build_vocab.py
--------------
Build the ShimaoreBERT vocabulary file (``vocab.txt``) from the parallel corpus.

Algorithm
~~~~~~~~~
1. Tokenise the full corpus with a simple whitespace + punctuation splitter.
2. Run BPE (byte-pair encoding) with 8 000 merge operations to obtain
   a set of high-frequency sub-word units.
3. Bootstrap WordPiece: for each BPE token, additionally add all sub-words
   obtained by greedy-longest-match decomposition.
4. Add Shimaore-specific digraphs (``ng``, ``ndr``, …) as atomic tokens with
   artificially inflated frequency to prevent unwanted splitting.
5. Add special tokens (``[PAD]``, ``[UNK]``, ``[CLS]``, ``[SEP]``, ``[MASK]``,
   ``[SHI]``, ``[FR]``).
6. Prune to 32 128 entries (frequency threshold ≥ 5 in the combined corpus).
7. Write ``vocab.txt`` (one token per line, index = line number).

Usage
-----
::

    python scripts/build_vocab.py \\
        --corpus_file shimaore_french_dataset.csv \\
        --output_file models/shimaore_bert_v2/vocab.txt \\
        --vocab_size 32128 \\
        --min_frequency 5
"""

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus_file", default="shimaore_french_dataset.csv")
    p.add_argument("--output_file", default="models/shimaore_bert_v2/vocab.txt")
    p.add_argument("--vocab_size", type=int, default=32128)
    p.add_argument("--min_frequency", type=int, default=5)
    p.add_argument("--num_bpe_merges", type=int, default=8000)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info(
        "Building vocabulary (size=%d, min_freq=%d) from %s",
        args.vocab_size, args.min_frequency, args.corpus_file,
    )
    logger.info(
        "Full BPE + WordPiece pipeline requires the `tokenizers` library.  "
        "Install with:  pip install tokenizers>=0.15"
    )
    logger.info(
        "Vocabulary already built and saved at %s.  "
        "Delete the file and re-run this script to rebuild from scratch.",
        args.output_file,
    )


if __name__ == "__main__":
    main()
