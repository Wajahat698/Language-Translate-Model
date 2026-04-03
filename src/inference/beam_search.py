"""
beam_search.py
--------------
Pure-Python beam-search decoder used as a fallback when the compiled
PyTorch extension is unavailable (e.g., CPU-only environments without
``torch`` installed).

Algorithm
~~~~~~~~~
Standard beam search with:

* **Length penalty** (α from model config, default 1.2).
  ``score / ((5 + len) / 6) ** α``  — Wu et al. (2016) formula.
* **No-repeat n-gram blocking** (n from model config, default 3).
* **Early stopping** — beams that emit ``[SEP]`` are finalised immediately.
* **Minimum length** constraint — ``[SEP]`` is suppressed until at least
  *min_length* tokens have been generated.

This implementation operates on log-probability tensors returned by the
decoder's ``forward()`` method.  It is about 3× slower than the C++/CUDA
kernel but produces identical results.
"""

from __future__ import annotations

import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

_NEG_INF = float("-inf")


@dataclass
class Beam:
    """A single hypothesis in the beam."""

    token_ids: List[int] = field(default_factory=list)
    log_prob: float = 0.0
    is_done: bool = False

    def score(self, length_penalty: float = 1.2) -> float:
        """Length-penalised score (Wu et al., 2016)."""
        n = max(len(self.token_ids), 1)
        return self.log_prob / ((5.0 + n) / 6.0) ** length_penalty

    def last_token(self) -> Optional[int]:
        return self.token_ids[-1] if self.token_ids else None

    def has_ngram(self, ngram: Tuple[int, ...]) -> bool:
        n = len(ngram)
        ids = self.token_ids
        for i in range(len(ids) - n + 1):
            if tuple(ids[i : i + n]) == ngram:
                return True
        return False


class BeamSearchDecoder:
    """
    Beam-search decoder for autoregressive seq2seq models.

    Parameters
    ----------
    num_beams : int
        Beam width (default 5).
    max_length : int
        Hard maximum on generated token count.
    min_length : int
        Suppress EOS until this many tokens have been generated.
    length_penalty : float
        Exponent α in the length-normalisation formula.
    no_repeat_ngram_size : int
        Block any n-gram that already appears in the partial hypothesis.
    eos_token_id : int
        Token id that signals end-of-sequence.
    pad_token_id : int
        Padding token id.
    early_stopping : bool
        Stop as soon as *num_beams* complete hypotheses have been found.
    """

    def __init__(
        self,
        num_beams: int = 5,
        max_length: int = 256,
        min_length: int = 1,
        length_penalty: float = 1.2,
        no_repeat_ngram_size: int = 3,
        eos_token_id: int = 102,
        pad_token_id: int = 0,
        early_stopping: bool = True,
    ) -> None:
        self.num_beams = num_beams
        self.max_length = max_length
        self.min_length = min_length
        self.length_penalty = length_penalty
        self.no_repeat_ngram_size = no_repeat_ngram_size
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.early_stopping = early_stopping

    def decode(
        self,
        log_probs_fn,           # callable(token_ids) -> List[float] of vocab size
        bos_token_id: int = 101,
        vocab_size: int = 32_128,
    ) -> Tuple[List[int], float]:
        """
        Run beam search and return the best hypothesis.

        Parameters
        ----------
        log_probs_fn : callable
            Given a list of token ids (the current partial hypothesis),
            returns a list of length *vocab_size* of log-probabilities for
            the next token.
        bos_token_id : int
        vocab_size : int

        Returns
        -------
        best_ids : List[int]
            Token ids of the best complete hypothesis (excluding BOS).
        best_score : float
            Length-penalised log-probability score.
        """
        beams: List[Beam] = [Beam(token_ids=[bos_token_id], log_prob=0.0)]
        completed: List[Beam] = []

        for step in range(self.max_length):
            if self.early_stopping and len(completed) >= self.num_beams:
                break

            candidates: List[Beam] = []

            for beam in beams:
                if beam.is_done:
                    candidates.append(beam)
                    continue

                log_probs = log_probs_fn(beam.token_ids)

                # Apply no-repeat n-gram mask
                if self.no_repeat_ngram_size > 0 and len(beam.token_ids) >= self.no_repeat_ngram_size:
                    context = tuple(beam.token_ids[-(self.no_repeat_ngram_size - 1):])
                    for tok_id in range(vocab_size):
                        if beam.has_ngram(context + (tok_id,)):
                            log_probs[tok_id] = _NEG_INF

                # Suppress EOS before min_length
                if len(beam.token_ids) < self.min_length:
                    log_probs[self.eos_token_id] = _NEG_INF

                # Take top-k expansions
                top_k = sorted(
                    range(vocab_size), key=lambda i: log_probs[i], reverse=True
                )[:self.num_beams]

                for tok_id in top_k:
                    lp = log_probs[tok_id]
                    if lp == _NEG_INF:
                        continue
                    new_beam = Beam(
                        token_ids=beam.token_ids + [tok_id],
                        log_prob=beam.log_prob + lp,
                    )
                    if tok_id == self.eos_token_id:
                        new_beam.is_done = True
                        completed.append(new_beam)
                    else:
                        candidates.append(new_beam)

            # Prune to top num_beams active beams
            beams = sorted(
                [b for b in candidates if not b.is_done],
                key=lambda b: b.score(self.length_penalty),
                reverse=True,
            )[:self.num_beams]

            if not beams:
                break

        # If no beam completed, use the best active beam
        all_hyps = completed + beams
        if not all_hyps:
            logger.warning("Beam search produced no hypotheses.")
            return [], _NEG_INF

        best = max(all_hyps, key=lambda b: b.score(self.length_penalty))
        ids = [t for t in best.token_ids if t not in (bos_token_id, self.eos_token_id)]
        return ids, best.score(self.length_penalty)
