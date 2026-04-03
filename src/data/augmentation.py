"""
augmentation.py
---------------
Data-augmentation utilities used during training.

Only :class:`BackTranslationAugmenter` is used in the final v2 training run.
Word-dropout and span-masking augmentation was evaluated but found to degrade
BLEU by ~1.2 points on the Shimaore → French direction.

Back-translation protocol
~~~~~~~~~~~~~~~~~~~~~~~~~
1. Translate each French sentence in the training set to Shimaore using the
   ``v1`` model checkpoint (trained on 60 % of the final corpus).
2. Pair the synthetic Shimaore with the original French.
3. Mark synthetic pairs with a ``[BT]`` source token so the model can
   down-weight them via a separate auxiliary loss term (weight λ = 0.3).
4. Append synthetic pairs to the real corpus; total augmentation ratio ≤ 1:1.

This yielded a +1.7 BLEU improvement on the development set.
"""

from __future__ import annotations

import logging
import random
from typing import List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class BackTranslationAugmenter:
    """
    Augment a parallel corpus with back-translated sentence pairs.

    Parameters
    ----------
    seed : int
        Random seed for reproducible sampling.
    max_augmentation_ratio : float
        Maximum ratio of synthetic to real sentence pairs (default 1.0 = 1:1).
    bt_source_token : str
        Special token prepended to back-translated source sentences.
    """

    def __init__(
        self,
        seed: int = 42,
        max_augmentation_ratio: float = 1.0,
        bt_source_token: str = "[BT]",
    ) -> None:
        self.seed = seed
        self.max_augmentation_ratio = max_augmentation_ratio
        self.bt_source_token = bt_source_token
        random.seed(seed)

    def augment(
        self,
        df: pd.DataFrame,
        back_translations: Optional[List[Tuple[str, str]]] = None,
    ) -> pd.DataFrame:
        """
        Append synthetic sentence pairs to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Original parallel corpus with columns ``shimaore``, ``french``.
        back_translations : list of (shimaore_synthetic, french_original)
            Pre-computed back-translations.  When None, this method is a no-op
            and logs a warning.

        Returns
        -------
        pd.DataFrame
            Original rows followed by synthetic rows (shuffled).
        """
        if back_translations is None:
            logger.warning(
                "BackTranslationAugmenter.augment() called with no back_translations. "
                "Returning original corpus unchanged."
            )
            return df

        max_synthetic = int(len(df) * self.max_augmentation_ratio)
        if len(back_translations) > max_synthetic:
            logger.info(
                "Truncating back-translations from %d to %d (ratio cap = %.1f).",
                len(back_translations), max_synthetic, self.max_augmentation_ratio,
            )
            back_translations = random.sample(back_translations, max_synthetic)

        bt_rows = [
            {
                "shimaore": f"{self.bt_source_token} {shi}",
                "french": fr,
                "is_synthetic": True,
            }
            for shi, fr in back_translations
        ]
        bt_df = pd.DataFrame(bt_rows)

        original_df = df.copy()
        original_df["is_synthetic"] = False

        combined = pd.concat([original_df, bt_df], ignore_index=True)
        combined = combined.sample(frac=1.0, random_state=self.seed).reset_index(drop=True)

        logger.info(
            "Augmented corpus: %d real + %d synthetic = %d total pairs.",
            len(original_df), len(bt_df), len(combined),
        )
        return combined
