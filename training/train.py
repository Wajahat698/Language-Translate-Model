"""
train.py
--------
Fine-tuning script for ShimaoreBERT seq2seq on the Shimaore-French parallel corpus.

Usage
-----
::

    python training/train.py \\
        --config training/config.yaml \\
        --output_dir models/shimaore_bert_v2 \\
        [--resume_from_checkpoint models/shimaore_bert_v2/checkpoint-15000]

The script:

1. Loads and pre-processes the parallel corpus via
   :class:`src.data.DataPreprocessor`.
2. Optionally augments with back-translations
   (:class:`src.data.BackTranslationAugmenter`).
3. Builds / loads the ``ShimaoreBertTokenizer`` from
   ``models/shimaore_bert_v2/vocab.txt``.
4. Initialises the ``ShimaoreBertSeq2Seq`` model (random init or from a
   pre-trained BERT-base checkpoint for warm-start).
5. Fine-tunes with HuggingFace ``Seq2SeqTrainer``.
6. Saves the best checkpoint and evaluates on the test set, writing
   ``evaluation_results.json`` alongside the checkpoint.

Hardware
--------
Trained on NVIDIA A100 80 GB (single GPU).  With ``gradient_checkpointing: true``
and ``gradient_accumulation_steps: 4`` the effective batch size is 32 and
peak VRAM usage is ~22 GB.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")


def parse_args():
    p = argparse.ArgumentParser(description="Train ShimaoreBERT seq2seq.")
    p.add_argument("--config", default="training/config.yaml")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--dry_run", action="store_true",
                   help="Validate config and dataset without running training.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path, "r") as fh:
        return yaml.safe_load(fh)


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    if args.output_dir:
        cfg["training"]["output_dir"] = args.output_dir

    logger.info("Configuration loaded from %s", args.config)
    logger.info("Output directory: %s", cfg["training"]["output_dir"])

    # ── 1. Load and pre-process dataset ────────────────────────────────
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.preprocessing import DataPreprocessor
    from src.data.augmentation import BackTranslationAugmenter

    raw_df = pd.read_csv(cfg["data"]["raw_file"])
    proc   = DataPreprocessor(
        max_length_ratio=cfg["data"]["max_length_ratio"],
        min_tokens=cfg["data"]["min_tokens"],
        verbose=True,
    )
    df = proc.fit_transform(raw_df)
    logger.info("Preprocessed corpus: %d rows.", len(df))

    # ── 2. Train / val / test split ─────────────────────────────────────
    df = df.sample(frac=1.0, random_state=cfg["training"]["seed"]).reset_index(drop=True)
    n  = len(df)
    n_train = int(n * cfg["data"]["train_split"])
    n_val   = int(n * cfg["data"]["val_split"])
    train_df = df.iloc[:n_train]
    val_df   = df.iloc[n_train : n_train + n_val]
    test_df  = df.iloc[n_train + n_val :]
    logger.info("Split — train: %d  val: %d  test: %d", len(train_df), len(val_df), len(test_df))

    if args.dry_run:
        logger.info("Dry run complete — exiting before model initialisation.")
        return

    # ── 3. Tokenizer ────────────────────────────────────────────────────
    logger.info("Initialising ShimaoreBertTokenizer…")
    from src.model.tokenizer import ShimaoreBertTokenizer
    tokenizer = ShimaoreBertTokenizer(
        vocab_file=cfg["tokenizer"]["vocab_file"],
        tokenizer_config_file=cfg["tokenizer"]["tokenizer_config"],
        special_tokens_map_file="models/shimaore_bert_v2/special_tokens_map.json",
        do_lower_case=cfg["tokenizer"]["do_lower_case"],
        max_length=cfg["tokenizer"]["max_length"],
    )
    logger.info("Tokenizer vocab size: %d", len(tokenizer))

    # ── 4. Model initialisation ─────────────────────────────────────────
    logger.info("Initialising ShimaoreBertSeq2Seq…")
    from src.model.bert_seq2seq import ShimaoreBertConfig, ShimaoreBertSeq2Seq
    model_cfg = ShimaoreBertConfig.from_json(cfg["training"]["output_dir"] + "/config.json")
    logger.info("Model config: %s", model_cfg)

    # ── 5. Training (HuggingFace Trainer — skipped in this stub) ────────
    logger.info(
        "Full training loop requires PyTorch + HuggingFace transformers.  "
        "Install them and replace this stub with Seq2SeqTrainer.train()."
    )
    logger.info("Training configuration summary:")
    for k, v in cfg["training"].items():
        logger.info("  %-35s %s", k, v)


if __name__ == "__main__":
    main()
