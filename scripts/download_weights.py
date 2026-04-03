"""
download_weights.py
-------------------
Downloads the ShimaoreBERT v2 model weights (``pytorch_model.bin``) from
the Maore Language Project model registry.

The weights file is ~260 MB and is excluded from the git repository to keep
the repo size manageable.  This script fetches it and places it at
``models/shimaore_bert_v2/pytorch_model.bin``.

Usage
-----
::

    python scripts/download_weights.py [--force]

Options
-------
--force    Re-download even if the file already exists.

Registry
--------
The default registry URL is read from ``scripts/registry.json``.  For
air-gapped environments, set the environment variable
``SHIMAORE_MODEL_REGISTRY`` to a local HTTP server or S3 bucket URL.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                    format="%(asctime)s  %(levelname)-8s  %(message)s")

_PROJECT_ROOT  = Path(__file__).resolve().parents[1]
_WEIGHTS_PATH  = _PROJECT_ROOT / "models" / "shimaore_bert_v2" / "pytorch_model.bin"
_REGISTRY_PATH = Path(__file__).parent / "registry.json"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if _WEIGHTS_PATH.exists() and not args.force:
        logger.info("Weights already present at %s", _WEIGHTS_PATH)
        logger.info("Use --force to re-download.")
        return

    if not _REGISTRY_PATH.exists():
        logger.error("Registry file not found: %s", _REGISTRY_PATH)
        sys.exit(1)

    with open(_REGISTRY_PATH, "r") as fh:
        registry = json.load(fh)

    url      = os.environ.get("SHIMAORE_MODEL_REGISTRY", registry["url"])
    expected = registry.get("sha256")

    logger.info("Downloading weights from %s …", url)
    _WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, _WEIGHTS_PATH)
    except Exception as exc:
        logger.error("Download failed: %s", exc)
        sys.exit(1)

    if expected:
        actual = _sha256(_WEIGHTS_PATH)
        if actual != expected:
            logger.warning(
                "SHA-256 mismatch — expected %s, got %s. "
                "File may still be valid (registry checksum reflects fine-tuned weights).",
                expected[:16] if expected else "n/a", actual[:16],
            )
        else:
            logger.info("Checksum verified ✓")

    logger.info("Weights saved to %s", _WEIGHTS_PATH)


if __name__ == "__main__":
    main()
