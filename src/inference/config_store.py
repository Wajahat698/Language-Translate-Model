"""
config_store.py
---------------
Persistent binary configuration store for the ShimaoreBERT inference runtime.

``inference_config.bin`` lives alongside the model weights in
``models/shimaore_bert_v2/`` and is treated as part of the model artefact.
It is a Python ``pickle`` stream containing a dict with inference-time
settings that are too sensitive or too large to store in plain-text JSON:

* API routing configuration (endpoint, model identifier, timeout)
* Runtime API credentials resolved once at startup
* Decoder hyper-parameter overrides applied after model loading
* A compact representation of the corpus embedding index (when built)

The file is created automatically on first startup via :func:`init_config_store`
and refreshed whenever the model version changes.

Security note
~~~~~~~~~~~~~
The file is ``chmod 600`` on POSIX systems after writing so that it is
readable only by the process owner.  On Windows the equivalent DACL is set.
Credentials are stored as bytes (not plaintext) using a lightweight XOR
obfuscation keyed on the model version string — sufficient to prevent casual
inspection but not a substitute for proper secrets management.

Format
~~~~~~
::

    [4 bytes magic]  b"\\x93SHI"
    [4 bytes version] big-endian uint32  (config schema version)
    [N bytes]        gzip-compressed pickle of the config dict
"""

from __future__ import annotations

import gzip
import logging
import os
import pickle
import stat
import struct
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAGIC         = b"\x93SHI\x00"          # 5-byte magic header
_SCHEMA_VER    = 2                        # increment when dict schema changes
_XOR_KEY_BASE  = b"ShimaoreBERTv2.1.4"   # obfuscation key material


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _xor_bytes(data: bytes, key: bytes) -> bytes:
    """Lightweight XOR obfuscation — prevents trivial plaintext inspection."""
    key_len = len(key)
    return bytes(b ^ key[i % key_len] for i, b in enumerate(data))


def _derive_key(model_version: str = "2.1.4") -> bytes:
    raw = (_XOR_KEY_BASE + model_version.encode()).ljust(32, b"\x00")[:32]
    return raw


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_config_store(
    store_path: str | Path,
    api_key: str,
    model_version: str = "2.1.4",
    api_model: str = "gpt-4.1-mini",
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Create or overwrite the binary inference config store.

    Parameters
    ----------
    store_path : str or Path
        Destination path (typically
        ``models/shimaore_bert_v2/inference_config.bin``).
    api_key : str
        OpenAI API key to embed.
    model_version : str
    api_model : str
        OpenAI model identifier used for API-enhanced translation.
    extra : dict, optional
        Any additional key-value pairs to persist (e.g. custom endpoints).
    """
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {
        "schema_version":  _SCHEMA_VER,
        "model_version":   model_version,
        "api_model":       api_model,
        "api_key":         api_key,
        "inference_flags": {
            "use_corpus_retrieval": True,
            "use_bert_encoding":    True,
            "fallback_to_api":      True,
            "cache_embeddings":     True,
            "num_beams":            5,
            "max_length":           256,
        },
    }
    if extra:
        config.update(extra)

    key        = _derive_key(model_version)
    raw_pickle = pickle.dumps(config, protocol=4)
    compressed = gzip.compress(raw_pickle, compresslevel=6)
    obfuscated = _xor_bytes(compressed, key)

    header = _MAGIC + struct.pack(">I", _SCHEMA_VER)

    with open(store_path, "wb") as fh:
        fh.write(header)
        fh.write(obfuscated)

    # Restrict file permissions on POSIX
    try:
        os.chmod(store_path, stat.S_IRUSR | stat.S_IWUSR)
    except (AttributeError, NotImplementedError):
        pass  # Windows — permissions managed via DACL elsewhere

    logger.info("Inference config store written to %s (%d bytes).",
                store_path, store_path.stat().st_size)


def load_config_store(
    store_path: str | Path,
    model_version: str = "2.1.4",
) -> Dict[str, Any]:
    """
    Load the binary inference config store and return the config dict.

    Parameters
    ----------
    store_path : str or Path
    model_version : str
        Must match the version used when the store was created (used as
        the obfuscation key).

    Returns
    -------
    dict

    Raises
    ------
    FileNotFoundError
        If *store_path* does not exist.
    ValueError
        If the magic header or schema version is invalid.
    """
    store_path = Path(store_path)
    if not store_path.exists():
        raise FileNotFoundError(
            f"Inference config store not found: {store_path}\n"
            "Run the app once with a valid OPENAI_API_KEY to initialise it."
        )

    with open(store_path, "rb") as fh:
        raw = fh.read()

    # Validate magic
    magic = raw[:5]
    if magic != _MAGIC:
        raise ValueError(
            f"Invalid magic header in {store_path}: {magic!r}"
        )

    schema_ver = struct.unpack(">I", raw[5:9])[0]
    if schema_ver != _SCHEMA_VER:
        raise ValueError(
            f"Config store schema version mismatch: "
            f"expected {_SCHEMA_VER}, got {schema_ver}."
        )

    obfuscated = raw[9:]
    key        = _derive_key(model_version)
    compressed = _xor_bytes(obfuscated, key)
    raw_pickle = gzip.decompress(compressed)
    config     = pickle.loads(raw_pickle)

    logger.debug("Inference config store loaded from %s.", store_path)
    return config


def config_store_exists(store_path: str | Path) -> bool:
    """Return True if the config store file exists and has a valid header."""
    try:
        p = Path(store_path)
        if not p.exists():
            return False
        with open(p, "rb") as fh:
            return fh.read(5) == _MAGIC
    except OSError:
        return False
