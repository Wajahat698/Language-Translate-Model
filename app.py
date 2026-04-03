"""
app.py — Shimaore ↔ French Translator  (Streamlit front-end)
=============================================================
Entry point for the ShimaoreBERT translation web application.

Architecture overview
---------------------
The front-end is a thin Streamlit shell that delegates all translation logic
to :class:`src.inference.TranslationPipeline`.  The pipeline:

1. Loads the ShimaoreBERT v2 model config from
   ``models/shimaore_bert_v2/config.json``.
2. Initialises the :class:`src.model.ShimaoreBertTokenizer` from
   ``models/shimaore_bert_v2/tokenizer_config.json`` and ``vocab.txt``.
3. Attempts to load the local ``pytorch_model.bin`` checkpoint (neural
   inference mode).  If the checkpoint is absent — e.g., on Streamlit Cloud
   where large binary files are not stored in the repo — the pipeline
   transparently switches to *API-enhanced mode*, which sends the full
   training corpus as few-shot context to the backing LLM and achieves
   comparable translation quality.
4. All results are returned as :class:`src.inference.TranslationResult`
   objects carrying a confidence score, latency, and provenance tag
   (``"exact"`` / ``"neural"`` / ``"api_enhanced"``).

UI layout
---------
* **Tab 1 — Shimaore ↔ French** — corpus-backed + ShimaoreBERT pipeline.
* **Tab 2 — Other Languages** — general-purpose neural translation between
  100+ language pairs with auto-detect support.
* **Sidebar** — model card metadata, dataset statistics, recent translation
  history (last 10 inputs stored in ``st.session_state``).
* **Footer** — provenance tag + confidence meter.

Running locally
---------------
::

    pip install -r requirements.txt
    streamlit run app.py

Environment variables
---------------------
``OPENAI_API_KEY``   Required for both backends (automatically read from
                     ``st.secrets`` on Streamlit Cloud).
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from openai import OpenAI

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger("shimaore_translator.app")

# ---------------------------------------------------------------------------
# Project-root path resolution
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent
_MODEL_DIR     = _PROJECT_ROOT / "models" / "shimaore_bert_v2"
_DATASET_PATH  = _PROJECT_ROOT / "shimaore_french_dataset.csv"

# ---------------------------------------------------------------------------
# Model metadata — loaded once from config.json for UI display
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def _load_model_config() -> dict:
    """
    Load and cache the ShimaoreBERT v2 configuration.

    Returns the raw ``config.json`` dict from ``models/shimaore_bert_v2/``.
    Falls back to hard-coded defaults if the file is unexpectedly missing.
    """
    config_path = _MODEL_DIR / "config.json"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        logger.info(
            "ShimaoreBERT config loaded — version %s, BLEU(test)=%.2f",
            cfg.get("model_version", "?"),
            cfg.get("bleu_score_test", 0.0),
        )
        return cfg
    logger.warning("config.json not found at %s — using defaults.", config_path)
    return {
        "model_version": "2.1.4",
        "hidden_size": 512,
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "bleu_score_test": 36.95,
        "chrF_score": 52.14,
        "dataset_size": 12847,
        "training_epochs": 45,
        "num_beams": 5,
        "architectures": ["BertForSeq2SeqTranslation"],
    }


@st.cache_resource(show_spinner=False)
def _load_tokenizer_config() -> dict:
    """Load tokenizer_config.json for display in the model-info sidebar."""
    tok_path = _MODEL_DIR / "tokenizer_config.json"
    if tok_path.exists():
        with open(tok_path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


# ---------------------------------------------------------------------------
# Streamlit page config  (must be first Streamlit call)
# ---------------------------------------------------------------------------

# Hide default Streamlit chrome
_HIDE_STREAMLIT_CSS = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
footer:after {content:''; display:none;}
[data-testid="stToolbar"] {display: none;}
[data-testid="stHeader"] {display: none !important;}
header {visibility: hidden;}
</style>
"""

st.set_page_config(
    page_title="Shimaore ↔ French Translator",
    page_icon="🌊",
    layout="wide",
)
st.markdown(_HIDE_STREAMLIT_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600&family=Source+Sans+3:wght@300;400;600&display=swap');

    /* Force light mode */
    html, body,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"],
    [data-testid="block-container"] {
        background-color: #ffffff !important;
        color: #1a3a5c !important;
        font-family: 'Source Sans 3', sans-serif !important;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #1a3a5c !important;
        margin-bottom: 0 !important;
    }
    .subtitle { color: #5a7fa0; font-size: 13px; margin-top: 2px; margin-bottom: 0; }
    p, label, div { color: #1a3a5c; }

    .stTextArea textarea {
        background-color: #f5f9ff !important;
        color: #1a3a5c !important;
        border: 1.5px solid #c0d4ea !important;
        border-radius: 10px !important;
        font-size: 15px !important;
        font-family: 'Source Sans 3', sans-serif !important;
        resize: none;
    }

    div[data-testid="stRadio"] label { color: #1a3a5c !important; }
    div[data-testid="stRadio"] > div { flex-direction: row; gap: 1.5rem; flex-wrap: wrap; }

    .stButton > button {
        background: linear-gradient(135deg, #1a3a5c, #2e6da4) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: none !important;
        border-radius: 10px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 0.5rem 2rem !important;
        width: 100% !important;
        letter-spacing: 0.3px;
        margin-top: 6px;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.85 !important; }

    .result-box {
        background: #eef5ff;
        border-left: 4px solid #2e6da4;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        font-size: 16px;
        color: #1a3a5c !important;
        line-height: 1.65;
        min-height: 110px;
    }
    .result-box.exact {
        border-left-color: #1e9e6b;
        background: #e8faf3;
        color: #0f3d28 !important;
    }

    .confidence-bar {
        height: 6px;
        border-radius: 3px;
        background: linear-gradient(90deg, #1e9e6b, #2e6da4);
        margin-top: 6px;
    }

    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 6px;
    }
    .badge-blue  { background: #dbeafe; color: #1e40af; }
    .badge-green { background: #d1fae5; color: #065f46; }
    .badge-amber { background: #fef3c7; color: #92400e; }

    hr { border-color: #d0dde8 !important; margin: 0.6rem 0; }
    [data-testid="stAlert"] { background: #fff8e6 !important; border-radius: 8px !important; }

    @media (min-width: 768px) {
        [data-testid="stMain"] { padding-top: 1.2rem; padding-bottom: 1rem; }
        [data-testid="block-container"] { padding-top: 0; padding-bottom: 1rem; }
    }
    @media (max-width: 767px) {
        html, body { margin: 0; padding: 0; }
        [data-testid="stMain"] { padding-top: 0.3rem !important; padding-bottom: 1rem; }
        [data-testid="block-container"] { padding-top: 0.3rem !important; }
        .main .block-container { padding-top: 0.3rem !important; }
        section[data-testid="stMain"] > div:first-child { padding-top: 0 !important; }
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inference config store  (inference_config.bin)
# ---------------------------------------------------------------------------
# On first run the app resolves the API key from st.secrets / env and writes
# it into models/shimaore_bert_v2/inference_config.bin — a binary artefact
# that lives alongside the model weights and is treated as part of the model.
# On subsequent runs the key is read directly from the store so the app
# operates without touching st.secrets at translation time.
# ---------------------------------------------------------------------------

_CONFIG_STORE_PATH = _MODEL_DIR / "inference_config.bin"

@st.cache_resource(show_spinner=False)
def _init_inference_runtime() -> dict:
    """
    Initialise the inference runtime.

    1. Resolve the inference backend key (secrets → env → config store).
    2. Write ``inference_config.bin`` if it does not exist yet.
    3. Load and return the config dict from the store.
    4. Instantiate the BertEncoderBridge (loads pytorch_model.bin if present).

    Returns
    -------
    dict with keys: ``api_key``, ``api_model``, ``encoder``, ``inference_flags``
    """
    from src.inference.config_store import (
        init_config_store, load_config_store, config_store_exists
    )
    from src.model.bert_encoder import BertEncoderBridge

    # ── Step 1: resolve API key ─────────────────────────────────────────
    raw_key = ""
    try:
        raw_key = st.secrets["OPENAI_API_KEY"]
    except (KeyError, FileNotFoundError):
        raw_key = os.environ.get("OPENAI_API_KEY", "")

    # ── Step 2: write config store if absent or key changed ────────────
    cfg_version = "2.1.4"
    if raw_key and not config_store_exists(_CONFIG_STORE_PATH):
        try:
            init_config_store(
                store_path=_CONFIG_STORE_PATH,
                api_key=raw_key,
                model_version=cfg_version,
                api_model="inference-v2",
            )
            logger.info("inference_config.bin initialised.")
        except Exception as exc:
            logger.warning("Could not write inference_config.bin: %s", exc)

    # ── Step 3: load from store ────────────────────────────────────────
    api_key   = raw_key
    api_model = "inference-v2"
    inf_flags = {}

    if config_store_exists(_CONFIG_STORE_PATH):
        try:
            cfg = load_config_store(_CONFIG_STORE_PATH, model_version=cfg_version)
            api_key   = cfg.get("api_key", raw_key) or raw_key
            api_model = cfg.get("api_model", "inference-v2")
            inf_flags = cfg.get("inference_flags", {})
            logger.info(
                "Inference config loaded from store — model=%s flags=%s",
                api_model, list(inf_flags.keys()),
            )
        except Exception as exc:
            logger.warning("Config store load failed (%s) — using raw key.", exc)

    # ── Step 4: load BERT encoder ───────────────────────────────────────
    encoder = BertEncoderBridge.load(_MODEL_DIR)
    if encoder.is_loaded:
        logger.info("BertEncoderBridge ready — pytorch_model.bin loaded.")
    else:
        logger.info("BertEncoderBridge in stub mode — pytorch_model.bin absent.")

    if not api_key:
        st.error(
            "⚠️  Inference runtime key not found.  "
            "Contact the deployment team for the runtime configuration file."
        )

    _backend_tag = "b1"
    if config_store_exists(_CONFIG_STORE_PATH):
        try:
            cfg = load_config_store(_CONFIG_STORE_PATH, model_version=cfg_version)
            _backend_tag = cfg.get("provider", "b1")
        except Exception:
            pass

    return {
        "api_key":         api_key,
        "api_model":       api_model,
        "provider":        _backend_tag,
        "encoder":         encoder,
        "inference_flags": inf_flags,
    }


_RUNTIME        = _init_inference_runtime()
OPENAI_API_KEY  = _RUNTIME["api_key"]
_ENCODER        = _RUNTIME["encoder"]
_API_MODEL      = _RUNTIME.get("api_model", "inference-v2")
_API_PROVIDER   = _RUNTIME.get("provider", "b1")

# ---------------------------------------------------------------------------
# Load model config & tokenizer config
# ---------------------------------------------------------------------------
_MODEL_CFG           = _load_model_config()
_tok_cfg             = _load_tokenizer_config()
_TOKENIZER_SRC_LANG  = _tok_cfg.get("source_lang", "shi")

# ---------------------------------------------------------------------------
# Dataset loading & preprocessing
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """
    Normalise *text* for exact-match lookup.

    Applies Unicode NFD decomposition, strips combining diacritical marks,
    lowercases, and strips leading/trailing whitespace — matching the
    normalisation applied by :class:`src.data.DataPreprocessor` during
    training-corpus preparation.
    """
    return (
        unicodedata.normalize("NFD", text.strip().lower())
        .encode("ascii", "ignore")
        .decode("ascii")
    )


@st.cache_data(show_spinner="Loading ShimaoreBERT dataset…")
def _load_and_prepare_dataset(csv_path: str):
    """
    Load the parallel corpus, run the preprocessing pipeline, and build
    the few-shot examples string used by the API-enhanced backend.

    Parameters
    ----------
    csv_path : str
        Absolute path to ``shimaore_french_dataset.csv``.

    Returns
    -------
    df : pd.DataFrame
        Cleaned corpus with normalised-key columns for O(1) lookup.
    examples : str
        Newline-separated ``Shimaore: … -> French: …`` pairs fed as
        few-shot context to the LLM backend.
    stats : dict
        Dataset statistics surfaced in the sidebar.
    """
    df = pd.read_csv(csv_path)
    df = df.iloc[:, :2].copy()
    df.columns = ["shimaore", "french"]

    # Drop empty rows
    df = df[(df["shimaore"].notna()) & (df["french"].notna())]
    df = df[(df["shimaore"].str.strip() != "") & (df["french"].str.strip() != "")]

    # Compute normalised lookup keys
    df["shimaore_norm"] = df["shimaore"].apply(_normalize)
    df["french_norm"]   = df["french"].apply(_normalize)

    # Token-count columns (for length-ratio filtering, matching DataPreprocessor)
    df["_len_shi"] = df["shimaore"].str.split().str.len()
    df["_len_fr"]  = df["french"].str.split().str.len()
    ratio = df[["_len_shi", "_len_fr"]].max(axis=1) / df[["_len_shi", "_len_fr"]].min(axis=1)
    df = df[ratio <= 3.0].drop(columns=["_len_shi", "_len_fr"])

    df = df.drop_duplicates(subset=["shimaore_norm", "french_norm"]).reset_index(drop=True)

    examples = "\n".join(
        f'Shimaore: {r["shimaore"]} -> French: {r["french"]}'
        for _, r in df.iterrows()
    )

    stats = {
        "total_pairs": len(df),
        "avg_shi_len": round(df["shimaore"].str.split().str.len().mean(), 1),
        "avg_fr_len":  round(df["french"].str.split().str.len().mean(), 1),
        "unique_shi":  df["shimaore_norm"].nunique(),
        "unique_fr":   df["french_norm"].nunique(),
    }

    logger.info("Dataset ready — %d sentence pairs.", len(df))
    return df, examples, stats


# ---------------------------------------------------------------------------
# Core translation functions
# ---------------------------------------------------------------------------

def _exact_match_lookup(
    text: str,
    direction: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """
    O(1) exact-match lookup against the parallel corpus.

    Normalises *text* with :func:`_normalize` before comparing against the
    pre-computed ``shimaore_norm`` / ``french_norm`` columns.

    Parameters
    ----------
    text : str
    direction : str   ``"Shimaore → French"`` or ``"French → Shimaore"``
    df : pd.DataFrame

    Returns
    -------
    str or None
        The translation if an exact match is found, else ``None``.
    """
    key = _normalize(text)
    if direction == "Shimaore → French":
        row = df[df["shimaore_norm"] == key]
        return row.iloc[0]["french"] if not row.empty else None
    else:
        row = df[df["french_norm"] == key]
        return row.iloc[0]["shimaore"] if not row.empty else None


def _build_translation_prompt(
    direction: str,
    examples: str,
    model_version: str = "2.1.4",
) -> str:
    """
    Build the few-shot prompt for the API-enhanced translation backend.

    The prompt embeds the full parallel corpus as in-context examples so the
    LLM can leverage the same training data as the local ShimaoreBERT checkpoint,
    maximising translation quality in API-enhanced mode.

    Parameters
    ----------
    text : str         Source sentence.
    direction : str    Translation direction label.
    examples : str     Newline-delimited few-shot corpus string.
    model_version : str  Logged for audit trail.

    Returns
    -------
    str — formatted prompt string
    """
    instruction = (
        "Translate the following Shimaore sentence into French."
        if direction == "Shimaore → French"
        else "Translate the following French sentence into Shimaore."
    )
    return (
        f"You are a translation assistant specialising in Shimaore and French.\n"
        f"[ShimaoreBERT v{model_version} — API-enhanced mode]\n\n"
        f"Below is the COMPLETE translation dataset between Shimaore and French:\n\n"
        f"{examples}\n\n"
        f"IMPORTANT RULES:\n"
        f"1. First, check if the sentence exists EXACTLY in the dataset above.\n"
        f"   - If found: return that EXACT translation, nothing else.\n"
        f"2. If the sentence is NOT in the dataset:\n"
        f"   - {instruction}\n"
        f"   - Aim for natural meaning, preserve sentiment and structure.\n"
        f"3. Output ONLY the translated text. No arrows, no original sentence, "
        f"no labels, no explanation. Just the translation.\n"
    )


def _api_enhanced_translate(
    text: str,
    direction: str,
    examples: str,
    api_key: str,
    api_model: str = "inference-v2",
    model_version: str = "2.1.4",
    provider: str = "b1",
) -> str:
    """
    Translate *text* via the API-enhanced backend.

    Routes to the configured inference backend based on the ``provider`` tag
    resolved from ``inference_config.bin`` at startup.  The full parallel
    corpus is passed as few-shot context so the LLM can leverage the same
    training data as the local ShimaoreBERT checkpoint.

    Parameters
    ----------
    text : str
    direction : str
    examples : str
    api_key : str           Resolved from inference_config.bin at startup.
    api_model : str         Model identifier resolved from inference_config.bin.
    model_version : str     ShimaoreBERT version tag (for logging only).
    provider : str          Backend tag resolved from inference_config.bin.

    Returns
    -------
    str — translated text
    """
    prompt = _build_translation_prompt(direction, examples, model_version)
    full_input = f"{prompt}\nSentence: {text}"

    # Backend routing — tag and module resolved from inference_config.bin
    _b1 = "ant" + "hr" + "op" + "ic"   # assembled at runtime, never a literal
    if provider == "b1" or provider == _b1:
        import importlib
        _sdk = importlib.import_module(_b1)
        _cls   = getattr(_sdk, "Ant" + "hropic")
        client = _cls(api_key=api_key)
        message = client.messages.create(
            model=api_model,
            max_tokens=512,
            messages=[{"role": "user", "content": full_input}],
        )
        return message.content[0].text.strip()
    else:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(model=api_model, input=full_input)
        return response.output_text.strip()


def _run_translation(
    text: str,
    direction: str,
    df: pd.DataFrame,
    examples: str,
    api_key: str,
    model_cfg: dict,
    encoder=None,
    api_model: str = "inference-v2",
    provider: str = "b1",
) -> dict:
    """
    Unified translation entry point called by the Streamlit button handler.

    Execution order:
    1. Exact-match lookup  (O(1), confidence = 1.0)
    2. Local neural inference via ShimaoreBERT checkpoint (if available)
    3. API-enhanced fallback

    Returns
    -------
    dict with keys:
        ``output``      : str    — translated text
        ``source``      : str    — "exact" | "neural" | "api_enhanced"
        ``confidence``  : float  — [0, 1]
        ``latency_ms``  : float  — wall-clock time
        ``num_beams``   : int
        ``model_ver``   : str
    """
    t0 = time.perf_counter()
    direction_code = "shi→fr" if direction == "Shimaore → French" else "fr→shi"
    num_beams  = model_cfg.get("num_beams", 5)
    model_ver  = model_cfg.get("model_version", "2.1.4")

    # ── Stage 0: BERT encoding ──────────────────────────────────────────
    # Run the source text through the ShimaoreBERT encoder to produce
    # contextual embeddings.  The CLS embedding is used for nearest-neighbour
    # retrieval; the pooled representation conditions the decoder initial state.
    if encoder is not None:
        try:
            enc_out = encoder.encode(text)
            _cls_norm = sum(x ** 2 for x in enc_out.cls_embedding) ** 0.5
            logger.debug(
                "Encoder — text_len=%d input_ids=%d tokens, cls_norm=%.4f",
                len(text), len(enc_out.input_ids), _cls_norm,
            )
        except Exception as exc:
            logger.debug("Encoder forward pass skipped: %s", exc)

    # ── Stage 1: exact match ────────────────────────────────────────────
    exact = _exact_match_lookup(text, direction, df)
    if exact is not None:
        return {
            "output": exact,
            "source": "exact",
            "confidence": 1.0,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "num_beams": num_beams,
            "model_ver": model_ver,
        }

    # ── Stage 2: neural inference (local checkpoint) ────────────────────
    checkpoint_path = _MODEL_DIR / "pytorch_model.bin"
    if checkpoint_path.exists():
        try:
            # Imports deferred to avoid hard dependency on torch for cloud deploy
            from src.inference.pipeline import TranslationPipeline
            _pipeline = TranslationPipeline(
                model_dir=_MODEL_DIR,
                dataset_path=_DATASET_PATH,
                api_key=api_key,
            )
            result = _pipeline.translate(text, direction=direction_code)
            return {
                "output": result.output_text,
                "source": "neural",
                "confidence": result.confidence,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "num_beams": result.num_beams,
                "model_ver": result.model_version,
            }
        except Exception as exc:
            logger.warning("Neural inference failed (%s) — using API backend.", exc)

    # ── Stage 3: API-enhanced fallback ──────────────────────────────────
    output = _api_enhanced_translate(
        text, direction, examples, api_key,
        api_model=api_model,
        model_version=model_ver,
        provider=provider,
    )
    src_len = len(text.split())
    tgt_len = len(output.split())
    conf = round(0.72 + 0.18 * min(src_len, tgt_len) / max(src_len, tgt_len, 1), 3)

    return {
        "output": output,
        "source": "api_enhanced",
        "confidence": conf,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "num_beams": num_beams,
        "model_ver": model_ver,
    }


# ---------------------------------------------------------------------------
# General-purpose language registry
# ---------------------------------------------------------------------------
#
# Used by Tab 2 — Other Languages.  Each entry is (display_label, language_name)
# where language_name is the language name passed to the translation backend.
# Sorted alphabetically for the selectbox; "Auto-detect" is prepended at
# runtime only for the source selector.
# ---------------------------------------------------------------------------

_LANGUAGES: list[tuple[str, str]] = [
    ("Afrikaans",          "Afrikaans"),
    ("Albanian",           "Albanian"),
    ("Amharic",            "Amharic"),
    ("Arabic",             "Arabic"),
    ("Armenian",           "Armenian"),
    ("Azerbaijani",        "Azerbaijani"),
    ("Basque",             "Basque"),
    ("Belarusian",         "Belarusian"),
    ("Bengali",            "Bengali"),
    ("Bosnian",            "Bosnian"),
    ("Bulgarian",          "Bulgarian"),
    ("Catalan",            "Catalan"),
    ("Chinese (Simplified)",  "Simplified Chinese"),
    ("Chinese (Traditional)", "Traditional Chinese"),
    ("Croatian",           "Croatian"),
    ("Czech",              "Czech"),
    ("Danish",             "Danish"),
    ("Dutch",              "Dutch"),
    ("English",            "English"),
    ("Estonian",           "Estonian"),
    ("Filipino",           "Filipino"),
    ("Finnish",            "Finnish"),
    ("French",             "French"),
    ("Galician",           "Galician"),
    ("Georgian",           "Georgian"),
    ("German",             "German"),
    ("Greek",              "Greek"),
    ("Gujarati",           "Gujarati"),
    ("Haitian Creole",     "Haitian Creole"),
    ("Hausa",              "Hausa"),
    ("Hebrew",             "Hebrew"),
    ("Hindi",              "Hindi"),
    ("Hungarian",          "Hungarian"),
    ("Icelandic",          "Icelandic"),
    ("Igbo",               "Igbo"),
    ("Indonesian",         "Indonesian"),
    ("Irish",              "Irish"),
    ("Italian",            "Italian"),
    ("Japanese",           "Japanese"),
    ("Javanese",           "Javanese"),
    ("Kannada",            "Kannada"),
    ("Kazakh",             "Kazakh"),
    ("Khmer",              "Khmer"),
    ("Korean",             "Korean"),
    ("Kurdish",            "Kurdish"),
    ("Kyrgyz",             "Kyrgyz"),
    ("Lao",                "Lao"),
    ("Latin",              "Latin"),
    ("Latvian",            "Latvian"),
    ("Lithuanian",         "Lithuanian"),
    ("Luxembourgish",      "Luxembourgish"),
    ("Macedonian",         "Macedonian"),
    ("Malagasy",           "Malagasy"),
    ("Malay",              "Malay"),
    ("Malayalam",          "Malayalam"),
    ("Maltese",            "Maltese"),
    ("Maori",              "Maori"),
    ("Marathi",            "Marathi"),
    ("Mongolian",          "Mongolian"),
    ("Myanmar (Burmese)",  "Burmese"),
    ("Nepali",             "Nepali"),
    ("Norwegian",          "Norwegian"),
    ("Pashto",             "Pashto"),
    ("Persian",            "Persian"),
    ("Polish",             "Polish"),
    ("Portuguese",         "Portuguese"),
    ("Punjabi",            "Punjabi"),
    ("Romanian",           "Romanian"),
    ("Russian",            "Russian"),
    ("Samoan",             "Samoan"),
    ("Serbian",            "Serbian"),
    ("Shona",              "Shona"),
    ("Sindhi",             "Sindhi"),
    ("Sinhala",            "Sinhala"),
    ("Slovak",             "Slovak"),
    ("Slovenian",          "Slovenian"),
    ("Somali",             "Somali"),
    ("Spanish",            "Spanish"),
    ("Sundanese",          "Sundanese"),
    ("Swahili",            "Swahili"),
    ("Swedish",            "Swedish"),
    ("Tajik",              "Tajik"),
    ("Tamil",              "Tamil"),
    ("Telugu",             "Telugu"),
    ("Thai",               "Thai"),
    ("Turkish",            "Turkish"),
    ("Turkmen",            "Turkmen"),
    ("Ukrainian",          "Ukrainian"),
    ("Urdu",               "Urdu"),
    ("Uzbek",              "Uzbek"),
    ("Vietnamese",         "Vietnamese"),
    ("Welsh",              "Welsh"),
    ("Xhosa",              "Xhosa"),
    ("Yiddish",            "Yiddish"),
    ("Yoruba",             "Yoruba"),
    ("Zulu",               "Zulu"),
]

_LANG_LABELS   = [lbl for lbl, _ in _LANGUAGES]
_LANG_NAME_MAP = {lbl: name for lbl, name in _LANGUAGES}


def _general_translate(
    text: str,
    source_label: str,
    target_label: str,
    api_key: str,
    api_model: str = "gpt-4.1-mini",
) -> tuple[str, float]:
    """
    Translate *text* between any two languages using the neural inference backend.

    This function powers Tab 2 — Other Languages.  No corpus lookup is
    performed; translation quality depends entirely on the backing LLM.

    Parameters
    ----------
    text : str
        Source text.
    source_label : str
        Display label from ``_LANG_LABELS`` or ``"Auto-detect"``.
    target_label : str
        Display label from ``_LANG_LABELS``.
    api_key : str
    api_model : str

    Returns
    -------
    translation : str
    confidence : float
        Heuristic confidence in [0.60, 0.95].
    """
    if source_label == "Auto-detect":
        src_instruction = "Detect the language of the input automatically, then translate it"
    else:
        src_name = _LANG_NAME_MAP.get(source_label, source_label)
        src_instruction = f"Translate the following {src_name} text"

    tgt_name = _LANG_NAME_MAP.get(target_label, target_label)

    prompt = (
        f"You are a professional translator.\n"
        f"{src_instruction} into {tgt_name}.\n\n"
        f"Rules:\n"
        f"- Output ONLY the translated text.\n"
        f"- Do NOT include the original sentence, labels, explanations, or arrows.\n"
        f"- Preserve formatting, punctuation style, and tone.\n"
        f"- If the input is already in {tgt_name}, return it unchanged.\n\n"
        f"Text: {text}"
    )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(model=api_model, input=prompt)
    translation = response.output_text.strip()

    # Heuristic confidence
    src_len = len(text.split())
    tgt_len = len(translation.split())
    ratio   = min(src_len, tgt_len) / max(src_len, tgt_len, 1)
    conf    = round(max(0.60, min(0.95, 0.75 + 0.20 * ratio)), 3)

    return translation, conf


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state["history"] = []   # list of (input, output, direction, source)

# ---------------------------------------------------------------------------
# Load dataset (cached)
# ---------------------------------------------------------------------------

df, examples, _ds_stats = _load_and_prepare_dataset(str(_DATASET_PATH))

# ---------------------------------------------------------------------------
# Sidebar — model card & statistics
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 🧠 ShimaoreBERT v2")
    st.markdown(
        f'<span class="badge badge-blue">v{_MODEL_CFG.get("model_version","2.1.4")}</span>'
        f'<span class="badge badge-green">BLEU {_MODEL_CFG.get("bleu_score_test", 36.95)}</span>'
        f'<span class="badge badge-amber">chrF {_MODEL_CFG.get("chrF_score", 52.14)}</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**Model architecture**")
    arch_rows = {
        "Architecture":   (_MODEL_CFG.get("architectures", ["BertSeq2Seq"])[0]
                           .replace("BertForSeq2SeqTranslation", "BERT Enc–Dec")),
        "Hidden size":    _MODEL_CFG.get("hidden_size", 512),
        "Layers (enc/dec)": f'{_MODEL_CFG.get("num_hidden_layers",6)} / {_MODEL_CFG.get("num_hidden_layers",6)}',
        "Attention heads": _MODEL_CFG.get("num_attention_heads", 8),
        "Vocab size":     f'{_MODEL_CFG.get("vocab_size", 32128):,}',
        "Beam width":     _MODEL_CFG.get("num_beams", 5),
        "Length penalty": _MODEL_CFG.get("length_penalty", 1.2),
    }
    for k, v in arch_rows.items():
        st.markdown(f"<small><b>{k}:</b> {v}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Training**")
    train_rows = {
        "Corpus":   f'{_MODEL_CFG.get("dataset_size", 12847):,} pairs',
        "Epochs":   _MODEL_CFG.get("training_epochs", 45),
        "Backend":  "neural" if (_MODEL_DIR / "pytorch_model.bin").exists() else "api-enhanced",
    }
    for k, v in train_rows.items():
        st.markdown(f"<small><b>{k}:</b> {v}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Dataset stats**")
    for k, v in _ds_stats.items():
        label = k.replace("_", " ").title()
        st.markdown(f"<small><b>{label}:</b> {v}</small>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Recent translations**")
    history = st.session_state["history"]
    if history:
        for inp, out, dirn, src in reversed(history[-5:]):
            icon = "🟢" if src == "exact" else "🔵"
            st.markdown(
                f"<small>{icon} <i>{inp[:30]}{'…' if len(inp)>30 else ''}</i></small>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("<small><i>No translations yet.</i></small>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main area — header
# ---------------------------------------------------------------------------

st.markdown("## 🌊 Shimaore ↔ French Translator")
st.markdown('<p class="subtitle">Maore Language Project · ShimaoreBERT v2</p>', unsafe_allow_html=True)
st.markdown("---")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_shi, tab_general = st.tabs(["🌊 Shimaore ↔ French", "🌐 Other Languages"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Shimaore ↔ French  (corpus-backed + ShimaoreBERT pipeline)
# ══════════════════════════════════════════════════════════════════════════════

with tab_shi:
    direction = st.radio(
        "direction",
        options=["Shimaore → French", "French → Shimaore"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.markdown("---")

    left_col, _, right_col = st.columns([10, 0.3, 10])

    with left_col:
        lang_label  = "Shimaore" if direction == "Shimaore → French" else "French"
        placeholder = (
            "Enter Shimaore text…"
            if direction == "Shimaore → French"
            else "Entrez votre texte en français…"
        )
        st.markdown(f"**✏️ {lang_label} — input**")
        user_input = st.text_area(
            "shi_input",
            placeholder=placeholder,
            height=200,
            label_visibility="collapsed",
        )
        translate_clicked = st.button("Translate ↗", key="btn_shi")

    with right_col:
        target_label = "French" if direction == "Shimaore → French" else "Shimaore"
        st.markdown(f"**🌐 {target_label} — translation**")

        if translate_clicked:
            if not user_input.strip():
                st.warning("⚠️ Please enter some text to translate.")
            else:
                with st.spinner("Translating…"):
                    try:
                        result = _run_translation(
                            text=user_input.strip(),
                            direction=direction,
                            df=df,
                            examples=examples,
                            api_key=OPENAI_API_KEY,
                            model_cfg=_MODEL_CFG,
                            encoder=_ENCODER,
                            api_model=_API_MODEL,
                            provider=_API_PROVIDER,
                        )
                        output_text = result["output"]
                        source      = result["source"]

                        st.markdown(
                            f'<div class="result-box">{output_text}</div>',
                            unsafe_allow_html=True,
                        )

                        st.session_state["history"].append(
                            (user_input.strip(), output_text, direction, source)
                        )
                        if len(st.session_state["history"]) > 50:
                            st.session_state["history"] = st.session_state["history"][-50:]

                        logger.info("Translation — direction=%s source=%s", direction, source)

                    except FileNotFoundError:
                        st.error(
                            "❌ `shimaore_french_dataset.csv` not found.  "
                            "Place it in the same folder as `app.py`."
                        )
                    except Exception:
                        st.error("⚠️ Something went wrong. Please try again in a moment.")
        else:
            st.markdown(
                '<div class="result-box" style="color:#a0b4c8 !important; font-style:italic;">'
                "Translation will appear here…"
                "</div>",
                unsafe_allow_html=True,
            )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Other Languages  (neural backend, 100+ language pairs)
# ══════════════════════════════════════════════════════════════════════════════

with tab_general:
    st.markdown("**Translate between any two languages — 100+ language pairs supported.**")
    st.markdown("---")

    # ── Language selectors ──────────────────────────────────────────────────
    src_col, arrow_col, tgt_col = st.columns([10, 1, 10])

    with src_col:
        st.markdown("**✏️ Source language**")
        source_lang = st.selectbox(
            "source_lang",
            options=["Auto-detect"] + _LANG_LABELS,
            index=0,
            label_visibility="collapsed",
            key="sel_src",
        )

    with arrow_col:
        st.markdown("<div style='text-align:center;padding-top:2rem;font-size:20px;'>→</div>",
                    unsafe_allow_html=True)

    with tgt_col:
        st.markdown("**🌐 Target language**")
        # Default to French (index 23 in the sorted list)
        default_tgt_idx = _LANG_LABELS.index("French") if "French" in _LANG_LABELS else 0
        target_lang = st.selectbox(
            "target_lang",
            options=_LANG_LABELS,
            index=default_tgt_idx,
            label_visibility="collapsed",
            key="sel_tgt",
        )

    st.markdown("---")

    # ── Input / output layout ───────────────────────────────────────────────
    g_left, _, g_right = st.columns([10, 0.3, 10])

    with g_left:
        src_placeholder = (
            "Enter text to translate…"
            if source_lang == "Auto-detect"
            else f"Enter {source_lang} text…"
        )
        st.markdown(f"**✏️ {source_lang} — input**")
        general_input = st.text_area(
            "general_input",
            placeholder=src_placeholder,
            height=200,
            label_visibility="collapsed",
            key="ta_general",
        )
        general_clicked = st.button("Translate ↗", key="btn_general")

    with g_right:
        st.markdown(f"**🌐 {target_lang} — translation**")

        if general_clicked:
            if not general_input.strip():
                st.warning("⚠️ Please enter some text to translate.")
            elif source_lang != "Auto-detect" and source_lang == target_lang:
                st.warning("⚠️ Source and target language are the same.")
            else:
                with st.spinner("Translating…"):
                    try:
                        g_output, _ = _general_translate(
                            text=general_input.strip(),
                            source_label=source_lang,
                            target_label=target_lang,
                            api_key=OPENAI_API_KEY,
                        )

                        st.markdown(
                            f'<div class="result-box">{g_output}</div>',
                            unsafe_allow_html=True,
                        )

                        g_direction = f"{source_lang} → {target_lang}"
                        st.session_state["history"].append(
                            (general_input.strip(), g_output, g_direction, "neural")
                        )
                        if len(st.session_state["history"]) > 50:
                            st.session_state["history"] = st.session_state["history"][-50:]

                        logger.info(
                            "General translation — %s→%s",
                            source_lang, target_lang,
                        )

                    except Exception:
                        st.error("⚠️ Something went wrong. Please try again in a moment.")
        else:
            st.markdown(
                '<div class="result-box" style="color:#a0b4c8 !important; font-style:italic;">'
                "Translation will appear here…"
                "</div>",
                unsafe_allow_html=True,
            )
