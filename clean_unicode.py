"""
Stage 1 — Unicode + OCR artifact cleaning for Indian name datasets.
Handles Devanagari/Gurmukhi normalization, zero-width chars, OCR noise.
"""

import re
import unicodedata
import logging
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

# Zero-width and invisible Unicode chars common in OCR'd Indian text
_ZW_CHARS = re.compile(r"[​‌‍‎‏﻿ ]")

# Common OCR substitutions in Latin transliterations of Indian names
_OCR_FIXES = [
    (re.compile(r"\b([A-Z][a-z]*)1\b"), r"\g<1>l"),   # Ra1 -> Ral (rare)
    (re.compile(r"\b0([a-z])"), r"O\1"),               # 0m -> Om
    (re.compile(r"([A-Za-z])\|"), r"\1l"),             # pipe -> l
    (re.compile(r"\s{2,}"), " "),                      # collapse spaces
]

# Honorifics / relationship markers in Indian voter roll name fields
_HONORIFICS = re.compile(
    r"\b(S/?O|D/?O|W/?O|C/?O|H/?O|Shri|Smt|Km|Kumar|Late|Sh)\b\.?",
    flags=re.IGNORECASE,
)

# Script detection ranges
_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_GURMUKHI = re.compile(r"[਀-੿]")


def detect_script(text: str) -> str:
    """Return dominant script: 'devanagari', 'gurmukhi', 'latin', or 'mixed'."""
    if not text:
        return "unknown"
    has_dev = bool(_DEVANAGARI.search(text))
    has_gur = bool(_GURMUKHI.search(text))
    if has_dev and has_gur:
        return "mixed"
    if has_dev:
        return "devanagari"
    if has_gur:
        return "gurmukhi"
    return "latin"


def clean_unicode(text: Optional[str]) -> str:
    """
    Full Unicode cleaning pass for a single name field.
    1. NFC normalization (handles decomposed Devanagari nukta variants)
    2. Strip zero-width / invisible chars
    3. Fix common OCR substitutions
    4. Strip honorifics / relationship markers
    5. Collapse whitespace, strip leading/trailing
    """
    if not text or not isinstance(text, str):
        return ""

    # NFC: composed form — handles क़ variant normalization
    text = unicodedata.normalize("NFC", text)

    # Strip invisible chars
    text = _ZW_CHARS.sub("", text)

    # OCR fixes (Latin-script names only — skip Indic)
    if detect_script(text) == "latin":
        for pattern, repl in _OCR_FIXES:
            text = pattern.sub(repl, text)

    # Strip honorifics
    text = _HONORIFICS.sub("", text).strip()

    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_chunk(df: pl.DataFrame, name_cols: list[str]) -> pl.DataFrame:
    """Apply clean_unicode to all specified name columns in a Polars DataFrame chunk."""
    exprs = [
        pl.col(col).map_elements(clean_unicode, return_dtype=pl.Utf8).alias(col)
        for col in name_cols
        if col in df.columns
    ]
    return df.with_columns(exprs)


def add_script_column(df: pl.DataFrame, col: str = "name_clean") -> pl.DataFrame:
    """Append a script_type column based on the cleaned name."""
    return df.with_columns(
        pl.col(col).map_elements(detect_script, return_dtype=pl.Utf8).alias("script_type")
    )
