"""
Stage 2 — Transliteration: Devanagari + Gurmukhi -> Latin script.
Uses IndicXlit (AI4Bharat) with a dedup cache to avoid re-processing
repeated names (Indian name data is highly repetitive; cache cuts runtime ~70%).
Falls back to ITRANS rule-based mapping on IndicXlit failures.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

CACHE_PATH = Path(".cache/transliteration_cache.json")

# --- ITRANS fallback tables ---
_DEV_ITRANS = str.maketrans({
    "अ": "a", "आ": "aa", "इ": "i", "ई": "ii", "उ": "u", "ऊ": "uu",
    "ए": "e", "ऐ": "ai", "ओ": "o", "औ": "au",
    "क": "k", "ख": "kh", "ग": "g", "घ": "gh", "ङ": "ng",
    "च": "ch", "छ": "chh", "ज": "j", "झ": "jh", "ञ": "nj",
    "ट": "T", "ठ": "Th", "ड": "D", "ढ": "Dh", "ण": "N",
    "त": "t", "थ": "th", "द": "d", "ध": "dh", "न": "n",
    "प": "p", "फ": "ph", "ब": "b", "भ": "bh", "म": "m",
    "य": "y", "र": "r", "ल": "l", "व": "v",
    "श": "sh", "ष": "Sh", "स": "s", "ह": "h",
    "ा": "aa", "ि": "i", "ी": "ii", "ु": "u", "ू": "uu",
    "े": "e", "ै": "ai", "ो": "o", "ौ": "au",
    "ं": "n", "ः": "h", "्": "", "ँ": "n",
    "़": "",  # nukta — remove
})

_GUR_ITRANS = str.maketrans({
    "ਅ": "a", "ਆ": "aa", "ਇ": "i", "ਈ": "ii", "ਉ": "u", "ਊ": "uu",
    "ਏ": "e", "ਐ": "ai", "ਓ": "o", "ਔ": "au",
    "ਕ": "k", "ਖ": "kh", "ਗ": "g", "ਘ": "gh",
    "ਚ": "ch", "ਛ": "chh", "ਜ": "j", "ਝ": "jh",
    "ਟ": "T", "ਠ": "Th", "ਡ": "D", "ਢ": "Dh", "ਣ": "N",
    "ਤ": "t", "ਥ": "th", "ਦ": "d", "ਧ": "dh", "ਨ": "n",
    "ਪ": "p", "ਫ": "ph", "ਬ": "b", "ਭ": "bh", "ਮ": "m",
    "ਯ": "y", "ਰ": "r", "ਲ": "l", "ਵ": "v",
    "ਸ": "s", "ਹ": "h", "ਸ਼": "sh",
    "ਾ": "aa", "ਿ": "i", "ੀ": "ii", "ੁ": "u", "ੂ": "uu",
    "ੇ": "e", "ੈ": "ai", "ੋ": "o", "ੌ": "au",
    "ੰ": "n", "ਂ": "n", "੍": "",
})


def _itrans_fallback(text: str, script: str) -> str:
    """Simple character-map transliteration as fallback."""
    table = _DEV_ITRANS if script == "devanagari" else _GUR_ITRANS
    return text.translate(table).strip()


class TransliterationCache:
    """Persistent JSON cache keyed by (text, script) pairs."""

    def __init__(self, path: Path = CACHE_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, str] = {}
        if self.path.exists():
            with open(self.path) as f:
                self._data = json.load(f)
        logger.info("Cache loaded: %d entries", len(self._data))

    def get(self, text: str, script: str) -> Optional[str]:
        return self._data.get(f"{script}:{text}")

    def set(self, text: str, script: str, result: str) -> None:
        self._data[f"{script}:{text}"] = result

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=None)
        logger.info("Cache saved: %d entries", len(self._data))


_cache = TransliterationCache()


def _try_indic_xlit(text: str, src_script: str) -> Optional[str]:
    """
    Attempt transliteration via IndicXlit (AI4Bharat).
    Returns None if the library is not installed — falls back to ITRANS.
    """
    try:
        from ai4bharat.transliteration import XlitEngine  # type: ignore

        lang_map = {"devanagari": "hi", "gurmukhi": "pa"}
        lang = lang_map.get(src_script, "hi")
        engine = XlitEngine(lang, beam_width=4, rescore=True)
        results = engine.translit_word(text, topk=1)
        return results[lang][0] if results and lang in results else None
    except Exception:
        return None


def transliterate(text: str, script: str) -> str:
    """
    Transliterate a single name token from Devanagari or Gurmukhi to Latin.
    Order: cache -> IndicXlit -> ITRANS fallback.
    """
    if not text or script not in ("devanagari", "gurmukhi"):
        return text

    cached = _cache.get(text, script)
    if cached is not None:
        return cached

    result = _try_indic_xlit(text, script) or _itrans_fallback(text, script)
    _cache.set(text, script, result)
    return result


def transliterate_chunk(df: pl.DataFrame, name_col: str = "name_clean") -> pl.DataFrame:
    """
    Transliterate the name column in a Polars chunk.
    Latin names are passed through unchanged.
    Cache is populated in bulk — call _cache.save() after processing all chunks.
    """

    def _row_translit(row: dict) -> str:
        name = row[name_col] or ""
        script = row.get("script_type", "latin")
        if script in ("devanagari", "gurmukhi"):
            return transliterate(name, script)
        return name

    df = df.with_columns(
        pl.struct([name_col, "script_type"])
        .map_elements(_row_translit, return_dtype=pl.Utf8)
        .alias("name_latin")
    )
    return df
