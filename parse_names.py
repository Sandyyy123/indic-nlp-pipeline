"""
Stage 3 — Name parsing: extract given name and surname from full name
and parental name fields in Indian voter roll / administrative datasets.
Handles S/O, D/O, W/O patterns, initials, joint names, multi-word surnames.
"""

import re
from dataclasses import dataclass
from typing import Optional

import polars as pl

# Relationship markers that appear in parental name fields
_REL_MARKERS = re.compile(
    r"^(S/?O|D/?O|W/?O|C/?O|H/?O|Son of|Daughter of|Wife of)\s*:?\s*",
    flags=re.IGNORECASE,
)

# Single initial (e.g. "R. Kumar" or "R Kumar")
_INITIAL = re.compile(r"^[A-Za-z]\.?\s+")

# Common Indian surnames (partial list for heuristic splitting)
_KNOWN_SURNAMES = {
    "singh", "kumar", "sharma", "verma", "gupta", "patel", "rao", "nair",
    "reddy", "joshi", "kaur", "mehta", "shah", "yadav", "mishra", "pandey",
    "tiwari", "chauhan", "bhatia", "malhotra", "kapoor", "arora", "sinha",
    "das", "devi", "bai", "devi", "lal", "ram", "prasad", "naidu",
}


@dataclass
class ParsedName:
    given_name: str
    surname: str
    raw: str
    notes: str = ""


def _strip_relation(text: str) -> str:
    """Remove leading relationship marker (S/O, D/O, etc.)."""
    return _REL_MARKERS.sub("", text).strip()


def _split_given_surname(name: str) -> tuple[str, str]:
    """
    Heuristic split of a normalized Latin name into given name + surname.
    Strategy:
    1. If single token -> given_name only, surname empty
    2. If last token is a known surname -> split there
    3. If first token is an initial -> given_name = initial, surname = rest
    4. Default: first token = given_name, rest = surname
    """
    name = name.strip()
    tokens = name.split()

    if not tokens:
        return "", ""

    if len(tokens) == 1:
        return tokens[0], ""

    # Last token known surname
    if tokens[-1].lower() in _KNOWN_SURNAMES:
        return " ".join(tokens[:-1]), tokens[-1]

    # First token is initial
    if _INITIAL.match(name):
        return tokens[0], " ".join(tokens[1:])

    # Default: first = given, rest = surname
    return tokens[0], " ".join(tokens[1:])


def parse_name(full_name: Optional[str], parental_name: Optional[str] = None) -> ParsedName:
    """
    Parse a single record's name fields into structured given_name + surname.
    Uses parental_name field to improve surname identification when available.
    """
    raw = full_name or ""
    clean = _strip_relation(raw)

    if not clean:
        return ParsedName(given_name="", surname="", raw=raw, notes="empty")

    given, surname = _split_given_surname(clean)

    # If parental name available, try to extract shared surname
    notes = ""
    if parental_name:
        parent_clean = _strip_relation(parental_name)
        parent_tokens = parent_clean.split()
        if parent_tokens and parent_tokens[-1].lower() in _KNOWN_SURNAMES:
            candidate = parent_tokens[-1]
            if not surname or surname.lower() != candidate.lower():
                notes = f"surname_from_parent:{candidate}"
                # Only override if our split produced no surname
                if not surname:
                    surname = candidate

    return ParsedName(given_name=given, surname=surname, raw=raw, notes=notes)


def parse_names_chunk(
    df: pl.DataFrame,
    full_name_col: str = "name_latin",
    parental_col: Optional[str] = "parental_name_latin",
) -> pl.DataFrame:
    """
    Apply name parsing to a Polars chunk.
    Adds columns: given_name, surname, parse_notes.
    """

    def _parse_row(row: dict) -> dict:
        full = row.get(full_name_col, "")
        parent = row.get(parental_col) if parental_col and parental_col in row else None
        parsed = parse_name(full, parent)
        return {
            "given_name": parsed.given_name,
            "surname": parsed.surname,
            "parse_notes": parsed.notes,
        }

    cols = [full_name_col]
    if parental_col and parental_col in df.columns:
        cols.append(parental_col)

    parsed = df.select(cols).map_rows(
        lambda row: tuple(_parse_row(dict(zip(cols, row))).values()),
        return_dtype=pl.Struct({"given_name": pl.Utf8, "surname": pl.Utf8, "parse_notes": pl.Utf8}),
    )

    return df.with_columns([
        parsed.struct.field("given_name"),
        parsed.struct.field("surname"),
        parsed.struct.field("parse_notes"),
    ])
