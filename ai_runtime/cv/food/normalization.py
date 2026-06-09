from __future__ import annotations

import re

_PARENTHETICAL_RE = re.compile(r"\([^)]*\)|\[[^\]]*\]|（[^）]*）")
_SEPARATOR_RE = re.compile(r"[_/·,]+")
_WHITESPACE_RE = re.compile(r"\s+")
_MATCH_KEY_RE = re.compile(r"[\s_\-/·,().\[\]{}]+")


def cleanup_food_query(value: str) -> str:
    stripped = value.strip()
    without_parenthetical = _PARENTHETICAL_RE.sub("", stripped)
    separated = _SEPARATOR_RE.sub(" ", without_parenthetical)
    compact = _WHITESPACE_RE.sub(" ", separated).strip()
    return compact.lower() if compact.isascii() else compact


def normalize_food_name(value: str) -> str:
    return _MATCH_KEY_RE.sub("", cleanup_food_query(value)).lower()
