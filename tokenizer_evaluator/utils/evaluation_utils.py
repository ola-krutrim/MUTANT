"""Metric helpers used by tokenizer evaluators."""

from __future__ import annotations

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover - exercised only in minimal test envs
    from difflib import SequenceMatcher

    class _FuzzModule:
        @staticmethod
        def ratio(original: str, decoded: str) -> float:
            return SequenceMatcher(None, original, decoded).ratio() * 100

    fuzz = _FuzzModule()


def exact_match(original: str, decoded: str) -> float:
    return float(original.strip() == decoded.strip())


def fuzzy_score(original: str, decoded: str) -> float:
    return float(fuzz.ratio(original, decoded))


def lcs(original: str, decoded: str) -> float:
    if not original:
        return 0.0
    m, n = len(original), len(decoded)
    table = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if original[i] == decoded[j]:
                table[i + 1][j + 1] = table[i][j] + 1
            else:
                table[i + 1][j + 1] = max(table[i + 1][j], table[i][j + 1])
    return 1 - (table[m][n] / len(original))


BYTE_TOKENS = {f"<0x{b:02X}>" for b in range(256)}


def get_byte_fallback(tokens, _text=None):
    count_fallback = 0
    i = 0
    while i < len(tokens):
        if tokens[i] in BYTE_TOKENS:
            while i < len(tokens) and tokens[i] in BYTE_TOKENS:
                i += 1
            count_fallback += 1
        else:
            i += 1
    return count_fallback
