"""Configuration defaults for the tokenizer evaluator package."""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency at runtime
    load_dotenv = None


if load_dotenv is not None:
    load_dotenv()


DEFAULT_DATA_PATH = os.getenv("TOK_EVAL_DATA_PATH", "./data")
DEFAULT_OUTPUT_PATH = os.getenv("TOK_EVAL_OUTPUT_PATH", "./results")
DEFAULT_HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_KEY")
DEFAULT_TOKENIZER_PATHS = tuple(
    path.strip() for path in os.getenv("TOK_EVAL_TOKENIZER_PATHS", "").split(",") if path.strip()
)
DEFAULT_LANGUAGE = os.getenv("TOK_EVAL_LANGUAGE")
