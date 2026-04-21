"""Configurable Flask server for the tokenizer visualizer."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from tokenizers.decoders import ByteLevel
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import sentencepiece as spm


LOGGER = logging.getLogger(__name__)
ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT_DIR / "tokenizers.example.json"
DEFAULT_HOST = os.getenv("TOKENIZER_VISUALIZER_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.getenv("TOKENIZER_VISUALIZER_PORT", "5000"))
BYTE_LEVEL_DECODER = ByteLevel()


@dataclass(frozen=True)
class TokenizerSpec:
    id: str
    label: str
    source: str
    path: str
    description: str = ""
    trust_remote_code: bool = False
    token: str | None = None
    decode_mode: str = "byte_level"


class SentencePieceWrapper:
    def __init__(self, model_path: str) -> None:
        self.processor = spm.SentencePieceProcessor()
        self.processor.load(model_path)

    def encode(self, text: str) -> list[str]:
        return list(self.processor.encode_as_pieces(text))


def _build_app_config() -> dict[str, Any]:
    config_path = Path(os.getenv("TOKENIZER_VISUALIZER_CONFIG", DEFAULT_CONFIG_PATH))
    return {
        "TOKENIZER_CONFIG_PATH": config_path,
        "FRONTEND_DIR": ROOT_DIR,
    }


def load_tokenizer_specs(config_path: Path) -> list[TokenizerSpec]:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Tokenizer config file not found: {config_path}. "
            "Set TOKENIZER_VISUALIZER_CONFIG to a valid JSON file."
        )

    with open(config_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    specs = []
    for item in data.get("tokenizers", []):
        specs.append(
            TokenizerSpec(
                id=item["id"],
                label=item["label"],
                source=item["source"],
                path=item["path"],
                description=item.get("description", ""),
                trust_remote_code=item.get("trust_remote_code", False),
                token=item.get("token"),
                decode_mode=item.get("decode_mode", "byte_level"),
            )
        )

    if not specs:
        raise ValueError(f"No tokenizers configured in {config_path}")
    return specs

def create_app() -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    app = Flask(__name__, static_folder=None)
    app.config.update(_build_app_config())
    CORS(app)

    @lru_cache(maxsize=1)
    def get_specs() -> list[TokenizerSpec]:
        return load_tokenizer_specs(Path(app.config["TOKENIZER_CONFIG_PATH"]))

    @lru_cache(maxsize=32)
    def get_tokenizer(tokenizer_id: str):
        spec = next((item for item in get_specs() if item.id == tokenizer_id), None)
        if spec is None:
            raise KeyError(f"Unknown tokenizer id: {tokenizer_id}")

        LOGGER.info("Loading tokenizer %s from %s", spec.id, spec.path)
        if spec.source == "hf":
            load_kwargs = {"token": spec.token, "trust_remote_code": spec.trust_remote_code}
            load_kwargs = {key: value for key, value in load_kwargs.items() if value}
            return AutoTokenizer.from_pretrained(spec.path, **load_kwargs)
        if spec.source == "fast":
            tokenizer_path = Path(spec.path)
            tokenizer_file = tokenizer_path / "tokenizer.json" if tokenizer_path.is_dir() else tokenizer_path
            return PreTrainedTokenizerFast(tokenizer_file=str(tokenizer_file))
        if spec.source == "spm":
            return SentencePieceWrapper(spec.path)
        raise ValueError(f"Unsupported tokenizer source: {spec.source}")

    def decode_token_strings(tokenizer, token_ids: list[int], spec: TokenizerSpec) -> list[str]:
        if spec.decode_mode == "token":
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            return [token.replace("▁", " ").replace("Ġ", " ") for token in tokens]

        decoded_tokens = []
        for token_id in token_ids:
            decoded_piece = tokenizer.decode([token_id], skip_special_tokens=False)
            decoded_tokens.append(BYTE_LEVEL_DECODER.decode([decoded_piece]))
        return decoded_tokens

    def tokenize_with_spec(tokenizer_id: str, text: str) -> list[str]:
        spec = next((item for item in get_specs() if item.id == tokenizer_id), None)
        if spec is None:
            raise KeyError(f"Unknown tokenizer id: {tokenizer_id}")
        tokenizer = get_tokenizer(tokenizer_id)
        if isinstance(tokenizer, SentencePieceWrapper):
            return tokenizer.encode(text)

        token_ids = tokenizer.encode(text, add_special_tokens=False)
        return decode_token_strings(tokenizer, token_ids, spec)

    @app.get("/")
    def index():
        return send_from_directory(app.config["FRONTEND_DIR"], "index.html")

    @app.get("/api/tokenizers")
    def list_tokenizers():
        return jsonify(
            {
                "tokenizers": [
                    {
                        "id": spec.id,
                        "label": spec.label,
                        "description": spec.description,
                        "source": spec.source,
                    }
                    for spec in get_specs()
                ]
            }
        )

    @app.post("/api/tokenize")
    def tokenize_text():
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400

        payload = request.get_json(silent=True) or {}
        text = payload.get("text", "")
        tokenizer_id = payload.get("tokenizerId")

        if tokenizer_id is None:
            return jsonify({"error": "Missing 'tokenizerId' in request"}), 400
        if not isinstance(text, str):
            return jsonify({"error": "'text' must be a string"}), 400
        if not text.strip():
            return jsonify({"tokens": [], "word_count": 0})

        try:
            tokens = tokenize_with_spec(tokenizer_id, text)
        except KeyError as exc:
            return jsonify({"error": str(exc)}), 404
        except Exception as exc:  # pragma: no cover - integration/runtime errors
            LOGGER.exception("Tokenizer request failed")
            return jsonify({"error": f"Failed to tokenize input: {exc}"}), 500

        return jsonify({"tokens": tokens, "word_count": len(text.strip().split())})

    return app


app = create_app()


if __name__ == "__main__":
    app.run(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=False)
