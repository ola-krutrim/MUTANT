"""Tokenizer loading and decoding logic."""

from __future__ import annotations

import os

import sentencepiece as spm
from transformers import AutoTokenizer, PreTrainedTokenizerFast, T5Tokenizer

from tokenizer_evaluator.utils.autotiktokenizer import AutoTikTokenizer


def load_tokenizer(tokenizer_path, tokenizer_type="hf", hf_key=None, trust_remote_code=False):
    if tokenizer_type == "hf":
        load_kwargs = {"token": hf_key, "trust_remote_code": trust_remote_code}
        load_kwargs = {key: value for key, value in load_kwargs.items() if value}
        try:
            return AutoTokenizer.from_pretrained(tokenizer_path, **load_kwargs)
        except Exception:
            return T5Tokenizer.from_pretrained(tokenizer_path, token=hf_key)

    if tokenizer_type == "spm":
        processor = spm.SentencePieceProcessor()
        processor.load(tokenizer_path)
        return processor

    if tokenizer_type == "ptf":
        tokenizer_file = tokenizer_path
        if os.path.isdir(tokenizer_path):
            tokenizer_file = os.path.join(tokenizer_path, "tokenizer.json")
        return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file)

    if tokenizer_type == "tkt":
        return AutoTikTokenizer.from_pretrained(tokenizer_path, token=hf_key)

    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")


def decode_text(tokenizer, tokens, tokenizer_type="hf"):
    if tokenizer_type == "hf":
        return tokenizer.decode(tokens, skip_special_tokens=True)
    if tokenizer_type == "spm":
        return tokenizer.decode_pieces(tokens)
    if tokenizer_type == "ptf":
        return tokenizer.decode(tokens)
    if tokenizer_type == "tkt":
        return tokenizer.decode(tokens)
    raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
