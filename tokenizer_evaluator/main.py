"""Command-line entrypoint for tokenizer evaluation."""

from __future__ import annotations

import argparse
import glob
import importlib
import os
import re
from pathlib import Path
from typing import Iterable, Sequence

from tokenizer_evaluator.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_HF_TOKEN,
    DEFAULT_LANGUAGE,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_TOKENIZER_PATHS,
)
from tokenizer_evaluator.evaluators.base import EvaluationInput

SUPPORTED_TOKENIZER_TYPES = ("hf", "spm", "ptf", "tkt")
SUPPORTED_LANGUAGES = (
    "as",
    "bn",
    "brx",
    "code",
    "doi",
    "eng",
    "gom",
    "gu",
    "hi",
    "kn",
    "mai",
    "ml",
    "mr",
    "nep",
    "or",
    "pa",
    "san",
    "sat",
    "snd",
    "ta",
    "te",
    "urd",
)
SUPPORTED_DATA_SOURCES = ("files", "hf")


def get_evaluator(tokenizer_path, tokenizer_type, data_files, out_dir, hf_token=None, trust_remote_code=False):
    if tokenizer_type == "hf":
        from tokenizer_evaluator.evaluators.hf_eval import HFEvaluator

        evaluator_cls = HFEvaluator
    elif tokenizer_type == "spm":
        from tokenizer_evaluator.evaluators.spm_eval import SPMEvaluator

        evaluator_cls = SPMEvaluator
    elif tokenizer_type == "ptf":
        from tokenizer_evaluator.evaluators.ptf_eval import PTFEvaluator

        evaluator_cls = PTFEvaluator
    elif tokenizer_type == "tkt":
        from tokenizer_evaluator.evaluators.tkt_eval import TikTokenEvaluator

        evaluator_cls = TikTokenEvaluator
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")
    return evaluator_cls(
        tokenizer_path=tokenizer_path,
        tokenizer_type=tokenizer_type,
        data_files=data_files,
        out_dir=out_dir,
        hf_token=hf_token,
        trust_remote_code=trust_remote_code,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate tokenizer fidelity and efficiency on text corpora.")
    parser.add_argument(
        "--tokenizer_paths",
        nargs="+",
        default=list(DEFAULT_TOKENIZER_PATHS),
        help="Tokenizer model names or local paths.",
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_PATH, help="Directory containing .txt evaluation files.")
    parser.add_argument("--data_source", choices=SUPPORTED_DATA_SOURCES, default="files")
    parser.add_argument("--out_dir", default=DEFAULT_OUTPUT_PATH, help="Directory for CSV and JSON outputs.")
    parser.add_argument("--tokenizer_type", choices=SUPPORTED_TOKENIZER_TYPES, required=True)
    parser.add_argument("--hf_token", default=DEFAULT_HF_TOKEN, help="Hugging Face token for gated repos if needed.")
    parser.add_argument("--lang", choices=SUPPORTED_LANGUAGES, default=DEFAULT_LANGUAGE)
    parser.add_argument("--hf_dataset", default=None, help="Hugging Face dataset repo id.")
    parser.add_argument("--hf_split", default="test", help="Dataset split to evaluate for --data_source hf.")
    parser.add_argument(
        "--hf_text_columns",
        nargs="+",
        default=["text"],
        help="Text columns to evaluate for --data_source hf. Defaults to: text",
    )
    parser.add_argument(
        "--enable-nsl",
        action="store_true",
        help="Compute NSL against a base tokenizer. Currently supported for the ptf evaluator.",
    )
    parser.add_argument(
        "--nsl-base-tokenizer",
        default=None,
        help="Base tokenizer path or repo name used for NSL when --enable-nsl is set.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow Hugging Face tokenizers that require trust_remote_code=True.",
    )
    return parser


def _sanitize_output_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._") or "dataset"


def discover_file_inputs(data_dir: str) -> Sequence[EvaluationInput]:
    data_files = sorted(glob.glob(os.path.join(data_dir, "**", "*.txt"), recursive=True))
    return [
        EvaluationInput(
            source_id=file_path,
            label=os.path.basename(file_path),
            output_stem=Path(file_path).stem,
            path=file_path,
        )
        for file_path in data_files
    ]


def _infer_text_columns(dataset_split, requested_columns: Sequence[str] | None) -> Sequence[str]:
    if requested_columns:
        missing_columns = [column for column in requested_columns if column not in dataset_split.features]
        if missing_columns:
            raise ValueError(
                f"Requested text columns not found in dataset split: {', '.join(missing_columns)}"
            )
        return list(requested_columns)
    string_columns = [
        column_name
        for column_name, feature in dataset_split.features.items()
        if getattr(feature, "dtype", None) == "string" or feature.__class__.__name__ == "Value" and getattr(feature, "dtype", None) == "string"
    ]
    if not string_columns:
        raise ValueError("Could not infer any string text columns from the Hugging Face dataset split.")
    return string_columns


def discover_hf_inputs(dataset_name: str, split: str, lang: str | None, text_columns: Sequence[str] | None) -> Sequence[EvaluationInput]:
    datasets_module = importlib.import_module("datasets")
    load_dataset = datasets_module.load_dataset
    get_dataset_config_names = getattr(datasets_module, "get_dataset_config_names", None)

    inputs = []
    config_names = get_dataset_config_names(dataset_name) if get_dataset_config_names else []
    if config_names:
        available_lang_configs = [config for config in config_names if config in SUPPORTED_LANGUAGES]
        if lang is not None:
            selected_configs = [lang]
        elif available_lang_configs:
            selected_configs = available_lang_configs
        else:
            selected_configs = list(config_names)
    else:
        selected_configs = [None]

    for config_name in selected_configs:
        dataset_split = load_dataset(dataset_name, name=config_name, split=split)
        selected_columns = _infer_text_columns(dataset_split, text_columns)
        for column_name in selected_columns:
            values = [value for value in dataset_split[column_name] if isinstance(value, str) and value.strip()]
            if not values:
                continue
            prefix = config_name or "default"
            output_stem = _sanitize_output_stem(f"{prefix}_{column_name}")
            inputs.append(
                EvaluationInput(
                    source_id=f"{dataset_name}:{config_name or 'default'}:{column_name}:{split}",
                    label=f"{prefix}:{column_name}",
                    output_stem=output_stem,
                    language=config_name if config_name in SUPPORTED_LANGUAGES else None,
                    lines=tuple(values),
                )
            )
    return inputs


def discover_data_inputs(args: argparse.Namespace) -> Sequence[EvaluationInput]:
    if args.data_source == "files":
        return discover_file_inputs(args.data_dir)
    return discover_hf_inputs(args.hf_dataset, args.hf_split, args.lang, args.hf_text_columns)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if not args.tokenizer_paths:
        parser.error("--tokenizer_paths is required unless TOK_EVAL_TOKENIZER_PATHS is set.")
    if args.data_source == "files" and not os.path.isdir(args.data_dir):
        parser.error(f"--data_dir does not exist or is not a directory: {args.data_dir}")
    if args.data_source == "hf" and not args.hf_dataset:
        parser.error("--hf_dataset is required when --data_source hf is set.")
    if args.enable_nsl and not args.nsl_base_tokenizer:
        parser.error("--nsl-base-tokenizer is required when --enable-nsl is set.")
def run(args: argparse.Namespace) -> None:
    data_inputs = discover_data_inputs(args)
    if not data_inputs:
        if args.data_source == "files":
            raise FileNotFoundError(f"No .txt files found under {args.data_dir}")
        raise FileNotFoundError(f"No text samples discovered in dataset {args.hf_dataset} split {args.hf_split}")

    for tok_path in args.tokenizer_paths:
        evaluator = get_evaluator(
            tok_path,
            args.tokenizer_type,
            data_inputs,
            args.out_dir,
            args.hf_token,
            args.trust_remote_code,
        )
        evaluator.nsl = args.enable_nsl
        evaluator.nsl_base_tokenizer = args.nsl_base_tokenizer
        evaluator.lang = args.lang
        evaluator.evaluate()


def main(argv: Iterable[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    validate_args(args, parser)
    run(args)


if __name__ == "__main__":
    main()
