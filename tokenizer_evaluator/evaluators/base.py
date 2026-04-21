"""Shared evaluator utilities."""

from __future__ import annotations

import json
import os
import csv
import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass
class EvaluationPaths:
    tokenizer_dir: Path
    csv_path: Path
    json_path: Path


@dataclass
class EvaluationInput:
    source_id: str
    label: str
    output_stem: str
    language: str | None = None
    path: str | None = None
    lines: tuple[str, ...] = ()


class BaseEvaluator:
    def __init__(
        self,
        tokenizer_path,
        tokenizer_type,
        data_inputs,
        out_dir,
        hf_token=None,
        trust_remote_code=False,
    ):
        self.tokenizer_path = tokenizer_path
        self.tokenizer_type = tokenizer_type
        self.data_inputs = list(data_inputs)
        self.out_dir = out_dir
        self.hf_token = hf_token
        self.trust_remote_code = trust_remote_code
        self.lang = None
        self.nsl = False
        self.nsl_base_tokenizer = None
        self._nsl_base_tokenizer_instance = None

    @property
    def tokenizer_slug(self) -> str:
        normalized_path = self.tokenizer_path.rstrip("/").rstrip(os.sep)
        readable = os.path.basename(normalized_path) or normalized_path.replace("/", "_")
        safe_readable = re.sub(r"[^A-Za-z0-9._-]+", "_", readable).strip("._") or "tokenizer"
        suffix = hashlib.sha1(self.tokenizer_path.encode("utf-8")).hexdigest()[:8]
        return f"{safe_readable}-{suffix}"

    def should_process_input(self, data_input: EvaluationInput) -> bool:
        if self.lang is None:
            return True
        if data_input.language is not None:
            return self.lang.lower() == data_input.language.lower()
        normalized = Path(data_input.path or data_input.label)
        candidate_parts = [normalized.stem, *normalized.parts, data_input.label]
        tokens = set()
        for part in candidate_parts:
            tokens.update(token for token in re.split(r"[^A-Za-z0-9]+", part.lower()) if token)
        return self.lang.lower() in tokens

    def iter_corpus_lines(self) -> Iterator[Tuple[EvaluationInput, str]]:
        for data_input in self.data_inputs:
            if not self.should_process_input(data_input):
                continue
            for line in self.iter_input_lines(data_input):
                yield data_input, line

    def iter_input_lines(self, data_input: EvaluationInput) -> Iterator[str]:
        if data_input.path is not None:
            with open(data_input.path, "r", encoding="utf-8") as handle:
                for raw_line in handle:
                    line = raw_line.strip()
                    if line:
                        yield line
            return
        for line in data_input.lines:
            stripped = line.strip()
            if stripped:
                yield stripped

    def build_output_paths(self, data_input: EvaluationInput, suffix: str = "_results") -> EvaluationPaths:
        tokenizer_dir = Path(self.out_dir) / self.tokenizer_slug
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        return EvaluationPaths(
            tokenizer_dir=tokenizer_dir,
            csv_path=tokenizer_dir / f"{data_input.output_stem}{suffix}.csv",
            json_path=tokenizer_dir / f"{data_input.output_stem}{suffix}.json",
        )

    def write_results(
        self,
        results: Iterable[Dict],
        data_input: EvaluationInput,
        summary: Dict,
        suffix: str = "_results",
    ) -> None:
        result_list = list(results)
        paths = self.build_output_paths(data_input, suffix)
        with open(paths.csv_path, "w", newline="", encoding="utf-8") as handle:
            if result_list:
                writer = csv.DictWriter(handle, fieldnames=list(result_list[0].keys()))
                writer.writeheader()
                writer.writerows(result_list)
            else:
                handle.write("")
        with open(paths.json_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    def write_jsonl_record(self, filename: str, payload: Dict) -> None:
        tokenizer_dir = Path(self.out_dir) / self.tokenizer_slug
        tokenizer_dir.mkdir(parents=True, exist_ok=True)
        with open(tokenizer_dir / filename, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def load_nsl_base_tokenizer(self):
        if not self.nsl:
            return None
        if self.nsl_base_tokenizer is None:
            raise ValueError("--nsl-base-tokenizer is required when NSL is enabled.")
        if self._nsl_base_tokenizer_instance is None:
            from tokenizer_evaluator.utils.tokenizer_utils import load_tokenizer

            self._nsl_base_tokenizer_instance = load_tokenizer(
                self.nsl_base_tokenizer,
                "hf",
                hf_key=self.hf_token,
                trust_remote_code=self.trust_remote_code,
            )
        return self._nsl_base_tokenizer_instance

    @staticmethod
    def base_token_count(base_tokenizer, line: str) -> int:
        encoded = base_tokenizer(line, add_special_tokens=False).input_ids
        return len(encoded)

    @staticmethod
    def mean_metric(results: Iterable[Dict], key: str) -> float:
        values = [row[key] for row in results if key in row]
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def ratio_from_means(results: Iterable[Dict], numerator_key: str, denominator_key: str) -> float:
        rows = list(results)
        if not rows:
            return 0.0
        numerator = sum(row.get(numerator_key, 0) for row in rows)
        denominator = sum(row.get(denominator_key, 0) for row in rows)
        return _safe_divide(numerator, denominator)

    @staticmethod
    def build_reyni_stats(token_counts: Counter, token_sequence_length: Iterable[int]) -> Dict:
        total_tokens = sum(token_counts.values())
        if total_tokens == 0:
            return {
                "total_tokens": 0,
                "unique_tokens": 0,
                "reyni_entropy": 0.0,
                "avg_sequence_length": 0.0,
                "reyni_efficiency": 0.0,
            }

        alpha = 2
        token_probabilities = {token_id: count / total_tokens for token_id, count in token_counts.items()}
        sum_p = sum(probability**alpha for probability in token_probabilities.values())
        reyni_entropy = (1 / (1 - alpha)) * math.log(sum_p) if sum_p else 0.0
        sequence_lengths = list(token_sequence_length)
        avg_sequence_length = sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0.0
        reyni_efficiency = reyni_entropy / avg_sequence_length if avg_sequence_length else 0.0
        return {
            "total_tokens": total_tokens,
            "unique_tokens": len(token_counts),
            "reyni_entropy": reyni_entropy,
            "avg_sequence_length": avg_sequence_length,
            "reyni_efficiency": reyni_efficiency,
        }

    def write_reyni_stats(self, token_counts: Counter, token_sequence_length: Iterable[int], sample_input: EvaluationInput) -> None:
        tokenizer_dir = self.build_output_paths(sample_input).tokenizer_dir
        with open(tokenizer_dir / "reyni_logging_info.json", "w", encoding="utf-8") as handle:
            json.dump(self.build_reyni_stats(token_counts, token_sequence_length), handle, indent=2)

    def evaluate(self):
        raise NotImplementedError
