"""Evaluator for PreTrainedTokenizerFast tokenizer.json artifacts."""

from __future__ import annotations

import json
import os
from collections import Counter

from tqdm import tqdm

from tokenizer_evaluator.evaluators.base import BaseEvaluator
from tokenizer_evaluator.utils.evaluation_utils import exact_match, fuzzy_score, get_byte_fallback
from tokenizer_evaluator.utils.tokenizer_utils import decode_text, load_tokenizer


class PTFEvaluator(BaseEvaluator):
    def evaluate(self):
        tokenizer = load_tokenizer(self.tokenizer_path, "ptf")
        nsl_base_tokenizer = self.load_nsl_base_tokenizer() if self.nsl else None

        token_counts = Counter()
        token_sequence_length = []
        first_processed_input = None

        for data_input in tqdm(self.data_inputs, desc="inputs"):
            if not self.should_process_input(data_input):
                continue
            if first_processed_input is None:
                first_processed_input = data_input

            results = []
            aggregated_token_ratio = 0.0
            total_seq = 0
            failed_samples = 0

            for line in tqdm(self.iter_input_lines(data_input), desc=data_input.label, leave=False):
                try:
                    encoded = tokenizer(line, add_special_tokens=False).input_ids
                except Exception:
                    failed_samples += 1
                    continue

                num_tokens = len(encoded)
                if num_tokens == 0:
                    continue

                token_counts.update(encoded)
                token_sequence_length.append(num_tokens)

                if self.nsl and nsl_base_tokenizer is not None:
                    base_count = self.base_token_count(nsl_base_tokenizer, line)
                    if base_count:
                        aggregated_token_ratio += num_tokens / base_count
                        total_seq += 1

                decoded = decode_text(tokenizer, encoded, "ptf")
                num_words = len(line.split())
                decoded_tokens = [tokenizer.convert_ids_to_tokens(token_id) for token_id in encoded]
                byte_fallback_count = get_byte_fallback(decoded_tokens, line)

                results.append(
                    {
                        "file": data_input.label,
                        "original": line,
                        "decoded": decoded,
                        "exact_match": exact_match(line, decoded),
                        "fuzzy_score": fuzzy_score(line, decoded),
                        "encoded_token_count": num_tokens,
                        "input_word_count": num_words,
                        "token_to_word_ratio": num_tokens / num_words if num_words else 0.0,
                        "byte_fallback_count": byte_fallback_count,
                    }
                )

            if not results:
                continue

            summary = {
                "file": data_input.output_stem,
                "avg_exact_match": self.mean_metric(results, "exact_match"),
                "avg_fuzzy_score": self.mean_metric(results, "fuzzy_score"),
                "avg_token_to_word_ratio": self.ratio_from_means(results, "encoded_token_count", "input_word_count"),
                "nsl_score": aggregated_token_ratio / total_seq if total_seq else 0.0,
                "byte_fallback_ratio": self.ratio_from_means(results, "byte_fallback_count", "input_word_count"),
                "byte_fallback_percentage": self.ratio_from_means(results, "byte_fallback_count", "input_word_count"),
                "failed_samples": failed_samples,
            }
            self.write_results(results, data_input, summary)

        if first_processed_input is None:
            return

        self.write_reyni_stats(token_counts, token_sequence_length, first_processed_input)
