"""Evaluator for tiktoken-compatible tokenizers."""

from __future__ import annotations

import os
import time
from collections import Counter
from datetime import datetime

from tqdm import tqdm

from tokenizer_evaluator.evaluators.base import BaseEvaluator
from tokenizer_evaluator.utils.evaluation_utils import exact_match, fuzzy_score
from tokenizer_evaluator.utils.tokenizer_utils import decode_text, load_tokenizer


class TikTokenEvaluator(BaseEvaluator):
    def evaluate(self):
        tokenizer = load_tokenizer(self.tokenizer_path, "tkt", hf_key=self.hf_token)
        nsl_base_tokenizer = self.load_nsl_base_tokenizer() if self.nsl else None

        total_tokens = 0
        total_time_taken = 0.0
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
            for line in tqdm(self.iter_input_lines(data_input), desc=data_input.label, leave=False):
                start_time = time.perf_counter()
                encoded = tokenizer.encode(line)
                end_time = time.perf_counter()

                num_tokens = len(encoded)
                if num_tokens == 0:
                    continue

                total_time_taken += end_time - start_time
                total_tokens += num_tokens
                token_counts.update(encoded)
                token_sequence_length.append(num_tokens)

                decoded = decode_text(tokenizer, encoded, "tkt")
                num_words = len(line.split())
                if nsl_base_tokenizer is not None:
                    base_count = self.base_token_count(nsl_base_tokenizer, line)
                    if base_count:
                        aggregated_token_ratio += num_tokens / base_count
                        total_seq += 1

                results.append(
                    {
                        "file": data_input.label,
                        "original": line,
                        "decoded": decoded,
                        "exact_match": exact_match(line, decoded),
                        "fuzzy_score": fuzzy_score(line, decoded),
                        "input_word_count": num_words,
                        "encoded_token_count": num_tokens,
                        "token_to_word_ratio": num_tokens / num_words if num_words else 0.0,
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
            }
            self.write_results(results, data_input, summary, suffix="_tkt_results")

            self.write_jsonl_record(
                "tkt_tokenizer_stats.jsonl",
                {
                    "total_time_taken": total_time_taken,
                    "total_tokens": total_tokens,
                    "tokens_per_second": total_tokens / total_time_taken if total_time_taken else 0.0,
                    "file_processed": data_input.output_stem,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )

        if first_processed_input is not None:
            self.write_reyni_stats(token_counts, token_sequence_length, first_processed_input)
