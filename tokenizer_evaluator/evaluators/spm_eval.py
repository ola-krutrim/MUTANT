"""Evaluator for SentencePiece tokenizers."""

from __future__ import annotations

import os
from collections import Counter

from tokenizer_evaluator.evaluators.base import BaseEvaluator
from tokenizer_evaluator.utils.evaluation_utils import exact_match, fuzzy_score, lcs
from tokenizer_evaluator.utils.tokenizer_utils import decode_text, load_tokenizer


class SPMEvaluator(BaseEvaluator):
    def evaluate(self):
        tokenizer = load_tokenizer(self.tokenizer_path, "spm")
        nsl_base_tokenizer = self.load_nsl_base_tokenizer() if self.nsl else None
        token_counts = Counter()
        token_sequence_length = []
        first_processed_input = None

        for data_input in self.data_inputs:
            if not self.should_process_input(data_input):
                continue
            if first_processed_input is None:
                first_processed_input = data_input

            results = []
            aggregated_token_ratio = 0.0
            total_seq = 0
            for line in self.iter_input_lines(data_input):
                encoded = tokenizer.EncodeAsPieces(line)
                decoded = decode_text(tokenizer, encoded, "spm")
                num_tokens = len(encoded)
                num_words = len(line.split())
                token_counts.update(encoded)
                token_sequence_length.append(num_tokens)
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
                        "lcs_distance": lcs(line, decoded),
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
                "avg_lcs_distance": self.mean_metric(results, "lcs_distance"),
                "avg_token_to_word_ratio": self.ratio_from_means(results, "encoded_token_count", "input_word_count"),
                "nsl_score": aggregated_token_ratio / total_seq if total_seq else 0.0,
            }
            self.write_results(results, data_input, summary)

        if first_processed_input is not None:
            self.write_reyni_stats(token_counts, token_sequence_length, first_processed_input)
