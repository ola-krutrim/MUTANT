"""Utilities for aggregating per-language summary files."""

from __future__ import annotations

import argparse
import json
import os


LANGUAGE_ORDER = [
    "as",
    "as_cleaned",
    "bn",
    "bn_cleaned",
    "brx",
    "code",
    "doi",
    "eng",
    "eng_cleaned",
    "gom",
    "gu",
    "gu_cleaned",
    "hi",
    "hi_cleaned",
    "kas",
    "kn",
    "kn_cleaned",
    "mai",
    "ml",
    "ml_cleaned",
    "mni",
    "mr",
    "mr_cleaned",
    "nep",
    "or",
    "or_cleaned",
    "pa",
    "pa_cleaned",
    "san",
    "sat",
    "snd",
    "ta",
    "ta_cleaned",
    "te",
    "te_cleaned",
    "urd",
]


def generate_language_scores(directory):
    scores = {}
    for filename in os.listdir(directory):
        if not filename.endswith("_results.json"):
            continue
        lang = filename.split("_results.json")[0]
        with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
            data = json.load(file)
            scores[lang] = data.get("avg_token_to_word_ratio", 0)

    with open(os.path.join(directory, "language_scores.txt"), "w", encoding="utf-8") as output_file:
        for lang in LANGUAGE_ORDER:
            output_file.write(f"{scores.get(lang, f'{lang}: No data')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    args = parser.parse_args()
    generate_language_scores(args.file_path)
