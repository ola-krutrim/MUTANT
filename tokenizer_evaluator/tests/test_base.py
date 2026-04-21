from pathlib import Path

from tokenizer_evaluator.evaluators.base import BaseEvaluator, EvaluationInput


def test_build_output_paths_creates_tokenizer_dir(tmp_path: Path):
    evaluator = BaseEvaluator("models/my-tokenizer", "hf", [], str(tmp_path))
    paths = evaluator.build_output_paths(
        EvaluationInput(source_id="eng.txt", label="eng.txt", output_stem="eng", path="eng.txt")
    )

    assert paths.tokenizer_dir.name.startswith("my-tokenizer-")
    assert paths.tokenizer_dir.parent == tmp_path
    assert paths.csv_path == paths.tokenizer_dir / "eng_results.csv"
    assert paths.json_path == paths.tokenizer_dir / "eng_results.json"


def test_ratio_from_means_handles_zero_denominator():
    rows = [{"encoded_token_count": 10, "input_word_count": 0}]
    assert BaseEvaluator.ratio_from_means(rows, "encoded_token_count", "input_word_count") == 0.0


def test_should_process_input_uses_explicit_language():
    evaluator = BaseEvaluator("models/my-tokenizer", "hf", [], "/tmp")
    evaluator.lang = "eng"
    data_input = EvaluationInput(source_id="x", label="demo", output_stem="demo", language="eng", lines=("hello",))

    assert evaluator.should_process_input(data_input) is True


def test_build_reyni_stats_handles_empty_counts():
    stats = BaseEvaluator.build_reyni_stats({}, [])

    assert stats["total_tokens"] == 0
    assert stats["reyni_entropy"] == 0.0
    assert stats["reyni_efficiency"] == 0.0
