import argparse
from pathlib import Path

import pytest

from tokenizer_evaluator.main import (
    build_parser,
    discover_file_inputs,
    discover_hf_inputs,
    validate_args,
)


def test_discover_file_inputs_recurses(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    (nested / "b.txt").write_text("world\n", encoding="utf-8")

    files = discover_file_inputs(str(tmp_path))

    assert [entry.path for entry in files] == [str(tmp_path / "a.txt"), str(nested / "b.txt")]


def test_validate_args_requires_nsl_base_tokenizer(tmp_path: Path):
    parser = build_parser()
    args = argparse.Namespace(
        tokenizer_paths=["tok"],
        data_dir=str(tmp_path),
        data_source="files",
        out_dir=str(tmp_path / "out"),
        tokenizer_type="ptf",
        hf_token=None,
        lang=None,
        hf_dataset=None,
        hf_split="test",
        hf_text_columns=None,
        enable_nsl=True,
        nsl_base_tokenizer=None,
        trust_remote_code=False,
    )

    with pytest.raises(SystemExit):
        validate_args(args, parser)


def test_validate_args_allows_nsl_for_non_ptf(tmp_path: Path):
    parser = build_parser()
    args = argparse.Namespace(
        tokenizer_paths=["tok"],
        data_dir=str(tmp_path),
        data_source="files",
        out_dir=str(tmp_path / "out"),
        tokenizer_type="hf",
        hf_token=None,
        lang=None,
        hf_dataset=None,
        hf_split="test",
        hf_text_columns=None,
        enable_nsl=True,
        nsl_base_tokenizer="base",
        trust_remote_code=False,
    )

    validate_args(args, parser)


def test_discover_hf_inputs_uses_lang_configs(monkeypatch):
    class FakeValue:
        def __init__(self, dtype):
            self.dtype = dtype

    class FakeSplit:
        features = {"text": FakeValue("string"), "meta": FakeValue("int64")}

        def __getitem__(self, key):
            if key == "text":
                return ["hello", "", "world"]
            if key == "meta":
                return [1, 2, 3]
            raise KeyError(key)

    class FakeDatasetsModule:
        @staticmethod
        def get_dataset_config_names(_dataset_name):
            return ["eng", "hi", "extra"]

        @staticmethod
        def load_dataset(_dataset_name, name=None, split=None):
            assert split == "test"
            assert name in {"eng", "hi"}
            return FakeSplit()

    monkeypatch.setattr("importlib.import_module", lambda _name: FakeDatasetsModule)

    inputs = discover_hf_inputs("demo/dataset", "test", None, None)

    assert [entry.language for entry in inputs] == ["eng", "hi"]
    assert [entry.label for entry in inputs] == ["eng:text", "hi:text"]
