from tokenizer_evaluator.utils.evaluation_utils import exact_match, fuzzy_score, get_byte_fallback, lcs


def test_exact_match_is_binary():
    assert exact_match("hello", "hello") == 1.0
    assert exact_match("hello", "hello ") == 1.0
    assert exact_match("hello", "world") == 0.0


def test_lcs_handles_empty_original():
    assert lcs("", "anything") == 0.0


def test_byte_fallback_groups_adjacent_bytes():
    tokens = ["A", "<0xE2>", "<0x82>", "<0xAC>", "B", "<0x61>"]
    assert get_byte_fallback(tokens) == 2


def test_fuzzy_score_is_numeric():
    assert fuzzy_score("tokenizer", "tokenizer") == 100.0
