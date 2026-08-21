from darija_translator.config import PreferenceConfig
from darija_translator.preference import (
    build_preference_record,
    chrf_similarity,
    is_useful_pair,
    select_rejected,
)

SYSTEM_PROMPT = "You are a professional English to Darija translator."


def test_chrf_similarity_is_high_for_identical_strings():
    assert chrf_similarity("salam khouya", "salam khouya") > 99


def test_select_rejected_keeps_the_candidate_furthest_from_the_reference():
    candidates = ["salam khouya", "completely different output"]

    rejected, similarity = select_rejected(candidates, "salam khouya")

    assert rejected == "completely different output"
    assert similarity < 50


def test_select_rejected_ignores_blank_candidates():
    rejected, _ = select_rejected(["", "   ", "chi haja"], "salam khouya")

    assert rejected == "chi haja"


def test_select_rejected_returns_none_when_nothing_was_generated():
    assert select_rejected(["", "  "], "salam khouya") is None


def test_build_preference_record_uses_the_conversational_dpo_format():
    record = build_preference_record("Hello", "salam", ["ahlan"],
                                     SYSTEM_PROMPT)

    assert record["prompt"] == [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": "Hello"
        },
    ]
    assert record["chosen"] == [{"role": "assistant", "content": "salam"}]
    assert record["rejected"] == [{"role": "assistant", "content": "ahlan"}]


def test_build_preference_record_keeps_the_flat_columns_too():
    record = build_preference_record("Hello", "salam", ["ahlan"],
                                     SYSTEM_PROMPT)

    assert record["english"] == "Hello"
    assert record["darija"] == "salam"
    assert record["generated"] == "ahlan"
    assert record["chrf"] < 100


def test_build_preference_record_carries_metadata():
    record = build_preference_record("Hello",
                                     "salam", ["ahlan"],
                                     SYSTEM_PROMPT,
                                     metadata={"row_index": 7})

    assert record["row_index"] == 7


def test_build_preference_record_skips_rows_without_a_reference():
    assert build_preference_record("Hello", "  ", ["ahlan"],
                                   SYSTEM_PROMPT) is None
    assert build_preference_record("  ", "salam", ["ahlan"],
                                   SYSTEM_PROMPT) is None


def test_build_preference_record_skips_rows_with_no_generation():
    assert build_preference_record("Hello", "salam", [""],
                                   SYSTEM_PROMPT) is None


def test_is_useful_pair_drops_generations_that_match_the_reference():
    config = PreferenceConfig()
    record = build_preference_record("Hello", "salam khouya",
                                     ["salam khouya"], SYSTEM_PROMPT)

    assert is_useful_pair(record, config) is False


def test_is_useful_pair_keeps_a_genuinely_bad_generation():
    config = PreferenceConfig()
    record = build_preference_record("Hello", "salam khouya",
                                     ["totally unrelated text"],
                                     SYSTEM_PROMPT)

    assert is_useful_pair(record, config) is True
