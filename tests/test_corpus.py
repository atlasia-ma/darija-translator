import pytest

from darija_translator.config import CorpusConfig
from darija_translator.corpus import (
    cap_repeated_openings,
    contains_phrase,
    decontaminate,
    dedupe,
    detokenise,
    length_bucket,
    normalise,
    parse_source,
    phrase_pattern,
    stratified_sample,
    within_length,
    word_count,
)


def test_normalise_collapses_whitespace():
    assert normalise("  Hello   there\n world ") == "Hello there world"


def test_word_count_counts_words_not_characters():
    assert word_count("  Hello   there world ") == 3


def test_within_length_uses_the_configured_window():
    config = CorpusConfig(min_words=3, max_words=5)

    assert within_length("one two three", config) is True
    assert within_length("one two", config) is False
    assert within_length("one two three four five six", config) is False


def test_dedupe_drops_repeats_ignoring_case_and_spacing():
    texts = ["Whatever floats your boat.", "whatever  floats your boat.",
             "Something else."]

    assert dedupe(texts) == ["Whatever floats your boat.", "Something else."]


def test_dedupe_keeps_the_first_spelling_seen():
    assert dedupe(["Hello there", "HELLO THERE"]) == ["Hello there"]


def test_cap_repeated_openings_limits_one_sentence_pattern():
    config = CorpusConfig(max_repeated_openings=2)
    texts = [
        "Sami told Layla a story.",
        "Sami told Layla the truth.",
        "Sami told Layla nothing.",
        "The weather is nice today.",
    ]

    kept = cap_repeated_openings(texts, config)

    assert kept == [
        "Sami told Layla a story.",
        "Sami told Layla the truth.",
        "The weather is nice today.",
    ]


def test_length_bucket_places_sentences_in_bands():
    buckets = (5, 10, 20)

    assert length_bucket("one two three", buckets) == 0
    assert length_bucket(" ".join(["w"] * 8), buckets) == 1
    assert length_bucket(" ".join(["w"] * 15), buckets) == 2
    assert length_bucket(" ".join(["w"] * 30), buckets) == 3


def test_stratified_sample_spreads_across_length_bands():
    config = CorpusConfig(length_buckets=(5, 10, 20))
    short = [f"short sentence number {i}" for i in range(10)]
    long = [" ".join(["word"] * 15) + f" {i}" for i in range(10)]

    sample = stratified_sample(short + long, 4, config)

    buckets = {length_bucket(text, config.length_buckets) for text in sample}
    assert len(sample) == 4
    assert buckets == {0, 2}


def test_stratified_sample_is_deterministic_for_a_seed():
    config = CorpusConfig()
    texts = [f"sentence number {i} here" for i in range(50)]

    assert stratified_sample(texts, 10, config) == stratified_sample(
        texts, 10, config)


def test_stratified_sample_returns_everything_when_asked_for_too_much():
    config = CorpusConfig()
    texts = ["one two three", "four five six"]

    assert sorted(stratified_sample(texts, 100, config)) == sorted(texts)


def test_decontaminate_drops_sentences_the_model_trained_on():
    seen = {"hello there"}

    assert decontaminate(["Hello there", "Something new"],
                         seen) == ["Something new"]


def test_parse_source_reads_a_bare_dataset_id():
    spec = parse_source("Gooogr/pie_idioms")

    assert spec.dataset == "Gooogr/pie_idioms"
    assert spec.config is None
    assert spec.column == "english"
    assert spec.count is None
    assert spec.filter_field is None


def test_parse_source_reads_config_column_and_count():
    spec = parse_source(
        "sentence-transformers/parallel-sentences-tatoeba:en-de:english:20000")

    assert spec.config == "en-de"
    assert spec.column == "english"
    assert spec.count == 20000


def test_parse_source_skips_empty_segments():
    spec = parse_source("Gooogr/pie_idioms::tokens:2000")

    assert spec.config is None
    assert spec.column == "tokens"
    assert spec.count == 2000


def test_parse_source_reads_a_row_filter():
    spec = parse_source("Gooogr/pie_idioms::tokens:2000:is_pie=true")

    assert spec.filter_field == "is_pie"
    assert spec.filter_value == "true"


@pytest.mark.parametrize("spec", [
    "",
    "   ",
    "dataset:::not_a_number",
    "dataset::tokens:10:missing_equals",
])
def test_parse_source_rejects_malformed_specs(spec):
    with pytest.raises(ValueError):
        parse_source(spec)


def test_detokenise_reattaches_punctuation():
    assert detokenise(["We", "should", "call", "it", "a", "day",
                       "."]) == "We should call it a day."


def test_detokenise_rejoins_contractions():
    assert detokenise(["It", "'s", "raining", "cats", "and", "dogs",
                       "."]) == "It's raining cats and dogs."
    assert detokenise(["They", "did", "n't", "come",
                       "."]) == "They didn't come."


def test_detokenise_handles_brackets_and_quotes():
    assert detokenise(["He", "said", "(", "quietly", ")",
                       "yes"]) == "He said (quietly) yes"


def test_detokenise_accepts_plain_strings_unchanged():
    assert detokenise("Already a sentence.") == "Already a sentence."


def test_phrase_pattern_matches_on_word_boundaries():
    pattern = phrase_pattern(["call it a day", "over the moon"])

    assert contains_phrase("We should call it a day.", pattern) is True
    assert contains_phrase("They were over the moon!", pattern) is True
    assert contains_phrase("Nothing figurative here.", pattern) is False


def test_phrase_pattern_ignores_case_and_spacing():
    pattern = phrase_pattern(["  Call It A Day "])

    assert contains_phrase("we should CALL it a day", pattern) is True


def test_phrase_pattern_does_not_match_inside_words():
    pattern = phrase_pattern(["a day"])

    assert contains_phrase("holiday", pattern) is False


def test_phrase_pattern_skips_phrases_too_short_to_be_meaningful():
    pattern = phrase_pattern(["a", "of", "over the moon"])

    assert contains_phrase("a bird flew over the trees", pattern) is False
    assert contains_phrase("they were over the moon", pattern) is True


def test_phrase_pattern_is_none_without_usable_phrases():
    assert phrase_pattern(["", "  ", "a"]) is None


def test_contains_phrase_is_true_when_no_filter_is_configured():
    assert contains_phrase("anything at all", None) is True
