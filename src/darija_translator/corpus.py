"""Assembling one English corpus out of several sources.

Diversity is the point: the model is weakest on registers its training set
was thin on, so length bands are sampled evenly and one sentence pattern is
not allowed to dominate.
"""
import re
from collections.abc import Iterable
from dataclasses import dataclass
from random import Random

from darija_translator.config import CorpusConfig


@dataclass(frozen=True)
class SourceSpec:
    """dataset[:config[:column[:count[:field=value]]]]"""
    dataset: str
    config: str | None = None
    column: str = "english"
    count: int | None = None
    filter_field: str | None = None
    filter_value: str | None = None


def normalise(text: str) -> str:
    return " ".join(str(text).split())


def word_count(text: str) -> int:
    return len(normalise(text).split())


def detokenise(value) -> str:
    """Idiom corpora ship token lists; "day ." and "It 's" are not English."""
    if not isinstance(value, list):
        return normalise(value)
    text = " ".join(str(token) for token in value)
    text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
    text = re.sub(r"([([{])\s+", r"\1", text)
    text = re.sub(r"\s+('(?:s|re|ve|ll|d|m|t)\b|n't\b)", r"\1", text)
    return normalise(text)


def phrase_pattern(phrases: Iterable[str]):
    """Literal alternation over the phrases worth selecting sentences for."""
    cleaned = {
        normalise(phrase).casefold()
        for phrase in phrases if len(normalise(phrase)) > 2
    }
    if not cleaned:
        return None
    alternation = "|".join(
        re.escape(phrase) for phrase in sorted(cleaned, key=len, reverse=True))
    return re.compile(r"\b(?:" + alternation + r")\b")


def contains_phrase(text: str, pattern) -> bool:
    """No pattern means no filter, so everything passes."""
    if pattern is None:
        return True
    return bool(pattern.search(normalise(text).casefold()))


def within_length(text: str, config: CorpusConfig) -> bool:
    return config.min_words <= word_count(text) <= config.max_words


def dedupe(texts: Iterable[str]) -> list[str]:
    seen, kept = set(), []
    for text in texts:
        normalised = normalise(text)
        key = normalised.casefold()
        if key and key not in seen:
            seen.add(key)
            kept.append(normalised)
    return kept


def opening(text: str, words: int = 3) -> str:
    return " ".join(normalise(text).casefold().split()[:words])


def cap_repeated_openings(texts: Iterable[str],
                          config: CorpusConfig) -> list[str]:
    """One prolific contributor's sentence frame should not become the corpus."""
    counts, kept = {}, []
    for text in texts:
        key = opening(text)
        counts[key] = counts.get(key, 0) + 1
        if counts[key] <= config.max_repeated_openings:
            kept.append(text)
    return kept


def length_bucket(text: str, buckets: tuple) -> int:
    words = word_count(text)
    for index, upper in enumerate(buckets):
        if words <= upper:
            return index
    return len(buckets)


def stratified_sample(texts: Iterable[str], total: int,
                      config: CorpusConfig) -> list[str]:
    """Round-robin across length bands, so no band drowns out the others."""
    bands: dict[int, list[str]] = {}
    for text in texts:
        bands.setdefault(length_bucket(text, config.length_buckets),
                         []).append(text)

    shuffler = Random(config.seed)
    for band in bands.values():
        shuffler.shuffle(band)

    sample = []
    while len(sample) < total and any(bands.values()):
        for index in sorted(bands):
            if bands[index] and len(sample) < total:
                sample.append(bands[index].pop())
    return sample


def decontaminate(texts: Iterable[str], seen: set[str]) -> list[str]:
    """Sentences the model trained on cannot fail, so they teach nothing."""
    return [
        text for text in texts if normalise(text).casefold() not in seen
    ]


def to_seen_set(texts: Iterable[str]) -> set[str]:
    return {normalise(text).casefold() for text in texts}


def parse_source(spec: str) -> SourceSpec:
    parts = [part.strip() for part in str(spec).split(":")]
    if not parts[0]:
        raise ValueError(f"no dataset in source spec {spec!r}")

    dataset = parts[0]
    config = parts[1] if len(parts) > 1 and parts[1] else None
    column = parts[2] if len(parts) > 2 and parts[2] else "english"

    count = None
    if len(parts) > 3 and parts[3]:
        if not parts[3].isdigit():
            raise ValueError(f"count must be a number in source spec {spec!r}")
        count = int(parts[3])

    field = value = None
    if len(parts) > 4 and parts[4]:
        if "=" not in parts[4]:
            raise ValueError(
                f"filter must look like field=value in source spec {spec!r}")
        field, value = parts[4].split("=", 1)

    return SourceSpec(dataset, config, column, count, field, value)
