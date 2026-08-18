"""Turn model generations into DPO preference pairs.

The reference darija from the dataset is the `chosen` answer, the model's own
generation is the `rejected` one — on-policy bad samples for DPO.
"""
import json
import os
from collections.abc import Iterator

from sacrebleu import CHRF

from darija_translator.config import DataConfig, InferenceConfig, PreferenceConfig
from darija_translator.inference import build_translation_messages, translate

_chrf = CHRF()


def chrf_similarity(hypothesis: str, reference: str) -> float:
    return _chrf.sentence_score(hypothesis, [reference]).score


def select_rejected(candidates: list[str],
                    reference: str) -> tuple[str, float] | None:
    """Keep the candidate furthest from the reference — the clearest bad sample."""
    scored = [(c, chrf_similarity(c, reference)) for c in candidates
              if c.strip()]
    if not scored:
        return None
    return min(scored, key=lambda pair: pair[1])


def to_preference_record(english: str,
                         reference: str,
                         rejected: str,
                         similarity: float,
                         system_prompt: str,
                         metadata: dict | None = None) -> dict:
    return {
        "prompt": build_translation_messages(english, system_prompt),
        "chosen": [{
            "role": "assistant",
            "content": reference.strip()
        }],
        "rejected": [{
            "role": "assistant",
            "content": rejected.strip()
        }],
        "english": english.strip(),
        "darija": reference.strip(),
        "generated": rejected.strip(),
        "chrf": similarity,
        **(metadata or {}),
    }


def is_useful_pair(record: dict, config: PreferenceConfig) -> bool:
    """A generation that already matches the reference teaches DPO nothing."""
    return record["chrf"] <= config.max_chrf_similarity


def write_record(handle, record: dict) -> None:
    """One JSON object per line, unescaped so darija stays readable."""
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def last_row_index(path: str) -> int | None:
    """Where a previous interrupted run stopped, from the records it wrote."""
    if not os.path.exists(path):
        return None
    last = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last = line
    if last is None:
        return None
    return json.loads(last).get("row_index")


def build_preference_record(english: str,
                            reference: str,
                            candidates: list[str],
                            system_prompt: str,
                            metadata: dict | None = None) -> dict | None:
    if not english.strip() or not reference.strip():
        return None
    selected = select_rejected(candidates, reference)
    if selected is None:
        return None
    rejected, similarity = selected
    return to_preference_record(english, reference, rejected, similarity,
                                system_prompt, metadata)


def generate_preference_pairs(
    model,
    tokenizer,
    rows: list[dict],
    data_config: DataConfig,
    inference_config: InferenceConfig,
    preference_config: PreferenceConfig,
    keep_all: bool = False,
) -> Iterator[dict]:
    """rows carry "english", "darija" and any metadata to keep (e.g. row_index)."""
    sources = [row["english"] for row in rows]
    generations = translate(model, tokenizer, sources,
                            data_config.system_prompt, inference_config)
    for row, (english, candidates) in zip(rows, generations):
        metadata = {
            key: value
            for key, value in row.items() if key not in ("english", "darija")
        }
        record = build_preference_record(english, row["darija"], candidates,
                                         data_config.system_prompt, metadata)
        if record is None:
            continue
        if keep_all or is_useful_pair(record, preference_config):
            yield record
