"""Filter generated translations and push them as a dataset.

    python scripts/filter_translations.py data/translations.jsonl \
        --corpus data/corpus.jsonl \
        --out data/translations.clean.jsonl \
        --push-to-hub atlasia/darija-dpo-negatives

Drops rows that are bad for reasons unrelated to translation quality: empty
output, the English copied back, Latin script, repetition loops, runaway length
and duplicate sources. Wrong translations are kept, since those are the point.

The output keeps prompt, english, generated and source, where source comes
from the corpus file the sentences were drawn from. Which adapter and decoding
produced the text belongs in the dataset card.
"""
import argparse
import json
import re
from collections import Counter

ARABIC = re.compile("[\u0600-\u06FF\u0750-\u077F]")

COLUMNS = ("prompt", "english", "generated", "source")

SYSTEM_PROMPT = "You are a professional English to Darija translator."


def normalise(text: str) -> str:
    return " ".join(str(text).split())


def arabic_fraction(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(ARABIC.match(char)) for char in letters) / len(letters)


def repeats_itself(text: str, window: int = 3, limit: int = 3) -> bool:
    """A loop the model could not break out of, not a translation."""
    words = text.split()
    if len(words) < window * limit:
        return False
    grams = Counter(" ".join(words[i:i + window])
                    for i in range(len(words) - window + 1))
    return max(grams.values()) >= limit


def rejection_reason(record: dict, args) -> str | None:
    generated = normalise(record.get("generated", ""))
    english = normalise(record.get("english", ""))
    if not generated:
        return "empty"
    if generated.casefold() == english.casefold():
        return "copied the English"
    if arabic_fraction(generated) < args.min_arabic:
        return "not Arabic script"
    if repeats_itself(generated):
        return "repetition loop"
    if len(generated.split()) > args.max_words:
        return "runaway length"
    return None


def prompt_for(record: dict, english: str) -> list[dict]:
    """Kept as generated; rebuilt only if an older file lacks the column."""
    if record.get("prompt"):
        return record["prompt"]
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": english
        },
    ]


def read_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_sources(path: str | None) -> dict:
    """english -> which corpus it came from, since translate drops the column."""
    if not path:
        return {}
    return {
        normalise(record["english"]).casefold(): record.get("source", "unknown")
        for record in read_records(path) if record.get("english")
    }


def filter_records(records: list[dict], sources: dict,
                   args) -> tuple[list[dict], Counter]:
    kept, dropped, seen = [], Counter(), set()
    for record in records:
        english = normalise(record.get("english", ""))
        key = english.casefold()
        if key in seen:
            dropped["duplicate source"] += 1
            continue
        seen.add(key)
        reason = rejection_reason(record, args)
        if reason:
            dropped[reason] += 1
            continue
        kept.append({
            "prompt": prompt_for(record, english),
            "english": english,
            "generated": normalise(record["generated"]),
            "source": sources.get(key, "unknown"),
        })
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path",
                        help="jsonl written by darija-translator translate")
    parser.add_argument("--corpus",
                        default=None,
                        help="corpus jsonl to recover the source column from")
    parser.add_argument("--out", default=None, help="write the kept rows here")
    parser.add_argument("--min-arabic",
                        type=float,
                        default=0.5,
                        help="minimum fraction of letters in Arabic script")
    parser.add_argument("--max-words",
                        type=int,
                        default=120,
                        help="drop generations longer than this")
    parser.add_argument("--push-to-hub", default=None, metavar="REPO_ID")
    parser.add_argument("--private",
                        action="store_true",
                        help="create the repo private (ignored if it exists)")
    args = parser.parse_args()

    records = read_records(args.path)
    sources = load_sources(args.corpus)
    kept, dropped = filter_records(records, sources, args)

    print(f"{len(records)} rows in, {len(kept)} kept")
    for reason, count in dropped.most_common():
        print(f"  dropped {count:>6}  {reason}")

    unknown = sum(1 for record in kept if record["source"] == "unknown")
    if unknown:
        print(f"  warning: {unknown} rows have no source; "
              f"pass --corpus to recover it")
    else:
        for source, count in Counter(r["source"] for r in kept).most_common():
            print(f"  {count:>6}  {source}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")

    if args.push_to_hub:
        from datasets import Dataset
        dataset = Dataset.from_list(kept)
        if args.private:
            dataset.push_to_hub(args.push_to_hub, private=True)
        else:
            dataset.push_to_hub(args.push_to_hub)
        print(f"pushed {len(dataset)} rows {list(COLUMNS)} "
              f"to {args.push_to_hub}")


if __name__ == "__main__":
    main()
