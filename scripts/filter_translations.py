"""Filter generated translations, then optionally push them to the Hub.

    python scripts/filter_translations.py data/translations.jsonl \
        --out data/translations.clean.jsonl \
        --push-to-hub atlasia/darija-dpo-negatives

Drops rows that are bad for reasons unrelated to translation quality: empty
output, the English copied back, Latin script, repetition loops, and runaway
length. Everything else is kept, wrong translations included, since those are
the point.
"""
import argparse
import json
import re
from collections import Counter

ARABIC = re.compile("[\u0600-\u06FF\u0750-\u077F]")


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
    generated = str(record.get("generated", "")).strip()
    english = str(record.get("english", "")).strip()
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


def read_records(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_records(records: list[dict], args) -> tuple[list[dict], Counter]:
    kept, dropped, seen = [], Counter(), set()
    for record in records:
        key = " ".join(str(record.get("english", "")).split()).casefold()
        if key in seen:
            dropped["duplicate source"] += 1
            continue
        seen.add(key)
        reason = rejection_reason(record, args)
        if reason:
            dropped[reason] += 1
            continue
        kept.append(record)
    return kept, dropped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="jsonl written by darija-translator translate")
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
    parser.add_argument("--public",
                        action="store_true",
                        help="push publicly instead of privately")
    args = parser.parse_args()

    records = read_records(args.path)
    kept, dropped = filter_records(records, args)

    print(f"{len(records)} rows in, {len(kept)} kept")
    for reason, count in dropped.most_common():
        print(f"  dropped {count:>6}  {reason}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            for record in kept:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"wrote {args.out}")

    if args.push_to_hub:
        from datasets import Dataset
        dataset = Dataset.from_list(kept)
        dataset.push_to_hub(args.push_to_hub, private=not args.public)
        visibility = "public" if args.public else "private"
        print(f"pushed {len(dataset)} rows to {args.push_to_hub} ({visibility})")


if __name__ == "__main__":
    main()
