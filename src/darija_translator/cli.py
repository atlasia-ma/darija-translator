import argparse
import os
from dataclasses import replace

from datasets import Dataset, load_dataset

from darija_translator.config import (
    CorpusConfig,
    DataConfig,
    InferenceConfig,
    ModelConfig,
    PreferenceConfig,
    TrainConfig,
)
from darija_translator.corpus import (
    cap_repeated_openings,
    decontaminate,
    dedupe,
    detokenise,
    length_bucket,
    parse_source,
    stratified_sample,
    to_seen_set,
    within_length,
)
from darija_translator.data import split_dataset
from darija_translator.evaluate import compute_translation_metrics
from darija_translator.inference import load_for_inference, to_generation_record, translate
from darija_translator.jsonl import (
    last_row_index,
    open_for_records,
    write_record,
)
from darija_translator.preference import generate_preference_pairs
from darija_translator.model import attach_lora, load_model_and_tokenizer
from darija_translator.train import build_trainer, prepare_data, save_model
from dotenv import load_dotenv

load_dotenv()


def run_train(args):
    data_config, model_config, train_config = DataConfig(), ModelConfig(
    ), TrainConfig()
    model, tokenizer = load_model_and_tokenizer(model_config)
    model = attach_lora(model, model_config)
    train_dataset, eval_dataset = prepare_data(args.dataset, data_config,
                                               tokenizer)
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset,
                            train_config)
    trainer.train()
    save_model(model, tokenizer, train_config)


def run_evaluate(args):
    data_config = DataConfig()
    inference_config = replace(InferenceConfig(),
                               adapter_model_id=args.adapter,
                               adapter_subfolder=args.subfolder,
                               batch_size=args.batch_size,
                               num_generations=1)
    model, tokenizer = load_for_inference(inference_config)
    _, eval_dataset = prepare_data(args.dataset,
                                   data_config,
                                   tokenizer,
                                   remove_columns=False)
    if args.limit:
        eval_dataset = eval_dataset.select(
            range(min(args.limit, len(eval_dataset))))
    predictions = [
        candidates[0]
        for _, candidates in translate(model, tokenizer,
                                       eval_dataset["english"],
                                       data_config.system_prompt,
                                       inference_config)
    ]
    metrics = compute_translation_metrics(predictions, eval_dataset["darija"])
    print(f"{len(predictions)} sentences: {metrics}")


def select_source_rows(dataset_name: str, data_config: DataConfig,
                       split: str, start: int, limit: int | None):
    """Prompts to translate, taken from the same split the SFT run used."""
    dataset = load_dataset(dataset_name, split="train")
    train_dataset, eval_dataset = split_dataset(dataset, data_config)
    pool = {"train": train_dataset, "eval": eval_dataset, "all": dataset}[split]
    end = len(pool) if limit is None else min(start + limit, len(pool))
    return pool.select(range(start, max(start, end)))


def run_generate_dpo(args):
    data_config = DataConfig()
    inference_config = replace(
        InferenceConfig(),
        adapter_model_id=args.adapter,
        adapter_subfolder=args.subfolder,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=not args.greedy,
        num_generations=args.num_generations,
    )
    preference_config = replace(PreferenceConfig(),
                                max_chrf_similarity=args.max_chrf,
                                output_path=args.out)

    start = args.start
    if args.resume:
        stopped_at = last_row_index(preference_config.output_path)
        if stopped_at is not None:
            start = stopped_at + 1
            print(f"resuming after row {stopped_at}")

    dataset = select_source_rows(args.dataset, data_config, args.split, start,
                                 args.limit)
    rows = [{
        "english": english,
        "darija": darija,
        "row_index": start + offset,
    } for offset, (english, darija) in enumerate(
        zip(dataset["english"], dataset["darija"]))]
    if not rows:
        print("nothing left to generate")
        return

    model, tokenizer = load_for_inference(inference_config)

    kept = 0
    with open_for_records(preference_config.output_path,
                          args.resume) as handle:
        for record in generate_preference_pairs(model, tokenizer, rows,
                                                data_config, inference_config,
                                                preference_config,
                                                args.keep_all):
            write_record(handle, record)
            handle.flush()
            kept += 1
            if kept % 100 == 0:
                print(f"{kept} pairs written (row {record['row_index']})")

    print(f"wrote {kept} pairs from {len(rows)} prompts "
          f"to {preference_config.output_path}")

    if args.push_to_hub:
        pairs = Dataset.from_json(preference_config.output_path)
        pairs.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed {len(pairs)} pairs to {args.push_to_hub}")


def load_english_rows(source: str,
                      split: str,
                      column: str,
                      start: int,
                      limit: int | None,
                      config: str | None = None):
    """An English corpus with no darija labels: a Hub dataset or a local file."""
    if os.path.exists(source):
        builder = {
            ".jsonl": "json",
            ".json": "json",
            ".csv": "csv",
            ".tsv": "csv",
            ".parquet": "parquet",
            ".txt": "text",
        }.get(os.path.splitext(source)[1], "text")
        dataset = load_dataset(builder, data_files=source, split="train")
    else:
        # multi-config corpora (language pairs, domains) need the config name
        dataset = load_dataset(source, config, split=split)

    if column not in dataset.column_names:
        raise SystemExit(
            f"no {column!r} column in {source}; found {dataset.column_names}. "
            f"pass --column with one of those")

    end = len(dataset) if limit is None else min(start + limit, len(dataset))
    return dataset.select(range(start, max(start, end)))


def run_translate(args):
    data_config = DataConfig()
    inference_config = replace(
        InferenceConfig(),
        adapter_model_id=args.adapter,
        adapter_subfolder=args.subfolder,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.sample,
        temperature=args.temperature,
        num_generations=args.num_generations,
    )

    start = args.start
    if args.resume:
        stopped_at = last_row_index(args.out)
        if stopped_at is not None:
            start = stopped_at + 1
            print(f"resuming after row {stopped_at}")

    dataset = load_english_rows(args.dataset, args.split, args.column, start,
                                args.limit, args.config)
    sources = dataset[args.column]
    if not sources:
        print("nothing left to translate")
        return

    model, tokenizer = load_for_inference(inference_config)

    written = 0
    with open_for_records(args.out, args.resume) as handle:
        for offset, (english, candidates) in enumerate(
                translate(model, tokenizer, sources, data_config.system_prompt,
                          inference_config)):
            record = to_generation_record(english, candidates,
                                          data_config.system_prompt,
                                          inference_config,
                                          {"row_index": start + offset})
            if record is None:
                continue
            write_record(handle, record)
            handle.flush()
            written += 1
            if written % 100 == 0:
                print(f"{written} translated (row {record['row_index']})")

    print(f"wrote {written} translations of {len(sources)} sources "
          f"to {args.out}")

    if args.push_to_hub:
        translations = Dataset.from_json(args.out)
        translations.push_to_hub(args.push_to_hub, private=True)
        print(f"pushed {len(translations)} rows to {args.push_to_hub}")


def texts_from_source(spec, split: str, config: CorpusConfig) -> list[str]:
    """Everything usable one source contributes, before the global mix."""
    dataset = load_dataset(spec.dataset, spec.config, split=split)
    if spec.column not in dataset.column_names:
        raise SystemExit(f"no {spec.column!r} column in {spec.dataset}; "
                         f"found {dataset.column_names}")
    if spec.filter_field:
        if spec.filter_field not in dataset.column_names:
            raise SystemExit(f"no {spec.filter_field!r} column to filter on "
                             f"in {spec.dataset}")
        wanted = spec.filter_value.casefold()
        dataset = dataset.filter(
            lambda row: str(row[spec.filter_field]).casefold() == wanted)

    # token-list columns (idiom corpora ship these) become plain sentences
    texts = [detokenise(value) for value in dataset[spec.column]]
    texts = dedupe(texts)
    texts = [text for text in texts if within_length(text, config)]
    texts = cap_repeated_openings(texts, config)
    if spec.count:
        texts = stratified_sample(texts, spec.count, config)
    return texts


def excluded_texts(dataset_name: str, column: str) -> set:
    print(f"loading {dataset_name} to decontaminate against")
    dataset = load_dataset(dataset_name, split="train")
    return to_seen_set(dataset[column])


def report_lengths(texts: list[str], config: CorpusConfig) -> None:
    edges = list(config.length_buckets)
    labels = [f"<={edges[0]}"] + [
        f"{a + 1}-{b}" for a, b in zip(edges, edges[1:])
    ] + [f">{edges[-1]}"]
    counts = [0] * len(labels)
    for text in texts:
        counts[length_bucket(text, config.length_buckets)] += 1
    print("length bands: " +
          "  ".join(f"{label} {count}" for label, count in zip(labels, counts)))


def run_prepare_corpus(args):
    config = replace(CorpusConfig(),
                     min_words=args.min_words,
                     max_words=args.max_words,
                     output_path=args.out)

    collected = []
    for raw in args.source:
        spec = parse_source(raw)
        texts = texts_from_source(spec, args.split, config)
        print(f"{spec.dataset}: {len(texts)} sentences")
        collected.extend((text, spec.dataset) for text in texts)

    by_text = dict(collected)
    texts = dedupe([text for text, _ in collected])
    print(f"{len(texts)} after dropping cross-source duplicates")

    if not args.no_decontaminate:
        seen = excluded_texts(args.exclude_dataset, args.exclude_column)
        texts = decontaminate(texts, seen)
        print(f"{len(texts)} after removing sentences already in "
              f"{args.exclude_dataset}")

    if args.total:
        texts = stratified_sample(texts, args.total, config)

    report_lengths(texts, config)

    with open_for_records(config.output_path, resume=False) as handle:
        for text in texts:
            write_record(handle, {
                "english": text,
                "source": by_text.get(text, "unknown"),
            })
    print(f"wrote {len(texts)} sentences to {config.output_path}")


def main():
    parser = argparse.ArgumentParser(prog="darija-translator")
    subparsers = parser.add_subparsers(required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--dataset",
        default="atlasia/english-to-darija-arabic-script-formatted")
    train_parser.set_defaults(func=run_train)

    eval_parser = subparsers.add_parser("evaluate")
    eval_parser.add_argument(
        "--dataset",
        default="atlasia/english-to-darija-arabic-script-formatted")
    eval_parser.add_argument("--adapter",
                             default=InferenceConfig.adapter_model_id)
    eval_parser.add_argument("--subfolder",
                             default=None,
                             help="adapter subfolder, e.g. last-checkpoint")
    eval_parser.add_argument("--batch-size",
                             type=int,
                             default=InferenceConfig.batch_size)
    eval_parser.add_argument("--limit",
                             type=int,
                             default=1000,
                             help="sentences from the held-out split; "
                             "0 evaluates all of it")
    eval_parser.set_defaults(func=run_evaluate)

    corpus_parser = subparsers.add_parser(
        "prepare-corpus",
        help="combine English sources into one corpus for translate")
    corpus_parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="DATASET[:CONFIG[:COLUMN[:COUNT[:FIELD=VALUE]]]]",
        help="repeatable; e.g. "
        "sentence-transformers/parallel-sentences-tatoeba:en-de:english:20000")
    corpus_parser.add_argument("--split",
                               default="train",
                               help="split read from every source")
    corpus_parser.add_argument("--total",
                               type=int,
                               default=None,
                               help="cap the mixed corpus, sampled evenly "
                               "across length bands")
    corpus_parser.add_argument("--min-words",
                               type=int,
                               default=CorpusConfig.min_words)
    corpus_parser.add_argument("--max-words",
                               type=int,
                               default=CorpusConfig.max_words)
    corpus_parser.add_argument(
        "--exclude-dataset",
        default="atlasia/english-to-darija-arabic-script-formatted",
        help="drop sentences the model already trained on")
    corpus_parser.add_argument("--exclude-column", default="english")
    corpus_parser.add_argument("--no-decontaminate",
                               action="store_true",
                               help="skip the overlap check")
    corpus_parser.add_argument("--out", default=CorpusConfig.output_path)
    corpus_parser.set_defaults(func=run_prepare_corpus)

    translate_parser = subparsers.add_parser(
        "translate",
        help="translate an unlabelled English corpus with the trained adapter")
    translate_parser.add_argument(
        "--dataset",
        required=True,
        help="Hub dataset id, or a local .jsonl/.csv/.parquet/.txt file")
    translate_parser.add_argument("--split", default="train")
    translate_parser.add_argument(
        "--config",
        default=None,
        help="config name, for datasets that have several "
        "(e.g. en-de for a language-pair corpus)")
    translate_parser.add_argument("--column",
                                  default="english",
                                  help="column holding the English text")
    translate_parser.add_argument("--adapter",
                                  default=InferenceConfig.adapter_model_id)
    translate_parser.add_argument("--subfolder",
                                  default=None,
                                  help="adapter subfolder, e.g. last-checkpoint")
    translate_parser.add_argument("--start", type=int, default=0)
    translate_parser.add_argument("--limit",
                                  type=int,
                                  default=None,
                                  help="default: the whole corpus")
    translate_parser.add_argument("--batch-size",
                                  type=int,
                                  default=InferenceConfig.batch_size)
    translate_parser.add_argument("--max-new-tokens",
                                  type=int,
                                  default=InferenceConfig.max_new_tokens)
    translate_parser.add_argument("--sample",
                                  action="store_true",
                                  help="sample instead of decoding greedily")
    translate_parser.add_argument("--temperature",
                                  type=float,
                                  default=InferenceConfig.temperature)
    translate_parser.add_argument("--num-generations",
                                  type=int,
                                  default=InferenceConfig.num_generations,
                                  help="candidates per source, all are kept")
    translate_parser.add_argument("--out", default="data/translations.jsonl")
    translate_parser.add_argument(
        "--resume",
        action="store_true",
        help="append to --out, continuing after its last row")
    translate_parser.add_argument("--push-to-hub",
                                  default=None,
                                  metavar="REPO_ID",
                                  help="push the translations as a HF dataset")
    translate_parser.set_defaults(func=run_translate)

    dpo_parser = subparsers.add_parser(
        "generate-dpo",
        help="translate prompts with the trained adapter and keep the "
        "generations as rejected samples")
    dpo_parser.add_argument(
        "--dataset",
        default="atlasia/english-to-darija-arabic-script-formatted")
    dpo_parser.add_argument("--adapter",
                            default=InferenceConfig.adapter_model_id)
    dpo_parser.add_argument("--subfolder",
                            default=None,
                            help="adapter subfolder, e.g. last-checkpoint")
    dpo_parser.add_argument("--split",
                            choices=("eval", "train", "all"),
                            default="eval",
                            help="which side of the SFT split to sample from")
    dpo_parser.add_argument("--start", type=int, default=0)
    dpo_parser.add_argument("--limit", type=int, default=1000)
    dpo_parser.add_argument("--batch-size",
                            type=int,
                            default=InferenceConfig.batch_size)
    dpo_parser.add_argument("--max-new-tokens",
                            type=int,
                            default=InferenceConfig.max_new_tokens)
    dpo_parser.add_argument("--temperature",
                            type=float,
                            default=InferenceConfig.temperature)
    dpo_parser.add_argument("--greedy",
                            action="store_true",
                            help="decode deterministically instead of sampling")
    dpo_parser.add_argument("--num-generations",
                            type=int,
                            default=InferenceConfig.num_generations,
                            help="candidates per prompt, the worst is kept")
    dpo_parser.add_argument("--max-chrf",
                            type=float,
                            default=PreferenceConfig.max_chrf_similarity,
                            help="drop pairs whose generation is this close "
                            "to the reference")
    dpo_parser.add_argument("--keep-all",
                            action="store_true",
                            help="skip the similarity filter")
    dpo_parser.add_argument("--out", default=PreferenceConfig.output_path)
    dpo_parser.add_argument(
        "--resume",
        action="store_true",
        help="append to --out, continuing after its last row")
    dpo_parser.add_argument("--push-to-hub",
                            default=None,
                            metavar="REPO_ID",
                            help="push the finished pairs as a HF dataset")
    dpo_parser.set_defaults(func=run_generate_dpo)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
