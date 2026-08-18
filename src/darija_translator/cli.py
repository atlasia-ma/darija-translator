import argparse
import os
from dataclasses import replace

from datasets import Dataset, load_dataset

from darija_translator.config import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    PreferenceConfig,
    TrainConfig,
)
from darija_translator.data import (
    format_conversations,
    is_darija_script,
    is_within_length,
    split_dataset,
    to_conversations,
)
from darija_translator.evaluate import compute_translation_metrics, generate_translations
from darija_translator.inference import load_for_inference
from darija_translator.preference import (
    generate_preference_pairs,
    last_row_index,
    write_record,
)
from darija_translator.model import attach_lora, load_model_and_tokenizer
from darija_translator.train import build_trainer, save_model
from dotenv import load_dotenv

load_dotenv()


def prepare_data(dataset_name: str,
                 data_config: DataConfig,
                 tokenizer,
                 remove_columns: bool = True) -> tuple:
    dataset = load_dataset(dataset_name, split="train")
    # dataset = dataset.filter(is_darija_script)
    # dataset = dataset.map(lambda b: to_conversations(b, data_config),
    #                       batched=True)
    # dataset = dataset.map(lambda b: format_conversations(b, tokenizer),
    #                       batched=True)
    # dataset = dataset.filter(lambda ex: is_within_length(ex, data_config))
    if remove_columns:
        dataset = dataset.remove_columns(
            [c for c in dataset.column_names if c != "text"])
    return split_dataset(dataset, data_config)


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
    model_config, data_config = ModelConfig(), DataConfig()
    model, tokenizer = load_model_and_tokenizer(model_config)
    _, eval_dataset = prepare_data(args.dataset,
                                   data_config,
                                   tokenizer,
                                   remove_columns=False)
    predictions = generate_translations(
        model,
        tokenizer,
        eval_dataset["english"],
        data_config.system_prompt,
    )
    metrics = compute_translation_metrics(predictions, eval_dataset["darija"])
    print(metrics)


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

    directory = os.path.dirname(preference_config.output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    mode = "a" if args.resume and os.path.exists(
        preference_config.output_path) else "w"

    kept = 0
    with open(preference_config.output_path, mode, encoding="utf-8") as handle:
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
    eval_parser.set_defaults(func=run_evaluate)

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
