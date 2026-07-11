import argparse

from datasets import load_dataset

from darija_translator.config import DataConfig, ModelConfig, TrainConfig
from darija_translator.data import (
    format_conversations,
    is_darija_script,
    is_within_length,
    split_dataset,
    to_conversations,
)
from darija_translator.evaluate import compute_translation_metrics, generate_translations
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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
