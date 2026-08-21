from unsloth.chat_templates import train_on_responses_only

import os

from datasets import load_dataset

from darija_translator.data import split_dataset
from trl import SFTConfig, SFTTrainer

from darija_translator.config import DataConfig, TrainConfig


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


def build_trainer(model, tokenizer, train_dataset, eval_dataset,
                  config: TrainConfig):
    if config.report_to == "wandb" or (isinstance(config.report_to, list)
                                       and "wandb" in config.report_to):
        os.environ["WANDB_PROJECT"] = config.wandb_project

    sft_args = SFTConfig(
        dataset_text_field="text",
        dataloader_num_workers=config.dataloader_num_workers,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        packing=False,
        max_seq_length=config.max_seq_length,
        warmup_ratio=config.warmup_ratio,
        num_train_epochs=config.num_train_epochs,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        eval_strategy="epoch",
        learning_rate=config.learning_rate,
        logging_steps=config.logging_steps,
        optim=config.optim,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        seed=config.seed,
        report_to=config.report_to,
        output_dir=config.output_dir,
        padding_free=False,
        save_strategy="steps",  # Save checkpoints at step intervals
        save_steps=400,
        save_total_limit=3,
        push_to_hub=True,  # Enable auto-uploading to HF
        hub_model_id=config.hub_model_id,
        hub_strategy="checkpoint",

        # group_by_length=config.group_by_length,
    )
    sft_args.group_by_length = True

    trainer = SFTTrainer(model=model,
                         tokenizer=tokenizer,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         args=sft_args)

    return train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )


def save_model(model, tokenizer, config: TrainConfig):
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
