import os

from trl import SFTConfig, SFTTrainer
from unsloth.chat_templates import train_on_responses_only

from darija_translator.config import TrainConfig


def build_trainer(model, tokenizer, train_dataset, eval_dataset,
                  config: TrainConfig):
    if config.report_to == "wandb" or (isinstance(config.report_to, list)
                                       and "wandb" in config.report_to):
        os.environ["WANDB_PROJECT"] = config.wandb_project

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            packing=config.packing,
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
            # group_by_length=config.group_by_length,
        ),
    )
    return train_on_responses_only(
        trainer,
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    )


def save_model(model, tokenizer, config: TrainConfig):
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
