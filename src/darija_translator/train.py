from unsloth.chat_templates import train_on_responses_only

import os

from datasets import load_dataset

from darija_translator.model import attach_lora, load_model_and_tokenizer
from darija_translator.data import split_dataset
from trl import SFTConfig, SFTTrainer

from darija_translator.config import DataConfig, ModelConfig, TrainConfig


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


if __name__ == "__main__":
    data_config, model_config, train_config = DataConfig(), ModelConfig(
    ), TrainConfig()
    model, tokenizer = load_model_and_tokenizer(model_config)
    model = attach_lora(model, model_config)
    train_dataset, eval_dataset = prepare_data(
        "atlasia/english-to-darija-arabic-script-formatted", data_config,
        tokenizer)
    trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset,
                            train_config)
    for i in range(min(10, len(trainer.train_dataset))):
        row = trainer.train_dataset[i]
        print(row)
        input_ids = row["input_ids"]
        labels = row["labels"]
        print(f"\n--- row {i} ---")
        print(f"input_ids: {input_ids}")
        print(f"labels: {labels}")
        print(f"decoded input: {tokenizer.decode(input_ids)}")
        print(
            f"decoded labels: {tokenizer.decode([lab for lab in labels if lab != -100])}"
        )

# # 1. Run your training
# trainer.train()

# # 2. Push the highly optimized LoRA adapter to your repo
# model.push_to_hub("your-username/darija-translator", token=True)
# tokenizer.push_to_hub("your-username/darija-translator", token=True)

# # Merges LoRA weights back into the base structure and pushes the whole thing
# model.push_to_hub_merged(
#     "your-username/darija-translator-merged",
#     tokenizer,
#     save_method="merged_16bit"
# )
# from unsloth import FastLanguageModel

# max_seq_length = 2048
# dtype = None # None for auto detection. Float16 for Tesla T4/V100, Bfloat16 for Ampere+
# load_in_4bit = True # Use True if you want to keep VRAM footprint low

# # 1. Load the base model and tokenizer
# model, tokenizer = FastLanguageModel.from_pretrained(
#     model_name = "unsloth/llama-3-8b-Instruct", # Use whatever base model you started with
#     max_seq_length = max_seq_length,
#     dtype = dtype,
#     load_in_4bit = load_in_4bit,
# )

# # 2. Layer your tiny checkpoint adapter right on top from the Hugging Face Hub
# # You can reference specific checkpoint folders directly using the 'subfolder' argument!
# model = FastLanguageModel.for_inference(model)
# model.load_adapter(
#     "your-username/darija-translator",
#     subfolder="checkpoint-400" # Change this to checkpoint-800, checkpoint-1200, etc.
# )

# # 3. Test your translator!
# inputs = tokenizer(
#     [
#         "<|im_start|>system\nYou are a professional English to Darija translator.<|im_end|>\n<|im_start|>user\nHow are you doing today?<|im_end|>\n<|im_start|>assistant\n"
#     ],
#     return_tensors = "pt"
# ).to("cuda")

# outputs = model.generate(**inputs, max_new_tokens = 64, use_cache = True)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])
