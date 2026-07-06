from unsloth import FastLanguageModel

from darija_translator.config import ModelConfig


def load_model_and_tokenizer(config: ModelConfig):
    return FastLanguageModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=config.load_in_16bit,
        full_finetuning=False,
    )


def attach_lora(model, config: ModelConfig):
    FastLanguageModel.for_training(model)
    return FastLanguageModel.get_peft_model(
        model,
        r=config.lora_r,
        target_modules=config.target_modules,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=config.random_state,
        use_rslora=False,
        loftq_config=None,
    )
