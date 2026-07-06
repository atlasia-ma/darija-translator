from dataclasses import dataclass, field


@dataclass(frozen=True)
class DataConfig:
    system_prompt: str = "You are a professional English to Darija translator."
    max_text_length: int = 2000
    test_size: float = 0.1
    seed: int = 3407


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = "LiquidAI/LFM2.5-230M"
    max_seq_length: int = 2048
    load_in_16bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: int = 0
    random_state: int = 3407
    target_modules: tuple = (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "in_proj",
        "w1",
        "w2",
        "w3",
    )


@dataclass(frozen=True)
class TrainConfig:
    per_device_train_batch_size: int = 16
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 3
    warmup_steps: int = 5
    learning_rate: float = 2e-4
    logging_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    seed: int = 3407
    packing: bool = True
    group_by_length: bool = True
    max_seq_length: int = 2048
    output_dir: str = "lora_model"
