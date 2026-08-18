"""Frozen config objects, one per pipeline stage.

Anything two stages have to agree on — the base model, the context length,
the seed, the adapter training pushes — lives here as a single constant so
the stages cannot drift apart.
"""
from dataclasses import dataclass

SEED = 3407
BASE_MODEL_NAME = "LiquidAI/LFM2.5-230M"
MAX_SEQ_LENGTH = 2048
HUB_MODEL_ID = "atlasia/edge-device-darija-translator"
# the exact prompt the SFT dataset was formatted with
SYSTEM_PROMPT = "You are a professional English to Darija translator."


@dataclass(frozen=True)
class DataConfig:
    system_prompt: str = SYSTEM_PROMPT
    max_text_length: int = 2000
    test_size: float = 0.1
    seed: int = SEED


@dataclass(frozen=True)
class ModelConfig:
    model_name: str = BASE_MODEL_NAME
    max_seq_length: int = MAX_SEQ_LENGTH
    load_in_16bit: bool = True
    lora_r: int = 16
    lora_alpha: int = 16
    lora_dropout: int = 0
    random_state: int = SEED
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
    dataloader_num_workers: int = 4
    per_device_train_batch_size: int = 64
    gradient_accumulation_steps: int = 2
    per_device_eval_batch_size: int = 8
    num_train_epochs: int = 3
    warmup_ratio: float = 0.1
    learning_rate: float = 2e-4
    logging_steps: int = 100
    weight_decay: float = 0.01
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    seed: int = SEED
    packing: bool = True
    group_by_length: bool = True
    max_seq_length: int = MAX_SEQ_LENGTH
    output_dir: str = "lora_model"
    report_to: str = "wandb"
    wandb_project: str = "darija-translator"
    hub_model_id: str = HUB_MODEL_ID


@dataclass(frozen=True)
class InferenceConfig:
    base_model_name: str = BASE_MODEL_NAME
    adapter_model_id: str = HUB_MODEL_ID
    adapter_subfolder: str | None = None
    max_seq_length: int = MAX_SEQ_LENGTH
    load_in_16bit: bool = True
    batch_size: int = 32
    max_new_tokens: int = 256
    do_sample: bool = True
    temperature: float = 0.9
    top_p: float = 0.95
    # candidates generated per prompt, the worst one becomes the bad sample
    num_generations: int = 1
    seed: int = SEED


@dataclass(frozen=True)
class PreferenceConfig:
    """DPO pairs: the dataset darija is chosen, the model output is rejected."""
    # a generation this close to the reference is not a useful bad sample
    max_chrf_similarity: float = 90.0
    output_path: str = "data/dpo_pairs.jsonl"
    hub_dataset_id: str = "atlasia/english-to-darija-dpo"
