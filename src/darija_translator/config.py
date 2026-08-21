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


def _require(condition, message: str) -> None:
    """Fail at construction, not forty minutes into a training run."""
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class DataConfig:
    system_prompt: str = SYSTEM_PROMPT
    max_text_length: int = 2000
    test_size: float = 0.1
    seed: int = SEED

    def __post_init__(self):
        _require(self.system_prompt.strip(), "system_prompt must not be empty")
        _require(self.max_text_length > 0, "max_text_length must be positive")
        _require(0 < self.test_size < 1, "test_size must be between 0 and 1")


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

    def __post_init__(self):
        _require(self.max_seq_length > 0, "max_seq_length must be positive")
        _require(self.lora_r > 0, "lora_r must be positive")
        _require(self.lora_alpha > 0, "lora_alpha must be positive")
        _require(0 <= self.lora_dropout < 1, "lora_dropout must be in [0, 1)")
        _require(self.target_modules, "target_modules must not be empty")


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

    def __post_init__(self):
        _require(self.per_device_train_batch_size > 0,
                 "per_device_train_batch_size must be positive")
        _require(self.per_device_eval_batch_size > 0,
                 "per_device_eval_batch_size must be positive")
        _require(self.gradient_accumulation_steps > 0,
                 "gradient_accumulation_steps must be positive")
        _require(self.num_train_epochs > 0,
                 "num_train_epochs must be positive")
        _require(self.learning_rate > 0, "learning_rate must be positive")
        _require(0 <= self.warmup_ratio <= 1,
                 "warmup_ratio must be between 0 and 1")
        _require(self.max_seq_length > 0, "max_seq_length must be positive")


@dataclass(frozen=True)
class InferenceConfig:
    base_model_name: str = BASE_MODEL_NAME
    adapter_model_id: str = HUB_MODEL_ID
    adapter_subfolder: str | None = None
    max_seq_length: int = MAX_SEQ_LENGTH
    load_in_16bit: bool = True
    batch_size: int = 32
    max_new_tokens: int = 256
    # greedy: the mode of the policy, and what an edge device will decode with
    do_sample: bool = False
    temperature: float = 0.9
    top_p: float = 0.95
    # candidates generated per prompt, the worst one becomes the bad sample
    num_generations: int = 1
    seed: int = SEED

    def __post_init__(self):
        _require(self.base_model_name, "base_model_name must not be empty")
        _require(self.adapter_model_id, "adapter_model_id must not be empty")
        _require(self.batch_size > 0, "batch_size must be positive")
        _require(self.max_new_tokens > 0, "max_new_tokens must be positive")
        _require(self.max_new_tokens < self.max_seq_length,
                 "max_new_tokens must fit inside max_seq_length")
        _require(self.num_generations > 0,
                 "num_generations must be at least 1")
        if self.do_sample:
            _require(self.temperature > 0,
                     "temperature must be positive when sampling")
            _require(0 < self.top_p <= 1, "top_p must be in (0, 1]")


@dataclass(frozen=True)
class PreferenceConfig:
    """DPO pairs: the dataset darija is chosen, the model output is rejected."""
    # a generation this close to the reference is not a useful bad sample
    max_chrf_similarity: float = 90.0
    output_path: str = "data/dpo_pairs.jsonl"
    hub_dataset_id: str = "atlasia/english-to-darija-dpo"

    def __post_init__(self):
        _require(0 <= self.max_chrf_similarity <= 100,
                 "max_chrf_similarity must be between 0 and 100")
        _require(self.output_path, "output_path must not be empty")
