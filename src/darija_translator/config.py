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
