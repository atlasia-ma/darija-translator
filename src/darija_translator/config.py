from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    system_prompt: str = "You are a professional English to Darija translator."
    max_text_length: int = 2000
