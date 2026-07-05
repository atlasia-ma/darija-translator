from dataclasses import dataclass


@dataclass
class DataConfig:
    system_prompt: str = "You are a professional English to Darija translator."
