"""Manual smoke test — run on a GPU machine, not part of the automated test suite.

Confirms load_model_and_tokenizer and attach_lora actually work against the
real Unsloth/HF stack, since that can't be meaningfully unit-tested.
"""
from darija_translator.config import ModelConfig
from darija_translator.model import attach_lora, load_model_and_tokenizer

config = ModelConfig()
model, tokenizer = load_model_and_tokenizer(config)
model = attach_lora(model, config)
print("Loaded model:", type(model))
print("Loaded tokenizer:", type(tokenizer))
print("Trainable params:",
      sum(p.numel() for p in model.parameters() if p.requires_grad))
