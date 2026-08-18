"""Manual smoke test — run on a GPU machine, not part of the automated test suite.

Confirms the trained adapter loads and generates darija for a few English
sentences, the same way generate-dpo will call it.
"""
from darija_translator.config import DataConfig, InferenceConfig
from darija_translator.inference import load_for_inference, translate

data_config = DataConfig()
config = InferenceConfig(batch_size=4, num_generations=2)
model, tokenizer = load_for_inference(config)

sentences = [
    "How are you doing today?",
    "I was wondering why you are not talking to me",
    "Where were you going?",
    "Can you send me the file before tomorrow morning?",
]

for english, candidates in translate(model, tokenizer, sentences,
                                     data_config.system_prompt, config):
    print(f"EN: {english}")
    for candidate in candidates:
        print(f"AR: {candidate}")
    print("-" * 60)
