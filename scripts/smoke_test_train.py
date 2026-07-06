"""Manual smoke test — run on GPU with a tiny dataset slice."""
from darija_translator.config import DataConfig, ModelConfig, TrainConfig
from darija_translator.model import attach_lora, load_model_and_tokenizer
from darija_translator.train import build_trainer, save_model

# trainer = build_trainer(model, tokenizer, train_dataset, eval_dataset, TrainConfig())
# trainer.train()
# save_model(model, tokenizer, TrainConfig())
