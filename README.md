# Darija Translator

Fine-tuning LFM2.5-230M for English↔Darija translation with LoRA (Unsloth + TRL).

## Architecture

Pipeline-stage layout, not layered/hexagonal — this is a training pipeline, not a
domain-rich application, so the structure follows the stages data moves through
rather than DDD-style layers:

    src/darija_translator/
    ├── config.py     # DataConfig, ModelConfig, TrainConfig — frozen dataclasses
    ├── data.py        # filtering, chat formatting, length filtering, split
    ├── model.py        # load base model + attach LoRA (Unsloth)
    ├── train.py         # SFTTrainer wiring, wandb tracking
    ├── evaluate.py       # BLEU/chrF scoring + generation
    └── cli.py             #`darija-translator train` / `evaluate`

Pure logic (`data.py`, `evaluate.py`'s `compute_translation_metrics`, all of
`config.py`) is unit-tested. Model/training code (`model.py`, `train.py`,
`generate_translations`) isn't — it's an integration point with a real model
and GPU, verified instead via manual smoke-test scripts in `scripts/`.

## Setup

    uv sync --group dev --group eval    # run the test suite, no GPU needed
    uv run pytest -v

For training/evaluation on GPU, also install:

    uv sync --group dev --group eval --group train

### W&B tracking

Training reports to Weights & Biases by default.

    cp .env.example .env
    # fill in WANDB_API_KEY from https://wandb.ai/authorize

## Hugging face checkpointing

huggingface-cli login

## Training (requires GPU)

    uv run darija-translator train

## Evaluation

    uv run darija-translator evaluate

Reports BLEU and chrF on the held-out split.

## Results

<!-- fill in once trained: BLEU/chrF on held-out set, training curves link -->

## Original exploration

The initial SFT experiment (before this repo existed) is kept at
`notebooks/01_exploration.ipynb` for reference.
