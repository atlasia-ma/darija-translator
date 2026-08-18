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
    ├── inference.py       # load trained adapter + batched generation
    ├── preference.py       # DPO pairs from model generations
    └── cli.py               # `darija-translator train` / `evaluate` / `generate-dpo`

Pure logic (`data.py`, `evaluate.py`'s `compute_translation_metrics`, all of
`config.py`, the prompt/pair helpers in `inference.py` and `preference.py`) is
unit-tested. Model/training code (`model.py`, `train.py`, `load_for_inference`,
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

## DPO data generation

Builds preference pairs for a DPO round: the dataset darija is the `chosen`
answer, the model's own generation is the `rejected` one — on-policy bad
samples from the SFT checkpoint.

    uv run darija-translator generate-dpo --limit 5000

Prompts are drawn from the held-out `eval` side of the SFT split by default
(`--split train|all` for the rest), so the model isn't graded on sentences it
memorised. Pairs are streamed to `data/dpo_pairs.jsonl` as they're generated;
`--resume` continues after the last row a previous run wrote.

Generations that already match the reference teach DPO nothing, so pairs with
chrF above `--max-chrf` (default 90) are dropped — `--keep-all` keeps them.
`--num-generations 4` samples several candidates per prompt and keeps the one
furthest from the reference, which gives harder negatives at 4x the compute.

    uv run darija-translator generate-dpo         --split eval --limit 20000 --batch-size 64         --num-generations 4 --out data/dpo_pairs.jsonl

Each line is ready for `trl`'s `DPOTrainer` in conversational format
(`prompt` / `chosen` / `rejected`), plus flat `english` / `darija` /
`generated` / `chrf` / `row_index` columns for filtering and inspection.
Add `--push-to-hub <repo_id>` to upload the finished file as a private
dataset.

The adapter defaults to `atlasia/edge-device-darija-translator`; use
`--adapter` / `--subfolder last-checkpoint` to generate from another
checkpoint.

## Results

<!-- fill in once trained: BLEU/chrF on held-out set, training curves link -->

## Original exploration

The initial SFT experiment (before this repo existed) is kept at
`notebooks/01_exploration.ipynb` for reference.
