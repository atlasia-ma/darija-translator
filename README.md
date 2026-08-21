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
    ├── evaluate.py       # BLEU/chrF scoring
    ├── corpus.py         # mix English sources into one balanced corpus
    ├── inference.py       # load trained adapter + batched generation
    ├── jsonl.py            # streaming record IO, resume support
    ├── preference.py        # DPO pairs from model generations
    └── cli.py                # train / evaluate / translate / generate-dpo

Pure logic (`data.py`, `evaluate.py`, `corpus.py`, all of `config.py`,
`jsonl.py`, the
prompt/record helpers in `inference.py` and `preference.py`) is unit-tested.
Model/training code (`model.py`, `train.py`, `load_for_inference`) isn't — it's an integration point with a real model
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

Reports BLEU and chrF for the trained adapter on the held-out split,
1000 sentences by default (`--limit 0` for all ~71k). Point it at another
checkpoint with `--adapter` / `--subfolder`.

## Building the English corpus

Mixes several English sources into one file for `translate`. Diversity is the
point — the model is weakest on registers the SFT set was thin on, so length
bands are sampled evenly and no single sentence pattern is allowed to dominate.

    uv run darija-translator prepare-corpus         --source sentence-transformers/parallel-sentences-opus-100:en-ar:english:50000         --source sentence-transformers/parallel-sentences-tatoeba:en-de:english:25000         --source sentence-transformers/parallel-sentences-global-voices:en-ar:english:20000         --source Gooogr/pie_idioms::tokens:10000:is_pie=true         --total 70000 --out data/corpus.jsonl

These are English→X parallel corpora with the other side discarded: sentences
written to be translated. They cover different ground on purpose — OPUS-100 is
itself a balanced mix across dozens of corpora (short to medium, many
questions), Tatoeba is clean everyday sentences, Global Voices is
human-translated journalism for the long band, and PIE supplies figurative
language. The `en-ar` configs are the English that someone chose to translate
into Arabic, which sits closer to Darija's world than `en-de`.

Sources are capped well above `--total` so the length-band sampling has room
to balance them; roughly 1.5x the target works.

Each `--source` is `dataset[:config[:column[:count[:field=value]]]]`; leave a
segment empty to skip it. The trailing `field=value` filters rows — above, it
keeps only instances where the expression is used figuratively. Token-list
columns are detokenised into real sentences.

Every source is deduped, filtered to `--min-words`/`--max-words`, capped so one
opening ("Sami told Layla...") can't repeat more than a few times, then sampled
across length bands. Sentences already present in the SFT training data are
dropped — the model can't fail on what it memorised, so those rows teach
nothing — via `--exclude-dataset` (`--no-decontaminate` skips the check, which
also skips downloading it).

The output has one text column, `english`, plus the source each sentence came
from. After `translate` it becomes two: `english` and `generated`.

### Selecting for idioms

Idiom-bearing sentences are sparse — around 3% of a general corpus — and the
ones in a dedicated idiom corpus tend to be long excerpts, where the model
fails because of the length rather than the figurative phrase. `--must-contain`
selects instead: keep only sentences containing one of a list of phrases.

    uv run darija-translator prepare-corpus         --source sentence-transformers/parallel-sentences-opus-100:en-ar:english:8000         --source sentence-transformers/parallel-sentences-tatoeba:en-de:english:4000         --must-contain Gooogr/pie_idioms::idiom         --max-words 15 --total 10000         --exclude-dataset data/corpus.jsonl --out data/idioms.jsonl

That pulls short idiom sentences out of corpora you already have, skipping
anything in the main corpus, and `cat data/idioms.jsonl >> data/corpus.jsonl`
merges the two. `--must-contain` also accepts a local `.txt` file of one phrase
per line. Matching is literal, so "call it a day" won't catch "called it a
day".

## Translating an unlabelled corpus

Runs the trained adapter over English text that has no darija labels, which is
the first half of building a DPO set: these generations are the candidate
`rejected` answers, waiting for a `chosen` counterpart from a teacher model or
a human pass.

    uv run darija-translator translate --dataset my-corpus.jsonl --column english

`--dataset` takes a Hub dataset id or a local `.jsonl` / `.csv` / `.parquet` /
`.txt` file. Multi-config corpora need `--config`:

    uv run darija-translator translate         --dataset sentence-transformers/parallel-sentences-tatoeba         --config en-de --column english --limit 30000 Nothing is filtered except empty generations — with no reference
there is nothing to score against, so every row is kept as-is. Each record
carries the prompt, the translation, every candidate, and which adapter and
decoding produced it, so the pairing pass can join against it later.

Decoding is greedy by default: the model's most likely output is what an edge
device will actually emit, so that is the output worth correcting. `--sample`
with `--num-generations 4` produces diverse candidates instead, which is what
you want if a human is going to rank them.

## DPO data generation

Builds preference pairs for a DPO round from a corpus that **does** have
reference translations: the reference darija is the `chosen` answer, the
model's own generation is the `rejected` one.

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
