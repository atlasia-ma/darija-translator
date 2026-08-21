"""Batched generation with the fine-tuned adapter.

Model loading imports unsloth/peft lazily so the pure helpers here stay
importable (and unit-testable) on a machine without the training stack.
"""
from collections.abc import Iterator

from darija_translator.config import InferenceConfig


def load_for_inference(config: InferenceConfig):
    from unsloth import FastLanguageModel
    from peft import PeftModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config.base_model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        load_in_8bit=False,
        load_in_16bit=config.load_in_16bit,
        full_finetuning=False,
    )
    model = PeftModel.from_pretrained(
        model,
        config.adapter_model_id,
        subfolder=config.adapter_subfolder,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    return model, prepare_tokenizer_for_generation(tokenizer)


def prepare_tokenizer_for_generation(tokenizer):
    """Decoder-only batched generation needs left padding to stay aligned."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_translation_messages(english: str, system_prompt: str) -> list[dict]:
    return [
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": english.strip()
        },
    ]


def encode_prompts(tokenizer, sources: list[str], system_prompt: str):
    texts = tokenizer.apply_chat_template(
        [build_translation_messages(s, system_prompt) for s in sources],
        tokenize=False,
        add_generation_prompt=True,
    )
    # the chat template already emits bos — add_special_tokens would double it
    return tokenizer(texts,
                     add_special_tokens=False,
                     padding=True,
                     return_tensors="pt")


def group_candidates(decoded: list[str],
                     num_generations: int) -> list[list[str]]:
    """generate() returns num_generations consecutive rows per prompt."""
    return [
        decoded[i:i + num_generations]
        for i in range(0, len(decoded), num_generations)
    ]


def translate_batch(model, tokenizer, sources: list[str], system_prompt: str,
                    config: InferenceConfig) -> list[list[str]]:
    import torch

    inputs = encode_prompts(tokenizer, sources, system_prompt).to(model.device)
    generate_kwargs = {
        "max_new_tokens": config.max_new_tokens,
        "do_sample": config.do_sample,
        "num_return_sequences": config.num_generations,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if config.do_sample:
        generate_kwargs["temperature"] = config.temperature
        generate_kwargs["top_p"] = config.top_p

    with torch.inference_mode():
        outputs = model.generate(**inputs, **generate_kwargs)

    new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
    decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
    return group_candidates([text.strip() for text in decoded],
                            config.num_generations)


def describe_decoding(config: InferenceConfig) -> str:
    """Provenance: months later this file is half a preference dataset."""
    if not config.do_sample:
        return "greedy"
    return f"sample(temperature={config.temperature}, top_p={config.top_p})"


def describe_adapter(config: InferenceConfig) -> str:
    if not config.adapter_subfolder:
        return config.adapter_model_id
    return f"{config.adapter_model_id}/{config.adapter_subfolder}"


def to_generation_record(english: str,
                         candidates: list[str],
                         system_prompt: str,
                         config: InferenceConfig,
                         metadata: dict | None = None) -> dict | None:
    """A translated row, with no claim about whether the translation is good."""
    kept = [candidate.strip() for candidate in candidates if candidate.strip()]
    if not english.strip() or not kept:
        return None
    return {
        "prompt": build_translation_messages(english, system_prompt),
        "english": english.strip(),
        "generated": kept[0],
        "candidates": kept,
        "adapter": describe_adapter(config),
        "decoding": describe_decoding(config),
        **(metadata or {}),
    }


def batched(items: list, batch_size: int) -> Iterator[list]:
    for start in range(0, len(items), batch_size):
        yield items[start:start + batch_size]


def translate(model, tokenizer, sources: list[str], system_prompt: str,
              config: InferenceConfig) -> Iterator[tuple[str, list[str]]]:
    """Yield (source, candidates) so callers can stream results to disk."""
    for batch in batched(sources, config.batch_size):
        for source, candidates in zip(
                batch,
                translate_batch(model, tokenizer, batch, system_prompt,
                                config)):
            yield source, candidates
