import dataclasses

import pytest

from darija_translator.config import (
    DataConfig,
    InferenceConfig,
    ModelConfig,
    PreferenceConfig,
    TrainConfig,
)

ALL_CONFIGS = [
    DataConfig, ModelConfig, TrainConfig, InferenceConfig, PreferenceConfig
]


def test_system_prompt_matches_the_one_the_dataset_was_formatted_with():
    # the fine-tuned model only behaves if inference reproduces this exactly
    assert DataConfig().system_prompt == (
        "You are a professional English to Darija translator.")


def test_inference_loads_the_base_model_that_was_fine_tuned():
    assert InferenceConfig().base_model_name == ModelConfig().model_name


def test_inference_defaults_to_the_adapter_training_pushes():
    assert InferenceConfig().adapter_model_id == TrainConfig().hub_model_id


def test_training_and_model_agree_on_sequence_length():
    assert TrainConfig().max_seq_length == ModelConfig().max_seq_length


def test_generation_fits_inside_the_trained_context():
    assert InferenceConfig().max_new_tokens < InferenceConfig().max_seq_length


def test_every_stage_shares_one_seed():
    seeds = {
        DataConfig().seed,
        ModelConfig().random_state,
        TrainConfig().seed,
        InferenceConfig().seed,
    }
    assert len(seeds) == 1


@pytest.mark.parametrize("config_class", ALL_CONFIGS)
def test_configs_are_frozen(config_class):
    config = config_class()
    first_field = dataclasses.fields(config_class)[0].name

    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(config, first_field, "changed")


def test_configs_accept_valid_overrides():
    assert DataConfig(test_size=0.2).test_size == 0.2
    assert ModelConfig(lora_r=32).lora_r == 32
    assert InferenceConfig(num_generations=4).num_generations == 4
    assert PreferenceConfig(max_chrf_similarity=75.0).max_chrf_similarity == 75.0


@pytest.mark.parametrize("kwargs", [
    {
        "test_size": 0
    },
    {
        "test_size": 1
    },
    {
        "test_size": 1.5
    },
    {
        "max_text_length": 0
    },
    {
        "system_prompt": "   "
    },
])
def test_data_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        DataConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {
        "lora_r": 0
    },
    {
        "lora_alpha": 0
    },
    {
        "lora_dropout": 1
    },
    {
        "max_seq_length": 0
    },
    {
        "target_modules": ()
    },
])
def test_model_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        ModelConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {
        "per_device_train_batch_size": 0
    },
    {
        "gradient_accumulation_steps": 0
    },
    {
        "num_train_epochs": 0
    },
    {
        "learning_rate": 0
    },
    {
        "warmup_ratio": 1.5
    },
])
def test_train_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        TrainConfig(**kwargs)


@pytest.mark.parametrize("kwargs", [
    {
        "batch_size": 0
    },
    {
        "max_new_tokens": 0
    },
    {
        "num_generations": 0
    },
    {
        "temperature": 0
    },
    {
        "top_p": 1.5
    },
    {
        "adapter_model_id": ""
    },
])
def test_inference_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        InferenceConfig(**kwargs)


def test_greedy_inference_ignores_the_sampling_parameters():
    config = InferenceConfig(do_sample=False, temperature=0.0)

    assert config.do_sample is False


@pytest.mark.parametrize("kwargs", [
    {
        "max_chrf_similarity": -1
    },
    {
        "max_chrf_similarity": 101
    },
    {
        "output_path": ""
    },
])
def test_preference_config_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        PreferenceConfig(**kwargs)
