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
