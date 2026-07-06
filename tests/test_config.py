from darija_translator.config import DataConfig, ModelConfig


def test_data_config_default_system_prompt():
    cfg = DataConfig()
    assert cfg.system_prompt == "You are a professional English to Darija translator."


def test_data_config_system_prompt_is_overridable():
    cfg = DataConfig(system_prompt="Translate casually.")
    assert cfg.system_prompt == "Translate casually."


def test_data_config_default_max_text_length():
    cfg = DataConfig()
    assert cfg.max_text_length == 2000


def test_data_config_max_text_length_is_overridable():
    cfg = DataConfig(max_text_length=500)
    assert cfg.max_text_length == 500


def test_data_config_default_test_size():
    cfg = DataConfig()
    assert cfg.test_size == 0.1


def test_data_config_default_seed():
    cfg = DataConfig()
    assert cfg.seed == 3407


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.model_name == "LiquidAI/LFM2.5-230M"
    assert cfg.max_seq_length == 2048
    assert cfg.load_in_16bit is True
    assert cfg.lora_r == 16
    assert cfg.lora_alpha == 16
    assert cfg.lora_dropout == 0
    assert cfg.random_state == 3407
    assert cfg.target_modules == (
        "q_proj",
        "k_proj",
        "v_proj",
        "out_proj",
        "in_proj",
        "w1",
        "w2",
        "w3",
    )
