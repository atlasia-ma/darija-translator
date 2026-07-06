from darija_translator.config import DataConfig


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
