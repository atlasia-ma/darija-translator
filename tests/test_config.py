from darija_translator.config import DataConfig


def test_data_config_default_system_prompt():
    cfg = DataConfig()
    assert cfg.system_prompt == "You are a professional English to Darija translator."


def test_data_config_system_prompt_is_overridable():
    cfg = DataConfig(system_prompt="Translate casually.")
    assert cfg.system_prompt == "Translate casually."
