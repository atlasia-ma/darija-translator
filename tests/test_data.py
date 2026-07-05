from darija_translator.config import DataConfig
from darija_translator.data import to_conversations


def test_to_conversations_builds_system_user_assistant_turns():
    batch = {"english": ["Hello"], "darija": ["Salam"]}
    config = DataConfig()

    result = to_conversations(batch, config)

    assert result["conversations"] == [[
        {
            "role": "system",
            "content": config.system_prompt
        },
        {
            "role": "user",
            "content": "Hello"
        },
        {
            "role": "assistant",
            "content": "Salam"
        },
    ]]


def test_to_conversations_strips_whitespace():
    batch = {"english": ["  Hello  "], "darija": ["  Salam  "]}
    config = DataConfig()

    result = to_conversations(batch, config)

    assert result["conversations"][0][1]["content"] == "Hello"
    assert result["conversations"][0][2]["content"] == "Salam"


def test_to_conversations_handles_multiple_pairs():
    batch = {"english": ["Hi", "Bye"], "darija": ["Ahlan", "Bslama"]}
    config = DataConfig()

    result = to_conversations(batch, config)

    assert len(result["conversations"]) == 2
