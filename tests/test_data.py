from darija_translator.config import DataConfig
from darija_translator.data import to_conversations, format_conversations, is_darija_script


class FakeTokenizer:
    bos_token = "<bos>"

    def apply_chat_template(self,
                            conversations,
                            tokenize=False,
                            add_generation_prompt=False):
        # mimic real behavior: one formatted string per conversation, bos-prefixed
        return [
            self.bos_token + "".join(f"[{turn['role']}]{turn['content']}"
                                     for turn in convo)
            for convo in conversations
        ]


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


def test_format_conversations_returns_text_field():
    examples = {
        "conversations": [[{
            "role": "user",
            "content": "Hi"
        }, {
            "role": "assistant",
            "content": "Salam"
        }]]
    }
    tokenizer = FakeTokenizer()

    result = format_conversations(examples, tokenizer)

    assert "text" in result
    assert len(result["text"]) == 1


def test_format_conversations_strips_bos_token_prefix():
    examples = {
        "conversations": [[{
            "role": "user",
            "content": "Hi"
        }, {
            "role": "assistant",
            "content": "Salam"
        }]]
    }
    tokenizer = FakeTokenizer()

    result = format_conversations(examples, tokenizer)

    assert not result["text"][0].startswith(tokenizer.bos_token)
    assert result["text"][0] == "[user]Hi[assistant]Salam"


def test_format_conversations_handles_multiple_examples():
    examples = {
        "conversations": [
            [{
                "role": "user",
                "content": "Hi"
            }],
            [{
                "role": "user",
                "content": "Bye"
            }],
        ]
    }
    tokenizer = FakeTokenizer()

    result = format_conversations(examples, tokenizer)

    assert len(result["text"]) == 2


def test_format_conversations_handles_tokenizer_without_bos_token():

    class NoBosTokenizer(FakeTokenizer):
        bos_token = None

        def apply_chat_template(self,
                                conversations,
                                tokenize=False,
                                add_generation_prompt=False):
            return [
                "".join(f"[{t['role']}]{t['content']}" for t in convo)
                for convo in conversations
            ]

    examples = {"conversations": [[{"role": "user", "content": "Hi"}]]}

    result = format_conversations(examples, NoBosTokenizer())

    assert result["text"][0] == "[user]Hi"


def test_is_darija_script_accepts_darija():
    example = {"script_type": "darija"}
    assert is_darija_script(example) is True


def test_is_darija_script_accepts_both():
    example = {"script_type": "both"}
    assert is_darija_script(example) is True


def test_is_darija_script_rejects_other_script_types():
    example = {"script_type": "latin"}
    assert is_darija_script(example) is False
