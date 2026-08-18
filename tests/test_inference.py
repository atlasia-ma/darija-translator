from darija_translator.config import InferenceConfig
from darija_translator.inference import (
    batched,
    build_translation_messages,
    encode_prompts,
    group_candidates,
    prepare_tokenizer_for_generation,
)


class FakeTokenizer:
    pad_token = "<pad>"
    eos_token = "<eos>"
    padding_side = "right"

    def __init__(self):
        self.received_kwargs = None

    def apply_chat_template(self,
                            conversations,
                            tokenize=False,
                            add_generation_prompt=False):
        assert add_generation_prompt, "generation needs the assistant prefix"
        return [
            "".join(f"[{turn['role']}]{turn['content']}" for turn in convo) +
            "[assistant]" for convo in conversations
        ]

    def __call__(self, texts, **kwargs):
        self.received_kwargs = kwargs
        return {"input_ids": texts}


def test_build_translation_messages_stops_before_the_answer():
    messages = build_translation_messages("  Hello  ", "Translate.")

    assert messages == [
        {
            "role": "system",
            "content": "Translate."
        },
        {
            "role": "user",
            "content": "Hello"
        },
    ]


def test_prepare_tokenizer_switches_to_left_padding():
    tokenizer = prepare_tokenizer_for_generation(FakeTokenizer())

    assert tokenizer.padding_side == "left"


def test_prepare_tokenizer_falls_back_to_eos_when_no_pad_token():
    tokenizer = FakeTokenizer()
    tokenizer.pad_token = None

    prepare_tokenizer_for_generation(tokenizer)

    assert tokenizer.pad_token == tokenizer.eos_token


def test_encode_prompts_does_not_re_add_special_tokens():
    tokenizer = FakeTokenizer()

    encode_prompts(tokenizer, ["Hello"], "Translate.")

    assert tokenizer.received_kwargs["add_special_tokens"] is False
    assert tokenizer.received_kwargs["padding"] is True


def test_encode_prompts_templates_every_source():
    tokenizer = FakeTokenizer()

    encoded = encode_prompts(tokenizer, ["Hello", "Bye"], "Translate.")

    assert encoded["input_ids"] == [
        "[system]Translate.[user]Hello[assistant]",
        "[system]Translate.[user]Bye[assistant]",
    ]


def test_group_candidates_keeps_generations_with_their_prompt():
    decoded = ["a1", "a2", "b1", "b2"]

    assert group_candidates(decoded, 2) == [["a1", "a2"], ["b1", "b2"]]


def test_group_candidates_with_a_single_generation_per_prompt():
    assert group_candidates(["a", "b"], 1) == [["a"], ["b"]]


def test_batched_splits_into_chunks_of_batch_size():
    assert list(batched([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]


def test_inference_config_batch_size_drives_batching():
    config = InferenceConfig(batch_size=3)

    assert list(batched(list(range(7)), config.batch_size)) == [[0, 1, 2],
                                                               [3, 4, 5], [6]]
