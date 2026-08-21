from darija_translator.config import InferenceConfig
from darija_translator.inference import (
    batched,
    build_translation_messages,
    encode_prompts,
    describe_decoding,
    group_candidates,
    prepare_tokenizer_for_generation,
    to_generation_record,
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


def test_describe_decoding_names_greedy():
    assert describe_decoding(InferenceConfig()) == "greedy"


def test_describe_decoding_records_the_sampling_parameters():
    config = InferenceConfig(do_sample=True, temperature=0.7, top_p=0.9)

    assert describe_decoding(config) == "sample(temperature=0.7, top_p=0.9)"


def test_generation_record_keeps_the_prompt_and_the_translation():
    record = to_generation_record("Hello", ["salam"], "Translate.",
                                  InferenceConfig())

    assert record["prompt"] == [
        {
            "role": "system",
            "content": "Translate."
        },
        {
            "role": "user",
            "content": "Hello"
        },
    ]
    assert record["english"] == "Hello"
    assert record["generated"] == "salam"
    assert record["candidates"] == ["salam"]


def test_generation_record_keeps_every_candidate():
    config = InferenceConfig(do_sample=True, num_generations=2)

    record = to_generation_record("Hello", ["salam", "ahlan"], "Translate.",
                                  config)

    assert record["generated"] == "salam"
    assert record["candidates"] == ["salam", "ahlan"]


def test_generation_record_drops_blank_candidates():
    record = to_generation_record("Hello", ["", "  ", "salam"], "Translate.",
                                  InferenceConfig())

    assert record["candidates"] == ["salam"]


def test_generation_record_is_none_when_nothing_was_generated():
    assert to_generation_record("Hello", ["", "  "], "Translate.",
                                InferenceConfig()) is None


def test_generation_record_is_none_without_a_source():
    assert to_generation_record("   ", ["salam"], "Translate.",
                                InferenceConfig()) is None


def test_generation_record_records_what_produced_it():
    config = InferenceConfig(adapter_model_id="atlasia/edge",
                             adapter_subfolder="last-checkpoint")

    record = to_generation_record("Hello", ["salam"], "Translate.", config)

    assert record["adapter"] == "atlasia/edge/last-checkpoint"
    assert record["decoding"] == "greedy"


def test_generation_record_carries_metadata():
    record = to_generation_record("Hello", ["salam"],
                                  "Translate.",
                                  InferenceConfig(),
                                  metadata={"row_index": 3})

    assert record["row_index"] == 3
