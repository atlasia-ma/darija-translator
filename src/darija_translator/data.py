from darija_translator.config import DataConfig

from typing import Protocol


class ChatTemplateTokenizer(Protocol):
    bos_token: str

    def apply_chat_template(
        self,
        conversations: list,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[str]:
        ...


def to_conversations(batch: dict, config: DataConfig):

    conversations = []
    for english, darija in zip(batch["english"], batch["darija"]):
        conversations.append([
            {
                "role": "system",
                "content": config.system_prompt
            },
            {
                "role": "user",
                "content": english.strip()
            },
            {
                "role": "assistant",
                "content": darija.strip()
            },
        ])
    return {"conversations": conversations}


def format_conversations(examples: dict,
                         tokenizer: ChatTemplateTokenizer) -> dict:
    texts = tokenizer.apply_chat_template(
        examples["conversations"],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not tokenizer.bos_token:
        return {"text": texts}
    return {"text": [x.removeprefix(tokenizer.bos_token) for x in texts]}
