from darija_translator.config import DataConfig


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
