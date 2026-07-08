from sacrebleu import BLEU, CHRF


def compute_translation_metrics(predictions: list[str],
                                references: list[str]) -> dict:
    bleu_score = BLEU(effective_order=True).corpus_score(
        predictions, [references])
    chrf_score = CHRF().corpus_score(predictions, [references])
    return {"bleu": bleu_score.score, "chrf": chrf_score.score}


def generate_translations(model, tokenizer, sources: list[str],
                          system_prompt: str) -> list[str]:
    predictions = []
    for text in sources:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": text
            },
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        input_len = inputs["input_ids"].shape[1]
        outputs = model.generate(**inputs,
                                 max_new_tokens=256,
                                 do_sample=False,
                                 use_cache=True)
        new_tokens = outputs[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        predictions.append(decoded[0])
    return predictions
