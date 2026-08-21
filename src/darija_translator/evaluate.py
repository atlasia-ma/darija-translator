from sacrebleu import BLEU, CHRF


def compute_translation_metrics(predictions: list[str],
                                references: list[str]) -> dict:
    bleu_score = BLEU(effective_order=True).corpus_score(
        predictions, [references])
    chrf_score = CHRF().corpus_score(predictions, [references])
    return {"bleu": bleu_score.score, "chrf": chrf_score.score}
