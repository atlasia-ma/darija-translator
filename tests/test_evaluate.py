from darija_translator.evaluate import compute_translation_metrics


def test_identical_predictions_score_near_perfect():
    predictions = ["Salam khouya"]
    references = ["Salam khouya"]

    metrics = compute_translation_metrics(predictions, references)

    assert metrics["bleu"] > 99
    assert metrics["chrf"] > 99


def test_completely_wrong_predictions_score_low():
    predictions = ["Something totally unrelated"]
    references = ["Salam khouya labas"]

    metrics = compute_translation_metrics(predictions, references)

    assert metrics["bleu"] < 20


def test_returns_both_bleu_and_chrf_keys():
    metrics = compute_translation_metrics(["hi"], ["hi"])
    assert set(metrics.keys()) == {"bleu", "chrf"}
