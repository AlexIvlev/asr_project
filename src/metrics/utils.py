import editdistance


def calc_cer(target_text: str, predicted_text: str) -> float:
    if len(target_text) == 0:
        return 0.0 if len(predicted_text) == 0 else 1.0

    return editdistance.eval(target_text, predicted_text) / len(target_text)


def calc_wer(target_text: str, predicted_text: str) -> float:
    target_words = target_text.split()
    pred_words = predicted_text.split()

    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0

    return editdistance.eval(target_words, pred_words) / len(target_words)
