import re

from torch import Tensor

from src.metrics.base_metric import BaseMetric
from src.metrics.utils import calc_wer


class WERMetric(BaseMetric):
    def __init__(self, text_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.text_encoder = text_encoder

    @staticmethod
    def _normalize(text):
        return re.sub(r"[^a-z ]", "", text.lower())

    def __call__(
        self, log_probs: Tensor, log_probs_length: Tensor, text: list[str], **kwargs
    ):
        predictions = self.text_encoder(log_probs, log_probs_length)

        wers = [
            calc_wer(self._normalize(target_text), pred_texts[0])
            for pred_texts, target_text in zip(predictions, text)
        ]

        return sum(wers) / len(wers)
