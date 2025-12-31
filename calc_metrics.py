import argparse
import json
import logging
from pathlib import Path
from src.metrics.utils import calc_cer, calc_wer

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)

def read_output(path: Path):
    """Read JSON files in format {'prediction': ..., 'target': ...}"""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data["prediction"], data["target"]


def normalize_text(text: str):
    import re
    return re.sub(r"[^a-z ]", "", text.lower())


def calc_metrics(pred_dir: Path):
    files = sorted(pred_dir.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {pred_dir}")

    cer_sum, wer_sum = 0.0, 0.0

    for f in files:
        pred_text, target_text = read_output(f)
        target_text_norm = normalize_text(target_text)
        cer_sum += calc_cer(target_text_norm, pred_text)
        wer_sum += calc_wer(target_text_norm, pred_text)

    n = len(files)
    return cer_sum / n, wer_sum / n


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", type=Path, required=True, help="Directory with JSON outputs")
    args = parser.parse_args()

    cer, wer = calc_metrics(args.pred_dir)
    logging.info(f"CER: {cer*100:.2f}%")
    logging.info(f"WER: {wer*100:.2f}%")
