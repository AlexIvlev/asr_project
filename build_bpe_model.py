import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import youtokentome as yttm

LIBRI_PATH = "data/datasets/librispeech"
BPE_DATASETS = {"train-clean-100", "train-clean-360", "train-other-500"}
BPE_MODEL_PATH = "bpe_model/bpe.model"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def build_bpe_model(dataset_dir, datasets, save_path, vocab_size = 500):
    """Build a BPE model on text from given datasets."""
    texts = []

    for name in datasets:
        json_file = dataset_dir / f"{name}_index.json"
        if not json_file.exists():
            raise FileNotFoundError(f"Dataset not found: {json_file}")
        with json_file.open(encoding="utf-8") as f:
            texts.extend([item["text"] for item in json.load(f)])

    with NamedTemporaryFile("w", encoding="utf-8", delete=False) as tmp:
        tmp.write("\n".join(texts))
        tmp_path = tmp.name

    save_path.parent.mkdir(parents=True, exist_ok=True)
    yttm.BPE.train(data=tmp_path, vocab_size=vocab_size, model=str(save_path))

    Path(tmp_path).unlink()

    logging.info(f"Done! Build BPE model on {len(texts)} sentences, saved to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Build a BPE model from JSON datasets.")
    parser.add_argument("--datasets", type=str, nargs="+", default=BPE_DATASETS)
    parser.add_argument("--dataset-dir", type=Path, default=LIBRI_PATH)
    parser.add_argument("--model-save-path", type=Path, default=BPE_MODEL_PATH)
    parser.add_argument("--vocab-size", type=int, default=500)
    args = parser.parse_args()

    build_bpe_model(args.dataset_dir, args.datasets, args.model_save_path, args.vocab_size)
