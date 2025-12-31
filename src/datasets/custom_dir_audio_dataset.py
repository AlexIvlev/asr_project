from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset

ALLOWED_EXTS = {".mp3", ".wav", ".flac", ".m4a"}


class CustomDirAudioDataset(BaseDataset):
    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        audio_dir = Path(audio_dir)
        transcription_dir = Path(transcription_dir) if transcription_dir else None

        data = []
        for path in sorted(audio_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in ALLOWED_EXTS:
                continue

            entry = {"path": str(path)}

            if transcription_dir:
                transcription_path = transcription_dir / f"{path.stem}.txt"
                if transcription_path.exists():
                    entry["text"] = transcription_path.read_text(
                        encoding="utf-8"
                    ).strip()

            entry.setdefault("text", "")

            info = torchaudio.info(str(path))
            entry["audio_len"] = info.num_frames / info.sample_rate
            data.append(entry)

        super().__init__(data, *args, **kwargs)
