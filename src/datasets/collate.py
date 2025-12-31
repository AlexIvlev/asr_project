import torch


def collate_fn(dataset_items: list[dict], time_reduction: int = 2):
    filtered_items = []
    for item in dataset_items:
        spec_len = item["spectrogram"].shape[-1]
        text_len = item["text_encoded"].shape[-1]
        output_len = spec_len // time_reduction

        if output_len >= text_len:
            filtered_items.append(item)

    if not filtered_items:
        return None

    dataset_items = filtered_items

    text = [item["text"] for item in dataset_items]
    audio_path = [item["audio_path"] for item in dataset_items]
    text_encoded = [item["text_encoded"].squeeze(0) for item in dataset_items]
    audio = [item["audio"].squeeze(0) for item in dataset_items]
    spectrograms = [item["spectrogram"].squeeze(0) for item in dataset_items]

    audio_lengths = torch.LongTensor([x.shape[-1] for x in audio])
    spectrogram_length = torch.LongTensor([x.shape[-1] for x in spectrograms])
    text_encoded_length = torch.LongTensor([len(x) for x in text_encoded])

    text_encoded_max_length = int(text_encoded_length.max())
    mels_max_length = int(spectrogram_length.max())
    audio_max_length = int(audio_lengths.max())

    text_encoded_padded = torch.zeros(
        len(text_encoded), text_encoded_max_length, dtype=torch.long
    )
    audio_padded = torch.zeros(len(audio), audio_max_length, dtype=torch.float32)

    n_feats = spectrograms[0].shape[0]
    spectrogram_padded = torch.zeros(
        len(spectrograms), n_feats, mels_max_length, dtype=torch.float32
    )

    for i in range(len(dataset_items)):
        text_encoded_padded[i, : text_encoded[i].shape[0]] = text_encoded[i]
        audio_padded[i, : audio[i].shape[-1]] = audio[i]
        spectrogram_padded[i, :, : spectrograms[i].shape[-1]] = spectrograms[i]

    return {
        "text": text,
        "text_encoded": text_encoded_padded,
        "text_encoded_length": text_encoded_length,
        "audio_path": audio_path,
        "audio": audio_padded,
        "spectrogram": spectrogram_padded,
        "spectrogram_length": spectrogram_length,
    }
