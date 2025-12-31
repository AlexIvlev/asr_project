import torch
import youtokentome as yttm
from src.text_encoder.ctc_text_encoder import CTCTextEncoder, DecoderType


class BPECTCTextEncoder(CTCTextEncoder):
    EMPTY_TOK = "<PAD>"
    SIL_TOKEN = "▁"

    def __init__(  # noqa
            self,
            bpe_model_path,
            decoder_type = DecoderType.ARGMAX,
            **kwargs,  # noqa
    ):
        if isinstance(decoder_type, str):
            decoder_type = DecoderType(decoder_type)

        self.decoder_type = decoder_type
        self.bpe = yttm.BPE(model=bpe_model_path)

        self.vocab = self.bpe.vocab()
        self.ind2char = {i: token for i, token in enumerate(self.vocab)}
        self.char2ind = {token: i for i, token in enumerate(self.vocab)}

        self.blank_idx = self.char2ind[self.EMPTY_TOK]
        self.empty_id = self.blank_idx

    def encode(self, text):
        ids = self.bpe.encode([text], output_type=yttm.OutputType.ID)[0]
        return torch.tensor(ids).unsqueeze(0)

    def decode(self, inds):
        if not len(inds):
            return ""

        if isinstance(inds, torch.Tensor):
            inds = inds.tolist()
        elif not isinstance(inds, list):
            inds = list(inds)

        result = self.bpe.decode([inds])
        return result[0] if result else ""

    def ctc_decode(self, inds):
        decoded = []
        prev = None

        for idx in inds:
            idx = int(idx)
            if idx != self.blank_idx and idx != prev:
                decoded.append(idx)
            prev = idx

        return self.decode(decoded)

    @property
    def n_tokens(self):
        return len(self.vocab)
