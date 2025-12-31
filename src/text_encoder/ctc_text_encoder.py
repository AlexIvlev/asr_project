import re
from collections import defaultdict
from enum import Enum
from pathlib import Path
from string import ascii_lowercase

import torch
from pyctcdecode import build_ctcdecoder
from torchaudio.models.decoder import ctc_decoder, download_pretrained_files

LM_NAME = "librispeech-4-gram"


class DecoderType(Enum):
    ARGMAX = "argmax"
    BS = "bs"
    BS_TORCH = "bs_torch"
    BS_LM = "bs_lm"


class CTCTextEncoder:
    EMPTY_TOK = ""
    SIL_TOKEN = " "

    def __init__(
        self,
        alphabet=None,
        decoder_type=DecoderType.ARGMAX,
        beam_size=10,
        **kwargs,  # noqa
    ):
        if isinstance(decoder_type, str):
            decoder_type = DecoderType(decoder_type)

        self.decoder_type = decoder_type
        self.beam_size = beam_size

        self.alphabet = alphabet or list(ascii_lowercase + " ")
        self.vocab = [self.EMPTY_TOK] + self.alphabet

        self.ind2char = {i: c for i, c in enumerate(self.vocab)}
        self.char2ind = {c: i for i, c in self.ind2char.items()}

        if self.decoder_type == DecoderType.BS_TORCH:
            self._init_torch_decoder()
        elif self.decoder_type == DecoderType.BS_LM:
            self._init_lm_decoder()

    def _init_torch_decoder(self):
        self.ctc_decoder = ctc_decoder(
            lexicon=None,
            tokens=self.vocab,
            blank_token=self.EMPTY_TOK,
            sil_token=self.SIL_TOKEN,
            nbest=1,
            beam_size=self.beam_size,
            beam_threshold=5.0,
        )

    def _init_lm_decoder(self):
        files = download_pretrained_files(LM_NAME)

        with Path(files.lexicon).open() as f:
            unigrams = [
                line.split("\t")[0].replace("'", "").lower().strip() for line in f
            ]

        self.lm_decoder = build_ctcdecoder(
            self.vocab,
            kenlm_model_path=files.lm,
            unigrams=unigrams,
        )

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, item: int):
        assert type(item) is int
        return self.ind2char[item]

    def __call__(
        self, log_probs: torch.FloatTensor, log_probs_length: torch.FloatTensor
    ) -> list[str] | list[list[str]]:
        if self.decoder_type == DecoderType.ARGMAX:
            preds = log_probs.argmax(-1).cpu().numpy()
            return [[self.ctc_decode(p[:l])] for p, l in zip(preds, log_probs_length)]

        if self.decoder_type == DecoderType.BS:
            results = []
            for lp, l in zip(log_probs, log_probs_length):
                hyps = self.ctc_beam_search(lp.exp(), int(l), self.beam_size)
                results.append([text for text, prob in hyps])
            return results

        if self.decoder_type == DecoderType.BS_TORCH:
            return [
                [self.decode(h.tokens).strip()]
                for h in self.ctc_decoder(log_probs.cpu(), log_probs_length)
            ]

        if self.decoder_type == DecoderType.BS_LM:
            results = []
            for lp, l in zip(log_probs.cpu().numpy(), log_probs_length):
                beams = self.lm_decoder.decode_beams(lp[:l], beam_width=self.beam_size)
                results.append([b[0] for b in beams])
            return results

        raise ValueError("Unsupported decoder type!")

    def encode(self, text) -> torch.Tensor:
        text = self.normalize_text(text)
        try:
            return torch.Tensor([self.char2ind[char] for char in text]).unsqueeze(0)
        except KeyError:
            unknown_chars = set([char for char in text if char not in self.char2ind])
            raise Exception(
                f"Can't encode text '{text}'. Unknown chars: '{' '.join(unknown_chars)}'"
            )

    def decode(self, inds) -> str:
        """
        Raw decoding without CTC.
        Used to validate the CTC decoding implementation.

        Args:
            inds (list): list of tokens.
        Returns:
            raw_text (str): raw text with empty tokens and repetitions.
        """
        return "".join([self.ind2char[int(ind)] for ind in inds]).strip()

    def ctc_decode(self, inds) -> str:
        blank = self.char2ind[self.EMPTY_TOK]
        decoded = []
        prev = blank

        for i in inds:
            i = int(i)
            if i != prev and i != blank:
                decoded.append(self.ind2char[i])
            prev = i

        return "".join(decoded)

    def ctc_beam_search(
        self, probs: torch.FloatTensor, length: int, beam_size: int
    ) -> list[tuple[str, float]]:
        """
        Vanilla CTC beam search decoder.

        Args:
            probs: probability matrix of shape (T, vocab_size)
            length: actual length of the sequence
            beam_size: number of beams to keep
        Returns:
            list of (text, probability) tuples sorted by probability
        """
        dp = {("", self.EMPTY_TOK): 1.0}

        for t in range(length):
            next_dp = defaultdict(float)
            current_probs = probs[t].cpu().numpy()

            for (prefix, last_char), prefix_prob in dp.items():
                for char_idx, char_prob in enumerate(current_probs):
                    if char_prob < 1e-10:
                        continue

                    cur_char = self.ind2char[char_idx]
                    new_prob = prefix_prob * char_prob

                    if cur_char == self.EMPTY_TOK or cur_char == last_char:
                        new_prefix = prefix
                    else:
                        new_prefix = prefix + cur_char

                    next_dp[(new_prefix, cur_char)] += new_prob

            dp = dict(sorted(next_dp.items(), key=lambda x: -x[1])[:beam_size])

        final_beams = defaultdict(float)
        for (prefix, _), prob in dp.items():
            final_beams[prefix] += prob

        return sorted(final_beams.items(), key=lambda x: -x[1])

    @staticmethod
    def normalize_text(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z ]", "", text)
        return text
