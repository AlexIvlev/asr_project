from torch import Tensor, nn
import torchaudio.transforms as T


class FreqMask(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = T.FrequencyMasking(*args, **kwargs)

    def __call__(self, spec: Tensor):
        return self._aug(spec)
