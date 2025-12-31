from torch import Tensor, nn
import torchaudio.transforms as T


class TimeMask(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._aug = T.TimeMasking(*args, **kwargs)

    def __call__(self, spec: Tensor):
        return self._aug(spec)
