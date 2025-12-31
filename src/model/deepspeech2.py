import torch
from torch import nn


class DeepSpeech2(nn.Module):
    """
    Simplified DeepSpeech2-like model, inspired by https://proceedings.mlr.press/v48/amodei16.pdf
    """
    def __init__(
            self,
            n_tokens: int = 28,
            dim: int = 512,
            n_feats: int = 128,
            n_channels: int = 48,
            gru_layers: int = 3,
            dropout: float = 0.3,
    ):
        super().__init__()

        self.dim = dim

        self.extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=n_channels,
                kernel_size=(41, 11),
                stride=(2, 2),
                padding=(20, 5),
                bias=False,
            ),
            nn.BatchNorm2d(n_channels),
            nn.Hardtanh(0, 20, inplace=True),

            nn.Conv2d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(21, 11),
                stride=(2, 1),
                padding=(10, 5),
                bias=False,
            ),
            nn.BatchNorm2d(n_channels),
            nn.Hardtanh(0, 20, inplace=True),
        )

        self.freq_reduction = 4
        self.time_reduction = 2

        rnn_input_size = (n_feats // self.freq_reduction) * n_channels

        self.rnn_layers = nn.ModuleList()

        for i in range(gru_layers):
            gru = nn.GRU(
                input_size=rnn_input_size if i == 0 else dim,
                hidden_size=dim,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0
            )
            self.rnn_layers.append(gru)

            if i < gru_layers - 1:
                self.rnn_layers.append(nn.BatchNorm1d(dim))
                if dropout > 0:
                    self.rnn_layers.append(nn.Dropout(dropout))

        self.fc = nn.Linear(dim, n_tokens, bias=True)

        self._init_weights()

    def _init_weights(self):
        """
        Initialize model weights for stable and efficient training.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

    def forward(
            self,
            spectrogram: torch.FloatTensor,
            spectrogram_length: torch.LongTensor,
            **batch,
    ):
        """
        Forward pass

        Args:
            spectrogram: (B, n_feats, T) - input mel spectrogram
            spectrogram_length: (B,) - sequence lengths

        Returns:
            dict with:
                - log_probs: (B, T', n_tokens)
                - log_probs_length: (B,)
        """
        spectrogram = spectrogram.unsqueeze(1)

        feats = self.extractor(spectrogram)

        B, C, F, T = feats.shape
        feats = feats.permute(0, 3, 1, 2).contiguous()
        feats = feats.view(B, T, C * F)

        for layer in self.rnn_layers:
            if isinstance(layer, nn.GRU):
                feats, _ = layer(feats)
                feats = feats[..., :self.dim] + feats[..., self.dim:]
            elif isinstance(layer, nn.BatchNorm1d):
                feats = layer(feats.transpose(1, 2)).transpose(1, 2)

            elif isinstance(layer, nn.Dropout):
                feats = layer(feats)

        logits = self.fc(feats)
        log_probs = nn.functional.log_softmax(logits, dim=-1)

        log_probs_length = spectrogram_length // self.time_reduction

        return {
            "log_probs": log_probs,
            "log_probs_length": log_probs_length,
        }

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
