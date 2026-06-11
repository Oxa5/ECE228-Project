from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

class PilotNetEncoder(nn.Module):
    def __init__(self, feature_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, kernel_size=5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, kernel_size=3, stride=1), nn.ELU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ELU(),
        )
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.LazyLinear(feature_dim)
        self.feature_dim = feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x)
        z = torch.flatten(z, 1)
        z = self.dropout(z)
        z = F.elu(self.proj(z))
        return z

class PilotNet(nn.Module):
    def __init__(self, feature_dim=256, dropout=0.3, **_):
        super().__init__()
        self.encoder = PilotNetEncoder(feature_dim, dropout)
        self.head = nn.Sequential(
            nn.Linear(feature_dim, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, 1),
        )
        self.last_attention = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        last = x[:, -1]
        z = self.encoder(last)
        return self.head(z)


class CNN_LSTM(nn.Module):
    """CNN per frame -> LSTM; regress from the final hidden state (the proposal baseline)."""

    def __init__(self, feature_dim=256, lstm_hidden=128, lstm_layers=1, dropout=0.3, **_):
        super().__init__()
        self.encoder = PilotNetEncoder(feature_dim, dropout)
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.last_attention = None

    def _encode_sequence(self, x):
        B, T = x.shape[:2]
        flat = x.reshape(B * T, *x.shape[2:])
        feats = self.encoder(flat)
        return feats.view(B, T, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._encode_sequence(x)
        out, (h_n, _) = self.lstm(feats)
        last = out[:, -1]
        return self.head(last)

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim: int, attn_dim: int = 64):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, attn_dim), nn.Tanh(),
            nn.Linear(attn_dim, 1),
        )

    def forward(self, seq: torch.Tensor):
        # seq: (B, T, H)
        scores = self.score(seq).squeeze(-1)
        weights = torch.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)
        return context, weights

class CNN_LSTM_Attention(nn.Module):
    def __init__(self, feature_dim=256, lstm_hidden=128, lstm_layers=1,
                 attention_dim=64, dropout=0.3, **_):
        super().__init__()
        self.encoder = PilotNetEncoder(feature_dim, dropout)
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.attention = TemporalAttention(lstm_hidden, attention_dim)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.last_attention = None

    def _encode_sequence(self, x):
        B, T = x.shape[:2]
        flat = x.reshape(B * T, *x.shape[2:])
        feats = self.encoder(flat)
        return feats.view(B, T, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._encode_sequence(x)
        out, _ = self.lstm(feats)
        context, weights = self.attention(out)
        self.last_attention = weights.detach()
        return self.head(context)

class CNN_LSTM_FeatAttention(nn.Module):
    def __init__(self, feature_dim=256, lstm_hidden=128, lstm_layers=1,
                 attention_dim=64, dropout=0.3, **_):
        super().__init__()
        self.encoder = PilotNetEncoder(feature_dim, dropout)
        self.feat_attention = TemporalAttention(feature_dim, attention_dim)
        self.lstm = nn.LSTM(
            input_size=feature_dim, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden, 64), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.last_attention = None

    def _encode_sequence(self, x):
        B, T = x.shape[:2]
        flat = x.reshape(B * T, *x.shape[2:])
        feats = self.encoder(flat)
        return feats.view(B, T, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self._encode_sequence(x)
        _, weights = self.feat_attention(feats)
        self.last_attention = weights.detach()
        out, _ = self.lstm(feats)
        context = torch.bmm(weights.unsqueeze(1), out).squeeze(1)  # (B, H)
        return self.head(context)

_MODELS = {
    "pilotnet": PilotNet,
    "cnn_lstm": CNN_LSTM,
    "cnn_lstm_attention": CNN_LSTM_Attention,
    "cnn_lstm_featattn": CNN_LSTM_FeatAttention,
}

def build_model(name: str, model_cfg: dict) -> nn.Module:
    name = name.lower()
    if name not in _MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from {list(_MODELS)}.")
    kwargs = dict(
        feature_dim=model_cfg.get("cnn_feature_dim", 256),
        lstm_hidden=model_cfg.get("lstm_hidden", 128),
        lstm_layers=model_cfg.get("lstm_layers", 1),
        attention_dim=model_cfg.get("attention_dim", 64),
        dropout=model_cfg.get("dropout", 0.3),
    )
    return _MODELS[name](**kwargs)

def all_model_names():
    return list(_MODELS.keys())
