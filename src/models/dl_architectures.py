"""
Deep Learning Model Architectures for rPPG-based Deepfake Detection.

These architectures match exactly those trained in ml-dl.ipynb notebook.
All models take 35-dimensional rPPG feature vectors as input.

Model architectures:
    1. DeepfakeCNN1D - 1D CNN treating features as a signal
    2. DeepfakeBiLSTM - BiLSTM with attention mechanism
    3. DeepfakeCNNBiLSTM - Hybrid CNN + BiLSTM
    4. DeepfakeTransformer - Transformer encoder
    5. PhysNetMLP - Deep residual MLP (DeepFakesON-Phys inspired)
"""
import numpy as np
import torch
import torch.nn as nn


class DeepfakeCNN1D(nn.Module):
    """
    1D Convolutional Neural Network for rPPG feature classification.
    Treats the 35 features as a 1D signal and applies conv filters
    to capture local feature interactions.

    Architecture:
        Input (B, 35) -> reshape (B, 1, 35)
        -> Conv1d(1, 32) -> BN -> ReLU -> Dropout
        -> Conv1d(32, 64) -> BN -> ReLU -> Dropout
        -> Conv1d(64, 128) -> BN -> ReLU -> AdaptiveAvgPool
        -> Flatten -> Linear(128, 64) -> ReLU -> Dropout
        -> Linear(64, 2) -> Output
    """
    def __init__(self, n_features=35, dropout=0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # (batch, 1, 35)
        x = self.features(x)
        x = self.classifier(x)
        return x


class Attention(nn.Module):
    """Scaled dot-product attention over LSTM outputs."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        scores = self.attn(lstm_out).squeeze(-1)
        weights = torch.softmax(scores, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), lstm_out).squeeze(1)
        return context, weights


class DeepfakeBiLSTM(nn.Module):
    """
    Bidirectional LSTM with attention for rPPG features.
    Reshapes 35 features into 7 groups of 5 for sequential processing.

    Architecture:
        Input (B, 35) -> reshape (B, 7, 5)
        -> Linear(5, 64) -> BiLSTM(64, 64, 2 layers)
        -> Attention -> Linear(128, 64) -> ReLU -> Dropout
        -> Linear(64, 2) -> Output
    """
    def __init__(self, n_features=35, hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.seq_len = 7
        self.feat_per_step = 5

        self.input_proj = nn.Linear(self.feat_per_step, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            bidirectional=True, dropout=dropout if n_layers > 1 else 0,
        )
        self.attention = Attention(hidden_dim * 2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.feat_per_step)
        x = self.input_proj(x)
        lstm_out, _ = self.lstm(x)
        context, _ = self.attention(lstm_out)
        return self.classifier(context)


class DeepfakeCNNBiLSTM(nn.Module):
    """
    Hybrid: 1D-CNN extracts local feature patterns,
    BiLSTM captures sequential dependencies.

    Architecture:
        Input (B, 35) -> reshape (B, 1, 35)
        -> Conv1d(1, 32) -> BN -> ReLU
        -> Conv1d(32, 64) -> BN -> ReLU
        -> permute -> BiLSTM(64, 64, 2 layers) -> Attention
        -> Linear(128, 64) -> ReLU -> Dropout
        -> Linear(64, 2) -> Output
    """
    def __init__(self, n_features=35, cnn_channels=32, lstm_hidden=64,
                 lstm_layers=2, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(1, cnn_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm1d(cnn_channels * 2),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_channels * 2, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True,
            bidirectional=True, dropout=dropout if lstm_layers > 1 else 0,
        )
        self.attention = Attention(lstm_hidden * 2)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.permute(0, 2, 1)
        lstm_out, _ = self.lstm(cnn_out)
        context, _ = self.attention(lstm_out)
        return self.classifier(context)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer."""
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class DeepfakeTransformer(nn.Module):
    """
    Transformer Encoder for rPPG feature classification.
    Reshapes 35 features into 7 groups of 5, applies multi-head
    self-attention to learn cross-group feature interactions.

    Architecture:
        Input (B, 35) -> reshape (B, 7, 5)
        -> Linear(5, 32) -> PositionalEncoding
        -> TransformerEncoder(2 layers, 4 heads)
        -> GlobalAveragePool -> LayerNorm
        -> Linear(32, 64) -> GELU -> Dropout
        -> Linear(64, 2) -> Output
    """
    def __init__(self, n_features=35, d_model=32, nhead=4,
                 num_layers=2, dim_ff=128, dropout=0.3):
        super().__init__()
        self.seq_len = 7
        self.feat_per_step = 5

        self.input_proj = nn.Linear(self.feat_per_step, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=self.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, self.seq_len, self.feat_per_step)
        x = self.input_proj(x)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.mean(dim=1)  # Global average pooling
        return self.classifier(x)


class ResidualBlock(nn.Module):
    """Residual block with pre-activation BN -> GELU pattern."""
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(x + self.block(x))


class PhysNetMLP(nn.Module):
    """
    Deep Residual MLP inspired by DeepFakesON-Phys CAN architecture.
    Uses residual connections for stable training.

    Architecture:
        Input (B, 35) -> Linear(35, 128) -> BN -> GELU -> Dropout
        -> ResidualBlock x3 -> Linear(128, 64) -> GELU -> Dropout
        -> Linear(64, 2) -> Output
    """
    def __init__(self, n_features=35, hidden_dim=128, n_blocks=3, dropout=0.3):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, dropout * 0.7) for _ in range(n_blocks)]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.res_blocks(x)
        return self.classifier(x)


# Model name to class mapping (matches checkpoint filenames)
DL_MODEL_CLASSES = {
    "CNN_1D": DeepfakeCNN1D,
    "BiLSTM_Attention": DeepfakeBiLSTM,
    "CNN_BiLSTM": DeepfakeCNNBiLSTM,
    "Transformer": DeepfakeTransformer,
    "PhysNet_MLP": PhysNetMLP,
}


def get_dl_model(name, n_features=35):
    """
    Factory function to create a DL model by name.

    Args:
        name: Model name (CNN_1D, BiLSTM_Attention, CNN_BiLSTM,
              Transformer, PhysNet_MLP)
        n_features: Input feature dimension (default 35)

    Returns:
        Instantiated model (nn.Module)
    """
    if name not in DL_MODEL_CLASSES:
        raise ValueError(f"Unknown model: {name}. Available: {list(DL_MODEL_CLASSES.keys())}")
    return DL_MODEL_CLASSES[name](n_features=n_features)
