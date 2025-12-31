"""Transformer model for predicting match win probability from point sequences."""

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class PointImportanceModel(nn.Module):
    """
    Transformer model that predicts win probability at each point in a match.

    Input: (batch, seq_len, n_features) - sequence of point features
    Output: (batch, seq_len) - win probability after each point
    """

    def __init__(
        self,
        n_features: int = 9,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        max_len: int = 500,
    ):
        super().__init__()

        self.d_model = d_model

        self.input_proj = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, x, lengths=None):
        """
        Args:
            x: (batch, seq_len, n_features)
            lengths: (batch,) actual sequence lengths (optional, for masking)

        Returns:
            probs: (batch, seq_len) win probability at each point
        """
        batch_size, seq_len, _ = x.shape

        x = self.input_proj(x)
        x = self.pos_encoder(x)

        # Causal mask: each point only sees previous points
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device), diagonal=1
        ).bool()

        # Padding mask for variable-length sequences
        padding_mask = None
        if lengths is not None:
            padding_mask = torch.arange(seq_len, device=x.device).expand(
                batch_size, seq_len
            ) >= lengths.unsqueeze(1)

        x = self.transformer(x, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.output_head(x).squeeze(-1)
        probs = torch.sigmoid(logits)

        return probs


if __name__ == "__main__":
    model = PointImportanceModel()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    batch = torch.randn(4, 200, 9)
    lengths = torch.tensor([200, 150, 180, 190])

    probs = model(batch, lengths)
    print(f"Input shape: {batch.shape}")
    print(f"Output shape: {probs.shape}")
    print(f"Output range: [{probs.min():.3f}, {probs.max():.3f}]")
