import torch
import torch.nn as nn


class LatentPredictor(nn.Module):
    """
    Lightweight transformer that predicts next-frame scene tokens
    from current-frame scene tokens + ego state conditioning.
    """

    def __init__(self, d_model=256, nhead=4, num_layers=2, d_ffn=512, ego_dim=11):
        super().__init__()
        self.ego_proj = nn.Sequential(
            nn.Linear(ego_dim, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ffn,
            batch_first=True,
            dropout=0.1,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, scene_tokens_t, ego_t):
        """
        Args:
            scene_tokens_t: (B, N_tokens, D) scene tokens at time t
            ego_t: (B, ego_dim) ego state at time t
        Returns:
            (B, N_tokens, D) predicted scene tokens at time t+1
        """
        ego_token = self.ego_proj(ego_t).unsqueeze(1)  # (B, 1, D)
        x = torch.cat([ego_token, scene_tokens_t], dim=1)  # (B, N+1, D)
        x = self.transformer(x)
        return x[:, 1:, :]  # drop ego token
