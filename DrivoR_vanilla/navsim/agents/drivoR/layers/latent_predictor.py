import torch
import torch.nn as nn

from ..transformer_decoder import Block


class LatentPredictor(nn.Module):
    """Predict next-frame scene tokens via self-attn on scene tokens +
    cross-attn to ego state."""

    def __init__(self, d_model=256, nhead=4, num_layers=2, d_ffn=512, ego_dim=11):
        super().__init__()
        self.ego_proj = nn.Sequential(
            nn.Linear(ego_dim, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        mlp_ratio = d_ffn / d_model
        self.blocks = nn.ModuleList([
            Block(dim=d_model, num_heads=nhead, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, d_model)

    def forward(self, scene_tokens_t, ego_t):
        """
        Args:
            scene_tokens_t: (B, N_tokens, D) scene tokens at time t
            ego_t: (B, ego_dim) ego state at time t
        Returns:
            (B, N_tokens, D) predicted scene tokens at time t+1
        """
        ego_kv = self.ego_proj(ego_t).unsqueeze(1)  # (B, 1, D)
        x = scene_tokens_t
        for blk in self.blocks:
            x = blk(x, ego_kv)
        return self.head(x)
