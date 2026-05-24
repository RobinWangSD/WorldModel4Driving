import torch
import torch.nn as nn

from ..transformer_decoder import DiTBlock


class LatentPredictor(nn.Module):
    """Predict next-frame scene tokens via ego-conditioned DiT blocks."""

    def __init__(self, d_model=256, nhead=4, num_layers=2, d_ffn=512, ego_dim=11):
        super().__init__()
        self.ego_proj = nn.Sequential(
            nn.Linear(ego_dim, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        mlp_ratio = d_ffn / d_model
        self.blocks = nn.ModuleList([
            DiTBlock(dim=d_model, num_heads=nhead, mlp_ratio=mlp_ratio)
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
        ego_c = self.ego_proj(ego_t)  # (B, D)
        x = scene_tokens_t
        for blk in self.blocks:
            x = blk(x, ego_c)
        return self.head(x)


class WorldModelLatentPredictor(nn.Module):
    """Multi-step, action-conditioned scene-token predictor.

    Each of `n_pred_steps` future steps gets its own DiT pass starting from the
    current scene tokens, conditioned on (pooled ego history + per-step future
    action + learned per-step embedding). Steps are batched, so the cost scales
    linearly with `n_pred_steps`.
    """

    def __init__(self, d_model=256, nhead=4, num_layers=2, d_ffn=512, ego_dim=11,
                 n_pred_steps=10, n_tokens=16):
        super().__init__()
        self.n_pred_steps = n_pred_steps
        self.n_tokens = n_tokens

        self.ego_proj = nn.Sequential(
            nn.Linear(ego_dim, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.action_proj = nn.Sequential(
            nn.Linear(ego_dim, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )
        self.step_embed = nn.Parameter(torch.zeros(n_pred_steps, d_model))

        mlp_ratio = d_ffn / d_model
        self.blocks = nn.ModuleList([
            DiTBlock(dim=d_model, num_heads=nhead, mlp_ratio=mlp_ratio)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(d_model, d_model)

    def forward(self, cur_scene_tokens, ego_history, future_actions):
        """
        Args:
            cur_scene_tokens: (B, N, D) scene tokens at the current step
            ego_history:      (B, T_hist, ego_dim) past ego states
            future_actions:   (B, T_pred, ego_dim) future ego states (teacher forced)
        Returns:
            (B, T_pred, N, D) predicted scene-token latents per future step
        """
        B, N, D = cur_scene_tokens.shape
        T_pred = future_actions.shape[1]
        assert T_pred <= self.n_pred_steps, (
            f"future_actions has {T_pred} steps but predictor was built for at most {self.n_pred_steps}"
        )

        ego_proj = self.ego_proj(ego_history).mean(dim=1)            # (B, D)
        action_proj = self.action_proj(future_actions)               # (B, T_pred, D)
        step_cond = action_proj + self.step_embed[None, :T_pred]     # (B, T_pred, D)
        cond = ego_proj.unsqueeze(1) + step_cond                     # (B, T_pred, D)

        x = cur_scene_tokens.unsqueeze(1).expand(B, T_pred, N, D).reshape(B * T_pred, N, D)
        c = cond.reshape(B * T_pred, D)
        for blk in self.blocks:
            x = blk(x, c)
        return self.head(x).reshape(B, T_pred, N, D)
