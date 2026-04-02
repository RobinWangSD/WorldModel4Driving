import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentLoss(nn.Module):
    """
    Latent space auxiliary losses for temporal representation learning.

    Ported from temporal-straightening/models/visual_world_model.py.
    Computes:
      1. Embedding prediction loss (MSE)
      2. Temporal straightening loss (curvature regularization)
      3. VCReg losses (variance + covariance regularization)
    """

    def __init__(
        self,
        prediction_weight: float = 1.0,
        straightening_weight: float = 0.0,
        vcreg_std_weight: float = 0.0,
        vcreg_cov_weight: float = 0.0,
        stop_grad_target: bool = True,
    ):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.straightening_weight = straightening_weight
        self.vcreg_std_weight = vcreg_std_weight
        self.vcreg_cov_weight = vcreg_cov_weight
        self.stop_grad_target = stop_grad_target
        self.mse = nn.MSELoss()

    def forward(
        self,
        predicted_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        all_frame_tokens: torch.Tensor,
    ) -> dict:
        """
        Args:
            predicted_tokens: (B, T-1, N_tokens, D) predicted next-frame tokens
            target_tokens: (B, T-1, N_tokens, D) actual next-frame tokens
            all_frame_tokens: (B, T, N_tokens, D) tokens from all frames
        Returns:
            dict with 'loss' (scalar) and per-component losses
        """
        losses = {}
        total = torch.tensor(0.0, device=predicted_tokens.device)

        # 1. Embedding prediction loss
        target = target_tokens.detach() if self.stop_grad_target else target_tokens
        pred_loss = self.mse(predicted_tokens, target)
        losses["latent_prediction"] = pred_loss
        total = total + self.prediction_weight * pred_loss

        # 2. Temporal straightening (requires >= 3 frames)
        if self.straightening_weight > 0 and all_frame_tokens.shape[1] >= 3:
            straight_loss = self._curvature_loss(all_frame_tokens)
            losses["latent_straightening"] = straight_loss
            total = total + self.straightening_weight * straight_loss

        # 3. VCReg: variance regularization
        if self.vcreg_std_weight > 0:
            std_loss = self._vcreg_std_loss(all_frame_tokens)
            losses["latent_vcreg_std"] = std_loss
            total = total + self.vcreg_std_weight * std_loss

        # 4. VCReg: covariance regularization
        if self.vcreg_cov_weight > 0:
            cov_loss = self._vcreg_cov_loss(all_frame_tokens)
            losses["latent_vcreg_cov"] = cov_loss
            total = total + self.vcreg_cov_weight * cov_loss

        losses["loss"] = total
        return losses

    def _curvature_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Temporal straightening via cosine curvature on mean-pooled tokens."""
        # Mean-pool over tokens per frame: (B, T, N, D) -> (B, T, D)
        pooled = tokens.mean(dim=2)
        v1 = pooled[:, 1:-1] - pooled[:, :-2]
        v2 = pooled[:, 2:] - pooled[:, 1:-1]

        # Mask out near-zero steps to avoid degenerate gradients
        cos = F.cosine_similarity(v1, v2, dim=-1, eps=1e-6)
        loss = 1.0 - cos
        step1 = v1.norm(dim=-1)
        step2 = v2.norm(dim=-1)
        mask = (step1 > 1e-6) & (step2 > 1e-6)
        if mask.any():
            return loss[mask].mean()
        return loss.mean()

    def _vcreg_std_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encourage high variance across batch dimension to prevent collapse."""
        x = tokens.reshape(-1, tokens.shape[-1])
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        return torch.mean(F.relu(1.0 - std_x))

    def _vcreg_cov_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decorrelate feature dimensions via off-diagonal covariance penalty."""
        x = tokens.reshape(-1, tokens.shape[-1])
        _, d = x.shape
        x = x - x.mean(dim=0)
        cov_x = (x.T @ x) / (x.shape[0] - 1)
        # Off-diagonal elements
        off_diag = cov_x.flatten()[:-1].view(d - 1, d + 1)[:, 1:].flatten()
        return off_diag.pow(2).sum() / d
