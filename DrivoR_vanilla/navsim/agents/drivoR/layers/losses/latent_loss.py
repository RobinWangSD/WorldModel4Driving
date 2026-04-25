import torch
import torch.nn as nn


class LatentLoss(nn.Module):
    """MSE loss for latent transition model p(o_{t+1} | o_t, a_t)."""

    def __init__(self, prediction_weight: float = 1.0, stop_grad_target: bool = True):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.stop_grad_target = stop_grad_target
        self.mse = nn.MSELoss()

    def forward(self, predicted_tokens: torch.Tensor, target_tokens: torch.Tensor) -> dict:
        """
        Args:
            predicted_tokens: (B, 1, N_tokens, D)
            target_tokens:    (B, 1, N_tokens, D)
        Returns:
            dict with 'loss' (scalar) and 'latent_prediction' (scalar)
        """
        target = target_tokens.detach() if self.stop_grad_target else target_tokens
        pred_loss = self.mse(predicted_tokens, target)
        return {
            "loss": self.prediction_weight * pred_loss,
            "latent_prediction": pred_loss,
        }
