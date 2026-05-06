import torch
import torch.nn as nn


class LatentLoss(nn.Module):
    """MSE loss for latent transition model p(o_{t+1} | o_t, a_t)."""

    def __init__(self, prediction_weight: float = 1.0, stop_grad_target: bool = True):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.stop_grad_target = stop_grad_target
        self.mse = nn.MSELoss()

    def forward(
        self,
        predicted_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            predicted_tokens: (B, 1, N_tokens, D)
            target_tokens:    (B, 1, N_tokens, D)
            valid_mask:       optional (B,) mask. Invalid rows keep the graph
                              connected but contribute zero loss.
        Returns:
            dict with 'loss' (scalar) and 'latent_prediction' (scalar)
        """
        target = target_tokens.detach() if self.stop_grad_target else target_tokens
        if valid_mask is None:
            pred_loss = self.mse(predicted_tokens, target)
        else:
            per_sample_loss = (predicted_tokens - target).pow(2).flatten(1).mean(dim=1)
            valid = valid_mask.to(device=per_sample_loss.device, dtype=per_sample_loss.dtype).reshape(-1)
            if valid.numel() == 1 and per_sample_loss.numel() != 1:
                valid = valid.expand_as(per_sample_loss)
            elif valid.numel() != per_sample_loss.numel():
                valid = per_sample_loss.new_zeros(per_sample_loss.shape)
            pred_loss = (per_sample_loss * valid).sum() / valid.sum().clamp_min(1.0)
        return {
            "loss": self.prediction_weight * pred_loss,
            "latent_prediction": pred_loss,
        }
