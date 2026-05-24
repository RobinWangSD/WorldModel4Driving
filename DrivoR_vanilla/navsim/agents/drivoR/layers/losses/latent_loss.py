import torch
import torch.nn as nn

from .sigreg import SIGReg


class LatentLoss(nn.Module):
    """Latent transition loss for p(o_{t+1} | o_t, a_t)."""

    def __init__(
        self,
        prediction_weight: float = 1.0,
        stop_grad_target: bool = True,
        sigreg_weight: float = 0.0,
        sigreg_knots: int = 17,
        sigreg_num_proj: int = 1024,
    ):
        super().__init__()
        self.prediction_weight = prediction_weight
        self.stop_grad_target = stop_grad_target
        self.sigreg_weight = sigreg_weight
        self.mse = nn.MSELoss()
        self.sigreg = SIGReg(knots=sigreg_knots, num_proj=sigreg_num_proj) if sigreg_weight > 0 else None

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
            dict with scalar loss components.
        """
        target = target_tokens.detach() if self.stop_grad_target else target_tokens
        valid_bool = self._normalize_valid_mask(valid_mask, predicted_tokens.shape[0], predicted_tokens.device)
        if valid_mask is None:
            pred_loss = self.mse(predicted_tokens, target)
        else:
            per_sample_loss = (predicted_tokens - target).pow(2).flatten(1).mean(dim=1)
            valid = valid_bool.to(dtype=per_sample_loss.dtype)
            pred_loss = (per_sample_loss * valid).sum() / valid.sum().clamp_min(1.0)

        total = self.prediction_weight * pred_loss
        losses = {
            "loss": total,
            "latent_prediction": pred_loss,
        }

        if self.sigreg is not None:
            sigreg_loss = self._sigreg_loss(predicted_tokens, valid_bool)
            total = total + self.sigreg_weight * sigreg_loss
            losses["loss"] = total
            losses["latent_sigreg"] = sigreg_loss

        return losses

    @staticmethod
    def _normalize_valid_mask(valid_mask, batch_size: int, device: torch.device) -> torch.Tensor:
        if valid_mask is None:
            return torch.ones(batch_size, dtype=torch.bool, device=device)
        if isinstance(valid_mask, torch.Tensor):
            valid = valid_mask.to(device=device, dtype=torch.bool).reshape(-1)
        else:
            valid = torch.as_tensor(valid_mask, dtype=torch.bool, device=device).reshape(-1)
        if valid.numel() == 1 and batch_size != 1:
            return valid.expand(batch_size)
        if valid.numel() != batch_size:
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        return valid

    def _sigreg_loss(self, predicted_tokens: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if valid_mask.any():
            tokens = predicted_tokens[valid_mask]
        else:
            return predicted_tokens.sum() * 0.0

        if tokens.ndim != 4:
            raise ValueError(f"Expected predicted_tokens with shape (B, T, N, D), got {tokens.shape}")

        b, t, n, d = tokens.shape
        proj = tokens.permute(1, 0, 2, 3).reshape(t, b * n, d)
        return self.sigreg(proj)


class MultiStepLatentLoss(nn.Module):
    """Multi-step latent prediction loss.

    Computes MSE between predicted and target scene-token latents over T_pred
    future steps, masked per (batch, step) by an optional validity tensor.
    """

    def __init__(self, stop_grad_target: bool = True):
        super().__init__()
        self.stop_grad_target = stop_grad_target

    def forward(
        self,
        predicted_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
        valid_mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            predicted_tokens: (B, T_pred, N, D)
            target_tokens:    (B, T_pred, N, D)
            valid_mask:       optional (B, T_pred) bool. Invalid (b, t) entries
                              keep the graph connected but contribute zero loss.
        Returns:
            dict with keys {loss, latent_prediction}.
        """
        if predicted_tokens.shape != target_tokens.shape:
            raise ValueError(
                f"predicted vs target shape mismatch: {predicted_tokens.shape} vs {target_tokens.shape}"
            )
        if predicted_tokens.ndim != 4:
            raise ValueError(f"Expected (B, T, N, D); got {tuple(predicted_tokens.shape)}")

        target = target_tokens.detach() if self.stop_grad_target else target_tokens
        per_step_loss = (predicted_tokens - target).pow(2).flatten(2).mean(dim=2)  # (B, T_pred)

        if valid_mask is None:
            loss = per_step_loss.mean()
        else:
            valid = valid_mask.to(device=per_step_loss.device, dtype=per_step_loss.dtype)
            if valid.shape != per_step_loss.shape:
                # graceful degrade: if shapes mismatch, treat all as valid
                loss = per_step_loss.mean()
            else:
                loss = (per_step_loss * valid).sum() / valid.sum().clamp_min(1.0)

        return {"loss": loss, "latent_prediction": loss}
