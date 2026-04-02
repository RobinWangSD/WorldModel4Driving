import torch
from torch import nn


class GaussianKLReg(nn.Module):
    """KL divergence regularizer toward N(0, I).
    KL(N(mu, sigma^2) || N(0, I)) = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    """

    def forward(self, mu, log_var):
        """
        mu:      (B, T, D)
        log_var: (B, T, D)
        """
        return -0.5 * (1 + log_var - mu.pow(2) - log_var.exp()).mean()


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)

    Tests whether latent embeddings follow an isotropic Gaussian distribution
    using the Epps-Pulley goodness-of-fit statistic with random projections.
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D) — embeddings with time as first dim
        """
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
