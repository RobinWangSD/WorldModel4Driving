"""ViT Predictor with AdaLN-Zero action conditioning.

Adapted from le-wm's ConditionalBlock + temporal-straightening's causal masking.
Action embeddings modulate each transformer layer's normalization parameters
instead of being concatenated as extra tokens.
"""

import torch
from torch import nn
from einops import rearrange


# Global variables for causal mask generation (same pattern as vit.py)
NUM_FRAMES = 1
NUM_PATCHES = 1


def generate_mask_matrix(npatch, nwindow):
    zeros = torch.zeros(npatch, npatch)
    ones = torch.ones(npatch, npatch)
    rows = []
    for i in range(nwindow):
        row = torch.cat([ones] * (i + 1) + [zeros] * (nwindow - i - 1), dim=1)
        rows.append(row)
    mask = torch.cat(rows, dim=0).unsqueeze(0).unsqueeze(0)
    return mask


def modulate(x, shift, scale):
    """AdaLN-zero modulation: x * (1 + scale) + shift"""
    return x * (1 + scale) + shift


class CausalAttention(nn.Module):
    """Scaled dot-product attention with causal masking."""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )
        self.bias = generate_mask_matrix(NUM_PATCHES, NUM_FRAMES).to("cuda")

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv
        )
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots = dots.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        attn = self.attend(dots)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning.

    Action embeddings produce 6 modulation parameters (shift, scale, gate)
    for both the attention and feedforward sub-layers.
    """

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.attn = CausalAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )
        # Zero-initialize so conditioning starts as identity
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        """
        x: (B, T*num_patches, D) — visual tokens
        c: (B, T*num_patches, D) — conditioning signal (action embeddings broadcast to patches)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.ff(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class ConditionalTransformer(nn.Module):
    """Transformer with AdaLN-zero ConditionalBlocks."""

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList(
            [ConditionalBlock(dim, heads, dim_head, mlp_dim, dropout) for _ in range(depth)]
        )

    def forward(self, x, c):
        for block in self.layers:
            x = block(x, c)
        return self.norm(x)


class ConditionalViTPredictor(nn.Module):
    """ViT predictor with AdaLN-zero action conditioning.

    Instead of concatenating action embeddings as extra tokens, action
    embeddings modulate each transformer layer via AdaLN-zero.
    """

    def __init__(
        self,
        *,
        num_patches,
        num_frames,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        assert pool in {"cls", "mean"}

        global NUM_FRAMES, NUM_PATCHES
        NUM_FRAMES = num_frames
        NUM_PATCHES = num_patches

        self.num_patches = num_patches
        self.dim = dim
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames * num_patches, dim)
        )
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = ConditionalTransformer(
            dim, depth, heads, dim_head, mlp_dim, dropout
        )
        # Projection for action embeddings (action_emb_dim -> dim)
        # Lazily initialized on first forward to auto-detect action_emb_dim
        self.cond_proj = None
        self.pool = pool

    def forward(self, x, c):
        """
        x: (B, num_frames * num_patches, dim) — visual tokens
        c: (B, num_frames, action_emb_dim) — action conditioning per frame
        """
        b, n, _ = x.shape
        x = x + self.pos_embedding[:, :n]
        x = self.dropout(x)

        # Project action embeddings to predictor dim if needed
        if c.size(-1) != self.dim:
            if self.cond_proj is None:
                self.cond_proj = nn.Linear(c.size(-1), self.dim).to(c.device)
            c = self.cond_proj(c)

        # Broadcast action conditioning to all patches within each frame
        # c: (B, num_frames, dim) -> (B, num_frames * num_patches, dim)
        c = c.unsqueeze(2).expand(-1, -1, self.num_patches, -1)
        c = c.reshape(b, -1, c.size(-1))[:, :n]

        x = self.transformer(x, c)
        return x
