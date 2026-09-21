"""Zero-initialized cross-camera fusion for the V3.31 joint candidate."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class RobustCrossCameraFusion(nn.Module):
    """Exchange camera evidence while conditioning on task/state summaries."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        cameras: int = 2,
        *,
        residual_scale: float = 0.10,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must divide heads")
        if cameras != 2:
            raise ValueError("V3.31 currently requires exactly two cameras")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual_scale must be in (0, 1]")
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.cameras = cameras
        self.residual_scale = residual_scale
        self.query_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.source_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.condition = nn.Sequential(
            nn.RMSNorm(2 * hidden_dim, eps=1e-6),
            nn.Linear(2 * hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
        )
        self.q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, spatial: Tensor, semantic: Tensor, state: Tensor) -> Tensor:
        if spatial.ndim != 3 or spatial.shape[-1] != self.hidden_dim:
            raise ValueError("spatial must have shape [B,C*T,H]")
        if spatial.shape[1] % self.cameras:
            raise ValueError("spatial tokens must divide evenly across cameras")
        if tuple(semantic.shape) != (spatial.shape[0], self.hidden_dim):
            raise ValueError("semantic must have shape [B,H]")
        if tuple(state.shape) != tuple(semantic.shape):
            raise ValueError("state must have shape [B,H]")
        batch, total_tokens, width = spatial.shape
        tokens = spatial.reshape(batch, self.cameras, total_tokens // self.cameras, width)
        source = tokens.flip(1)
        condition = self.condition(torch.cat((semantic, state), dim=-1))[:, None, None, :]

        def split_heads(values: Tensor) -> Tensor:
            return values.reshape(
                batch * self.cameras,
                -1,
                self.heads,
                width // self.heads,
            ).transpose(1, 2)

        query = split_heads(self.q(self.query_norm(tokens + condition)))
        key = split_heads(self.k(self.source_norm(source + condition)))
        value = split_heads(self.v(self.source_norm(source)))
        attended = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        attended = attended.transpose(1, 2).reshape(batch, self.cameras, -1, width)
        residual = self.output(attended).reshape(batch, total_tokens, width)
        return spatial.float() + self.residual_scale * residual.float()
