"""V3.29 zero-initialized language-conditioned spatial attention."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class OrderedLanguageBridge(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, heads: int, cameras: int = 2, max_tokens: int = 32,
                 *, mode: str = "scalar_gate", residual_scale: float = 0.10):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError('hidden size must divide heads')
        if mode not in {"scalar_gate", "zero_output"}:
            raise ValueError("unsupported bridge mode")
        if not 0.0 < residual_scale <= 1.0:
            raise ValueError("residual scale must be in (0, 1]")
        if max_tokens < 32:
            raise ValueError("max_tokens must preserve the trained 32-token positions")
        self.heads, self.cameras, self.max_tokens = heads, cameras, max_tokens
        self.trained_position_tokens = 32
        self.mode, self.residual_scale = mode, residual_scale
        self.input_norm = nn.RMSNorm(feature_dim)
        self.projection = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.query_norm, self.language_norm = nn.RMSNorm(hidden_dim), nn.RMSNorm(hidden_dim)
        self.q, self.k, self.v, self.out = [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(4)]
        # Keep the historical parameter shape so a V3.29 checkpoint can be
        # loaded exactly. Positions beyond 32 use deterministic sinusoidal
        # coordinates and are learned through the surrounding projections.
        self.position = nn.Parameter(torch.randn(self.trained_position_tokens, hidden_dim) * 0.02)
        self.camera = nn.Parameter(torch.randn(cameras, hidden_dim) * 0.02)
        self.gate = nn.Parameter(torch.zeros(()))
        if mode == "zero_output":
            nn.init.zeros_(self.out.weight)

    def forward(self, spatial: Tensor, language: Tensor, mask: Tensor) -> Tensor:
        if (language.ndim != 4 or language.shape[1] != self.cameras
                or not 0 < language.shape[2] <= self.max_tokens):
            raise ValueError('ordered language exceeds its [B,C,T,D] token contract')
        if mask.dtype != torch.bool or mask.shape != language.shape[:3] or not mask.any(-1).all():
            raise ValueError('language mask must be bool, matching and nonempty per camera')
        if spatial.ndim != 3 or spatial.shape[0] != language.shape[0]:
            raise ValueError('spatial queries must be [B,T,H] with matching batch')
        # Clear padding before projection too, so arbitrary padding cannot contaminate results.
        language = language.masked_fill(~mask[..., None], 0)
        encoded = self.projection(self.input_norm(language))
        length, width = language.shape[2], encoded.shape[-1]
        if length <= self.trained_position_tokens:
            position = self.position[:length]
        else:
            extra_count = length - self.trained_position_tokens
            index = torch.arange(
                self.trained_position_tokens,
                length,
                device=encoded.device,
                dtype=torch.float32,
            )[:, None]
            frequency = torch.exp(
                torch.arange(0, width, 2, device=encoded.device, dtype=torch.float32)
                * (-torch.log(torch.tensor(10000.0, device=encoded.device)) / width)
            )[None]
            extra = torch.zeros(extra_count, width, device=encoded.device, dtype=torch.float32)
            extra[:, 0::2] = torch.sin(index * frequency)
            extra[:, 1::2] = torch.cos(index * frequency[:, : extra[:, 1::2].shape[1]])
            position = torch.cat((self.position, extra.to(self.position.dtype)), dim=0)
        encoded = encoded + position[None, None] + self.camera[None, :, None]
        encoded = self.language_norm(encoded.flatten(1, 2))
        b, t, h = spatial.shape
        def heads(x: Tensor) -> Tensor:
            return x.reshape(b, -1, self.heads, h // self.heads).transpose(1, 2)
        attended = F.scaled_dot_product_attention(
            heads(self.q(self.query_norm(spatial))), heads(self.k(encoded)), heads(self.v(encoded)),
            attn_mask=mask.flatten(1, 2)[:, None, None, :], dropout_p=0.,
        ).transpose(1, 2).reshape(b, t, h)
        residual = self.out(attended)
        scale = self.gate.tanh() if self.mode == "scalar_gate" else self.residual_scale
        # Preserve the residual addition in FP32.  The next consumer may use
        # autocast, so V3.30 also records the post-consumer numerical effect.
        return spatial.float() + float(scale) * residual.float() if not isinstance(scale, Tensor) else spatial.float() + scale.float() * residual.float()
