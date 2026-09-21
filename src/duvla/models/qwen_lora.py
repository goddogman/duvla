"""Audited, layer-local LoRA attachment for the Qwen backbone."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Sequence

import torch
from torch import Tensor
from torch import nn

if TYPE_CHECKING:
    from .qwen_backbone import QwenVLBackbone


@dataclass(frozen=True)
class QwenLoraSpec:
    """One explicit LoRA placement contract for Qwen3-VL."""

    layers: tuple[int, ...] = (16, 17, 18, 19)
    projections: tuple[str, ...] = ("q_proj", "v_proj")
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if not self.layers or len(set(self.layers)) != len(self.layers):
            raise ValueError("LoRA layers must be non-empty and unique")
        if min(self.layers) < 0:
            raise ValueError("LoRA layers must be non-negative")
        if not self.projections or len(set(self.projections)) != len(self.projections):
            raise ValueError("LoRA projections must be non-empty and unique")
        if any(not name or "." in name for name in self.projections):
            raise ValueError("LoRA projections must be direct module names")
        if self.rank <= 0 or self.alpha <= 0 or not 0.0 <= self.dropout < 1.0:
            raise ValueError("invalid LoRA rank, alpha, or dropout")


def qwen_language_lora_targets(model: nn.Module, spec: QwenLoraSpec) -> tuple[str, ...]:
    """Resolve exact language-transformer projections and reject silent spillover."""

    modules = dict(model.named_modules())
    targets = tuple(
        f"model.language_model.layers.{layer}.self_attn.{projection}"
        for layer in spec.layers
        for projection in spec.projections
    )
    missing = [name for name in targets if name not in modules]
    if missing:
        raise ValueError(f"Qwen LoRA targets are missing: {missing}")
    non_linear = [name for name in targets if not isinstance(modules[name], nn.Linear)]
    if non_linear:
        raise TypeError(f"Qwen LoRA targets are not linear projections: {non_linear}")
    return targets


def expected_lora_parameters(
    model: nn.Module,
    targets: tuple[str, ...],
    *,
    rank: int,
) -> int:
    """Return the exact A+B parameter count before PEFT wraps the modules."""

    modules = dict(model.named_modules())
    total = 0
    for name in targets:
        module = modules.get(name)
        if not isinstance(module, nn.Linear):
            raise TypeError(f"LoRA target is not a linear layer: {name}")
        total += rank * (module.in_features + module.out_features)
    return total


def attach_qwen_lora(
    backbone: QwenVLBackbone,
    spec: QwenLoraSpec,
) -> tuple[str, ...]:
    """Attach PEFT LoRA while keeping all original Qwen parameters frozen."""

    if backbone.model is None:
        raise RuntimeError("load Qwen before attaching LoRA")
    if backbone.config.freeze:
        raise ValueError("LoRA requires freeze=False so backbone forward keeps autograd enabled")
    try:
        from peft import LoraConfig, get_peft_model
    except ModuleNotFoundError as exc:  # pragma: no cover - integration environment
        raise RuntimeError("PEFT is required for Qwen LoRA") from exc
    targets = qwen_language_lora_targets(backbone.model, spec)
    backbone.model = get_peft_model(
        backbone.model,
        LoraConfig(
            r=spec.rank,
            lora_alpha=spec.alpha,
            lora_dropout=spec.dropout,
            bias="none",
            target_modules=list(targets),
        ),
    )
    trainable = [name for name, value in backbone.model.named_parameters() if value.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("PEFT left a non-LoRA Qwen parameter trainable")
    return targets


def forward_v331_lora_context(
    backbone: QwenVLBackbone,
    agent_images: Sequence[Any],
    wrist_images: Sequence[Any],
    instructions: Sequence[str],
    *,
    max_tokens: int = 256,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Encode two views in one camera-major Qwen batch for the V3.31 policy.

    The single combined forward is both faster and less memory hungry than
    retaining two independent autograd graphs.  It restores the exact policy
    layouts after Qwen: visual ``[B,L,C,T,D]``, semantic ``[B,L,1,D]`` and
    ordered language ``[B,C,T,D]`` plus its boolean mask.
    """

    batch_size = len(instructions)
    if batch_size <= 0 or len(agent_images) != batch_size or len(wrist_images) != batch_size:
        raise ValueError("two equally sized camera batches and instructions are required")
    examples = tuple(zip(agent_images, instructions)) + tuple(
        zip(wrist_images, instructions)
    )
    inputs = backbone.prepare_view_batch_inputs(examples)
    combined_instructions = tuple(instructions) + tuple(instructions)
    visual_all, semantic_rows, language_rows, mask_rows = backbone.forward_v331_context(
        (inputs,), combined_instructions, max_tokens=max_tokens
    )
    if visual_all.shape[0] != 2 * batch_size or visual_all.shape[2] != 1:
        raise RuntimeError("combined Qwen output does not contain two views per sample")
    visual = (
        visual_all[:, :, 0]
        .reshape(2, batch_size, *visual_all.shape[1:2], *visual_all.shape[3:])
        .permute(1, 2, 0, 3, 4)
    )
    semantic = semantic_rows.reshape(
        2, batch_size, *semantic_rows.shape[1:]
    ).mean(0)
    language = language_rows[:, 0].reshape(
        2, batch_size, *language_rows.shape[2:]
    ).permute(1, 0, 2, 3)
    mask = mask_rows[:, 0].reshape(
        2, batch_size, mask_rows.shape[-1]
    ).permute(1, 0, 2)
    if mask.dtype is not torch.bool:
        raise RuntimeError("Qwen language mask must remain boolean")
    return visual, semantic, language, mask
