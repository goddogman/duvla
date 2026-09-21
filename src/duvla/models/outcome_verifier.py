"""Action-conditioned outcome prediction for V101 candidate planning.

The verifier never changes an action chunk.  It ranks chunks proposed by a
frozen policy and predicts their likely future outcome so the rollout loop can
select, verify, and replan without benchmark task indices.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class OutcomeVerifierConfig:
    context_dim: int = 2048
    state_dim: int = 8
    action_dim: int = 7
    action_horizon: int = 8
    hidden_dim: int = 256
    future_dim: int = 256
    action_layers: int = 2
    attention_heads: int = 8
    dropout: float = 0.0
    failure_weight: float = 1.0
    uncertainty_weight: float = 0.1
    baseline_relative_scoring: bool = False
    baseline_candidate_index: int = -1
    plan_outcome_alignment: bool = False
    plan_alignment_weight: float = 1.0
    stepwise_context_grounding: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.context_dim,
            self.state_dim,
            self.action_dim,
            self.action_horizon,
            self.hidden_dim,
            self.future_dim,
            self.action_layers,
            self.attention_heads,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("all verifier dimensions must be positive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if (
            self.failure_weight < 0.0
            or self.uncertainty_weight < 0.0
            or self.plan_alignment_weight < 0.0
        ):
            raise ValueError("score penalties must be non-negative")
        if self.baseline_relative_scoring and self.baseline_candidate_index < 0:
            raise ValueError(
                "baseline-relative scoring requires a non-negative baseline candidate index"
            )


@dataclass(frozen=True)
class OutcomeVerifierOutput:
    future_latent: Tensor
    progress_logits: Tensor
    failure_logits: Tensor
    uncertainty: Tensor
    score: Tensor
    selected_index: Tensor
    planned_future_latent: Tensor | None = None
    plan_alignment: Tensor | None = None


def conservative_ensemble_scores(
    member_scores: Tensor,
    candidate_mask: Tensor,
    *,
    baseline_index: int,
    uncertainty_penalty: float = 1.0,
    minimum_consensus: int = 4,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build risk-adjusted scores from independently trained rankers.

    An alternative remains eligible only when enough members independently
    name it their best non-baseline candidate.  The returned score is a lower
    confidence bound (mean minus a multiple of ensemble standard deviation),
    while the unchanged baseline remains available as the safe fallback.
    """

    if member_scores.ndim != 3:
        raise ValueError("member_scores must have shape [members, batch, candidates]")
    members, batch, candidates = member_scores.shape
    if candidate_mask.shape != (batch, candidates) or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean with shape [batch, candidates]")
    if not 0 <= baseline_index < candidates:
        raise ValueError("baseline_index is outside candidate range")
    if uncertainty_penalty < 0.0:
        raise ValueError("uncertainty_penalty must be non-negative")
    if not 1 <= minimum_consensus <= members:
        raise ValueError("minimum_consensus must be inside the ensemble size")
    if not candidate_mask[:, baseline_index].all():
        raise ValueError("baseline candidate must be valid for every row")

    alternatives = candidate_mask.clone()
    alternatives[:, baseline_index] = False
    member_alternative_scores = member_scores.masked_fill(
        ~alternatives[None], torch.finfo(member_scores.dtype).min
    )
    member_votes = member_alternative_scores.argmax(dim=-1)
    vote_count = torch.zeros(
        batch, candidates, dtype=torch.long, device=member_scores.device
    )
    vote_count.scatter_add_(
        1,
        member_votes.transpose(0, 1),
        torch.ones_like(member_votes.transpose(0, 1), dtype=torch.long),
    )

    mean = member_scores.mean(dim=0)
    deviation = member_scores.std(dim=0, unbiased=False)
    robust = mean - uncertainty_penalty * deviation
    eligible = candidate_mask & (vote_count >= minimum_consensus)
    eligible[:, baseline_index] = True
    robust = robust.masked_fill(~eligible, torch.finfo(robust.dtype).min)
    return robust, deviation, vote_count


def select_with_baseline_fallback(
    score: Tensor,
    candidate_mask: Tensor,
    *,
    baseline_index: int,
    minimum_advantage: float,
) -> Tensor:
    """Choose an alternative only when it beats the stable baseline confidently.

    The comparison is entirely within the policy-generated candidate set.  It
    uses neither a benchmark task id nor simulator reward/success information.
    """

    if score.ndim != 2 or candidate_mask.shape != score.shape:
        raise ValueError("score and candidate_mask must have shape [batch, candidates]")
    if candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean")
    if not 0 <= baseline_index < score.shape[1]:
        raise ValueError("baseline_index is outside candidate range")
    if minimum_advantage < 0.0:
        raise ValueError("minimum_advantage must be non-negative")
    if not candidate_mask[:, baseline_index].all():
        raise ValueError("baseline candidate must be valid for every row")

    alternatives = candidate_mask.clone()
    alternatives[:, baseline_index] = False
    if not alternatives.any(dim=1).all():
        return torch.full(
            (score.shape[0],), baseline_index, dtype=torch.long, device=score.device
        )
    alternative_score = score.masked_fill(
        ~alternatives, torch.finfo(score.dtype).min
    )
    alternative_index = alternative_score.argmax(dim=1)
    best = score.gather(1, alternative_index[:, None]).squeeze(1)
    baseline = score[:, baseline_index]
    use_alternative = best - baseline > minimum_advantage
    fallback = torch.full_like(alternative_index, baseline_index)
    return torch.where(use_alternative, alternative_index, fallback)


def regret_weighted_listwise_losses(
    score: Tensor,
    candidate_error: Tensor,
    candidate_mask: Tensor,
    *,
    baseline_index: int,
    temperature_floor: float = 0.02,
    opportunity_scale: float = 0.05,
) -> dict[str, Tensor]:
    """Supervise candidate scores from reward-free physical branch regret.

    Near-tied candidates receive a soft target instead of an arbitrary hard
    winner.  Rows where the stable baseline has a large oracle gap receive
    more weight, because those rows contain the useful correction signal.
    The baseline-relative quality target keeps zero as the semantic score of
    the unchanged policy candidate and makes conservative fallback possible.
    """

    if score.ndim != 2 or candidate_error.shape != score.shape:
        raise ValueError("score and candidate_error must have shape [batch, candidates]")
    if candidate_mask.shape != score.shape or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean and match score")
    if not 0 <= baseline_index < score.shape[1]:
        raise ValueError("baseline_index is outside candidate range")
    if temperature_floor <= 0.0 or opportunity_scale <= 0.0:
        raise ValueError("listwise scales must be positive")
    if not candidate_mask.any(dim=1).all():
        raise ValueError("every row must contain a valid candidate")
    if not candidate_mask[:, baseline_index].all():
        raise ValueError("baseline candidate must be valid for every row")
    if not torch.isfinite(score[candidate_mask]).all() or not torch.isfinite(
        candidate_error[candidate_mask]
    ).all():
        raise ValueError("valid candidate scores and errors must be finite")

    maximum = torch.finfo(candidate_error.dtype).max
    masked_error = candidate_error.masked_fill(~candidate_mask, maximum)
    oracle_error = masked_error.min(dim=1).values
    regret = (candidate_error - oracle_error[:, None]).clamp_min(0.0)
    valid_count = candidate_mask.sum(dim=1).clamp_min(1).to(regret.dtype)
    row_temperature = (
        (regret * candidate_mask.to(regret.dtype)).sum(dim=1) / valid_count
    ).clamp_min(temperature_floor)
    target_logits = -(regret / row_temperature[:, None].detach())
    target_logits = target_logits.masked_fill(~candidate_mask, -torch.inf)
    target_probability = target_logits.softmax(dim=1)
    predicted_log_probability = score.masked_fill(
        ~candidate_mask, -torch.inf
    ).log_softmax(dim=1)
    listwise_per_row = -(
        target_probability * predicted_log_probability.masked_fill(~candidate_mask, 0.0)
    ).sum(dim=1)

    baseline_error = candidate_error[:, baseline_index]
    opportunity = (baseline_error - oracle_error).clamp_min(0.0)
    row_weight = (1.0 + opportunity / opportunity_scale).clamp(max=5.0).detach()
    listwise = (listwise_per_row * row_weight).sum() / row_weight.sum().clamp_min(1.0)

    denominator = baseline_error.clamp_min(0.05)
    quality_target = (
        (baseline_error[:, None] - candidate_error) / denominator[:, None]
    ).clamp(-2.0, 1.0)
    quality_element = F.smooth_l1_loss(score, quality_target, reduction="none")
    quality_per_row = (
        quality_element * candidate_mask.to(quality_element.dtype)
    ).sum(dim=1) / valid_count
    quality = (quality_per_row * row_weight).sum() / row_weight.sum().clamp_min(1.0)
    return {
        "listwise": listwise,
        "regret_quality": quality,
    }


def multi_horizon_action_error(
    candidate_actions: Tensor,
    expert_actions: Tensor,
    *,
    horizons: tuple[int, ...] = (1, 2, 4),
    weights: tuple[float, ...] = (0.2, 0.3, 0.5),
) -> Tensor:
    """Measure candidate imitation error over several execution horizons."""

    if candidate_actions.ndim != 4:
        raise ValueError("candidate_actions must have shape [batch, candidates, horizon, action_dim]")
    if expert_actions.ndim != 3:
        raise ValueError("expert_actions must have shape [batch, horizon, action_dim]")
    if (
        candidate_actions.shape[0] != expert_actions.shape[0]
        or candidate_actions.shape[2:] != expert_actions.shape[1:]
    ):
        raise ValueError("candidate and expert action contracts differ")
    if len(horizons) != len(weights) or not horizons:
        raise ValueError("horizons and weights must be non-empty and have equal length")
    if any(h <= 0 or h > candidate_actions.shape[2] for h in horizons):
        raise ValueError("multi-horizon prefix is outside the action chunk")
    if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
        raise ValueError("multi-horizon weights must be non-negative and non-empty")
    errors = []
    for horizon in horizons:
        errors.append(
            (
                candidate_actions[:, :, :horizon]
                - expert_actions[:, None, :horizon]
            )
            .abs()
            .mean(dim=(2, 3))
        )
    weight_tensor = candidate_actions.new_tensor(weights)
    stacked = torch.stack(errors, dim=-1)
    return (stacked * weight_tensor).sum(dim=-1) / weight_tensor.sum()


def compress_outcome_context(context: Tensor, future_dim: int) -> Tensor:
    """Create a deterministic future target from frozen Qwen context tokens.

    No learned target encoder is used: tokens are averaged and the 2048-wide
    vector is adaptively pooled to ``future_dim``.  This keeps cache targets
    reproducible and small while preserving the train/inference contract.
    """

    if future_dim <= 0:
        raise ValueError("future_dim must be positive")
    if context.ndim == 4:
        context = context.flatten(1, 2)
    if context.ndim != 3:
        raise ValueError("context must have rank 3 or 4")
    pooled = context.float().mean(dim=1)
    if pooled.shape[-1] == future_dim:
        return F.layer_norm(pooled, (future_dim,))
    compressed = F.adaptive_avg_pool1d(pooled[:, None, :], future_dim).squeeze(1)
    return F.layer_norm(compressed, (future_dim,))


def arrange_v2_policy_context(
    context: Tensor,
    *,
    camera_count: int,
    spatial_tokens_per_camera: int,
) -> Tensor:
    """Attach V2 shared semantic/history tokens to both ordered camera streams."""

    if context.ndim != 3:
        raise ValueError("V2 policy context must have shape [batch, tokens, width]")
    if camera_count != 2 or spatial_tokens_per_camera <= 0:
        raise ValueError("V2 outcome context requires two cameras and positive spatial tokens")
    spatial_count = camera_count * spatial_tokens_per_camera
    if context.shape[1] <= spatial_count:
        raise ValueError("V2 policy context has no shared semantic/history tokens")
    spatial = context[:, :spatial_count].reshape(
        context.shape[0], camera_count, spatial_tokens_per_camera, context.shape[-1]
    )
    shared = context[:, spatial_count:]
    shared = shared[:, None].expand(-1, camera_count, -1, -1)
    return torch.cat((spatial, shared), dim=2)


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    weights = weights.expand_as(values)
    denominator = weights.sum().clamp_min(1.0)
    return (values * weights).sum() / denominator


class ActionConditionedOutcomeVerifier(nn.Module):
    """Rank candidate action chunks using their predicted execution outcomes."""

    def __init__(self, config: OutcomeVerifierConfig) -> None:
        super().__init__()
        self.config = config
        self.context_projection = nn.Sequential(
            nn.LayerNorm(config.context_dim),
            nn.Linear(config.context_dim, config.hidden_dim),
        )
        self.state_projection = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.hidden_dim),
        )
        self.action_projection = nn.Linear(config.action_dim, config.hidden_dim)
        self.relative_action_projection = (
            nn.Linear(config.action_dim, config.hidden_dim, bias=False)
            if config.baseline_relative_scoring
            else None
        )
        self.action_positions = nn.Parameter(
            torch.randn(config.action_horizon, config.hidden_dim) * 0.02
        )
        action_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_encoder = nn.TransformerEncoder(
            action_layer, num_layers=config.action_layers
        )
        self.context_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.attention_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.step_context_attention = (
            nn.MultiheadAttention(
                config.hidden_dim,
                config.attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            if config.stepwise_context_grounding
            else None
        )
        self.outcome_token = (
            nn.Parameter(torch.randn(config.hidden_dim) * 0.02)
            if config.stepwise_context_grounding
            else None
        )
        self.previous_outcome_projection = nn.Linear(config.future_dim, config.hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(config.hidden_dim * 3),
            nn.Linear(config.hidden_dim * 3, config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
        )
        self.future_head = nn.Linear(config.hidden_dim, config.future_dim)
        self.progress_head = nn.Linear(config.hidden_dim, 1)
        self.failure_head = nn.Linear(config.hidden_dim, 1)
        self.uncertainty_head = nn.Linear(config.hidden_dim, 1)
        self.candidate_attention = (
            nn.MultiheadAttention(
                config.hidden_dim,
                config.attention_heads,
                dropout=config.dropout,
                batch_first=True,
            )
            if config.baseline_relative_scoring
            else None
        )
        self.quality_head = (
            nn.Linear(config.hidden_dim, 1)
            if config.baseline_relative_scoring
            else None
        )
        self.plan_fusion = (
            nn.Sequential(
                nn.LayerNorm(config.hidden_dim * 2),
                nn.Linear(config.hidden_dim * 2, config.hidden_dim),
                nn.GELU(),
            )
            if config.plan_outcome_alignment
            else None
        )
        self.plan_head = (
            nn.Linear(config.hidden_dim, config.future_dim)
            if config.plan_outcome_alignment
            else None
        )

    def _flatten_context(self, context: Tensor) -> Tensor:
        if context.ndim == 4:
            batch, cameras, tokens, width = context.shape
            context = context.reshape(batch, cameras * tokens, width)
        if context.ndim != 3 or context.shape[-1] != self.config.context_dim:
            raise ValueError(
                "context must have shape [batch, tokens, context_dim] or "
                "[batch, cameras, tokens, context_dim]"
            )
        return context

    def forward(
        self,
        context: Tensor,
        state: Tensor,
        candidate_actions: Tensor,
        *,
        candidate_mask: Tensor | None = None,
        previous_outcome: Tensor | None = None,
    ) -> OutcomeVerifierOutput:
        context = self._flatten_context(context)
        if state.ndim != 2 or state.shape[-1] != self.config.state_dim:
            raise ValueError("state must have shape [batch, state_dim]")
        if candidate_actions.ndim != 4:
            raise ValueError("candidate_actions must have shape [batch, candidates, horizon, action_dim]")
        batch, candidates, horizon, action_dim = candidate_actions.shape
        if state.shape[0] != batch or context.shape[0] != batch:
            raise ValueError("context, state, and candidate_actions batch sizes must match")
        if horizon != self.config.action_horizon or action_dim != self.config.action_dim:
            raise ValueError("candidate action shape does not match verifier config")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch, candidates, dtype=torch.bool, device=candidate_actions.device
            )
        if candidate_mask.shape != (batch, candidates) or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean with shape [batch, candidates]")
        if not candidate_mask.any(dim=1).all():
            raise ValueError("every batch item must contain at least one valid candidate")

        # Frozen Qwen returns BF16 features online, while verifier checkpoints
        # are stored in FP32.  Keep this boundary explicit so LayerNorm/Linear
        # never receive mixed parameter and activation dtypes during rollout.
        model_dtype = self.context_projection[0].weight.dtype
        context = context.to(dtype=model_dtype)
        state = state.to(dtype=model_dtype)
        candidate_actions = candidate_actions.to(dtype=model_dtype)
        if previous_outcome is not None:
            previous_outcome = previous_outcome.to(dtype=model_dtype)

        memory = self.context_projection(context)
        state_token = self.state_projection(state)
        action_tokens = self.action_projection(candidate_actions)
        if self.relative_action_projection is not None:
            baseline_index = self.config.baseline_candidate_index
            if baseline_index >= candidates:
                raise ValueError("configured baseline candidate is outside candidate range")
            relative = candidate_actions - candidate_actions[:, baseline_index : baseline_index + 1]
            action_tokens = action_tokens + self.relative_action_projection(relative)
        action_tokens = action_tokens + self.action_positions[None, None, :, :]
        flat_actions = action_tokens.reshape(
            batch * candidates, horizon, self.config.hidden_dim
        )
        if self.step_context_attention is None or self.outcome_token is None:
            encoded_actions = self.action_encoder(flat_actions)
            candidate_token = encoded_actions.mean(dim=1).reshape(
                batch, candidates, self.config.hidden_dim
            )
            candidate_token = candidate_token + state_token[:, None, :]
        else:
            expanded_memory = memory[:, None].expand(
                -1, candidates, -1, -1
            ).reshape(batch * candidates, memory.shape[1], self.config.hidden_dim)
            expanded_state = state_token[:, None].expand(-1, candidates, -1).reshape(
                batch * candidates, 1, self.config.hidden_dim
            )
            summary = self.outcome_token[None, None].expand(
                batch * candidates, 1, -1
            ) + expanded_state
            grounded_sequence = torch.cat(
                (summary, flat_actions + expanded_state), dim=1
            )
            grounded_context, _ = self.step_context_attention(
                grounded_sequence,
                expanded_memory,
                expanded_memory,
                need_weights=False,
            )
            grounded_sequence = self.action_encoder(
                grounded_sequence + grounded_context
            )
            candidate_token = grounded_sequence[:, 0].reshape(
                batch, candidates, self.config.hidden_dim
            )

        if previous_outcome is not None:
            if previous_outcome.shape == (batch, self.config.future_dim):
                previous_outcome = previous_outcome[:, None, :].expand(-1, candidates, -1)
            if previous_outcome.shape != (batch, candidates, self.config.future_dim):
                raise ValueError(
                    "previous_outcome must have shape [batch, future_dim] or "
                    "[batch, candidates, future_dim]"
                )
            candidate_token = candidate_token + self.previous_outcome_projection(previous_outcome)

        attended, _ = self.context_attention(candidate_token, memory, memory, need_weights=False)
        state_expanded = state_token[:, None, :].expand(-1, candidates, -1)
        fused = self.fusion(torch.cat((candidate_token, attended, state_expanded), dim=-1))
        if self.candidate_attention is not None:
            compared, _ = self.candidate_attention(
                fused,
                fused,
                fused,
                key_padding_mask=~candidate_mask,
                need_weights=False,
            )
            fused = fused + compared
        future_latent = self.future_head(fused)
        progress_logits = self.progress_head(fused).squeeze(-1)
        failure_logits = self.failure_head(fused).squeeze(-1)
        uncertainty = F.softplus(self.uncertainty_head(fused).squeeze(-1))
        if self.quality_head is None:
            score = (
                progress_logits.sigmoid()
                - self.config.failure_weight * failure_logits.sigmoid()
                - self.config.uncertainty_weight * uncertainty
            )
        else:
            score = self.quality_head(fused).squeeze(-1)
            score = score - score[:, self.config.baseline_candidate_index : self.config.baseline_candidate_index + 1]
        planned_future_latent = None
        plan_alignment = None
        if self.plan_head is not None and self.plan_fusion is not None:
            context_summary = memory.mean(dim=1)
            plan_token = self.plan_fusion(
                torch.cat((context_summary, state_token), dim=-1)
            )
            planned_future_latent = self.plan_head(plan_token)
            plan_alignment = -F.smooth_l1_loss(
                future_latent,
                planned_future_latent[:, None].expand_as(future_latent),
                reduction="none",
            ).mean(dim=-1)
            score = score + self.config.plan_alignment_weight * plan_alignment
            if self.config.baseline_relative_scoring:
                score = score - score[
                    :,
                    self.config.baseline_candidate_index : self.config.baseline_candidate_index
                    + 1,
                ]
        score = score.masked_fill(~candidate_mask, torch.finfo(score.dtype).min)
        return OutcomeVerifierOutput(
            future_latent=future_latent,
            progress_logits=progress_logits,
            failure_logits=failure_logits,
            uncertainty=uncertainty,
            score=score,
            selected_index=score.argmax(dim=1),
            planned_future_latent=planned_future_latent,
            plan_alignment=plan_alignment,
        )


    def losses(
        self,
        output: OutcomeVerifierOutput,
        *,
        future_target: Tensor,
        progress_target: Tensor,
        failure_target: Tensor,
        candidate_mask: Tensor,
        preferred_index: Tensor | None = None,
        quality_target: Tensor | None = None,
        planned_future_target: Tensor | None = None,
        future_target_mask: Tensor | None = None,
        planned_future_target_mask: Tensor | None = None,
        future_weight: float = 1.0,
        progress_weight: float = 1.0,
        failure_weight: float = 1.0,
        ranking_weight: float = 1.0,
        calibration_weight: float = 0.1,
        quality_weight: float = 1.0,
        plan_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        if future_target.shape != output.future_latent.shape:
            raise ValueError("future_target shape must match future_latent")
        if progress_target.shape != output.progress_logits.shape:
            raise ValueError("progress_target shape must match progress_logits")
        if failure_target.shape != output.failure_logits.shape:
            raise ValueError("failure_target shape must match failure_logits")
        if candidate_mask.shape != output.score.shape or candidate_mask.dtype != torch.bool:
            raise ValueError("candidate_mask must be boolean and match score shape")
        progress_target = progress_target.to(dtype=output.progress_logits.dtype)
        failure_target = failure_target.to(dtype=output.failure_logits.dtype)
        future_target = future_target.to(dtype=output.future_latent.dtype)

        effective_future_mask = (
            candidate_mask
            if future_target_mask is None
            else future_target_mask
        )
        if future_target_mask is not None and (
            future_target_mask.shape != future_target.shape
            or future_target_mask.dtype != torch.bool
        ):
            raise ValueError("future_target_mask must be boolean and match future_target")
        future = _masked_mean(
            F.smooth_l1_loss(output.future_latent, future_target, reduction="none"),
            effective_future_mask,
        )
        progress = _masked_mean(
            F.binary_cross_entropy_with_logits(
                output.progress_logits, progress_target, reduction="none"
            ),
            candidate_mask,
        )
        failure = _masked_mean(
            F.binary_cross_entropy_with_logits(
                output.failure_logits, failure_target, reduction="none"
            ),
            candidate_mask,
        )
        progress_error = (
            output.progress_logits.sigmoid().detach() - progress_target
        ).abs()
        calibration = _masked_mean(
            F.smooth_l1_loss(output.uncertainty, progress_error, reduction="none"),
            candidate_mask,
        )

        quality = output.score.new_zeros(())
        if quality_target is not None:
            if quality_target.shape != output.score.shape:
                raise ValueError("quality_target shape must match score")
            quality_target = quality_target.to(dtype=output.score.dtype)
            quality = _masked_mean(
                F.smooth_l1_loss(output.score, quality_target, reduction="none"),
                candidate_mask,
            )
            score_error = (output.score.detach() - quality_target).abs()
            calibration = _masked_mean(
                F.smooth_l1_loss(
                    output.uncertainty, score_error, reduction="none"
                ),
                candidate_mask,
            )

        plan = output.score.new_zeros(())
        if planned_future_target is not None:
            if output.planned_future_latent is None:
                raise ValueError("planned_future_target requires plan-outcome alignment")
            if planned_future_target.shape != output.planned_future_latent.shape:
                raise ValueError(
                    "planned_future_target shape must match planned future latent"
                )
            plan_error = F.smooth_l1_loss(
                output.planned_future_latent,
                planned_future_target.to(dtype=output.planned_future_latent.dtype),
                reduction="none",
            )
            if planned_future_target_mask is None:
                plan = plan_error.mean()
            else:
                if (
                    planned_future_target_mask.shape != planned_future_target.shape
                    or planned_future_target_mask.dtype != torch.bool
                ):
                    raise ValueError(
                        "planned_future_target_mask must be boolean and match target"
                    )
                plan = _masked_mean(plan_error, planned_future_target_mask)
        ranking = output.score.new_zeros(())
        if preferred_index is not None:
            batch, candidates = output.score.shape
            if preferred_index.shape != (batch,) or preferred_index.dtype != torch.long:
                raise ValueError("preferred_index must be int64 with shape [batch]")
            if ((preferred_index < 0) | (preferred_index >= candidates)).any():
                raise ValueError("preferred_index is outside candidate range")
            preferred_valid = candidate_mask.gather(1, preferred_index[:, None]).squeeze(1)
            if not preferred_valid.all():
                raise ValueError("preferred_index must reference a valid candidate")
            positive = output.score.gather(1, preferred_index[:, None])
            pair_mask = candidate_mask.clone()
            pair_mask.scatter_(1, preferred_index[:, None], False)
            pair_loss = F.softplus(-(positive - output.score))
            ranking = _masked_mean(pair_loss, pair_mask)

        total = (
            future_weight * future
            + progress_weight * progress
            + failure_weight * failure
            + ranking_weight * ranking
            + calibration_weight * calibration
            + quality_weight * quality
            + plan_weight * plan
        )
        return {
            "total": total,
            "future": future,
            "progress": progress,
            "failure": failure,
            "ranking": ranking,
            "calibration": calibration,
            "quality": quality,
            "plan": plan,
        }


class ActionConditionedOutcomeEnsemble(nn.Module):
    """Conservative V3.18 wrapper around independent outcome rankers."""

    def __init__(
        self,
        members: list[ActionConditionedOutcomeVerifier],
        *,
        uncertainty_penalty: float = 1.0,
        minimum_consensus: int = 4,
    ) -> None:
        super().__init__()
        if not members:
            raise ValueError("outcome ensemble requires at least one member")
        config = members[0].config
        if any(member.config != config for member in members[1:]):
            raise ValueError("outcome ensemble members must share one config")
        if not config.baseline_relative_scoring:
            raise ValueError("outcome ensemble requires baseline-relative members")
        if not 1 <= minimum_consensus <= len(members):
            raise ValueError("minimum_consensus must be inside the ensemble size")
        if uncertainty_penalty < 0.0:
            raise ValueError("uncertainty_penalty must be non-negative")
        self.members = nn.ModuleList(members)
        self.config = config
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.minimum_consensus = int(minimum_consensus)

    def forward(
        self,
        context: Tensor,
        state: Tensor,
        candidate_actions: Tensor,
        *,
        candidate_mask: Tensor | None = None,
        previous_outcome: Tensor | None = None,
    ) -> OutcomeVerifierOutput:
        if candidate_mask is None:
            candidate_mask = torch.ones(
                candidate_actions.shape[:2],
                dtype=torch.bool,
                device=candidate_actions.device,
            )
        outputs = [
            member(
                context,
                state,
                candidate_actions,
                candidate_mask=candidate_mask,
                previous_outcome=previous_outcome,
            )
            for member in self.members
        ]
        member_scores = torch.stack([output.score for output in outputs])
        score, score_deviation, _vote_count = conservative_ensemble_scores(
            member_scores,
            candidate_mask,
            baseline_index=self.config.baseline_candidate_index,
            uncertainty_penalty=self.uncertainty_penalty,
            minimum_consensus=self.minimum_consensus,
        )

        def mean(name: str) -> Tensor:
            return torch.stack([getattr(output, name) for output in outputs]).mean(dim=0)

        planned = [output.planned_future_latent for output in outputs]
        alignment = [output.plan_alignment for output in outputs]
        return OutcomeVerifierOutput(
            future_latent=mean("future_latent"),
            progress_logits=mean("progress_logits"),
            failure_logits=mean("failure_logits"),
            uncertainty=score_deviation,
            score=score,
            selected_index=score.argmax(dim=1),
            planned_future_latent=(
                torch.stack([value for value in planned if value is not None]).mean(dim=0)
                if all(value is not None for value in planned)
                else None
            ),
            plan_alignment=(
                torch.stack([value for value in alignment if value is not None]).mean(dim=0)
                if all(value is not None for value in alignment)
                else None
            ),
        )
