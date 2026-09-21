"""Closed-loop selection and post-action verification for Duvla V2.3.

This module contains no LIBERO reward, task id, or simulator success access.
It converts learned outcome predictions into an auditable execute/replan
decision using only candidate actions and the next real observation latent.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from duvla.models.outcome_verifier import OutcomeVerifierOutput


@dataclass(frozen=True)
class OutcomePlannerConfig:
    execute_steps: int = 2
    smoothness_weight: float = 0.05
    prediction_error_threshold: float = 0.75
    uncertainty_tolerance: float = 0.25
    minimum_predicted_progress: float = 0.05

    def __post_init__(self) -> None:
        if self.execute_steps <= 0:
            raise ValueError("execute_steps must be positive")
        if min(
            self.smoothness_weight,
            self.prediction_error_threshold,
            self.uncertainty_tolerance,
            self.minimum_predicted_progress,
        ) < 0.0:
            raise ValueError("planner thresholds and penalties must be non-negative")


@dataclass(frozen=True)
class OutcomePlan:
    selected_index: Tensor
    selected_chunk: Tensor
    execution_prefix: Tensor
    predicted_future_latent: Tensor
    predicted_progress: Tensor
    predicted_failure: Tensor
    uncertainty: Tensor
    score: Tensor


@dataclass(frozen=True)
class OutcomeVerification:
    prediction_error: Tensor
    permitted_error: Tensor
    predicted_progress: Tensor
    should_replan: Tensor
    reason_prediction_mismatch: Tensor
    reason_low_progress: Tensor


def _gather_candidates(values: Tensor, indices: Tensor) -> Tensor:
    if indices.ndim != 1 or values.shape[0] != indices.shape[0]:
        raise ValueError("candidate gather batch shapes disagree")
    view = [indices.shape[0], 1] + [1] * (values.ndim - 2)
    expanded = indices.view(*view).expand(-1, 1, *values.shape[2:])
    return values.gather(1, expanded).squeeze(1)


def select_outcome_plan(
    output: OutcomeVerifierOutput,
    candidate_actions: Tensor,
    *,
    previous_action: Tensor,
    candidate_mask: Tensor,
    config: OutcomePlannerConfig,
) -> OutcomePlan:
    """Select one verifier-ranked chunk after a task-agnostic smoothness penalty."""

    if candidate_actions.ndim != 4:
        raise ValueError("candidate_actions must have shape [batch, candidates, horizon, action]")
    batch, candidates, horizon, action_dim = candidate_actions.shape
    if config.execute_steps > horizon:
        raise ValueError("execute_steps cannot exceed candidate horizon")
    if previous_action.shape != (batch, action_dim):
        raise ValueError("previous_action must have shape [batch, action_dim]")
    if candidate_mask.shape != (batch, candidates) or candidate_mask.dtype != torch.bool:
        raise ValueError("candidate_mask must be boolean [batch, candidates]")
    if not bool(candidate_mask.any(dim=1).all()):
        raise ValueError("every row requires a valid candidate")
    if output.score.shape != (batch, candidates):
        raise ValueError("verifier output and candidate pool shapes disagree")
    first_delta = candidate_actions[:, :, 0] - previous_action[:, None, :]
    internal_delta = candidate_actions[:, :, 1:] - candidate_actions[:, :, :-1]
    first_jerk = first_delta[..., :-1].square().mean(dim=-1)
    if internal_delta.shape[2]:
        internal_jerk = internal_delta[..., :-1].square().mean(dim=(-1, -2))
        jerk = 0.5 * (first_jerk + internal_jerk)
    else:
        jerk = first_jerk
    score = output.score - config.smoothness_weight * jerk
    score = score.masked_fill(~candidate_mask, torch.finfo(score.dtype).min)
    selected = score.argmax(dim=1)
    chunk = _gather_candidates(candidate_actions, selected)
    return OutcomePlan(
        selected_index=selected,
        selected_chunk=chunk,
        execution_prefix=chunk[:, : config.execute_steps],
        predicted_future_latent=_gather_candidates(output.future_latent, selected),
        predicted_progress=_gather_candidates(
            output.progress_logits.sigmoid().unsqueeze(-1), selected
        ).squeeze(-1),
        predicted_failure=_gather_candidates(
            output.failure_logits.sigmoid().unsqueeze(-1), selected
        ).squeeze(-1),
        uncertainty=_gather_candidates(
            output.uncertainty.unsqueeze(-1), selected
        ).squeeze(-1),
        score=score.gather(1, selected[:, None]).squeeze(1),
    )


def verify_observed_outcome(
    plan: OutcomePlan,
    observed_future_latent: Tensor,
    *,
    config: OutcomePlannerConfig,
) -> OutcomeVerification:
    """Trigger replanning without reading reward or simulator success."""

    if observed_future_latent.shape != plan.predicted_future_latent.shape:
        raise ValueError("observed future latent shape differs from the prediction")
    error = (observed_future_latent - plan.predicted_future_latent).float().square().mean(
        dim=-1
    ).sqrt()
    permitted = config.prediction_error_threshold + (
        config.uncertainty_tolerance * plan.uncertainty.float()
    )
    mismatch = error > permitted
    low_progress = plan.predicted_progress.float() < config.minimum_predicted_progress
    return OutcomeVerification(
        prediction_error=error,
        permitted_error=permitted,
        predicted_progress=plan.predicted_progress,
        should_replan=mismatch | low_progress,
        reason_prediction_mismatch=mismatch,
        reason_low_progress=low_progress,
    )
