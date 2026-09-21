"""V85 candidate composition and candidate-pool upper-bound diagnostics.

These helpers deliberately contain no LIBERO environment logic.  They make
the current V85 inference rule explicit and analyse branch outcomes without
allowing diagnostic success labels to enter the V101 training cache.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class V85Decomposition:
    """Auditable components of the current five-flow V85 ensemble."""

    action: Tensor
    flow_base: Tensor
    direct_residual: Tensor
    task_residual: Tensor


@dataclass(frozen=True)
class CandidateOracleMetrics:
    """Success ceiling of a candidate pool, with and without baseline fallback."""

    state_count: int
    candidate_count: int
    baseline_successes: int
    candidate_oracle_successes: int
    fallback_oracle_successes: int
    recoverable_baseline_failures: int
    unrecoverable_baseline_failures: int

    @property
    def baseline_success_rate(self) -> float:
        return self.baseline_successes / self.state_count

    @property
    def candidate_oracle_success_rate(self) -> float:
        return self.candidate_oracle_successes / self.state_count

    @property
    def fallback_oracle_success_rate(self) -> float:
        return self.fallback_oracle_successes / self.state_count

    @property
    def maximum_absolute_gain(self) -> float:
        return self.fallback_oracle_success_rate - self.baseline_success_rate

    def to_dict(self) -> dict[str, int | float]:
        result: dict[str, int | float] = asdict(self)
        result.update(
            baseline_success_rate=self.baseline_success_rate,
            candidate_oracle_success_rate=self.candidate_oracle_success_rate,
            fallback_oracle_success_rate=self.fallback_oracle_success_rate,
            maximum_absolute_gain=self.maximum_absolute_gain,
        )
        return result


def _validate_v85_inputs(
    flow_candidates: Tensor,
    direct_actions: Tensor,
    task_residual: Tensor,
    direct_mix: float,
) -> None:
    if flow_candidates.ndim != 4:
        raise ValueError("flow_candidates must have shape [batch, candidates, horizon, action_dim]")
    expected = (
        flow_candidates.shape[0],
        flow_candidates.shape[2],
        flow_candidates.shape[3],
    )
    if tuple(direct_actions.shape) != expected:
        raise ValueError(f"direct_actions must have shape {expected}")
    if tuple(task_residual.shape) != expected:
        raise ValueError(f"task_residual must have shape {expected}")
    if not 0.0 <= direct_mix <= 1.0:
        raise ValueError("direct_mix must be in [0, 1]")


def compose_v85_candidates(
    flow_candidates: Tensor,
    direct_actions: Tensor,
    task_residual: Tensor,
    *,
    direct_mix: float = 0.25,
) -> Tensor:
    """Apply the exact per-candidate V51/V85 direct-action blend.

    The current model computes ``direct_prediction = direct + task_residual``
    and then blends that prediction with each Flow sample.  The task residual
    is therefore inside the direct mix; it is not added at full scale.
    """

    _validate_v85_inputs(flow_candidates, direct_actions, task_residual, direct_mix)
    direct_prediction = direct_actions + task_residual
    return (1.0 - direct_mix) * flow_candidates + direct_mix * direct_prediction[:, None]


def ensemble_v85_candidates(candidate_actions: Tensor, *, gripper_index: int = -1) -> Tensor:
    """Reproduce evaluator median aggregation and signed gripper voting."""

    if candidate_actions.ndim != 4:
        raise ValueError("candidate_actions must have shape [batch, candidates, horizon, action_dim]")
    if candidate_actions.shape[1] <= 0 or candidate_actions.shape[-1] <= 0:
        raise ValueError("candidate and action dimensions must be non-empty")
    index = gripper_index % candidate_actions.shape[-1]
    result = candidate_actions.median(dim=1).values
    votes = (candidate_actions[..., index] >= 0).sum(dim=1)
    result[..., index] = torch.where(
        votes * 2 >= candidate_actions.shape[1],
        torch.ones_like(result[..., index]),
        -torch.ones_like(result[..., index]),
    )
    return result


def compose_v1_101_candidate_pool(
    flow_candidates: Tensor, *, gripper_index: int = -1
) -> Tensor:
    """Build the V1.101 pool: signed Flow samples plus V1.85 median.

    Both cache collection and online rollout call this helper so the verifier
    never sees a candidate distribution that differs between training and
    evaluation.
    """

    if flow_candidates.ndim != 4:
        raise ValueError(
            "flow_candidates must have shape [batch, candidates, horizon, action_dim]"
        )
    if flow_candidates.shape[1] <= 0 or flow_candidates.shape[-1] <= 0:
        raise ValueError("candidate and action dimensions must be non-empty")
    index = gripper_index % flow_candidates.shape[-1]
    signed = flow_candidates.clone()
    signed[..., index] = torch.where(
        signed[..., index] >= 0,
        torch.ones_like(signed[..., index]),
        -torch.ones_like(signed[..., index]),
    )
    baseline = ensemble_v85_candidates(signed, gripper_index=index)
    return torch.cat((signed, baseline[:, None]), dim=1)


def decompose_v85_ensemble(
    flow_candidates: Tensor,
    direct_actions: Tensor,
    task_residual: Tensor,
    *,
    direct_mix: float = 0.25,
    gripper_index: int = -1,
) -> V85Decomposition:
    """Return an exact V85 action plus its progressive-residual interpretation.

    For arm dimensions, with ``B = median(flow_candidates)``, V85 is
    ``B + m * (direct - B) + m * task_residual``.  Gripper aggregation remains
    the evaluator's majority vote and is copied from the exact candidate
    composition rather than approximated by residual algebra.
    """

    candidates = compose_v85_candidates(
        flow_candidates, direct_actions, task_residual, direct_mix=direct_mix
    )
    exact = ensemble_v85_candidates(candidates, gripper_index=gripper_index)
    flow_base = flow_candidates.median(dim=1).values
    direct_residual = direct_actions - flow_base
    decomposition = flow_base + direct_mix * direct_residual + direct_mix * task_residual
    index = gripper_index % exact.shape[-1]
    decomposition[..., index] = exact[..., index]
    return V85Decomposition(
        action=decomposition,
        flow_base=flow_base,
        direct_residual=direct_residual,
        task_residual=task_residual,
    )


def candidate_oracle_metrics(
    candidate_success: Tensor,
    baseline_success: Tensor,
    *,
    candidate_mask: Tensor | None = None,
) -> CandidateOracleMetrics:
    """Measure whether a selector could improve over the median baseline.

    Success labels are intended only for a train-side branch diagnostic.  They
    must not be included in the gradient-training outcome cache.
    """

    if candidate_success.ndim != 2:
        raise ValueError("candidate_success must have shape [states, candidates]")
    states, candidates = candidate_success.shape
    if states <= 0 or candidates < 2:
        raise ValueError("at least one state and two candidates are required")
    if tuple(baseline_success.shape) != (states,):
        raise ValueError(f"baseline_success must have shape {(states,)}")
    successes = candidate_success.bool()
    baseline = baseline_success.bool()
    if candidate_mask is None:
        mask = torch.ones_like(successes)
    else:
        if tuple(candidate_mask.shape) != tuple(successes.shape):
            raise ValueError("candidate_mask must match candidate_success")
        mask = candidate_mask.bool()
        if not bool(mask.any(dim=1).all()):
            raise ValueError("every state must have at least one valid candidate")
    candidate_oracle = (successes & mask).any(dim=1)
    fallback_oracle = baseline | candidate_oracle
    recoverable = (~baseline) & candidate_oracle
    unrecoverable = (~baseline) & (~candidate_oracle)
    return CandidateOracleMetrics(
        state_count=states,
        candidate_count=candidates,
        baseline_successes=int(baseline.sum()),
        candidate_oracle_successes=int(candidate_oracle.sum()),
        fallback_oracle_successes=int(fallback_oracle.sum()),
        recoverable_baseline_failures=int(recoverable.sum()),
        unrecoverable_baseline_failures=int(unrecoverable.sum()),
    )
