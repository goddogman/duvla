"""Detached, episode-local execution history for the V3.26 feedback adapter.

The action argument is always the command actually sent to ``env.step`` in
environment units, after clipping.  This tracker standardizes it with the
training statistics; storing an unexecuted network prediction is not equivalent.
"""

from __future__ import annotations

from dataclasses import fields
from typing import TYPE_CHECKING, Mapping

import torch
from torch import Tensor

from .flow_contract import StandardizationStats

if TYPE_CHECKING:
    from duvla.models.duvla_v3_26 import ExecutionFeedback


def _finite_float(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains non-finite values")


def _bool_rows(name: str, value: Tensor, batch: int, device: torch.device) -> None:
    if value.dtype != torch.bool or value.shape != (batch,) or value.device != device:
        raise ValueError(f"{name} must be a boolean [batch] tensor on the input device")


def _copy_feedback(feedback: ExecutionFeedback) -> ExecutionFeedback:
    return type(feedback)(
        **{field.name: getattr(feedback, field.name).detach().clone() for field in fields(feedback)}
    )


class ExecutionFeedbackTracker:
    """Keep one previous decision and its observed execution, never its graph.

    Call ``reset()`` at every environment reset.  Batched environments may use
    ``reset(reset_mask=...)``.  Optional episode IDs additionally protect callers
    that reuse a tracker across episodes without a global reset.
    """

    def __init__(
        self,
        action_stats: StandardizationStats,
        *,
        spatial_tokens_per_camera: int = 64,
    ) -> None:
        if spatial_tokens_per_camera != 64:
            raise ValueError("V3.26 requires exactly 64 spatial tokens per camera")
        mean = torch.tensor(action_stats.mean, dtype=torch.float32)
        std = torch.tensor(action_stats.std, dtype=torch.float32)
        if mean.shape != (7,) or std.shape != (7,):
            raise ValueError("action statistics must have seven dimensions")
        if not bool(torch.isfinite(mean).all() and torch.isfinite(std).all() and (std > 0).all()):
            raise ValueError("action statistics must be finite with positive standard deviations")
        self._mean = mean
        self._std = std
        self._feedback: ExecutionFeedback | None = None
        self._episode_ids: Tensor | None = None

    def reset(self, *, reset_mask: Tensor | None = None) -> None:
        """Discard all history or just the rows belonging to reset environments."""
        if reset_mask is None:
            self._feedback = None
            self._episode_ids = None
            return
        if self._feedback is None:
            if reset_mask.dtype != torch.bool or reset_mask.ndim != 1:
                raise ValueError("reset_mask must be a boolean [batch] tensor")
            return
        feedback = self._feedback
        _bool_rows("reset_mask", reset_mask, feedback.valid.shape[0], feedback.valid.device)
        self._invalidate_rows(reset_mask)

    def _invalidate_rows(self, invalid: Tensor) -> None:
        if self._feedback is None:
            return
        feedback = self._feedback
        for field in fields(feedback):
            value = getattr(feedback, field.name)
            value[invalid] = 0
        if not bool(feedback.valid.any()):
            self._feedback = None
            self._episode_ids = None

    @staticmethod
    def _validate_episode_ids(episode_ids: Tensor, batch: int, device: torch.device) -> None:
        if (
            episode_ids.shape != (batch,)
            or episode_ids.dtype not in (torch.int32, torch.int64)
            or episode_ids.device != device
        ):
            raise ValueError("episode_ids must be an integer [batch] tensor on the input device")

    def record_execution(
        self,
        context: Tensor,
        state: Tensor,
        executed_actions: Tensor,
        executed_mask: Tensor,
        prediction: Mapping[str, Tensor] | None = None,
        *,
        prediction_valid: Tensor | None = None,
        episode_ids: Tensor | None = None,
    ) -> None:
        """Record a completed/partial prefix and the forecast made before it.

        ``context`` is [B,T,D], with the first 128 tokens in camera-major
        spatial order. ``state`` is normalized [B,8]. ``executed_actions`` is
        [B,2,7] in environment units, and ``executed_mask`` is a boolean prefix
        mask. A two-step forecast is invalidated for any shorter execution.
        """
        from duvla.models.duvla_v3_26 import ExecutionFeedback

        for name, value in (
            ("context", context), ("state", state), ("executed_actions", executed_actions)
        ):
            _finite_float(name, value)
        if (
            context.ndim != 3 or context.shape[0] == 0
            or context.shape[1] < 128 or context.shape[2] == 0
        ):
            raise ValueError("context must have shape [batch, at least 128 tokens, width]")
        batch, _, width = context.shape
        device = context.device
        if state.shape != (batch, 8) or executed_actions.shape != (batch, 2, 7):
            raise ValueError("state/actions must have shapes [batch,8] and [batch,2,7]")
        if state.device != device or executed_actions.device != device:
            raise ValueError("context, state, and executed actions must share one device")
        if (
            executed_mask.shape != (batch, 2)
            or executed_mask.dtype != torch.bool
            or executed_mask.device != device
        ):
            raise ValueError("executed_mask must be boolean [batch,2] on the input device")
        if bool((executed_mask[:, 1] & ~executed_mask[:, 0]).any()):
            raise ValueError("executed_mask must describe a contiguous executed prefix")
        if bool((executed_actions[executed_mask].abs() > 1.0 + 1e-6).any()):
            raise ValueError("executed actions must be actual clipped environment commands")
        if episode_ids is not None:
            self._validate_episode_ids(episode_ids, batch, device)
        valid = executed_mask.any(dim=1)
        complete = executed_mask.all(dim=1)
        if prediction_valid is not None:
            _bool_rows("prediction_valid", prediction_valid, batch, device)
        if prediction is None:
            state_delta = state.new_zeros(batch, 8)
            visual_delta = context.new_zeros(batch, 2, width)
            forecast_valid = torch.zeros_like(valid)
        else:
            if set(prediction) != {"state_delta", "visual_delta"}:
                raise ValueError("prediction must contain state_delta and visual_delta only")
            state_delta = prediction["state_delta"]
            visual_delta = prediction["visual_delta"]
            _finite_float("predicted state delta", state_delta)
            _finite_float("predicted visual delta", visual_delta)
            if state_delta.shape != (batch, 8) or visual_delta.shape != (batch, 2, width):
                raise ValueError("prediction shapes must be [batch,8] and [batch,2,width]")
            if state_delta.device != device or visual_delta.device != device:
                raise ValueError("predictions must share the context device")
            forecast_valid = complete if prediction_valid is None else complete & prediction_valid
        normalized = (
            executed_actions.float() - self._mean.to(device)
        ) / self._std.to(device)
        normalized = normalized.to(dtype=state.dtype)
        normalized = normalized.masked_fill(~executed_mask[..., None], 0)
        state_delta = state_delta.masked_fill(~forecast_valid[:, None], 0)
        visual_delta = visual_delta.masked_fill(~forecast_valid[:, None, None], 0)
        self._feedback = _copy_feedback(ExecutionFeedback(
            previous_context=context,
            previous_state=state,
            executed_actions=normalized,
            executed_mask=executed_mask,
            predicted_state_delta=state_delta,
            predicted_visual_delta=visual_delta,
            valid=valid,
            prediction_valid=forecast_valid,
        ))
        self._episode_ids = None if episode_ids is None else episode_ids.detach().clone()
        self._invalidate_rows(~valid)

    def get_feedback(self, *, episode_ids: Tensor | None = None) -> ExecutionFeedback | None:
        """Return an independent snapshot; new episodes cannot reuse stale rows."""
        if self._feedback is None:
            return None
        if episode_ids is not None:
            self._validate_episode_ids(
                episode_ids, self._feedback.valid.shape[0], self._feedback.valid.device
            )
            if self._episode_ids is None:
                raise ValueError("episode_ids must also be supplied when recording execution")
            self._invalidate_rows(episode_ids != self._episode_ids)
        return None if self._feedback is None else _copy_feedback(self._feedback)

    def diagnostics(
        self, current_context: Tensor, current_state: Tensor
    ) -> dict[str, float | int | bool | None]:
        """Compare actual changes to the pre-execution prediction for traces.

        The visual summary uses only two groups of 64 spatial tokens. Semantic
        and history tokens never enter the predicted-vs-actual visual change.
        These are normalized diagnostic errors, not physical task success.
        """
        from duvla.models.duvla_v3_26 import spatial_summary

        feedback = self.get_feedback()
        if feedback is None:
            return {
                "feedback_valid": False,
                "prediction_valid": False,
                "feedback_rows": 0,
                "forecast_rows": 0,
                "state_prediction_error": None,
                "visual_prediction_error": None,
            }
        _finite_float("current_context", current_context)
        _finite_float("current_state", current_state)
        if (
            current_context.shape != feedback.previous_context.shape
            or current_state.shape != feedback.previous_state.shape
        ):
            raise ValueError("current observation shapes must match the recorded decision")
        if (
            current_context.device != feedback.previous_context.device
            or current_state.device != feedback.previous_state.device
        ):
            raise ValueError("current observation must share the recorded device")
        actual_state = current_state.float() - feedback.previous_state.float()
        actual_visual = (
            spatial_summary(current_context.float())
            - spatial_summary(feedback.previous_context.float())
        )
        valid = feedback.valid
        forecast = feedback.prediction_valid & valid
        return {
            "feedback_valid": bool(valid.any()),
            "prediction_valid": bool(forecast.any()),
            "feedback_rows": int(valid.sum()),
            "forecast_rows": int(forecast.sum()),
            "actual_state_delta_l1": float(actual_state[valid].abs().mean()),
            "actual_visual_delta_l1": float(actual_visual[valid].abs().mean()),
            "state_prediction_error": (
                float(
                    (actual_state[forecast] - feedback.predicted_state_delta[forecast]).abs().mean()
                )
                if bool(forecast.any()) else None
            ),
            "visual_prediction_error": (
                float(
                    (actual_visual[forecast] - feedback.predicted_visual_delta[forecast]).abs().mean()
                )
                if bool(forecast.any()) else None
            ),
        }
