"""Small, testable V8 temporal and phase conditioning modules.

These modules deliberately sit at the V8 conditioning boundary.  They do not
run Qwen and do not change the LIBERO action contract; the training entrypoint
will be wired only after the history/phase data contract is tested.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class TemporalHierarchicalConfig:
    """Shape contract for the V8 conditioning sidecar."""

    state_dim: int = 8
    action_dim: int = 7
    hidden_dim: int = 768
    history_length: int = 4
    phase_classes: int = 5
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if min(
            self.state_dim,
            self.action_dim,
            self.hidden_dim,
            self.history_length,
            self.phase_classes,
        ) <= 0:
            raise ValueError("V8 dimensions and class counts must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class StateActionHistoryEncoder(nn.Module):
    """Encode aligned state and previously executed action history.

    ``states[:, t]`` is paired with ``previous_actions[:, t]``.  At the first
    timestep the caller supplies a zero action.  The output is one token per
    batch item, so no image-history cache is required.
    """

    def __init__(self, config: TemporalHierarchicalConfig) -> None:
        super().__init__()
        self.config = config
        input_dim = config.state_dim + config.action_dim
        self.input_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, config.hidden_dim),
            nn.SiLU(),
        )
        self.position = nn.Parameter(
            torch.randn(config.history_length, config.hidden_dim) * 0.02
        )
        self.gru = nn.GRU(
            config.hidden_dim,
            config.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(config.hidden_dim)

    def forward(self, states: Tensor, previous_actions: Tensor) -> Tensor:
        expected_state = (states.shape[0], self.config.history_length, self.config.state_dim)
        expected_action = (
            states.shape[0],
            self.config.history_length,
            self.config.action_dim,
        )
        if tuple(states.shape) != expected_state:
            raise ValueError(f"states must have shape {expected_state}, got {tuple(states.shape)}")
        if tuple(previous_actions.shape) != expected_action:
            raise ValueError(
                f"previous_actions must have shape {expected_action}, got {tuple(previous_actions.shape)}"
            )
        tokens = torch.cat((states, previous_actions), dim=-1)
        tokens = self.input_projection(tokens)
        tokens = tokens + self.position[None, :, :]
        sequence, _ = self.gru(tokens)
        return self.output_norm(sequence[:, -1:, :])


class PhaseSubgoalConditioner(nn.Module):
    """Predict a coarse action phase and return a differentiable phase token."""

    phase_names = ("approach", "grasp", "transport", "place", "release")

    def __init__(self, config: TemporalHierarchicalConfig) -> None:
        super().__init__()
        if config.phase_classes != len(self.phase_names):
            raise ValueError("the initial V8 phase contract requires exactly five phase classes")
        self.config = config
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.phase_classes),
        )
        self.phase_embedding = nn.Embedding(config.phase_classes, config.hidden_dim)

    def forward(self, conditioning_token: Tensor, phase_labels: Tensor | None = None) -> tuple[Tensor, Tensor]:
        expected = (conditioning_token.shape[0], 1, self.config.hidden_dim)
        if tuple(conditioning_token.shape) != expected:
            raise ValueError(
                f"conditioning_token must have shape {expected}, got {tuple(conditioning_token.shape)}"
            )
        logits = self.classifier(conditioning_token[:, 0])
        if phase_labels is not None:
            if tuple(phase_labels.shape) != (conditioning_token.shape[0],):
                raise ValueError("phase_labels must have shape [batch]")
            if phase_labels.dtype not in (torch.int32, torch.int64):
                raise ValueError("phase_labels must be an integer tensor")
            if bool(((phase_labels < 0) | (phase_labels >= self.config.phase_classes)).any()):
                raise ValueError("phase_labels contain an invalid phase id")
            phase_token = self.phase_embedding(phase_labels)[:, None, :]
        else:
            probabilities = logits.softmax(dim=-1)
            phase_token = probabilities @ self.phase_embedding.weight
            phase_token = phase_token[:, None, :]
        return phase_token, logits


class TemporalHierarchicalConditioner(nn.Module):
    """Compose history and phase tokens without changing the action target."""

    def __init__(self, config: TemporalHierarchicalConfig | None = None) -> None:
        super().__init__()
        self.config = config or TemporalHierarchicalConfig()
        self.history = StateActionHistoryEncoder(self.config)
        self.phase = PhaseSubgoalConditioner(self.config)

    def forward(
        self,
        states: Tensor,
        previous_actions: Tensor,
        phase_labels: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        history_token = self.history(states, previous_actions)
        phase_token, logits = self.phase(history_token, phase_labels)
        return torch.cat((history_token, phase_token), dim=1), logits


class SubtaskProgressConditioner(nn.Module):
    """Predict and embed coarse subtask progress for long-horizon correction."""

    def __init__(self, hidden_dim: int, *, index_classes: int = 4, cycle_classes: int = 4) -> None:
        super().__init__()
        self.index_classes = index_classes
        self.cycle_classes = cycle_classes
        input_dim = 3 * hidden_dim
        self.index_head = nn.Linear(input_dim, index_classes)
        self.progress_head = nn.Linear(input_dim, 1)
        self.cycle_head = nn.Linear(input_dim, cycle_classes)
        self.index_embedding = nn.Embedding(index_classes, hidden_dim)
        self.cycle_embedding = nn.Embedding(cycle_classes, hidden_dim)
        self.progress_projection = nn.Linear(1, hidden_dim)
        self.fusion = nn.Sequential(
            nn.LayerNorm(3 * hidden_dim),
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        nn.init.zeros_(self.fusion[-1].weight)
        nn.init.zeros_(self.fusion[-1].bias)

    def forward(
        self,
        pooled_context: Tensor,
        state_token: Tensor,
        history_token: Tensor,
        *,
        subtask_index: Tensor | None = None,
        subtask_progress: Tensor | None = None,
        cycle_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if state_token.ndim == 3:
            state_token = state_token[:, 0]
        if history_token.ndim == 3:
            history_token = history_token[:, 0]
        condition = torch.cat((pooled_context, state_token, history_token), dim=-1)
        index_logits = self.index_head(condition)
        progress_prediction = self.progress_head(condition).sigmoid()[:, 0]
        cycle_logits = self.cycle_head(condition)
        if subtask_index is None:
            index_embedding = index_logits.softmax(dim=-1) @ self.index_embedding.weight
        else:
            index_embedding = self.index_embedding(subtask_index.long())
        if cycle_state is None:
            cycle_embedding = cycle_logits.softmax(dim=-1) @ self.cycle_embedding.weight
        else:
            cycle_embedding = self.cycle_embedding(cycle_state.long())
        progress_value = progress_prediction if subtask_progress is None else subtask_progress.float()
        progress_embedding = self.progress_projection(progress_value[:, None])
        embedding = self.fusion(
            torch.cat((index_embedding, cycle_embedding, progress_embedding), dim=-1)
        )
        return embedding, index_logits, progress_prediction, cycle_logits


class PersistentProgressMemory(nn.Module):
    """Compress a long within-episode visual history into progress slots."""

    def __init__(self, hidden_dim: int, *, slots: int = 4, heads: int = 4) -> None:
        super().__init__()
        if hidden_dim <= 0 or slots <= 0 or heads <= 0:
            raise ValueError("persistent memory dimensions must be positive")
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by persistent memory heads")
        self.hidden_dim = hidden_dim
        self.slots = slots
        self.input_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.query = nn.Parameter(torch.randn(slots, hidden_dim) * 0.02)
        self.query_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.sequence_norm = nn.RMSNorm(hidden_dim, eps=1e-6)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.output = nn.Sequential(
            nn.RMSNorm(hidden_dim, eps=1e-6),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )

    def forward(self, sequence: Tensor) -> Tensor:
        if sequence.ndim != 3 or sequence.shape[-1] != self.hidden_dim:
            raise ValueError(
                "persistent memory sequence must have shape [batch, history, hidden_dim]"
            )
        if sequence.shape[1] <= 0:
            raise ValueError("persistent memory sequence cannot be empty")
        encoded, _ = self.gru(self.input_norm(sequence))
        queries = self.query_norm(self.query)[None, :, :].expand(sequence.shape[0], -1, -1)
        normalized_sequence = self.sequence_norm(encoded)
        attended, _ = self.attention(
            queries,
            normalized_sequence,
            normalized_sequence,
            need_weights=False,
        )
        return queries + self.output(attended)
