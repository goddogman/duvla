"""Paired closed-loop outcome statistics for fixed LIBERO task/state sets."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PairedOutcome:
    suite: str
    task_id: int
    init_state_id: int
    baseline_success: bool
    candidate_success: bool


def _mcnemar_exact(gains: int, regressions: int) -> float:
    discordant = gains + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, index) for index in range(min(gains, regressions) + 1))
    return min(1.0, 2.0 * tail / (2**discordant))


def _rate(values: list[PairedOutcome], attribute: str) -> float:
    return sum(bool(getattr(value, attribute)) for value in values) / max(len(values), 1)


def paired_outcome_summary(
    outcomes: list[PairedOutcome],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 73,
) -> dict[str, object]:
    """Summarize paired outcomes with a task-cluster bootstrap interval."""

    if not outcomes:
        raise ValueError("paired outcomes cannot be empty")
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    keys = [(item.suite, item.task_id, item.init_state_id) for item in outcomes]
    if len(keys) != len(set(keys)):
        raise ValueError("paired outcomes contain duplicate suite/task/state keys")
    gains = sum(not item.baseline_success and item.candidate_success for item in outcomes)
    regressions = sum(item.baseline_success and not item.candidate_success for item in outcomes)
    baseline_successes = sum(item.baseline_success for item in outcomes)
    candidate_successes = sum(item.candidate_success for item in outcomes)

    clusters: dict[tuple[str, int], list[float]] = defaultdict(list)
    for item in outcomes:
        clusters[(item.suite, item.task_id)].append(
            float(item.candidate_success) - float(item.baseline_success)
        )
    cluster_values = torch.tensor(
        [sum(values) / len(values) for _, values in sorted(clusters.items())],
        dtype=torch.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    draws = torch.randint(
        cluster_values.numel(),
        (bootstrap_samples, cluster_values.numel()),
        generator=generator,
    )
    bootstrap = cluster_values.index_select(0, draws.flatten()).reshape(draws.shape).mean(dim=1)
    lower, upper = torch.quantile(bootstrap, torch.tensor([0.025, 0.975], dtype=torch.float64))

    suites: dict[str, dict[str, object]] = {}
    for suite in sorted({item.suite for item in outcomes}):
        values = [item for item in outcomes if item.suite == suite]
        suite_gains = sum(not item.baseline_success and item.candidate_success for item in values)
        suite_regressions = sum(
            item.baseline_success and not item.candidate_success for item in values
        )
        suites[suite] = {
            "episodes": len(values),
            "baseline_successes": sum(item.baseline_success for item in values),
            "candidate_successes": sum(item.candidate_success for item in values),
            "baseline_rate": _rate(values, "baseline_success"),
            "candidate_rate": _rate(values, "candidate_success"),
            "gains": suite_gains,
            "regressions": suite_regressions,
            "net_successes": suite_gains - suite_regressions,
        }
    tasks: dict[str, dict[str, object]] = {}
    for cluster in sorted(clusters):
        suite, task_id = cluster
        values = [
            item for item in outcomes if item.suite == suite and item.task_id == task_id
        ]
        task_gains = sum(not item.baseline_success and item.candidate_success for item in values)
        task_regressions = sum(
            item.baseline_success and not item.candidate_success for item in values
        )
        tasks[f"{suite}/task-{task_id:02d}"] = {
            "episodes": len(values),
            "baseline_successes": sum(item.baseline_success for item in values),
            "candidate_successes": sum(item.candidate_success for item in values),
            "gains": task_gains,
            "regressions": task_regressions,
            "net_successes": task_gains - task_regressions,
        }
    episodes = len(outcomes)
    return {
        "episodes": episodes,
        "task_clusters": len(clusters),
        "baseline_successes": baseline_successes,
        "candidate_successes": candidate_successes,
        "baseline_rate": baseline_successes / episodes,
        "candidate_rate": candidate_successes / episodes,
        "absolute_rate_change": (candidate_successes - baseline_successes) / episodes,
        "gains": gains,
        "regressions": regressions,
        "net_successes": gains - regressions,
        "mcnemar_exact_two_sided_p": _mcnemar_exact(gains, regressions),
        "task_cluster_bootstrap_95ci": [float(lower), float(upper)],
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "suites": suites,
        "tasks": tasks,
    }
