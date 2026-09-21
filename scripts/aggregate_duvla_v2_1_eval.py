#!/usr/bin/env python3
"""Aggregate four strict Duvla V2.1 LIBERO suite summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duvla.training.feature_cache import atomic_write_json


SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suites: dict[str, object] = {}
    episodes = 0
    successes = 0
    checkpoint: str | None = None
    outcome_verifier: str | None = None
    outcome_verifier_initialized = False
    action_intent: str | None = None
    action_intent_initialized = False
    for suite in SUITES:
        path = args.root / suite / "summary.json"
        if not path.is_file():
            raise SystemExit(f"missing suite summary: {path}")
        payload = json.loads(path.read_text())
        if payload.get("complete") is not True:
            raise SystemExit(f"suite summary is incomplete: {path}")
        config = payload.get("run_config")
        aggregate = payload.get("aggregate")
        if not isinstance(config, dict) or not isinstance(aggregate, dict):
            raise SystemExit(f"suite summary has invalid structure: {path}")
        if config.get("benchmark_task_index") is not False or config.get("task_routing") != "natural_language":
            raise SystemExit(f"suite is not a strict natural-language evaluation: {path}")
        suite_checkpoint = str(config.get("checkpoint"))
        if checkpoint is None:
            checkpoint = suite_checkpoint
        elif checkpoint != suite_checkpoint:
            raise SystemExit("all four suite summaries must use one checkpoint")
        suite_verifier_raw = config.get("outcome_verifier")
        suite_verifier = None if suite_verifier_raw is None else str(suite_verifier_raw)
        if not outcome_verifier_initialized:
            outcome_verifier = suite_verifier
            outcome_verifier_initialized = True
        elif outcome_verifier != suite_verifier:
            raise SystemExit("all four suite summaries must use one outcome verifier")
        suite_intent_raw = config.get("action_intent")
        suite_intent = None if suite_intent_raw is None else str(suite_intent_raw)
        if not action_intent_initialized:
            action_intent = suite_intent
            action_intent_initialized = True
        elif action_intent != suite_intent:
            raise SystemExit("all four suite summaries must use one action-intent checkpoint")
        suite_episodes = int(aggregate["episode_count"])
        suite_successes = int(aggregate["success_count"])
        episodes += suite_episodes
        successes += suite_successes
        suites[suite] = aggregate
    if episodes != args.expected_episodes:
        raise SystemExit(f"expected {args.expected_episodes} episodes, found {episodes}")
    result = {
        "format": "duvla_v2_1_four_suite_aggregate",
        "checkpoint": checkpoint,
        "outcome_verifier": outcome_verifier,
        "action_intent": action_intent,
        "episode_count": episodes,
        "success_count": successes,
        "success_rate": successes / episodes,
        "suites": suites,
        "benchmark_task_index": False,
        "task_routing": "natural_language",
        "complete": True,
    }
    atomic_write_json(args.output, result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
