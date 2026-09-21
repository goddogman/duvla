#!/usr/bin/env python3
"""Compare two LIBERO runs on exactly matched suite/task/initial-state pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from duvla.evaluation.paired import PairedOutcome, paired_outcome_summary
from duvla.training.feature_cache import atomic_write_json


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument(
        "--suites",
        nargs="+",
        choices=SUITES,
        default=list(SUITES),
        help="Suites to compare; defaults to all four LIBERO suites.",
    )
    parser.add_argument("--init-state-start", type=int, default=0)
    parser.add_argument("--init-state-count", type=int)
    parser.add_argument(
        "--state-manifest",
        type=Path,
        help="Optional fixed-state manifest; use instead of a contiguous state interval.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _expected_from_manifest(
    path: Path, suites: tuple[str, ...]
) -> set[tuple[str, int, int]]:
    payload = json.loads(path.read_text())
    return {
        (suite, int(task), int(state))
        for suite, tasks in payload["suites"].items()
        if suite in suites
        for task, states in tasks.items()
        for state in states
    }


def _traces(
    root: Path,
    expected: set[tuple[str, int, int]],
    suites: tuple[str, ...],
) -> dict[tuple[str, int, int], bool]:
    result: dict[tuple[str, int, int], bool] = {}
    for suite in suites:
        trace_root = root / suite / "traces"
        if not trace_root.is_dir():
            trace_root = root / suite
        if not trace_root.is_dir():
            raise SystemExit(f"missing trace directory: {root / suite}")
        for path in trace_root.glob("task-*-episode-*.json"):
            payload = json.loads(path.read_text())
            summary = payload.get("summary", payload)
            task = int(summary["task_id"])
            state = int(summary["init_state_id"])
            key = (suite, task, state)
            if key not in expected:
                continue
            if key in result:
                raise SystemExit(f"duplicate trace key in {root}: {key}")
            result[key] = bool(summary["success"])
    missing = sorted(expected - set(result))
    extra = sorted(set(result) - expected)
    if missing or extra:
        raise SystemExit(
            f"trace set mismatch for {root}: missing={missing[:5]} extra={extra[:5]}"
        )
    return result


def main() -> None:
    args = parse_args()
    suites = tuple(args.suites)
    if args.state_manifest is not None:
        if args.init_state_count is not None:
            raise SystemExit("choose either --state-manifest or --init-state-count")
        expected = _expected_from_manifest(args.state_manifest, suites)
        state_selection: object = str(args.state_manifest.resolve())
    else:
        if args.init_state_start < 0 or not args.init_state_count or args.init_state_count <= 0:
            raise SystemExit("initial-state interval is invalid")
        stop = args.init_state_start + args.init_state_count
        expected = {
            (suite, task, state)
            for suite in suites
            for task in range(10)
            for state in range(args.init_state_start, stop)
        }
        state_selection = [args.init_state_start, stop - 1]
    baseline = _traces(args.baseline_root, expected, suites)
    candidate = _traces(args.candidate_root, expected, suites)
    if set(baseline) != set(candidate):  # pragma: no cover - protected by _traces
        raise SystemExit("baseline and candidate trace keys differ")
    outcomes = [
        PairedOutcome(
            suite=key[0],
            task_id=key[1],
            init_state_id=key[2],
            baseline_success=baseline[key],
            candidate_success=candidate[key],
        )
        for key in sorted(baseline)
    ]
    statistics = paired_outcome_summary(
        outcomes, bootstrap_samples=args.bootstrap_samples, seed=args.seed
    )
    payload = {
        "记录语言": "中文",
        "协议": "完全配对的 LIBERO suite/task/official-initial-state 闭环比较",
        "baseline_root": str(args.baseline_root.resolve()),
        "candidate_root": str(args.candidate_root.resolve()),
        "suites": list(suites),
        "state_selection": state_selection,
        "statistics": statistics,
    }
    atomic_write_json(args.output, payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
