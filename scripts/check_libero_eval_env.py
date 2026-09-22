#!/usr/bin/env python3
"""Fail fast unless the dedicated LIBERO evaluation runtime is compatible."""

from __future__ import annotations

import sys

import huggingface_hub
import tokenizers
import transformers

from duvla.evaluation.initial_states import load_libero_initial_states

EXPECTED = {
    "huggingface_hub": "0.36.2",
    "transformers": "4.57.6",
    "tokenizers": "0.22.2",
}


def main() -> None:
    detected = {
        "huggingface_hub": huggingface_hub.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
    }
    mismatches = {
        name: (EXPECTED[name], version)
        for name, version in detected.items()
        if version != EXPECTED[name]
    }
    if mismatches:
        raise SystemExit(f"incompatible LIBERO evaluation runtime: {mismatches}")
    try:
        from libero.libero import benchmark
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"LIBERO/Qwen evaluation import failed: {exc}") from exc
    print(f"python={sys.executable}")
    for name, version in detected.items():
        print(f"{name}={version}")
    print("libero/qwen imports=ok")
    suite = benchmark.get_benchmark("libero_10")(task_order_index=0)
    states = load_libero_initial_states(suite, 0)
    if len(states) != 50:
        raise SystemExit(f"expected 50 official initial states, found {len(states)}")
    print(f"libero_10/task0 initial states={states.shape}, dtype={states.dtype}, safe_load=ok")


if __name__ == "__main__":
    main()
