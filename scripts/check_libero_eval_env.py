#!/usr/bin/env python3
"""Fail fast unless the dedicated LIBERO evaluation runtime is compatible."""

from __future__ import annotations

import sys

import huggingface_hub
import tokenizers
import transformers


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
        import libero  # noqa: F401
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: F401
    except ImportError as exc:
        raise SystemExit(f"LIBERO/Qwen evaluation import failed: {exc}") from exc
    print(f"python={sys.executable}")
    for name, version in detected.items():
        print(f"{name}={version}")
    print("libero/qwen imports=ok")


if __name__ == "__main__":
    main()
