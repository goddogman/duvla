#!/usr/bin/env python3
"""Check a V3.31 policy bundle on CPU without loading Qwen or a simulator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from duvla.data import file_sha256
from duvla.evaluation.flow_contract import StandardizationStats
from evaluate_duvla_v2_1 import _load_policy, _validate_v331_deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    manifest = json.loads((args.model_dir / "train_manifest.json").read_text())
    policy, checkpoint = _load_policy(args.model_dir / "policy.pt", manifest, torch.device("cpu"))
    if checkpoint.get("version") != "V3.31" or checkpoint.get("format") != "duvla_v3_31":
        raise ValueError("expected the DuVLA V3.31 model bundle")
    protocol = argparse.Namespace(
        camera_size=128, fps=20, action_steps=2, flow_samples=5, flow_seed=23,
        no_flip_views=False, state_clip=None, outcome_verifier=None, action_intent=None,
        candidate_aggregation=None,
    )
    _validate_v331_deployment(checkpoint, protocol)
    config = json.loads((args.model_dir / "model_config.json").read_text())
    # JSON represents checkpoint tuples as lists; compare their JSON values.
    if config != json.loads(json.dumps(checkpoint["model_config"])):
        raise ValueError("model_config.json differs from the configuration inside policy.pt")
    state_stats = StandardizationStats.from_manifest(manifest, "state")
    action_stats = StandardizationStats.from_manifest(manifest, "action")
    if (state_stats.dim, action_stats.dim) != (policy.config.state_dim, policy.config.action_dim):
        raise ValueError("normalization dimensions differ from the policy configuration")
    print("V3.31 CPU model loading and bundle consistency: OK")
    print(f"policy_parameters={sum(p.numel() for p in policy.parameters()):,}")
    print(f"state_dim={state_stats.dim}, action_dim={action_stats.dim}")
    print(f"policy_sha256={file_sha256(args.model_dir / 'policy.pt')}")


if __name__ == "__main__":
    main()
