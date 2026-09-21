"""Sparse paired-observation sidecar for the V3.31 joint adaptation run."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import Tensor


class V331AugmentSidecar:
    """Load only augmented current observations; histories stay in the base batch."""

    def __init__(
        self,
        root: Path,
        base: Path,
        language_sidecar: Path,
        *,
        allow_smoke: bool = False,
    ) -> None:
        self.root = root
        self.manifest = json.loads((root / "manifest.json").read_text())
        self.signature = sha256((root / "manifest.json").read_bytes()).hexdigest()
        expected_base = sha256((base / "manifest.json").read_bytes()).hexdigest()
        expected_language = sha256((language_sidecar / "manifest.json").read_bytes()).hexdigest()
        if self.manifest.get("base_manifest_sha256") != expected_base:
            raise ValueError("V3.31 augmentation/base manifest mismatch")
        if self.manifest.get("language_manifest_sha256") != expected_language:
            raise ValueError("V3.31 augmentation/language manifest mismatch")
        if self.manifest.get("uses_evaluation_initial_states") is not False:
            raise ValueError("V3.31 augmentation provenance violates the training boundary")
        if not self.manifest.get("complete"):
            raise ValueError("V3.31 augmentation cache is incomplete")
        if not allow_smoke and self.manifest.get("formal") is not True:
            raise ValueError("formal V3.31 training rejects a smoke augmentation cache")
        indices = self.manifest.get("selected_indices")
        if not isinstance(indices, list) or not indices:
            raise ValueError("V3.31 manifest lacks selected_indices")
        if indices != sorted(set(int(value) for value in indices)):
            raise ValueError("V3.31 selected_indices must be sorted and unique")
        shard_size = int(self.manifest["shard_size"])
        self.locations = {
            int(index): (position // shard_size, position % shard_size)
            for position, index in enumerate(indices)
        }
        self.shards: OrderedDict[int, dict[str, object]] = OrderedDict()

    def _payload(self, shard: int) -> dict[str, object]:
        if shard not in self.shards:
            payload = torch.load(
                self.root / f"shard-{shard:06d}.pt",
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            if payload.get("contract_sha256") != self.manifest.get("contract_sha256"):
                raise ValueError("V3.31 augmentation shard contract mismatch")
            self.shards[shard] = payload
            if len(self.shards) > 8:
                self.shards.popitem(last=False)
        self.shards.move_to_end(shard)
        return self.shards[shard]

    def fetch(self, batch: dict[str, Tensor]) -> dict[str, Tensor] | None:
        slots: list[int] = []
        rows: list[tuple[dict[str, object], int]] = []
        for slot, index in enumerate(batch["dataset_indices"].tolist()):
            location = self.locations.get(int(index))
            if location is None:
                continue
            shard_id, local = location
            payload = self._payload(shard_id)
            stored = payload["dataset_indices"]
            if not isinstance(stored, Tensor) or int(stored[local]) != int(index):
                raise ValueError("V3.31 augmentation index mismatch")
            slots.append(slot)
            rows.append((payload, local))
        if not rows:
            return None
        lengths = [
            int(payload["language_offsets"][local + 1] - payload["language_offsets"][local])
            for payload, local in rows
        ]
        longest = max(lengths)
        feature_dim = int(rows[0][0]["language"].shape[-1])
        language = torch.zeros(
            len(rows), 2, longest, feature_dim, dtype=rows[0][0]["language"].dtype
        )
        language_mask = torch.zeros(len(rows), 2, longest, dtype=torch.bool)
        visuals, semantics = [], []
        for output_row, ((payload, local), length) in enumerate(zip(rows, lengths)):
            begin = int(payload["language_offsets"][local])
            end = int(payload["language_offsets"][local + 1])
            language[output_row, :, :length] = payload["language"][begin:end].transpose(0, 1)
            language_mask[output_row, :, :length] = True
            visuals.append(payload["visual"][local])
            semantics.append(payload["semantic"][local])
        return {
            "slots": torch.tensor(slots, dtype=torch.long),
            "visual": torch.stack(visuals),
            "semantic": torch.stack(semantics),
            "language_tokens": language,
            "language_mask": language_mask,
        }
