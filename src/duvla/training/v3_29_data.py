"""Immutable, source-bound V3.29 language/causal-action sidecars."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
import json
from pathlib import Path

import torch
from torch import Tensor


def causal_chunks(actions: Tensor, horizon: int = 8) -> tuple[Tensor, Tensor]:
    if actions.ndim != 2 or actions.shape[1] != 7 or not torch.isfinite(actions).all():
        raise ValueError('finite episode actions [N,7] required')
    n = len(actions)
    result = actions.new_zeros((n, horizon, 7))
    mask = torch.zeros(n, horizon, dtype=torch.bool)
    for step in range(horizon):
        count = max(n-step-1, 0)
        result[:count, step] = actions[step+1:step+1+count]
        mask[:count, step] = True
    return result, mask


class V329Sidecar:
    def __init__(self, root: Path, base: Path, *, allow_partial: bool = False):
        self.root = root
        self.manifest = json.loads((root/'manifest.json').read_text())
        self.signature = sha256((root/'manifest.json').read_bytes()).hexdigest()
        if self.manifest['base_manifest_sha256'] != sha256((base/'manifest.json').read_bytes()).hexdigest():
            raise ValueError('sidecar/base manifest mismatch')
        if self.manifest.get('uses_evaluation_initial_states') is not False or self.manifest.get('action_offset') != 1:
            raise ValueError('sidecar violates training provenance or causal offset')
        if not allow_partial and not self.manifest['complete']:
            raise ValueError('sidecar incomplete; formal training forbidden')
        self.shards: OrderedDict[int, dict] = OrderedDict()

    def transform(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        languages, masks, actions, valid, previous = [], [], [], [], []
        for index in batch['dataset_indices'].tolist():
            key = index // self.manifest['shard_size']
            if key not in self.shards:
                self.shards[key] = torch.load(self.root/f'shard-{key:06d}.pt', weights_only=True, mmap=True)
                if len(self.shards) > 8:
                    self.shards.popitem(last=False)
            payload = self.shards[key]
            self.shards.move_to_end(key)
            local = index - int(payload['dataset_indices'][0])
            if int(payload['dataset_indices'][local]) != index:
                raise ValueError('sidecar index mismatch')
            begin, end = payload['offsets'][local:local+2].tolist()
            n = end - begin
            if not 0 < n <= 32:
                raise ValueError('invalid packed instruction length')
            token = torch.zeros(2, 32, payload['language'].shape[-1], dtype=payload['language'].dtype)
            token[:, :n] = payload['language'][begin:end].transpose(0, 1)
            mask = torch.arange(32)[None].expand(2, -1) < n
            languages.append(token); masks.append(mask)
            actions.append(payload['actions'][local]); valid.append(payload['valid_mask'][local])
            previous.append(payload['previous_actions'][local])
        batch = dict(batch, language_tokens=torch.stack(languages), language_mask=torch.stack(masks),
                     actions=torch.stack(actions), valid_mask=torch.stack(valid), previous_actions=torch.stack(previous))
        keep = batch['valid_mask'].any(-1)
        return {name: value[keep] for name, value in batch.items()}
