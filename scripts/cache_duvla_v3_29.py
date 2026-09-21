#!/usr/bin/env python3
"""Extract exact ordered language once; reuse immutable V3.27 visual features."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from hashlib import sha256
import json
from pathlib import Path
import time

import h5py
import numpy as np
from PIL import Image
import torch

from cache_official_libero_hdf5_features import _dataset_audit
from duvla.evaluation.libero_contract import orient_libero_view
from duvla.models.qwen_backbone import QwenVLBackbone
from duvla.training.feature_cache import atomic_write_json, load_feature_shard
from duvla.training.v3_29_data import causal_chunks
from duvla.training.resource_budget import require_disk_budget


def atomic_save(path: Path, payload: dict) -> None:
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-cache', type=Path, required=True)
    parser.add_argument('--dataset-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-shards', type=int, default=0, help='isolated diagnostic only')
    args = parser.parse_args()
    if args.max_shards < 0:
        raise ValueError('max shards must be nonnegative')
    base_path = args.base_cache/'manifest.json'
    base = json.loads(base_path.read_text())
    audit = _dataset_audit(args.dataset_root)
    if base['source_contract_sha256'] != audit.source_contract_sha256 or not base['all_demonstrations']:
        raise ValueError('source provenance mismatch')
    if not args.max_shards:
        old_path = args.output/'manifest.json'
        old = json.loads(old_path.read_text()) if old_path.exists() else {}
        remaining = 1-min(1,old.get('completed_shards',0)/max(1,old.get('shard_count',5291)))
        require_disk_budget(args.base_cache,minimum_gib=30+35*remaining)
    backbone = QwenVLBackbone().load(device=torch.device('cuda'))
    total = base['selected_frames']; size = base['shard_size']
    count = (total+size-1)//size
    contract = {
        'version': 'V3.29', 'format': 'v329_ordered_language_causal_actions', 'action_offset': 1,
        'base_manifest_sha256': sha256(base_path.read_bytes()).hexdigest(),
        'source_contract_sha256': audit.source_contract_sha256,
        'code_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'backbone_code_sha256': sha256(Path('src/duvla/models/qwen_backbone.py').read_bytes()).hexdigest(),
        'source_rows': total, 'eligible_rows': total-len(audit.demos), 'shard_size': size,
        'shard_count': count, 'batch_size': 4, 'instruction_layer': -1, 'max_tokens': 32,
        'train_demonstrations': len(audit.demos), 'uses_evaluation_initial_states': False,
        'model_path': str(backbone.config.model_path),
    }
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if any(old.get(k) != v for k,v in contract.items()):
            raise ValueError('existing sidecar contract differs; do not overwrite')
        if old.get('complete'):
            print('V3.29缓存已完成，无需重复'); return
    starts = [d.dataset_start for d in audit.demos]
    opened = None; opened_path = None; active_demo = None; targets = None
    def source_item(index):
        nonlocal opened, opened_path, active_demo, targets
        meta = audit.demos[bisect_right(starts, index)-1]
        if opened_path != meta.task.path:
            if opened is not None: opened.close()
            opened = h5py.File(meta.task.path, 'r'); opened_path = meta.task.path
        if active_demo != (str(meta.task.path), meta.demo_key):
            demo = opened['data'][meta.demo_key]
            action = torch.from_numpy(np.asarray(demo['actions'], dtype=np.float32))
            action = (action-torch.tensor(base['action_mean']))/torch.tensor(base['action_std'])
            targets = causal_chunks(action)
            active_demo = (str(meta.task.path), meta.demo_key)
        frame = index-meta.dataset_start
        obs = opened['data'][meta.demo_key]['obs']
        images = [Image.fromarray(orient_libero_view(obs[key][frame])) for key in ('agentview_rgb','eye_in_hand_rgb')]
        current = torch.from_numpy(np.asarray(opened['data'][meta.demo_key]['actions'][frame], dtype=np.float32))
        current = (current-torch.tensor(base['action_mean']))/torch.tensor(base['action_std'])
        return images, meta.task.instruction, targets[0][frame].clone(), targets[1][frame].clone(), current
    limit = min(count, args.max_shards or count)
    completed = 0; started = time.monotonic(); new_rows = 0
    try:
        for shard_id in range(limit):
            path = args.output/f'shard-{shard_id:06d}.pt'
            start, stop = shard_id*size, min((shard_id+1)*size, total)
            if path.exists():
                existing = torch.load(path, weights_only=True, mmap=True)
                if existing.get('contract') != contract or existing['dataset_indices'].tolist() != list(range(start,stop)):
                    raise ValueError(f'invalid existing shard: {path}')
                completed += 1; continue
            language, offsets, token_ids, action, valid, previous = [], [0], [], [], [], []
            old_shard = load_feature_shard(args.base_cache/path.name, mmap=True) if shard_id % 128 == 0 else None
            audit_error = 0.
            for first in range(start, stop, 4):
                rows = [source_item(i) for i in range(first, min(first+4,stop))]
                instructions = [r[1] for r in rows]
                inputs = [backbone.prepare_view_batch_inputs([(r[0][camera],r[1]) for r in rows]) for camera in (0,1)]
                visual, semantic, words, mask = backbone.forward_v329_context(inputs,instructions)
                visual = visual[:, 1:2]
                if old_shard is not None:
                    local = first-start
                    audit_error = max(audit_error, float((visual.cpu()-old_shard.features[local:local+len(rows)]).abs().max()),
                                      float((semantic.cpu()-old_shard.semantic_features[local:local+len(rows)]).abs().max()))
                    if audit_error != 0:
                        raise ValueError('V3.29 extraction changed old spatial/summary features')
                for row, source in enumerate(rows):
                    length = int(mask[row,0].sum())
                    language.append(words[row,:,:length].transpose(0,1).cpu().contiguous())
                    offsets.append(offsets[-1]+length)
                    token_ids.append(backbone.processor.tokenizer.encode(source[1],add_special_tokens=False))
                    action.append(source[2]);valid.append(source[3]);previous.append(source[4])
            atomic_save(path, {'contract':contract,'dataset_indices':torch.arange(start,stop),
                'language':torch.cat(language),'offsets':torch.tensor(offsets), 'token_ids':token_ids,
                'actions':torch.stack(action),'valid_mask':torch.stack(valid),
                'previous_actions':torch.stack(previous),
                'old_fields_max_error':audit_error if old_shard is not None else None})
            completed += 1; new_rows += stop-start
            atomic_write_json(manifest_path,{**contract,'complete':False,'completed_shards':completed})
            atomic_write_json(args.output/'progress.json',{
                '阶段':'V3.29有序语言和因果标签缓存', 'completed_shards':completed,'total_shards':count,
                'rows_per_second':new_rows/max(time.monotonic()-started,1e-6), 'elapsed_seconds':time.monotonic()-started})
            if completed % 250 == 0 or completed == limit:
                print(f'V3.29缓存 {completed}/{count}',flush=True)
                if not args.max_shards:
                    require_disk_budget(args.base_cache,minimum_gib=30+35*(1-completed/count))
    finally:
        if opened is not None: opened.close()
    atomic_write_json(manifest_path,{**contract,'complete':limit==count,'completed_shards':completed})


if __name__ == '__main__': main()
