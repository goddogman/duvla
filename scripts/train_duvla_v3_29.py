#!/usr/bin/env python3
"""One preregistered 30E V3.29 run with deterministic, atomic recovery."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, replace
from hashlib import sha256
import json
import math
from pathlib import Path
import time

import torch

from train_duvla_v2_1 import _model_config, _configure_stage, _seed_everything, _atomic_torch_save, _assert_fp32_training_state, _task_weights
from duvla.models.duvla_v2_1 import DuvlaV21Policy
from duvla.training.feature_cache import atomic_write_json
from duvla.training.v3_29_data import V329Sidecar
from duvla.training.resource_budget import require_disk_budget
from duvla.training.v2_1_data import iter_v2_1_batches, prefetch_v2_1_batches
from duvla.training.loss_curve import append_loss_point, truncate_loss_points, write_loss_curve_artifacts


def make_policy(manifest: dict, seed: int, *, language: bool = True, tiny: bool = False,
                v330: bool = False):
    _seed_everything(seed)
    config = replace(
        _model_config(manifest, smoke=tiny, v3_28_fp32_amp_flow=True),
        ordered_language_bridge=language,
        ordered_language_bridge_mode="zero_output" if v330 else "scalar_gate",
        causal_action_attention=not v330,
        candidate_aggregation="trajectory_medoid" if v330 else "coordinate_median",
    )
    policy = DuvlaV21Policy(config)
    groups = _configure_stage(policy, 'flow')
    if policy.language_bridge is not None:
        policy.language_bridge.requires_grad_(True)
        if v330:
            policy.language_bridge.gate.requires_grad_(False)
        groups += ('language_bridge',)
    return policy, config, groups


def main(*, version: str = "V3.29", entrypoint: str = "scripts/train_duvla_v3_29.py") -> None:
    if version not in {"V3.29", "V3.30"}:
        raise ValueError("unsupported training recipe version")
    v330 = version == "V3.30"
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-cache', type=Path, required=True)
    parser.add_argument('--sidecar', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=192)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--without-language', action='store_true')
    parser.add_argument('--smoke-updates', type=int, default=0)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if min(args.batch_size,args.epochs) <= 0 or args.smoke_updates < 0:
        raise ValueError('invalid training budget')
    if not args.smoke_updates and args.epochs != 30:
        raise ValueError(f'formal {version} must plan 30E from initialization')
    if not args.smoke_updates:
        require_disk_budget(args.base_cache,minimum_gib=20)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise ValueError(f'{version} training requires the preflighted GPU')
    base = json.loads((args.base_cache/'manifest.json').read_text())
    sidecar = V329Sidecar(args.sidecar,args.base_cache,allow_partial=bool(args.smoke_updates))
    # A GPU smoke must exercise the formal-capacity model; the tiny option is
    # reserved for CPU unit tests that call make_policy directly.
    policy, config, groups = make_policy(base,args.seed,language=not args.without_language,tiny=False,v330=v330)
    policy.to(device=device,dtype=torch.float32)
    parameters = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters,lr=2e-4,weight_decay=.01,fused=True)
    _assert_fp32_training_state(parameters,optimizer)
    n = int(sidecar.manifest['eligible_rows']); per_epoch = math.ceil(n/args.batch_size)
    total_steps = per_epoch*args.epochs; warmup=max(1,round(total_steps*.03))
    def factor(step):
        if step < warmup: return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1,(step-warmup)/max(1,total_steps-warmup))))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,factor)
    code_files = [entrypoint,'scripts/train_duvla_v3_29.py','src/duvla/models/duvla_v2_1.py',
                  'src/duvla/models/ordered_language_bridge.py','src/duvla/training/v3_29_data.py',
                  'src/duvla/training/v2_1_data.py']
    code_files = list(dict.fromkeys(code_files))
    signature = {'version':version,'epochs':args.epochs,'batch_size':args.batch_size,'seed':args.seed,
        'sidecar_sha256':sidecar.signature,'base_signature':base['cache_signature'],
        'model_config':asdict(config),'smoke_updates':args.smoke_updates,
        'code_hashes':{p:sha256(Path(p).read_bytes()).hexdigest() for p in code_files}}
    args.output.mkdir(parents=True,exist_ok=True)
    resume_path=args.output/'resume.pt'; curve=args.output/'loss_curve.jsonl'
    if (args.output/'run_manifest.json').exists() and not args.resume:
        raise ValueError('existing run: explicit --resume required')
    epoch_start=0; skip_samples=0; step=0; exposures=0; rng=None
    if args.resume:
        saved=torch.load(resume_path,map_location='cpu',weights_only=True)
        if saved['run_signature'] != signature: raise ValueError('resume signature mismatch')
        policy.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer']); scheduler.load_state_dict(saved['scheduler'])
        epoch_start=saved['epoch_index'];skip_samples=saved['samples_in_epoch'];step=saved['step'];exposures=saved['exposures']
        rng=saved
        truncate_loss_points(curve,max_optimizer_step=step)
    atomic_write_json(args.output/'run_manifest.json',{**signature,'trainable_groups':groups,
        'selected_frames':n,'planned_optimizer_steps':total_steps,'formal':not bool(args.smoke_updates),
        'parameter_counts':{'total':sum(p.numel() for p in policy.parameters()),'trainable':sum(p.numel() for p in parameters)},
        'uses_evaluation_initial_states':False,'benchmark_task_index':False})
    def save(epoch, samples):
        _atomic_torch_save(resume_path,{'run_signature':signature,'model':policy.state_dict(),
            'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
            'epoch_index':epoch,'samples_in_epoch':samples,'step':step,'exposures':exposures,
            'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
    def checkpoint(epoch):
        return {'format':'duvla_v3_30' if v330 else 'duvla_v3_29','version':version,'stage':'flow','variant':'multilayer',
            'epoch':epoch,'optimizer_step':step,'effective_samples':exposures,'effective_epochs':exposures/n,
            'planned_epochs':30,'training_complete':epoch==30,'checkpoint_complete':True,'formal':not bool(args.smoke_updates),
            'model_config':asdict(config),'model_state_dict':policy.state_dict(),
            'cache_signature':base['cache_signature'],'sidecar_sha256':sidecar.signature,
            'data_contract':sidecar.manifest,'action_offset':1,'camera_size':128,
            'action_attention':'bidirectional_chunk' if v330 else 'causal_chunk',
            'candidate_aggregation':config.candidate_aggregation,
            'language_bridge_mode':config.ordered_language_bridge_mode,
            'benchmark_task_index':False,'task_routing':'natural_language','trainable_groups':groups,
            'uses_evaluation_initial_states':False,'uses_reward':False,'uses_success':False,
            'run_signature':signature}
    if rng is not None:
        torch.set_rng_state(rng['torch_rng']);torch.cuda.set_rng_state(rng['cuda_rng'])
    else:
        # New bridge construction must not alter the common Flow noise stream.
        _seed_everything(args.seed+1)
    task_weights = _task_weights(base)
    started=time.monotonic(); initial_exposures=exposures; window=Counter(); window_rows=0
    milestones={1,3,6,8,10,15,20,25,30}
    for epoch in range(epoch_start,args.epochs):
        if not args.smoke_updates:
            require_disk_budget(args.base_cache,minimum_gib=12)
        policy.train(); samples=0; tasks=Counter(); episodes=set(); unique=set()
        iterator=iter_v2_1_batches(args.base_cache,batch_size=args.batch_size,history_length=4,
            history_stride=2,seed=args.seed,epoch=epoch,shuffle=not bool(args.smoke_updates),shard_shuffle_block_size=4,
            mmap_shards=True,row_transform=sidecar.transform)
        batches=prefetch_v2_1_batches(iterator,pin_memory=True)
        try:
            for batch in batches:
                count=len(batch['dataset_indices'])
                tasks.update(batch['task_indices'].tolist());episodes.update(batch['episode_indices'].tolist());unique.update(batch['dataset_indices'].tolist())
                if epoch==epoch_start and samples < skip_samples:
                    samples += count
                    if samples>skip_samples: raise ValueError('resume cursor not on batch boundary')
                    continue
                moved={k:v.to(device=device,dtype=torch.bfloat16 if v.is_floating_point() else v.dtype,non_blocking=True) for k,v in batch.items()}
                kwargs={k:moved[k] for k in ('history_visual','history_semantic','history_states','previous_actions')}
                if not args.without_language:
                    kwargs.update(language_tokens=moved['language_tokens'],language_mask=moved['language_mask'])
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    losses=policy.flow_loss_components(moved['visual'],moved['semantic'],moved['states'],moved['actions'],moved['valid_mask'],
                        sample_weights=task_weights.index_select(0,batch['task_indices']).to(device,dtype=torch.bfloat16),**kwargs)
                if any(not torch.isfinite(v).all() for v in losses.values()): raise RuntimeError('nonfinite loss')
                total_loss = sum(losses.values())
                total_loss.backward()
                losses = {**losses, 'total':total_loss}
                norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
                if step==0 and policy.language_bridge is not None:
                    if config.ordered_language_bridge_mode == 'scalar_gate':
                        if policy.language_bridge.gate.grad is None:
                            raise RuntimeError('language gate has no gradient')
                    elif policy.language_bridge.out.weight.grad is None or not policy.language_bridge.out.weight.grad.abs().sum()>0:
                        raise RuntimeError('zero-output language projection has no gradient')
                optimizer.step();scheduler.step();step+=1;samples+=count;exposures+=count
                for key,value in losses.items(): window[key]+=float(value.detach())*count
                window_rows+=count
                if step%100==0 or (args.smoke_updates and step>=args.smoke_updates) or samples==n:
                    point={'optimizer_step':step,'effective_epochs':exposures/n,'learning_rate':optimizer.param_groups[0]['lr'],
                        'losses':{k:v/window_rows for k,v in window.items()}}
                    append_loss_point(curve,point);window.clear();window_rows=0
                    atomic_write_json(args.output/'training_state.json',{
                        '阶段':f'{version}训练','state':'running','optimizer_step':step,'planned_optimizer_steps':total_steps,
                        'effective_epochs':exposures/n,'samples_per_second':(exposures-initial_exposures)/max(time.monotonic()-started,1e-6),
                        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,'latest_loss':point['losses'],
                        'language_bridge_mode':None if policy.language_bridge is None else policy.language_bridge.mode,
                        'language_gate':None if policy.language_bridge is None or v330 else float(policy.language_bridge.gate.detach())})
                if step%500==0:
                    if not args.smoke_updates:
                        require_disk_budget(args.base_cache,minimum_gib=12)
                    save(epoch,samples)
                if step%5000==0: print(f'{version} {step}/{total_steps} updates, {exposures/n:.2f}E',flush=True)
                if args.smoke_updates and step>=args.smoke_updates:
                    save(epoch,samples)
                    atomic_write_json(args.output/'training_state.json',{
                        '阶段':'GPU冒烟结束，非正式训练','state':'smoke_completed',
                        'formal':False,'optimizer_step':step,'effective_epochs':exposures/n,
                        'planned_optimizer_steps':total_steps})
                    atomic_write_json(args.output/'smoke.json',{'passed':True,'updates':step,
                        'peak_gpu_gib':torch.cuda.max_memory_allocated()/2**30,
                        'seconds':time.monotonic()-started,'trainable_parameters':sum(p.numel() for p in parameters),
                        'language_bridge_mode':None if policy.language_bridge is None else policy.language_bridge.mode,
                        'language_gate':None if policy.language_bridge is None or v330 else float(policy.language_bridge.gate.detach())})
                    return
        finally:
            batches.close()
        if samples != n or len(unique)!=n or len(episodes)!=2000 or len(tasks)!=40:
            raise RuntimeError(f'incomplete epoch coverage: {samples}/{n}, episodes={len(episodes)} tasks={len(tasks)}')
        atomic_write_json(args.output/f'coverage_{epoch+1}e.json',{'rows':len(unique),'episodes':len(episodes),'tasks':dict(tasks)})
        save(epoch+1,0)
        if epoch+1 in milestones:
            slug = version.lower().replace('v', 'v')
            _atomic_torch_save(args.output/f'duvla-{slug}-{epoch+1}e.pt',checkpoint(epoch+1))
            write_loss_curve_artifacts(curve,csv_path=args.output/f'loss_curve_{epoch+1}e.csv',svg_path=args.output/f'loss_curve_{epoch+1}e.svg',title=f'Duvla {version} {epoch+1}E')
        write_loss_curve_artifacts(curve,csv_path=args.output/'loss_curve.csv',svg_path=args.output/'loss_curve.svg',title=f'Duvla {version}')
        skip_samples=0
    atomic_write_json(args.output/'training_state.json',{'state':'completed','optimizer_step':step,'effective_epochs':exposures/n,'planned_optimizer_steps':total_steps})


if __name__=='__main__': main()
