from __future__ import annotations

from dataclasses import replace
import json
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from duvla.models.ordered_language_bridge import OrderedLanguageBridge
from duvla.models.duvla_v2_1 import DuvlaV21Policy
from duvla.training.v3_29_data import causal_chunks, V329Sidecar
from scripts.train_duvla_v2_1 import _model_config
from scripts.evaluate_duvla_v2_1 import _validate_v329_deployment
import duvla.training.resource_budget as budget


def test_bridge_identity_padding_and_gradient() -> None:
    bridge=OrderedLanguageBridge(8,24,4)
    x=torch.randn(2,128,24); language=torch.randn(2,2,32,8)
    mask=torch.arange(32)[None,None,:].expand(2,2,-1)<5
    assert torch.equal(bridge(x,language,mask),x)
    bridge(x,language,mask).square().mean().backward()
    assert bridge.gate.grad is not None and bridge.gate.grad.abs()>0
    with torch.no_grad(): bridge.gate.fill_(.1)
    bridge.zero_grad()
    y=bridge(x,language,mask)
    other=language.clone();other[~mask]=1e5
    torch.testing.assert_close(y,bridge(x,other,mask))
    y.square().mean().backward()
    assert bridge.k.weight.grad.abs().sum()>0
    with pytest.raises(ValueError,match='nonempty'):
        bridge(x,language,torch.zeros_like(mask))


def test_causal_targets_and_episode_tail() -> None:
    actions=torch.arange(35,dtype=torch.float32).reshape(5,7)
    chunks,mask=causal_chunks(actions)
    assert torch.equal(chunks[0,:4],actions[1:])
    assert torch.equal(chunks[3,0],actions[4])
    assert mask.sum(1).tolist()==[4,3,2,1,0]
    assert chunks[~mask].eq(0).all()


def test_sidecar_filters_last_row_and_corrects_transition_reference(tmp_path: Path) -> None:
    base=tmp_path/'base';base.mkdir();(base/'manifest.json').write_text('{}')
    root=tmp_path/'sidecar';root.mkdir()
    manifest={'complete':True,'base_manifest_sha256':sha256(b'{}').hexdigest(),
              'uses_evaluation_initial_states':False,'action_offset':1,'shard_size':2}
    (root/'manifest.json').write_text(json.dumps(manifest))
    target,mask=causal_chunks(torch.ones(2,7))
    torch.save({'dataset_indices':torch.tensor([0,1]),'offsets':torch.tensor([0,3,6]),
        'language':torch.randn(6,2,8),'actions':target,'valid_mask':mask,
        'previous_actions':torch.full((2,7),5.)},root/'shard-000000.pt')
    store=V329Sidecar(root,base)
    result=store.transform({'dataset_indices':torch.tensor([0,1]),'previous_actions':torch.zeros(2,7)})
    assert result['dataset_indices'].tolist()==[0]
    assert result['language_tokens'].shape==(1,2,32,8)
    assert result['previous_actions'].eq(5).all()


def test_policy_shared_init_identity_and_flow_training() -> None:
    base=_model_config({'action_mean':[0.]*7,'action_std':[1.]*7},smoke=True,v3_28_fp32_amp_flow=True)
    torch.manual_seed(17); plain=DuvlaV21Policy(base)
    torch.manual_seed(17); enhanced=DuvlaV21Policy(replace(base,ordered_language_bridge=True))
    for key,value in plain.state_dict().items(): assert torch.equal(value,enhanced.state_dict()[key])
    visual=torch.randn(1,1,2,64,2048);semantic=torch.randn(1,4,1,2048);state=torch.randn(1,8)
    extra={'language_tokens':torch.randn(1,2,32,2048),'language_mask':torch.ones(1,2,32,dtype=torch.bool)}
    noise=torch.randn(1,2,8,7)
    a=plain.sample_actions(visual,semantic,state,noise=noise,apply_direct=False,apply_instruction=False)
    b=enhanced.sample_actions(visual,semantic,state,noise=noise,apply_direct=False,apply_instruction=False,**extra)
    assert torch.equal(a,b)
    losses=enhanced.flow_loss_components(visual,semantic,state,torch.randn(1,8,7),torch.ones(1,8,dtype=torch.bool),previous_actions=torch.zeros(1,7),**extra)
    sum(losses.values()).backward()
    assert enhanced.language_bridge.gate.grad.abs()>0


def test_deployment_budget_and_camera_guards() -> None:
    checkpoint={'format':'duvla_v3_29','formal':True,'checkpoint_complete':True,'epoch':30,
        'effective_epochs':30.,'planned_epochs':30,'action_offset':1,'training_complete':True,
        'uses_reward':False,'uses_success':False,'uses_evaluation_initial_states':False,
        'data_contract':{'complete':True,'train_demonstrations':2000},'sidecar_sha256':'test'}
    args=SimpleNamespace(v329_development_20e=False,camera_size=128,fps=20,action_steps=2,flow_samples=5,flow_seed=23,
        no_flip_views=False,state_clip=None,outcome_verifier=None,action_intent=None)
    _validate_v329_deployment(checkpoint,args)
    with pytest.raises(ValueError): _validate_v329_deployment(dict(checkpoint,epoch=20),args)
    args.v329_development_20e=True
    _validate_v329_deployment(dict(checkpoint,epoch=20,effective_epochs=20.,training_complete=False),args)
    args.camera_size=256
    with pytest.raises(ValueError): _validate_v329_deployment(dict(checkpoint,epoch=20),args)


def test_guest_write_budget_and_host_reserve_are_separate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    monkeypatch.setattr(budget, '_is_wsl', lambda: True)
    monkeypatch.setenv('DUVLA_WSL_HOST_DRIVE', 'D')
    monkeypatch.setattr(budget.shutil,'disk_usage',lambda p:SimpleNamespace(free=510*2**30))
    monkeypatch.setattr(Path,'exists',lambda p:True)
    monkeypatch.setattr(budget.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=str(43*2**30)))
    result=budget.require_disk_budget(tmp_path,minimum_gib=65)
    assert result['windows_d_free_gib']==43
    assert result['required_host_reserve_gib']==12
    monkeypatch.setattr(budget.subprocess,'run',lambda *args,**kwargs:SimpleNamespace(stdout=str(8*2**30)))
    with pytest.raises(RuntimeError,match='Windows D盘'):
        budget.require_disk_budget(tmp_path,minimum_gib=65)
    monkeypatch.setattr(budget.shutil,'disk_usage',lambda p:SimpleNamespace(free=60*2**30))
    with pytest.raises(RuntimeError,match='WSL可用空间不足'):
        budget.require_disk_budget(tmp_path,minimum_gib=65)
