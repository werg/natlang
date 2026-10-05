import json
from pathlib import Path
import pytest
from scripts.run_online_repair_round import snapshot_checkpoint,wait_for_training


def test_stable_repair_snapshot_contains_weights_config_and_state(tmp_path):
    source=tmp_path/'source';source.mkdir()
    for name in ['state.json','adapter_config.json','adapter_model.safetensors']:(source/name).write_bytes(b'pinned')
    receipt=snapshot_checkpoint(source,tmp_path/'snapshot')
    assert set(receipt)=={'state.json','adapter_config.json','adapter_model.safetensors'}
    with pytest.raises(FileExistsError):snapshot_checkpoint(source,tmp_path/'snapshot')


def test_training_handoff_uses_actual_supervisor_complete_state(tmp_path):
    state=tmp_path/'state.json';status=tmp_path/'status.json'
    state.write_text(json.dumps({'step':3,'trained_examples':20,'corpus':{'phase_manifest_sha256':'pin'}}))
    status.write_text(json.dumps({'state':'complete'}))
    plan={'completion':{'step':3,'trained_examples':20},'checkpoint_state':str(state),
          'status_file':str(status),'checkpoint_identity':{'phase_manifest_sha256':'pin'}}
    wait_for_training(plan)
    plan['checkpoint_identity']['phase_manifest_sha256']='other'
    with pytest.raises(ValueError,match='different phase'):wait_for_training(plan)


def test_snapshot_reads_actual_trainer_weights_subdirectory(tmp_path):
    source=tmp_path/'source';weights=source/'weights';weights.mkdir(parents=True)
    (source/'state.json').write_text('{}')
    for name in ['adapter_config.json','adapter_model.safetensors']:(weights/name).write_bytes(b'weights')
    receipt=snapshot_checkpoint(source,tmp_path/'snapshot')
    assert len(receipt)==3
    assert (tmp_path/'snapshot/adapter_model.safetensors').read_bytes()==b'weights'


def test_provider_readiness_retries_startup_without_task_deadline(monkeypatch):
    from scripts import run_online_repair_round as module
    from contextlib import nullcontext
    from types import SimpleNamespace
    calls=iter([OSError('server initializing'),nullcontext(SimpleNamespace(status=200))])
    def request(*args,**kwargs):
        value=next(calls)
        if isinstance(value,Exception):raise value
        return value
    monkeypatch.setattr(module.urllib.request,'urlopen',request)
    sleeps=[];monkeypatch.setattr(module.time,'sleep',sleeps.append)
    clock=iter([0,1]);monkeypatch.setattr(module.time,'monotonic',lambda:next(clock))
    module.wait_for_provider('http://127.0.0.1:8082',startup_budget_seconds=10)
    assert sleeps==[5]


def test_provider_readiness_reports_expired_deployment_budget(monkeypatch):
    from scripts import run_online_repair_round as module
    def unavailable(*args,**kwargs):raise OSError('initializing')
    monkeypatch.setattr(module.urllib.request,'urlopen',unavailable)
    clock=iter([0,1]);monkeypatch.setattr(module.time,'monotonic',lambda:next(clock))
    with pytest.raises(RuntimeError,match='deployment startup budget'):
        module.wait_for_provider('http://127.0.0.1:8082',startup_budget_seconds=.5)
