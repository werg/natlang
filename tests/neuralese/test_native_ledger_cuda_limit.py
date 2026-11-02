from types import SimpleNamespace
import pytest
import torch
from natlang_neuralese.train import memory_policy


@pytest.mark.parametrize('discrete',[False,True])
def test_allocator_limit_uses_actual_claim_and_shared_host_charge(monkeypatch,discrete):
    gib=2**30
    monkeypatch.setattr(memory_policy,'ledger_claim',lambda:{'budget_bytes':96*gib,'host_bytes':20*gib if not discrete else 0,'discrete_cuda':discrete})
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(total_memory=128*gib))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:10*gib)
    monkeypatch.setattr(torch.cuda,'memory_reserved',lambda device:100*gib)
    calls=[]
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:calls.append('cache-released'))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda value,device:calls.append(value))
    result=memory_policy.enforce_ledger_cuda_limit('cuda')
    expected=96*gib-(0 if discrete else 20*gib+memory_policy.ledger_claim_margin_bytes(96*gib))
    assert result['allocator_limit_bytes']==expected
    assert calls==['cache-released',expected/(128*gib)]


def test_overcommitted_claim_fails_before_more_cuda_work(monkeypatch):
    gib=2**30
    monkeypatch.setattr(memory_policy,'ledger_claim',lambda:{'budget_bytes':32*gib,'host_bytes':24*gib,'discrete_cuda':False})
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda device:SimpleNamespace(total_memory=128*gib))
    monkeypatch.setattr(torch.cuda,'memory_allocated',lambda device:10*gib)
    with pytest.raises(torch.cuda.OutOfMemoryError,match='remaining ledger allowance'):
        memory_policy.enforce_ledger_cuda_limit('cuda')


def test_cpu_and_unmanaged_jobs_keep_existing_allocator_policy(monkeypatch):
    monkeypatch.setattr(memory_policy,'ledger_claim',lambda:None)
    assert memory_policy.enforce_ledger_cuda_limit('cpu') is None
    assert memory_policy.enforce_ledger_cuda_limit('cuda') is None


@pytest.mark.parametrize('explicit',[None,'2'])
def test_unified_compiler_default_is_bounded_and_explicit_setting_is_retained(monkeypatch,explicit):
    import torch._inductor.config as config
    monkeypatch.setattr(memory_policy,'ledger_claim',lambda:{'discrete_cuda':False})
    monkeypatch.setattr(config,'compile_threads',config.compile_threads)
    if explicit is None:monkeypatch.delenv('TORCHINDUCTOR_COMPILE_THREADS',raising=False)
    else:monkeypatch.setenv('TORCHINDUCTOR_COMPILE_THREADS',explicit)
    assert memory_policy.configure_native_compilation()==(1 if explicit is None else 2)
    assert config.compile_threads==(1 if explicit is None else 2)
