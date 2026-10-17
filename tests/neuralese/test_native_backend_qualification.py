"""State/failure contracts only; actual FP8 kernels require destination CUDA evidence."""
import contextlib
import random

import pytest
import torch

from natlang_neuralese.eval import native_backend


class Backend:
    def __init__(self):
        self.row_counts = {'calls': 7, 'rows': 21}
        self.has_executed = True
        self.state = {'scale': [3]}
        self.active = False
    def extra_state(self): return self.state
    def restore_extra_state(self, state): self.state = state
    def collect_row_counts(self): return dict(self.row_counts)
    def configuration(self): return {'backend': 'test-state-contract'}
    @contextlib.contextmanager
    def autocast(self):
        self.active = True
        try: yield
        finally: self.active = False


@pytest.mark.parametrize('failure', [None, 'zero-execution', 'numerical', 'exception'])
def test_actual_scope_counters_and_rng_state_restore_on_every_exit(monkeypatch, failure):
    backend = Backend()
    rng = torch.get_rng_state().clone()
    python_rng = random.getstate()
    def check(*args, **kwargs):
        assert backend.active and kwargs['checkpoint_sha256'] == 'a'*64
        assert kwargs['forward_atol'] == 1e-4 and kwargs['gradient_atol'] == 1e-5
        random.random(); torch.rand(3)
        backend.state['scale'][0] = 99
        if failure == 'exception': raise RuntimeError('destination compiler failed')
        if failure != 'zero-execution': backend.row_counts.update(calls=10, rows=36)
        return {'passed': failure != 'numerical'}
    monkeypatch.setattr(native_backend, 'qualify_existing', check)
    args = dict(backend=backend, checkpoint_sha256='a'*64, forward_atol=1e-4, gradient_atol=1e-5, device='cpu')
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            native_backend.qualify_training_backend(None, None, None, {}, **args)
    else:
        report = native_backend.qualify_training_backend(None, None, None, {}, **args)
        assert report['executed_projections'] == {'calls': 3, 'rows': 15}
        assert report['optimizer_updates'] == 0 and not report['task_quality_qualified']
    assert torch.equal(torch.get_rng_state(), rng) and random.getstate() == python_rng
    assert backend.state == {'scale': [3]} and backend.row_counts == {'calls': 7, 'rows': 21}
    assert backend.has_executed and not backend.active


def test_backend_guard_rejects_disabled_configuration_before_source_or_model_loading(tmp_path):
    from natlang_neuralese.train.native_state_sft import main
    with pytest.raises(ValueError, match='enabled backend and declared tolerances'):
        main(['--native-graphs', str(tmp_path/'missing'), '--held-native-graphs', str(tmp_path/'missing-held'),
              '--out', str(tmp_path/'out'), '--qualify-training-backend'])
    assert not (tmp_path/'out').exists()


def test_recipe_exposes_backend_qualification_as_an_operational_check():
    from natlang_neuralese.train.recipe import HANDLERS, OPERATIONAL_STAGE_PARAMETERS
    assert 'qualify_training_backend' in HANDLERS['native_state_sft']['parameters']
    assert 'qualify_training_backend' in OPERATIONAL_STAGE_PARAMETERS


def test_complete_backend_failure_remains_structured_and_restores_owned_state(monkeypatch):
    backend=Backend();rng=torch.get_rng_state().clone()
    def failed(*args,**kwargs):
        assert kwargs['gradient_parameters']==[('actual',None)]
        backend.state['scale'][0]=91
        kwargs['before_graph']()
        assert backend.state=={'scale':[3]}
        backend.state['scale'][0]=92
        backend.row_counts.update(calls=8,rows=23)
        torch.rand(3)
        return {'passed':False,'complete_gradient_qualification':{'passed':False}}
    monkeypatch.setattr(native_backend,'qualify_existing',failed)
    report=native_backend.qualify_training_backend(None,None,None,{},backend=backend,
        checkpoint_sha256='a'*64,forward_atol=1e-4,gradient_atol=1e-5,device='cpu',
        gradient_parameters=[('actual',None)],raise_on_failed=False)
    assert report['passed']is False and report['executed_projections']=={'calls':1,'rows':2}
    assert report['runtime_reference']['complete_gradient_qualification']['passed']is False
    assert backend.state=={'scale':[3]}and backend.row_counts=={'calls':7,'rows':21}
    assert torch.equal(torch.get_rng_state(),rng)and backend.has_executed and not backend.active
