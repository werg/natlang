import copy
import json

import pytest
import torch

from natlang_neuralese.train.native_operation_report import (
    finite_training_state_receipt, supervised_progress_numerics_ok,
)
from natlang_neuralese.train.native_resident import binding


def trained():
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
    model(torch.ones(2, 3)).square().mean().backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_finite_check_preserves_actual_weights_optimizer_and_rng():
    model, optimizer = trained()
    parameters = [(p, p.detach().clone(), p._version) for p in model.parameters()]
    slots = [(v, v.clone(), v._version) for s in optimizer.state.values()
             for v in s.values() if torch.is_tensor(v)]
    rng = torch.get_rng_state().clone()
    receipt = finite_training_state_receipt(model, optimizer, 'a' * 64)
    assert receipt['passed'] and receipt['floating_tensors']['model'] == 2
    assert receipt['floating_tensors']['optimizer'] > 0
    assert receipt['runtime_qualification_granted'] is False
    for value, before, version in parameters + slots:
        assert torch.equal(value, before) and value._version == version
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize('fault', ['weight', 'moment'])
def test_nonfinite_saved_state_is_rejected(fault):
    model, optimizer = trained()
    with torch.no_grad():
        target = next(model.parameters()) if fault == 'weight' else next(iter(optimizer.state.values()))['exp_avg']
        target.flatten()[0] = float('nan')
    receipt = finite_training_state_receipt(model, optimizer, 'a' * 64)
    assert receipt['passed'] is False and receipt['nonfinite']


@pytest.fixture
def partial(tmp_path):
    model, optimizer = trained()
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
    parent = binding(checkpoint)
    certificate = tmp_path / 'runtime.json'
    certificate.write_text(json.dumps({'passed': False, 'deferred': True,
        'reason': 'shared qualification request pending', 'checkpoint_sha256': parent['sha256']}))
    report = {'numerics_ok': False, 'completed': False, 'qualified': False,
        'foundation_ready': False,
        'progress_numerics': finite_training_state_receipt(model, optimizer, parent['sha256']),
        'runtime_certificate': {**binding(certificate), 'passed': False}}
    return report, parent, certificate


def test_partial_finite_progress_grants_no_readiness(partial):
    report, checkpoint, _ = partial
    original = copy.deepcopy(report)
    assert supervised_progress_numerics_ok(report, checkpoint)
    assert report == original and not any(report[k] for k in ('completed', 'qualified', 'foundation_ready'))


@pytest.mark.parametrize('fault', ['missing', 'wrong-checkpoint', 'nonfinite', 'readiness', 'empty-model', 'bool-count'])
def test_missing_or_mismatched_numerics_cannot_promote_progress(partial, fault):
    report, checkpoint, _ = partial
    evidence = report['progress_numerics']
    if fault == 'missing': report.pop('progress_numerics')
    elif fault == 'wrong-checkpoint': evidence['checkpoint_sha256'] = '0' * 64
    elif fault == 'nonfinite': evidence['nonfinite'] = ['optimizer/slot']
    elif fault == 'readiness': evidence['runtime_qualification_granted'] = True
    elif fault == 'empty-model': evidence['floating_tensors']['model'] = 0
    else: evidence['floating_tensors']['optimizer'] = True
    assert not supervised_progress_numerics_ok(report, checkpoint)


@pytest.mark.parametrize('fault', ['error', 'not-deferred', 'wrong-reason', 'wrong-checkpoint'])
def test_actual_failed_runtime_is_not_a_deferred_check(partial, fault):
    report, checkpoint, path = partial
    value = json.loads(path.read_text())
    if fault == 'error': value['error'] = 'actual forward/gradient mismatch'
    elif fault == 'not-deferred': value['deferred'] = False
    elif fault == 'wrong-reason': value['reason'] = 'operational stop; qualification deferred until resume'
    else: value['checkpoint_sha256'] = '0' * 64
    path.write_text(json.dumps(value)); report['runtime_certificate'] = {**binding(path), 'passed': False}
    assert not supervised_progress_numerics_ok(report, checkpoint)


def test_certificate_still_requires_exact_bytes(partial):
    report, checkpoint, path = partial
    path.write_text('{}')
    with pytest.raises(ValueError): supervised_progress_numerics_ok(report, checkpoint)


def test_legacy_reports_keep_existing_contract():
    assert supervised_progress_numerics_ok({'numerics_ok': True}, {'sha256': 'a' * 64})
    assert not supervised_progress_numerics_ok({'numerics_ok': False}, {'sha256': 'a' * 64})


def test_new_nonfinite_evidence_cannot_hide_behind_older_qualified_flag(partial):
    report, checkpoint, _ = partial
    report['numerics_ok'] = True
    report['progress_numerics']['passed'] = False
    assert not supervised_progress_numerics_ok(report, checkpoint)


def test_supervised_dispatch_keeps_finite_partial_work_unqualified(tmp_path, monkeypatch):
    from natlang_neuralese.rl import production
    parent = tmp_path / 'parent.pt'; parent.write_bytes(b'original parent')
    config = {'native_state': True, 'mandatory_sft_only': True, 'out': str(tmp_path / 'job'),
        'checkpoint': str(parent), 'seed_command': ['shared-trainer', '{trainer_args}'],
        'trainer_parameters': {'refresh_at_checkpoint': True, 'checkpoint_every': 7}}
    config_path = tmp_path / 'config.json'; config_path.write_text(json.dumps(config))
    def trainer(command, values):
        model, optimizer = trained()
        out = tmp_path / 'job/learner'; out.mkdir(parents=True)
        checkpoint = out / 'checkpoint.pt'
        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, checkpoint)
        ref = binding(checkpoint); certificate = out / 'runtime.json'
        certificate.write_text(json.dumps({'passed': False, 'deferred': True,
            'reason': 'shared qualification request pending', 'checkpoint_sha256': ref['sha256']}))
        report = {'completed': False, 'qualified': False, 'foundation_ready': False,
            'numerics_ok': False, 'progress_numerics': finite_training_state_receipt(model, optimizer, ref['sha256']),
            'runtime_certificate': {**binding(certificate), 'passed': False},
            'continuation_boundary': {'kind': 'semantic-refresh', 'reached': True, 'new_optimizer_updates': 1},
            'mandatory_sft': {'complete': False, 'remaining_graph_visits': 19}}
        (out / 'report.json').write_text(json.dumps(report))
    monkeypatch.setattr(production, 'run_command', trainer)
    result = production.run(config_path, rounds=1, defer_native_refresh=True)
    assert result['phase'] == 'awaiting_semantic_refresh'
    assert result['mandatory_sft']['complete'] is False
    assert result['checkpoint'] == str(parent)
    report = json.loads((tmp_path / 'job/learner/report.json').read_text())
    assert report['runtime_certificate']['passed'] is False
    assert not any(report[k] for k in ('completed', 'qualified', 'foundation_ready'))
