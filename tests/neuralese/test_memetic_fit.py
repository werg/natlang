"""Memetic adapter fits (rl/memetic_fit.py) with small fakes: no model, no GPU."""
import hashlib
import json
import threading
from pathlib import Path

import pytest
import torch

from natlang_neuralese.rl import memetic_fit as MF
from natlang_neuralese.tune.learned import meta_bank

PARENT = 'a' * 64
IDENTITY = {'checkpoint_sha256': PARENT, 'tokenizer_fingerprint': 'b' * 64, 'embedding_weight_sha256': 'c' * 64,
            'architecture': 'lfm3'}
GROUPS = ['q0:attention', 'q0:convolution', 'q1:attention', 'q1:convolution']


def spec(tmp_path, **extra):
    return {'schema': 'natlang.memetic-episode/1', 'parent_sha256': PARENT,
            'program': {'id': 'memetic:fam@123'},
            'support': [{'id': 's1'}, {'id': 's2'}], 'query': [{'id': 'q1'}],
            'evidence': {'attempts': [], 'success': .5, **extra},
            'eval': {'samples': 2, 'workers': 1, 'case_seconds': 60, 'max_model_requests': 8, 'max_turns': 8,
                     'seed_base': 31000},
            'out': str(tmp_path / 'prog' / 'episodes' / PARENT[:16])}


INFO = {'schema': MF.FIT_SCHEMA, 'components': 3, 'steps': 12, 'groups': GROUPS, 'parent_sha256': PARENT}


def test_fit_spec_defaults_and_problems(tmp_path):
    assert MF.normalize_fit_spec(spec(tmp_path), {}, INFO) == {
        'worlds': ['s1', 's2'], 'components': 3, 'steps': 12, 'scope': 'all', 'init': 'learned'}
    with pytest.raises(MF.FitProblem) as error:
        MF.normalize_fit_spec(spec(tmp_path), {'worlds': ['q1'], 'steps': 99, 'scope': 'mlp', 'extra': 1}, INFO)
    text = ' '.join(error.value.problems)
    assert 'support worlds' in text and 'steps' in text and 'scope' in text and 'unknown fit fields' in text
    assert MF.normalize_fit_spec(spec(tmp_path), {'worlds': ['s2'], 'components': 1, 'scope': 'attention',
                                                  'init': 'zero'}, INFO)['worlds'] == ['s2']


def test_support_rollouts_recorded_and_layout(tmp_path):
    assert MF.support_rollouts(spec(tmp_path, rollouts='/x/r.jsonl')) == ['/x/r.jsonl']
    path = Path(MF.support_rollouts(spec(tmp_path))[0])
    assert path == tmp_path / 'prog' / 'baselines' / PARENT[:16] / 'evals' / PARENT[:16] / 'baseline-support' / \
        'collection' / 'rollouts.jsonl'


def test_mask_selects_components_and_scope():
    mask = MF.coefficient_mask(3, GROUPS, 2, 'attention')
    assert mask.tolist() == [[1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 0]]
    assert MF.coefficient_mask(3, GROUPS, 3, 'all').sum() == 12


def test_meta_sgd_matches_the_inner_law_and_keeps_masked_coefficients():
    torch.manual_seed(0)
    target, a0, log_step = torch.randn(3, 4), torch.randn(3, 4), torch.full((3, 4), -2.)
    mask = MF.coefficient_mask(3, GROUPS, 2, 'all')

    def support_pass(a):
        loss = ((a * mask - target) ** 2).sum()
        loss.backward()
        return float(loss.detach())
    fitted, travel, history = meta_bank.meta_sgd(a0 * mask, log_step, 20, support_pass)
    assert history[-1] < history[0]
    assert torch.equal(fitted[2], torch.zeros(4)) and torch.equal(travel[2], torch.zeros(4))
    # The reference law (BankMetaLearner.fit before the refactor), inline.
    a = (a0 * mask).clone().requires_grad_(True)
    first, second = torch.zeros_like(a), torch.zeros_like(a)
    for t in range(1, 21):
        a.grad = None
        ((a * mask - target) ** 2).sum().backward()
        with torch.no_grad():
            first.mul_(.9).add_(a.grad, alpha=.1)
            second.mul_(.999).addcmul_(a.grad, a.grad, value=.001)
            a.sub_(log_step.exp() * (first / (1 - .9 ** t)) / ((second / (1 - .999 ** t)).sqrt() + 1e-8))
    assert torch.allclose(fitted, a.detach())


class FakeBank:
    snapshot, basis_artifact = 'nbank1:fake', 'nnt1_basis'
    group_labels = GROUPS
    components = [object()] * 3

    def __init__(self):
        self.composed = []

    def composed_cores(self, slots, coefficients):
        return {'m': coefficients}

    def adapter_tensors(self, cores):
        return cores

    def compose_adapter(self, engine, cores, provenance=None):
        self.composed.append((cores['m'].detach().clone(), provenance))
        return type('Record', (), {'id': 'nnt1_adapter'})()


class FakeSupport:
    """Support loss: squared distance of the composed coefficients to a target (its gradient reaches ``a``)."""
    target = torch.full((3, 4), .5)

    def __init__(self, rows):
        self.rows = rows

    def accumulate(self, adapters, make_leaves, *, backward=True):
        leaves = make_leaves()
        value = ((leaves['nnt1_placeholder']['m'] - self.target) ** 2).sum()
        if backward and value.requires_grad:
            value.backward()
        return float(value)


def fake_fit(tmp_path, monkeypatch, rows):
    engine = type('Engine', (), {'checkpoint_sha256': PARENT, 'native_identity': dict(IDENTITY)})()
    service = object.__new__(MF.BankFit)
    service.engine, service.out, service.protected, service.max_tokens = engine, tmp_path / 'svc', None, 4096
    service.manifest = {'inner_fit': {'law': meta_bank.INNER_FIT_LAW, 'tensors': 'nnt1_law', 'steps': 12}}
    service.bank, service.scope, service.placeholder = FakeBank(), 'program', 'nnt1_placeholder'
    service.init, service.log_step, service.steps = torch.zeros(3, 4), torch.full((3, 4), -2.), 12
    service.lock = threading.Lock()
    seen = {}

    def rollout_rows(engine, paths, *, protected, max_tokens, policy):
        seen['policy'] = policy
        return list(rows), {'read': len(rows)}
    monkeypatch.setattr(meta_bank, 'rollout_rows', rollout_rows)
    monkeypatch.setattr(meta_bank, 'support_set', lambda engine, rows: FakeSupport(rows))
    monkeypatch.setattr(MF.BankFit, '_export', lambda self, ident: {'path': f'/x/{ident}', 'sha256': ident, 'bytes': 1})
    return service, seen


def row(world):
    return {'family': 'fam', 'world': world, 'graph': {'id': f'rollout:{world}'}}


def test_bank_fit_fits_support_rows_only_and_caches(tmp_path, monkeypatch):
    service, seen = fake_fit(tmp_path, monkeypatch, [row('s1'), row('s2'), row('q1')])
    request = {'parent_sha256': PARENT, 'rollouts': [str(tmp_path / 'none.jsonl')], 'program': 'memetic:fam',
               'fit': {'worlds': ['s1', 's2'], 'components': 2, 'steps': 12, 'scope': 'attention', 'init': 'learned'}}
    result = service.fit(request)
    assert seen['policy'] == PARENT
    binding = result['adapter']
    assert binding['support']['rows'] == ['rollout:s1', 'rollout:s2'] and binding['support']['counts']['outside_worlds'] == 1
    assert result['support_loss']['after'] < result['support_loss']['before']
    executed = service.bank.composed[-1][0]
    assert torch.equal(executed * (1 - MF.coefficient_mask(3, GROUPS, 2, 'attention')), torch.zeros(3, 4))
    assert binding['adapter_id'] == 'nnt1_adapter' and binding['parent_sha256'] == PARENT
    assert binding['expected_effective_checkpoint_sha256'] == MF.effective_checkpoint(IDENTITY, 'nnt1_adapter')
    saved = torch.load(binding['coefficients']['path'])
    assert hashlib.sha256(Path(binding['coefficients']['path']).read_bytes()).hexdigest() == binding['coefficients']['sha256']
    assert torch.equal(saved['coefficients'], executed)
    count = len(service.bank.composed)
    assert service.fit(request) == json.loads(json.dumps(result)) and len(service.bank.composed) == count


def test_bank_fit_problems(tmp_path, monkeypatch):
    service, _ = fake_fit(tmp_path, monkeypatch, [row('q1')])
    fit = {'worlds': ['s1'], 'components': 3, 'steps': 2, 'scope': 'all', 'init': 'zero'}
    assert 'no verified successful support run' in service.fit(
        {'parent_sha256': PARENT, 'rollouts': [], 'fit': fit})['problems'][0]
    assert 'another parent' in service.fit({'parent_sha256': 'd' * 64, 'rollouts': [], 'fit': fit})['problems'][0]


def test_bank_fitter_over_http(tmp_path):
    calls = []

    class Service:
        def info(self):
            return INFO

        def fit(self, body):
            calls.append(body)
            if body['fit']['worlds'] == ['s2']:
                return {'problems': ['nothing to fit']}
            return {'schema': MF.FIT_SCHEMA, 'adapter': {'schema': MF.ADAPTER_SCHEMA, 'adapter_id': 'nnt1_x'},
                    'support_loss': {'before': 2., 'after': 1., 'history': [2., 1.5]}}
    server = MF.serve_fit(Service())
    threading.Thread(target=server.serve_forever, daemon=True).start()
    endpoint = f'http://127.0.0.1:{server.server_address[1]}'
    try:
        out = MF.bank_fitter(spec(tmp_path), {'steps': 4}, endpoint=endpoint)
        assert out == {'adapter': {'schema': MF.ADAPTER_SCHEMA, 'adapter_id': 'nnt1_x'},
                       'support_loss': {'before': 2., 'after': 1.}}
        assert calls[0]['fit']['steps'] == 4 and calls[0]['parent_sha256'] == PARENT
        assert calls[0]['rollouts'] == MF.support_rollouts(spec(tmp_path))
        with pytest.raises(MF.FitProblem, match='nothing to fit'):
            MF.bank_fitter(spec(tmp_path), {'worlds': ['s2']}, endpoint=endpoint)
        with pytest.raises(MF.FitProblem, match='support worlds'):
            MF.bank_fitter(spec(tmp_path), {'worlds': ['q1']}, endpoint=endpoint)
        other = {**spec(tmp_path), 'parent_sha256': 'd' * 64}
        with pytest.raises(MF.FitProblem, match='another parent'):
            MF.bank_fitter(other, {}, endpoint=endpoint)
    finally:
        server.shutdown()
        server.server_close()


def adapter_binding(tmp_path):
    artifacts = {}
    for ident in ('nnt1_basis', 'nnt1_adapter'):
        path = tmp_path / f'{ident}.safetensors'
        path.write_bytes(ident.encode())
        artifacts[ident] = {'path': str(path), 'sha256': hashlib.sha256(ident.encode()).hexdigest(), 'bytes': 1}
    return {'schema': MF.ADAPTER_SCHEMA, 'bank': 'nbank1:fake', 'native_identity': dict(IDENTITY),
            'adapter_id': 'nnt1_adapter', 'basis_id': 'nnt1_basis', 'artifacts': artifacts,
            'coefficients': {'sha256': 'e' * 64},
            'expected_effective_checkpoint_sha256': MF.effective_checkpoint(IDENTITY, 'nnt1_adapter')}


def test_upload_order_and_custody(tmp_path):
    adapter, puts = adapter_binding(tmp_path), []
    MF.upload_adapter('http://unused', adapter, put=lambda path, data: puts.append((path, data)))
    assert [path for path, _ in puts] == ['/v1/neuralese/native-tensor-artifacts/nnt1_basis',
                                          '/v1/neuralese/native-tensor-artifacts/nnt1_adapter']
    Path(adapter['artifacts']['nnt1_adapter']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError, match='changed since the fit'):
        MF.upload_adapter('http://unused', adapter, put=lambda path, data: None)


def test_adapter_evaluator_collects_the_adapted_actor(tmp_path):
    adapter = adapter_binding(tmp_path)
    seen, uploads = {}, []
    effective = adapter['expected_effective_checkpoint_sha256']

    def collect(**kwargs):
        seen.update(kwargs)
        out = Path(kwargs['out'])
        (out / 'row-000' / 'sample-00').mkdir(parents=True)
        (out / 'row-000' / 'sample-00' / 'results.jsonl').write_text(json.dumps({'outcome': {}}) + '\n')
        (out / 'rewards.jsonl').write_text(json.dumps({'group': 0, 'sample': 0, 'reward': 1.0, 'outcome': 'task_success'}) + '\n')
        turn = {'sampling': {'policy': {'checkpoint_sha256': seen.get('turn_policy', effective)},
                             'adapter_binding': {'adapters': [{'id': 'nnt1_adapter', 'scale_hex': (1.).hex()}]}}}
        (out / 'rollouts.jsonl').write_text(json.dumps({'task_id': 's1', 'turns': [turn, turn]}) + '\n')
        return {'successes': 1, 'scored': 1, 'rollouts': 1}

    base_calls = []
    evaluate = MF.adapter_evaluator(spec(tmp_path), 'http://actor', collect=collect, identity=dict(IDENTITY),
                                    base=lambda *a, **k: base_calls.append((a, k)) or {'success': 0.},
                                    upload=lambda endpoint, value: uploads.append((endpoint, value['adapter_id'])))
    assert evaluate([{'id': 's1'}], None, None, name='plain') == {'success': 0.} and len(base_calls) == 1
    with pytest.raises(ValueError, match='without guidance'):
        evaluate([{'id': 's1'}], 'some guidance', adapter, name='both')
    summary = evaluate([{'id': 's1'}], None, adapter, name='fit')
    assert summary['success'] == 1.0 and summary['adapted_turns'] == 2 and summary['adapter_actor'] == effective
    assert seen['adapters']['entries'] == [{'id': 'nnt1_adapter', 'scale': 1.0}]
    assert seen['policy_id'] == PARENT and uploads == [('http://actor', 'nnt1_adapter')]
    out = Path(summary['dir'])
    assert out.parent.name == effective[:16]
    assert json.loads((out / 'adapter-actor.json').read_text())['effective_checkpoint_sha256'] == effective
    assert evaluate([{'id': 's1'}], None, adapter, name='fit')['success'] == 1.0 and len(uploads) == 1  # cached
    seen['turn_policy'] = PARENT  # a turn that ran the bare parent
    with pytest.raises(ValueError, match='did not run the bound adapter'):
        evaluate([{'id': 's1'}], None, adapter, name='fit-again')


def test_adapter_from_another_parent_is_refused(tmp_path):
    adapter = adapter_binding(tmp_path)
    with pytest.raises(ValueError, match='another parent'):
        MF.actor_binding(adapter, {**IDENTITY, 'checkpoint_sha256': 'd' * 64})


def test_collection_binding_records_adapters(tmp_path):
    from natlang_neuralese.rl.production import collection_binding
    cases = tmp_path / 'cases.jsonl'
    cases.write_text('{}\n')
    (tmp_path / 'runtime' / 'dist').mkdir(parents=True)
    common = dict(cases=cases, endpoints=['http://a'], policy_id=PARENT, rows=1, offset=0, group=2, root_mode='student',
                  runtime=tmp_path / 'runtime', seed_base=1, neuralese_server=True, case_seconds=1,
                  max_model_requests=1, max_turns=1)
    assert 'adapters' not in collection_binding(**common)
    binding = {'schema': MF.ACTOR_SCHEMA, 'entries': [{'id': 'nnt1_adapter', 'scale': 1.0}]}
    assert collection_binding(**common, adapters=binding)['adapters'] == binding
