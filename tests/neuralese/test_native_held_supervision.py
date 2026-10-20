"""Held source intent is explicit; dual CE uses one actual graph/readout."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
from torch.nn import functional as F

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.data.native_graph import parse_graph, graph_loss
from natlang_neuralese.data.native_sources import NativeGraphSource, _row_bindings
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import _evaluate
from natlang_neuralese.train.native_supervision import held_measurements, load_policy
from test_native_supervision_mass import ordinary, declared
from test_shared_state import tiny


def held_row():
    row = ordinary()
    row.update(id='protected:held-example', split='test', source_groups=['independent-held'])
    return row


def policy():
    value = declared()
    value['held_sources'] = [{**value['sources'][0], 'sha256':'b' * 64}]
    return value


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
def test_dual_held_ce_matches_exact_oracle_one_forward_readout_and_no_grad(family):
    row = held_row()
    raw = parse_graph(row, tokenizer_fingerprint='f' * 64)
    measured = held_measurements(raw, row, source_sha256='b' * 64, policy=policy())
    assert measured.target_weights == raw.target_weights
    assert _row_bindings([raw]) == _row_bindings([measured])
    model = SharedStateModel(tiny(family), marker_ids=(58,59,60,61), rank=4)
    model.eval()
    with torch.no_grad():
        hidden = model(raw.plan, retain_states=False)['h_final'][0]
        ce = F.cross_entropy(model.logits(hidden[:-1]).float(),
                             torch.tensor(raw.plan.invocations[0].token_ids[1:]), reduction='none')
        raw_weights = torch.tensor(raw.target_weights[1:])
        expected = {'raw': float((ce*raw_weights).sum()/raw_weights.sum())}
        for name, weights in measured.evaluation_weights.items():
            weights = torch.tensor(weights[1:])
            expected[name] = float((ce*weights).sum()/weights.sum())
    forwards = []
    hook = model.register_forward_hook(lambda *_: forwards.append(1))
    readouts = []
    original = model.logits
    def logits(states):
        readouts.append(len(states))
        return original(states)
    model.logits = logits
    model.train()
    rng = torch.get_rng_state().clone()
    result = _evaluate(model, [measured, raw], checkpoint_layers=False, chunk_tokens=3)
    hook.remove()
    assert forwards == [1,1]
    assert sum(readouts) == 2 * (len(raw.target_weights)-1)
    assert result['loss'] == pytest.approx(expected['raw'], abs=1e-6)
    assert result['ce_metric_kind'] == 'original-corpus-weighted-token-mean'
    for name in ['declared','target']:
        metric = result['additional_ce'][name]
        assert metric['loss'] == pytest.approx(expected[name], abs=1e-6)
        assert metric['examples'] == 1 and metric['unmeasured_examples'] == 1
    assert result['additional_ce_bindings'][0]['source_sha256'] == 'b' * 64
    assert model.training and torch.equal(torch.get_rng_state(), rng)
    assert all(parameter.grad is None for parameter in model.parameters())


def test_explicit_held_source_split_raw_bindings_and_uncovered_sources(tmp_path):
    row = held_row()
    path = tmp_path/'held.jsonl'
    path.write_text(json.dumps(row)+'\n')
    digest = sha256_file_hex(path)
    value = policy()
    value['held_sources'][0]['sha256'] = digest
    source = NativeGraphSource(path, 'f' * 64, required_split='test')
    bindings = copy.deepcopy(source.bindings)
    source.set_ce_policy(value)
    measured, = source
    assert measured.evaluation_weights and source.bindings == bindings
    assert measured.split == 'test'
    example = parse_graph(row, tokenizer_fingerprint='f' * 64)
    assert held_measurements(example,row,source_sha256='c'*64,policy=value) is example
    row['split'] = 'train'
    with pytest.raises(ValueError, match='protected test'):
        held_measurements(parse_graph(row,tokenizer_fingerprint='f'*64),row,source_sha256=digest,policy=value)
    bad = policy()
    bad['held_sources'][0]['sha256'] = bad['sources'][0]['sha256']
    declaration = tmp_path/'bad.json'
    declaration.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='unique SHA'):
        load_policy({'path':str(declaration),'sha256':sha256_file_hex(declaration)})


def test_materializer_separates_train_held_intent_and_rejects_wrong_split(tmp_path):
    from scripts.neuralese_data.build_native_ce_supervision_policy import build
    train = tmp_path/'train.jsonl'; held = tmp_path/'test.jsonl'
    train.write_text(json.dumps(ordinary())+'\n');held.write_text(json.dumps(held_row())+'\n')
    value = build([train],legacy_pair_context_weight=.05,held_sources=[held])
    assert value['sources'][0]['sha256'] == sha256_file_hex(train)
    assert value['held_sources'][0]['sha256'] == sha256_file_hex(held)
    assert value['held_inventory'][0]['sha256'] == sha256_file_hex(held)
    with pytest.raises(ValueError,match='wrong explicit split'):
        build([train],legacy_pair_context_weight=.05,held_sources=[train])


@pytest.mark.parametrize('active', [False,True])
def test_declared_policy_does_not_select_raw_prompt_ce_best_export(active,monkeypatch):
    from natlang_neuralese.train import native_state_sft as stage
    saved = []
    monkeypatch.setattr(stage,'model_state',lambda model:{'weights':'exact'})
    monkeypatch.setattr(stage,'collect_sft_progress',lambda training:{'credits':'preserved'})
    stage._save(SimpleNamespace(save=lambda state,metric:saved.append((state,metric))),None,
        SimpleNamespace(state_dict=lambda:{'moments':'exact'}),config={},identity={},step=2,
        epoch=0,cursor=2,source_receipts=[],source_hash='a',source_row_bindings={},held_row_bindings={},
        exposure_counts={'train':2},global_step=2,rng={'rng':'exact'},metrics={'held':{'loss':.1}},
        architecture='lfm',options={'ce_supervision_active':active})
    state,metric = saved[0]
    assert metric is None if active else metric == .1
    assert state['training']['optimizer'] == {'moments':'exact'}
    assert state['training']['rng'] == {'rng':'exact'}
    assert state['training']['exposure_counts'] == {'train':2}


def test_declared_held_loss_is_mean_of_original_graphs_not_prompt_mass():
    from dataclasses import replace
    model = SharedStateModel(tiny('lfm'),marker_ids=(58,59,60,61),rank=4)
    rows = [held_row(), held_row()]
    rows[1]['id'] += ':second'
    rows[1]['nodes'][0]['token_ids'] = [2,3,4,5,6]
    rows[1]['nodes'][0]['target_weights'] = [0.,.05,1.,1.,1.]
    rows[1]['conversion']['source_suffix_start'] = 2
    examples = [held_measurements(parse_graph(row,tokenizer_fingerprint='f'*64),row,
                                 source_sha256='b'*64,policy=policy()) for row in rows]
    model.eval()
    with torch.no_grad():
        details = [graph_loss(model,example,checkpoint_layers=False)[1] for example in examples]
    expected = sum(detail['evaluation_sums']['declared']['loss_sum']/
                   detail['evaluation_sums']['declared']['weight'] for detail in details)/2
    result = _evaluate(model,examples,checkpoint_layers=False,chunk_tokens=3)
    assert result['additional_ce']['declared']['loss'] == pytest.approx(expected,abs=1e-6)
    assert result['additional_ce']['declared']['measured_examples'] == 2


@pytest.mark.parametrize('family',['lfm','mellum'])
def test_extra_held_annotations_leave_raw_loss_and_every_derivative_exact(family):
    row = held_row()
    row['nodes'][0]['target_weights'][4] = 0.
    row['nodes'][0]['target_weights'][24] = 0.
    raw = parse_graph(row,tokenizer_fingerprint='f'*64)
    measured = held_measurements(raw,row,source_sha256='b'*64,policy=policy())
    assert measured.evaluation_weights['target'][4] == measured.evaluation_weights['target'][24] == 0
    model = SharedStateModel(tiny(family),marker_ids=(58,59,60,61),rank=4)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    expected = graph_loss(model,raw,checkpoint_layers=True,chunk_tokens=3)[0]
    expected_grad = torch.autograd.grad(expected,params,allow_unused=True)
    actual = graph_loss(model,measured,checkpoint_layers=True,chunk_tokens=3)[0]
    actual_grad = torch.autograd.grad(actual,params,allow_unused=True)
    assert torch.equal(actual,expected)
    for observed,reference in zip(actual_grad,expected_grad):
        assert (observed is None) == (reference is None)
        if observed is not None:assert torch.equal(observed,reference)


def test_no_observed_suffix_target_is_explicitly_unmeasured_not_zero_loss():
    row = held_row()
    row['conversion']['source_suffix_start'] = len(row['nodes'][0]['token_ids'])
    row['nodes'][0]['target_weights'] = [0.] + [.05]*(len(row['nodes'][0]['token_ids'])-1)
    raw = parse_graph(row,tokenizer_fingerprint='f'*64)
    example = held_measurements(raw,row,source_sha256='b'*64,policy=policy())
    model = SharedStateModel(tiny('lfm'),marker_ids=(58,59,60,61),rank=4)
    result = _evaluate(model,[example],checkpoint_layers=False,chunk_tokens=3)
    assert result['additional_ce']['target']['loss'] is None
    assert result['additional_ce']['target']['weight'] == 0
    assert result['additional_ce']['target']['measured_examples'] == 0


def test_actual_shared_sft_held_metrics_no_held_exposure_or_raw_best(tmp_path):
    from test_native_serve import _tiny_source
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as stage
    from scripts.neuralese_data.build_native_ce_supervision_policy import build
    base = tmp_path/'base'
    _tiny_source(base)
    _,_,_,config = load_native_model(base,dtype=torch.float32,rank=4)
    train = tmp_path/'train.jsonl'; held = tmp_path/'test.jsonl'
    for path,row in [(train,ordinary()),(held,held_row())]:
        row['tokenizer_fingerprint'] = config['tokenizer_fingerprint']
        path.write_text(json.dumps(row)+'\n')
    declaration = tmp_path/'ce-policy.json'
    declaration.write_text(json.dumps(build([train],legacy_pair_context_weight=.05,held_sources=[held])))
    out = tmp_path/'run'
    assert stage.main(['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(train),'--held-native-graphs',str(held),'--ce-supervision-policy',str(declaration),
        '--epochs','1','--eval-every','1','--checkpoint-every','1','--out',str(out)]) == 0
    state = torch.load(out/'checkpoint.pt',weights_only=False)['training']
    assert state['exposure_counts'] == {'mass:ordinary':1}
    assert state['optimizer'] and state['rng']
    metric = state['metrics']['held']
    assert metric['source_sha256'] == sha256_file_hex(held)
    assert metric['additional_ce']['target']['measured_examples'] == 1
    assert metric['additional_ce_policy']['sha256'] == sha256_file_hex(declaration)
    assert not (out/'best-checkpoint.pt').exists()
    assert state['metrics']['checkpoint_selection']['raw_ce_best_export'] is False
