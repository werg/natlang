"""Data continuation of an active neuralese_direct stage (owner 2026-10-12: "continue our training from what we have
with more data"): direct_sft continues its update counter, weights and optimizer on new text data, the documents of
its consumed updates counted as seen; the handoff accepts it only when declared (continued_text_data)."""
import copy
import json
from types import SimpleNamespace

import pytest

from natlang_neuralese.train import direct_sft as trainer
from natlang_neuralese.train import recipe as runner


def window(doc, n, length=40, identity=None):
    return {'ids': list(range(length)), 'prefix': 4, 'document': f'h-{doc}', 'document_id': identity or f'id-{doc}',
            'start': n}


def corpus(documents, windows_per_document=3, rename=None):
    rename = rename or {}
    return [window(rename.get(d, d), n, length=30 + 7 * ((i + n) % 5), identity=f'id-{d}')
            for i, d in enumerate(documents) for n in range(windows_per_document)]


def args(**overrides):
    return SimpleNamespace(**{'seed': 0, 'batch_tokens': 160, 'batch': 1, 'epochs': 1.0, 'steps': None, **overrides})


def quantization_stage():
    return {'id': 'direct_sft', 'progress': [0.0, 0.35], 'start': 0, 'end': 'derived'}


def test_continuation_keeps_the_consumed_prefix_and_trains_the_unseen_rest_once():
    a = args()
    previous = corpus(range(40))
    # The new data: the old documents (one re-rendered: new content hash, same id) plus 60 new ones.
    new = corpus(range(100), rename={3: '3-rerendered'})
    _, old_plan = trainer.direct_schedule(previous, a)
    consumed = 9
    record, plan = trainer.data_continuation(previous, previous[:2], new, new[:2], consumed=consumed, previous=None,
                                             quantization_stage=quantization_stage(), a=a)
    seen = {w['document'] for batch, _ in old_plan[:consumed] for w in batch}
    assert record['consumed_updates'] == consumed and record['seen_documents'] == len(seen)
    assert record['seen_windows'] == sum(len(batch) for batch, _ in old_plan[:consumed])
    assert record['updates'] == len(plan) and all(not batch for batch, _ in plan[:consumed])
    trained = [w for batch, _ in plan[consumed:] for w in batch]
    # Every unseen window exactly once; no window of a seen document (by hash or by id) again.
    seen_ids = {w['document_id'] for batch, _ in old_plan[:consumed] for w in batch}
    expected = [w for w in new if w['document'] not in seen and w['document_id'] not in seen_ids]
    assert sorted(map(id, trained)) == sorted(map(id, expected))
    covered = [c for _, c in plan]
    assert covered == sorted(covered) and covered[-1] == pytest.approx(1.0)
    # Deterministic: the record rebuilds the same plan (a later resume).
    _, again = trainer.direct_schedule(new, a, record)
    assert [list(map(id, b)) for b, _ in again] == [list(map(id, b)) for b, _ in plan]


def test_continuation_rebases_the_qat_progress_without_a_jump():
    a = args()
    previous, new = corpus(range(40)), corpus(range(120))
    _, old_plan = trainer.direct_schedule(previous, a)
    consumed = 11
    record, plan = trainer.data_continuation(previous, previous[:1], new, new[:1], consumed=consumed, previous=None,
                                             quantization_stage=quantization_stage(), a=a)
    rebase = record['quantization_rebase']
    old_progress = 0.35 * consumed / len(old_plan)
    assert rebase['progress'] == [pytest.approx(old_progress), 0.35]
    assert rebase['start'] == consumed and rebase['end'] == len(plan) > len(old_plan)
    assert record['lr_schedule'].startswith('constant')
    from natlang_neuralese.train.quantization import Quantization
    spec = {'schema': 'natlang.neuralese-quantization/1', 'loss': {'mode': 'sample'},
            'points': [{'name': 'q4', 'gate': 'report', 'weight': 1.0, 'ramp': {'start': 0.0, 'end': 1.0},
                        'modules': {'experts': {'format': 'int4', 'group': 32}}}]}
    before = Quantization({**spec, 'stage': {'id': 's', 'progress': [0.0, 0.35], 'start': 0, 'end': len(old_plan)}})
    after = Quantization({**spec, 'stage': {'id': 's', 'progress': rebase['progress'], 'start': consumed,
                                            'end': len(plan)}})
    assert after.progress(consumed) == pytest.approx(before.progress(consumed))
    assert after.progress(len(plan)) == pytest.approx(0.35)
    assert after.progress(consumed + 1) - after.progress(consumed) < before.progress(consumed) - before.progress(consumed - 1)


def test_held_rebase_and_chained_continuation():
    a = args()
    r2, r3, r4 = corpus(range(30)), corpus(range(60)), corpus(range(90))
    first, plan3 = trainer.data_continuation(r2, r2[:3], r3, r3[:3], consumed=5, previous=None,
                                             quantization_stage=quantization_stage(), a=a)
    assert first['held']['rebased'] is False
    stage = {**quantization_stage(), **{k: first['quantization_rebase'][k] for k in ('start', 'progress')}}
    second, plan4 = trainer.data_continuation(r3, r3[:3], r4, r4[3:6], consumed=9, previous=first,
                                              quantization_stage=stage, a=a)
    assert second['held']['rebased'] is True
    assert set(first['seen_document_sha256']) <= set(second['seen_document_sha256'])
    trained3 = {w['document'] for batch, _ in plan3[5:9] for w in batch}
    assert trained3 <= set(second['seen_document_sha256'])
    assert second['previous']['consumed_updates'] == 5 and 'seen_document_sha256' not in second['previous']
    assert second['quantization_rebase']['previous'] == {'start': 5, 'end': len(plan3),
                                                         'progress': first['quantization_rebase']['progress']}


def test_continuation_needs_a_whole_epoch_budget_inside_the_first_pass():
    previous, new = corpus(range(20)), corpus(range(40))
    with pytest.raises(ValueError, match='whole-epoch'):
        trainer.data_continuation(previous, [], new, [], consumed=2, previous=None, quantization_stage=None,
                                  a=args(epochs=0.5))
    _, two_pass = trainer.direct_schedule(previous, args(epochs=2.0))
    with pytest.raises(ValueError, match='first pass'):
        trainer.data_continuation(previous, [], new, [], consumed=len(two_pass) - 1, previous=None,
                                  quantization_stage=None, a=args(epochs=2.0))


def test_continuation_mode_requires_the_declared_previous_data(tmp_path):
    old, new, heads = tmp_path / 'r2.jsonl', tmp_path / 'r3.jsonl', tmp_path / 'heads.pt'
    for path, text in ((old, 'r2'), (new, 'r3'), (heads, 'h')):
        path.write_text(text)
    a = SimpleNamespace(heads=heads, records=None, pieces=None, text_data=new, continue_from=None,
                        continue_identity=None, continue_monitor_reference=None, continued_text_data=None)
    inputs = trainer.identity_inputs(a)
    trained_on_old = {'step': 7, 'identity': {'inputs': trainer.identity_inputs(SimpleNamespace(**{**vars(a),
                                                                                                  'text_data': old}))}}
    assert trainer.continuation_mode(a, None, inputs) == ('none', None)
    assert trainer.continuation_mode(a, trained_on_old, inputs) == ('none', None)  # identity check refuses it
    a.continued_text_data = old
    assert trainer.continuation_mode(a, trained_on_old, inputs) == ('continue', None)
    record = {'consumed_updates': 7}
    continued = {'step': 9, 'identity': {'inputs': inputs}, 'data_continuation': record}
    assert trainer.continuation_mode(a, continued, inputs) == ('resume', record)
    a.continued_text_data = heads  # not the data the checkpoint read
    assert trainer.continuation_mode(a, trained_on_old, inputs) == ('none', None)


def _handoff_fixture(tmp_path):
    out = tmp_path / 'run'
    frozen = out / 'runtime' / 'natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'a.py').write_text('x = 1\n')
    package = tmp_path / 'package'
    package.mkdir()
    (package / 'a.py').write_text('x = 2\n')
    (out / 'sft').mkdir()

    def plan_for(header='r2', continued=None):
        bindings = {'header': {'path': 'h2.jsonl', 'sha256': 'r2'}}
        inputs = {'text_data': 'header'}
        stage_inputs = {'text_data': {'binding': 'header', 'path': '/d/r2', 'sha256': 'r2'}}
        if header != 'r2':
            bindings['header_new'] = {'path': 'h3.jsonl', 'sha256': header}
            inputs = {'text_data': 'header_new'}
            stage_inputs = {'text_data': {'binding': 'header_new', 'path': '/d/' + header, 'sha256': header}}
        if continued:
            inputs['continued_text_data'] = 'header'
            stage_inputs['continued_text_data'] = {'binding': 'header', 'path': '/d/r2', 'sha256': continued}
        recipe = {'id': 'r', 'input_bindings': bindings,
                  'stages': [{'id': 'sft', 'kind': 'neuralese_direct', 'inputs': inputs,
                              'parameters': {'epochs': 1.0}}]}
        flat = {'/heads': 'H', **{v['path']: v['sha256'] for v in stage_inputs.values()}}
        return {'recipe': recipe, 'recipe_sha256': 'new', 'inputs': flat, 'device': 'cuda',
                'stage_inputs': {'sft': stage_inputs}}

    existing = dict(plan_for(), recipe_sha256='lineage', code={'a.py': runner.sha(frozen / 'a.py')})
    plan_path = out / 'recipe-plan.json'
    plan_path.write_text(json.dumps(existing))
    return out, frozen, package, plan_path, existing, plan_for


def test_handoff_moves_the_active_stage_to_new_data_only_by_a_declared_data_continuation(tmp_path):
    out, frozen, package, plan_path, existing, plan_for = _handoff_fixture(tmp_path)
    with pytest.raises(ValueError, match='only by a declared data continuation'):
        runner.lineage_handoff(copy.deepcopy(existing), plan_for('r3'), frozen, package, plan_path, out, dry_run=True)
    with pytest.raises(ValueError, match='only by a declared data continuation'):  # not the data it trained on
        runner.lineage_handoff(copy.deepcopy(existing), plan_for('r3', continued='other'), frozen, package, plan_path,
                               out, dry_run=True)
    updated = runner.lineage_handoff(copy.deepcopy(existing), plan_for('r3', continued='r2'), frozen, package,
                                     plan_path, out)
    record = updated['handoffs'][-1]
    assert record['stage_input_changes']['sft']['kind'] == 'data_continuation'
    assert record['previous_inputs'] == existing['inputs'] and updated['recipe_sha256'] == 'lineage'
    assert updated['stage_inputs']['sft']['text_data']['sha256'] == 'r3'
    # A finished stage never takes new data.
    (out / 'sft-report.json').write_text('{}')
    (package / 'a.py').write_text('x = 3\n')
    with pytest.raises(ValueError, match='inputs of started stage sft'):
        runner.lineage_handoff(updated, plan_for('r4', continued='r3'), frozen, package, plan_path, out)


def test_the_shared_handler_declares_the_continuation_role():
    assert 'continued_text_data' in runner.HANDLERS['neuralese_direct']['optional_inputs']
    assert runner.missing_stage_inputs('neuralese_direct', {'records', 'continued_text_data'})
    args_ = runner.stage_input_args({'text_data': {'path': '/new'}, 'continued_text_data': {'path': '/old'}},
                                    'neuralese_direct')
    assert args_[args_.index('--continued-text-data') + 1] == '/old'
    assert 'continued_text_data' in trainer.OPERATIONAL  # never part of the trainer's resume identity options


class _Tokenizer:
    unk_token_id = 0

    def __init__(self):
        self.vocab = {}

    def convert_tokens_to_ids(self, token):
        return self.unk_token_id

    def __len__(self):
        return 10_000

    def __call__(self, text, add_special_tokens=False):
        return {'input_ids': [3 + self.vocab.setdefault(word, len(self.vocab)) for word in text.split()]}


def test_streamed_plan_data_equals_the_trainers_windows(tmp_path):
    """stage_plan_data (a data-only plan, row by row) gives the trainer's direct_schedule exactly."""
    tokenizer = _Tokenizer()
    engine = SimpleNamespace(tokenizer=tokenizer, backbone=SimpleNamespace(controls=SimpleNamespace(open_id=1,
                                                                                                    close_id=2)),
                             _tokens=lambda text: tokenizer(text)['input_ids'])
    rows = []
    for i in range(30):
        words = ' '.join(f'w{(i * 7 + j) % 50}' for j in range(5 + (i * 11) % 40))
        rows.append({'id': f'd{i}', 'text': words, 'split': 'test' if i % 9 == 0 else 'train',
                     'source_groups': [f'g{i}'], **({'text_cohort': 'other'} if i % 4 == 0 else {})})
    rows.append(dict(rows[3]))  # an exact duplicate (dropped)
    rows.append({**rows[9], 'split': 'train', 'source_groups': ['g-dup']})  # a train copy of a held text (dropped)
    data = tmp_path / 'text.jsonl'
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    a = SimpleNamespace(records=None, pieces=None, tokens=16, prefix_tokens=4, target_tokens=6, held_documents=2,
                        seed=0, batch_tokens=48, batch=1, epochs=1.0, steps=None)
    _, targets, held, _, _, _ = trainer.stage_data(engine, a, data)
    light_targets, light_held, _ = trainer.stage_plan_data(engine, a, data)
    assert len(light_targets) == len(targets) and len(targets) > 30
    assert trainer.held_documents(light_held) == trainer.held_documents(held)
    _, plan = trainer.direct_schedule(targets, a)
    _, light_plan = trainer.direct_schedule(light_targets, a)
    assert trainer.plan_documents_sha256(plan) == trainer.plan_documents_sha256(light_plan)
    assert [len(b) for b, _ in plan] == [len(b) for b, _ in light_plan]
