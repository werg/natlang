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


def test_cohort_repeats_upweight_a_cohort_in_every_epoch_and_continue_with_new_data():
    """--cohort-repeats: each document of a named cohort appears that many times per epoch (agentic tool-loop
    sessions, union r4); a continuation onto data with the cohort repeats only its unseen documents."""
    a = args(cohort_repeats={'agentic-tool-loop': 3})
    data = corpus(range(30))
    for w in data:
        if w['document'] in {'h-1', 'h-2', 'h-3'}:
            w['cohort'] = 'agentic-tool-loop'
    _, plan = trainer.direct_schedule(data, a)
    trained = [w['document'] for batch, _ in plan for w in batch]
    assert all(trained.count(f'h-{d}') == 3 * 3 for d in (1, 2, 3))  # 3 windows x 3 repeats
    assert all(trained.count(f'h-{d}') == 3 for d in range(4, 30))
    assert [c for _, c in plan][-1] == pytest.approx(1.0)
    _, base = trainer.direct_schedule(data, args())
    assert len([w for b, _ in base for w in b]) == len(trained) - 2 * 9
    # Continuation from cohort-free data: the previous plan rebuilds without repeats (no cohort there).
    previous = corpus(range(10))
    record, cont = trainer.data_continuation(previous, previous[:2], data, data[:2], consumed=3, previous=None,
                                             quantization_stage=None, a=a, previous_a=args())
    unseen_agentic = {w['document'] for w in trainer.continued_unseen(data, record)} & {'h-1', 'h-2', 'h-3'}
    later = [w['document'] for batch, _ in cont[3:] for w in batch]
    assert all(later.count(d) == 9 for d in unseen_agentic)
    assert 'cohort_repeats' in trainer.WINDOWING_KEYS


class _FingerprintTokenizer(_Tokenizer):
    chat_template = None
    all_special_tokens = ['<s>']
    backend_tokenizer = SimpleNamespace(to_str=lambda: '{}')

    def get_vocab(self):
        return {'<s>': 0}


def test_a_second_pass_over_the_same_text_file_skips_what_the_first_decided(tmp_path, monkeypatch):
    """Row-index cache (common.row_index): the stream's validation and duplicate filtering is a function of the file bytes
    and the tokenizer, so a later call makes one pass over the file instead of two and yields exactly the same rows."""
    import pathlib
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    rows = []
    for i in range(12):
        rows.append({'id': f'd{i}', 'text': ' '.join(f'w{(i * 7 + j) % 30}' for j in range(5 + i)),
                     'split': 'test' if i % 5 == 0 else 'train', 'source_groups': [f'g{i}']})
    rows.append(dict(rows[3]))  # an exact duplicate
    rows.append({**rows[5], 'split': 'train', 'source_groups': ['g-dup']})  # a train copy of a held text
    data = tmp_path / 'text.jsonl'
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    opened = []
    real_open = pathlib.Path.open
    monkeypatch.setattr(pathlib.Path, 'open', lambda self, *a, **k: (opened.append(str(self)), real_open(self, *a, **k))[1])
    tokenizer = _FingerprintTokenizer()
    cold = list(trainer.stream_text_rows(data, tokenizer))
    cold_opens = opened.count(str(data))
    warm = list(trainer.stream_text_rows(data, tokenizer))
    warm_opens = opened.count(str(data)) - cold_opens
    assert cold == warm and len(cold) == len(rows) - 2
    assert (cold_opens, warm_opens) == (2, 1)
    data.write_text(data.read_text() + json.dumps({**rows[1], 'id': 'extra', 'text': 'brand new words here',
                                                    'source_groups': ['g-extra']}) + '\n')
    assert len(list(trainer.stream_text_rows(data, tokenizer))) == len(cold) + 1  # another file hash: no stale entry


def test_prepared_window_cache_restores_roles_invalidates_and_rebuilds_corruption(tmp_path, monkeypatch):
    """A warm direct resume restores exact windows and role masks without tokenizing rows again."""
    from natlang_neuralese.train import prepared_windows
    from natlang_neuralese.train.text_warmup import prepare_text_windows

    class RoleTokenizer(_FingerprintTokenizer):
        unk_token_id = -1
        special = {'<|im_start|>': 99, 'system': 10, 'user': 11, 'assistant': 12,
                   'tool': 13, '<think>': 14, '</think>': 15}

        def convert_tokens_to_ids(self, token):
            return self.special.get(token, self.unk_token_id)

    tokenizer = RoleTokenizer()
    engine = SimpleNamespace(tokenizer=tokenizer, backbone=SimpleNamespace(controls=SimpleNamespace(open_id=1,
                                                                                                    close_id=2)),
                             _tokens=lambda text: [40, 41])
    monkeypatch.setenv('NATLANG_PREPARED_WINDOW_CACHE', str(tmp_path / 'prepared-cache'))
    data = tmp_path / 'text.jsonl'
    rows = [
        {'id': 'train', 'text': '<|im_start|>system hello <|im_start|>user question <|im_start|>assistant answer',
         'token_ids': [99, 10, 30, 31, 99, 11, 32, 99, 12, 33, 34, 35, 36],
         'split': 'train', 'source_groups': ['train-group'], 'supervised_suffix_start': 9},
        {'id': 'held', 'text': '<|im_start|>system hello <|im_start|>user prompt <|im_start|>assistant held',
         'token_ids': [99, 10, 30, 31, 99, 11, 32, 99, 12, 33, 34, 35, 36],
         'split': 'test', 'source_groups': ['held-group'], 'supervised_suffix_start': 9},
    ]
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    for row in rows:
        row['tokenizer_sha256'] = tokenizer_fingerprint(tokenizer)
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    a = SimpleNamespace(records=None, pieces=None, text_data=data, tokens=10, prefix_tokens=3,
                        target_tokens=4, train_target_tokens=4, train_min_context_tokens=2,
                        held_documents=1, merge_prefix_rows=False, recovery_selection=None, e2e_mix=None,
                        out=tmp_path)
    # Missing pinned annotations must never share a cache key with an
    # unannotated preparation (or any previously prepared pinned input).
    a.recovery_selection = {'path': str(tmp_path / 'missing-recovery.jsonl'),
                            'sha256': '0' * 64, 'rules': ['failed'], 'rendering': 'full'}
    assert prepared_windows.cache_spec(engine, a, data) is None
    a.recovery_selection = None
    a.e2e_mix = {'neuralese_spans': {'path': str(tmp_path / 'missing-spans.jsonl'),
                                     'sha256': '0' * 64}}
    assert prepared_windows.cache_spec(engine, a, data) is None
    a.e2e_mix = None
    calls = []

    def counted(*args, **kwargs):
        calls.append(1)
        return prepare_text_windows(*args, **kwargs)

    monkeypatch.setattr('natlang_neuralese.train.text_warmup.prepare_text_windows', counted)
    cold = trainer.stage_data(engine, a, data)
    cold_calls = len(calls)
    assert cold_calls == len(rows)
    warm = trainer.stage_data(engine, a, data)
    assert len(calls) == cold_calls

    def shape(result):
        train, targets, held, selection, receipt, row_meta = result
        return (
            [[(list(w['ids']), list(w['roles']), {k: v for k, v in w.items() if k not in ('ids', 'roles')})
              for w in train],
             [list(w['ids']) for w in targets],
             [(list(w['ids']), list(w['roles'])) for w in held], selection, receipt, row_meta]
        )

    assert shape(warm) == shape(cold)

    spec = prepared_windows.cache_spec(engine, a, data)
    entry = prepared_windows.cache_path(spec)
    token_shard = next(entry.glob('tokens-*.bin'))
    damaged = bytearray(token_shard.read_bytes())
    damaged[-1] ^= 0xFF
    token_shard.write_bytes(damaged)
    rebuilt = trainer.stage_data(engine, a, data)
    assert len(calls) == cold_calls + len(rows)
    assert shape(rebuilt) == shape(cold)
    trainer.stage_data(engine, a, data)
    assert len(calls) == cold_calls + len(rows)

    a.target_tokens = 3
    trainer.stage_data(engine, a, data)
    assert len(calls) == cold_calls + len(rows) * 2

    rows.append({'id': 'new', 'text': '<|im_start|>system hello <|im_start|>user new <|im_start|>assistant reply',
                 'token_ids': [99, 10, 30, 31, 99, 11, 32, 99, 12, 33, 34, 35, 36],
                 'split': 'train', 'source_groups': ['new-group'], 'supervised_suffix_start': 9,
                 'tokenizer_sha256': tokenizer_fingerprint(tokenizer)})
    data.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    trainer.stage_data(engine, a, data)
    assert len(calls) == cold_calls + 2 * (len(rows) - 1) + len(rows)


def test_compressed_text_matches_plain_eager_and_cached_streams(tmp_path, monkeypatch):
    import subprocess
    from natlang_neuralese.train.text_warmup import load_text_rows
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    rows = [{'id': f'd{i}', 'text': f'words {i}\u2028tail\u2029end',
             'split': 'test' if i == 0 else 'train', 'source_groups': [f'g{i}']}
            for i in range(5)]
    rows += [dict(rows[2]), {**rows[0], 'split': 'train', 'source_groups': ['held-copy']}]
    raw = ''.join(json.dumps(r, ensure_ascii=False) + '\r\n' for r in rows).encode()
    plain, packed = tmp_path / 'text.jsonl', tmp_path / 'text.jsonl.zst'
    plain.write_bytes(raw)
    packed.write_bytes(subprocess.run(['zstd', '-q', '-c'], input=raw,
                                     capture_output=True, check=True).stdout)
    tokenizer = _FingerprintTokenizer()
    expected = list(trainer.stream_text_rows(plain, tokenizer))
    assert len(expected) == 5
    assert list(trainer.stream_text_rows(packed, tokenizer)) == expected
    assert list(trainer.stream_text_rows(packed, tokenizer)) == expected
    assert load_text_rows(None, text_data=plain) == load_text_rows(None, text_data=packed)


@pytest.mark.parametrize('has_edge', [True, False])
def test_edged_value_masks_precede_prefix_merge_and_have_distinct_cache_identity(tmp_path, monkeypatch, has_edge):
    import hashlib
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    tokenizer = _FingerprintTokenizer()
    base = list(range(10, 130))
    rows = [{'id': f'r{i}', 'text': f't{i}', 'split': 'train', 'source_groups': [f'g{i}'],
             'token_ids': base[:n], 'tokenizer_sha256': tokenizer_fingerprint(tokenizer),
             'supervised_suffix_start': n - 2} for i, n in enumerate((70, 90, 110))]
    data = tmp_path / 'data.jsonl'
    data.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    a = SimpleNamespace(merge_prefix_rows=True, recovery_selection=None, e2e_mix=None, out=tmp_path)
    assert len(list(trainer.stage_rows(a, data, tokenizer))) == 1
    for index in (0, 1):
        path = tmp_path / f'mask{index}.jsonl'
        path.write_text(json.dumps({'id': f'r{index}', 'spans': [[65, 70, 'value', has_edge]]}) + '\n')
        a.e2e_mix = {'neuralese_spans': {'path': str(path),
                                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}}
        cold = list(trainer.stage_rows(a, data, tokenizer))
        warm = list(trainer.stage_rows(a, data, tokenizer))
        assert cold == warm and len(cold) == 2
        assert {r['id'] for r in cold} == {f'r{index}', 'r2'}
        field = 'context_only_spans' if has_edge else 'token_choice_detached_spans'
        assert next(r for r in cold if r['id'] == f'r{index}')[field] == [[65, 70]]


def test_the_prefix_merge_decisions_are_cached_under_their_key(tmp_path, monkeypatch):
    from natlang_neuralese.train.text_warmup import stream_merge_prefix_rows
    monkeypatch.setenv('NATLANG_ROW_INDEX_CACHE', str(tmp_path / 'cache'))
    base = list(range(10, 10 + 120))
    rows = [{'id': f'r{i}', 'text': f't{i}', 'split': 'train', 'source_groups': [f'g{i}'], 'token_ids': base[:n],
             'supervised_suffix_start': n - 2} for i, n in enumerate((70, 90, 110, 80))]
    passes = []

    def source():
        passes.append(1)
        return (dict(r) for r in rows)
    key = tmp_path / 'cache' / 'merge-test.pkl'
    cold, cold_receipt = stream_merge_prefix_rows(source, cache_key=key)
    cold = list(cold)
    assert len(passes) == 2
    warm, warm_receipt = stream_merge_prefix_rows(source, cache_key=key)
    warm = list(warm)
    assert len(passes) == 3  # the warm call made only the yielding pass
    assert warm == cold and warm_receipt == cold_receipt and cold_receipt['contained_rows'] > 0


def test_combined_epoch_and_data_handoff_accepts_only_verified_consumed_placeholders():
    old = corpus(range(40))
    new = corpus(range(100))
    a = args(epochs=1.0)
    record, plan = trainer.data_continuation(old, [], new, [], consumed=9, previous=None,
                                            quantization_stage=None, a=a)
    sampler = trainer.direct_sampler(new)
    assert trainer.verify_epoch_budget_handoff(
        sampler, plan, a, previous_epochs=0.5, consumed=9, data_continuation=record
    ) == 'declared-data-continuation-prefix'
    with pytest.raises(ValueError, match='consumed update placeholders'):
        trainer.verify_epoch_budget_handoff(sampler, plan, a, previous_epochs=0.5, consumed=8,
                                            data_continuation=record)
    bad = list(plan)
    bad[0] = ([new[0]], 0.0)
    with pytest.raises(ValueError, match='consumed update placeholders'):
        trainer.verify_epoch_budget_handoff(sampler, bad, a, previous_epochs=0.5, consumed=9,
                                            data_continuation=record)


def test_epoch_only_handoff_still_rejects_consumed_batch_changes():
    data = corpus(range(40))
    a = args()
    sampler, plan = trainer.direct_schedule(data, a)
    bad = list(plan)
    bad[0] = ([], 0.0)
    with pytest.raises(ValueError, match='changes or discards consumed'):
        trainer.verify_epoch_budget_handoff(sampler, bad, a, previous_epochs=1.0, consumed=1)
