from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path

from natlang_neuralese.common.jsonio import open_jsonl_text, write_jsonl_text

ROOT = Path(__file__).resolve().parents[2]
RENDERER = ROOT / 'scripts/neuralese_data/render_native_turn_corpus.py'
spec = importlib.util.spec_from_file_location('streaming_trajectory_renderer', RENDERER)
renderer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = renderer
spec.loader.exec_module(renderer)


def _source(path: Path):
    rows = [
        {'id': 'a0', 'teacher_trajectory_id': 'trajectory-a', 'source_ref': {'trajectory_id': 'trajectory-a', 'invocation_id': 'i0'},
         'split': 'train', 'source_groups': ['g-a'], 'messages': [], 'target': {'role': 'assistant', 'content': 'one'}},
        {'id': 'b0', 'teacher_trajectory_id': 'trajectory-b', 'source_ref': {'trajectory_id': 'trajectory-b', 'invocation_id': 'i0'},
         'split': 'test', 'source_groups': ['g-b'], 'messages': [], 'target': {'role': 'assistant', 'content': 'held'}},
        {'id': 'a1', 'teacher_trajectory_id': 'trajectory-a', 'source_ref': {'trajectory_id': 'trajectory-a', 'invocation_id': 'i1'},
         'split': 'train', 'source_groups': ['g-a'], 'messages': [], 'target': {'role': 'assistant', 'content': 'two'}},
    ]
    with write_jsonl_text(path) as stream:
        for row in rows:
            stream.write(renderer.canonical(row) + '\n')
    return rows


def _read(path):
    with open_jsonl_text(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


class _CharacterChatTokenizer:
    """Small deterministic tokenizer exercising the real gold renderer."""
    all_special_tokens = ('<|im_start|>', '<|im_end|>', '<|im_sep|>', '<|tool_call_start|>', '<|tool_call_end|>')
    unk_token_id = 0
    chat_template = 'fixture-chat-template'

    def __init__(self):
        self.natlang_history_reasoning = {'policy': 'undeclared', 'backbone': 'fixture', 'probe': {}}
        self._special_ids = {value: index + 1 for index, value in enumerate(self.all_special_tokens)}
        self._special_pattern = re.compile('(' + '|'.join(re.escape(x) for x in
            sorted(self.all_special_tokens, key=len, reverse=True)) + ')')
        self.backend_tokenizer = type('Backend', (), {'to_str': lambda _self: '{"type":"fixture"}'})()

    def get_vocab(self):
        return {chr(code): 1000 + code for code in range(128)} | self._special_ids

    def apply_chat_template(self, messages, *, tools=None, tokenize=False, add_generation_prompt=False, **_kwargs):
        text = ''
        for message in messages:
            role = message.get('role', 'user')
            text += f'<|im_start|>{role}\n{message.get("content") or ""}'
            reasoning = message.get('reasoning_content')
            if isinstance(reasoning, str):
                text += '<think>' + reasoning + '</think>'
            calls = message.get('tool_calls') or []
            if calls:
                text += '<|tool_call_start|>' + json.dumps(calls, sort_keys=True) + '<|tool_call_end|>'
            text += '<|im_end|>\n'
        if add_generation_prompt:
            text += '<|im_start|>assistant\n'
        return text

    def convert_tokens_to_ids(self, token):
        return self._special_ids.get(token, self.unk_token_id)

    def _encode(self, text):
        ids, offsets = [], []
        cursor = 0
        for part in self._special_pattern.split(text):
            if not part:
                continue
            end = cursor + len(part)
            if part in self._special_ids:
                ids.append(self._special_ids[part])
                offsets.append((cursor, end))
            else:
                for index, char in enumerate(part):
                    ids.append(1000 + ord(char))
                    offsets.append((cursor + index, cursor + index + 1))
            cursor = end
        return ids, offsets

    def __call__(self, text, *, return_offsets_mapping=False, **_kwargs):
        ids, offsets = self._encode(text)
        result = {'input_ids': ids}
        if return_offsets_mapping:
            result['offset_mapping'] = offsets
        return result

    def encode(self, text, **_kwargs):
        return self._encode(text)[0]

    def decode(self, ids, *, skip_special_tokens=False, **_kwargs):
        pieces = []
        reverse = {value: key for key, value in self._special_ids.items()}
        for token in ids:
            if token in reverse:
                if not skip_special_tokens:
                    pieces.append(reverse[token])
            else:
                pieces.append(chr(token - 1000))
        return ''.join(pieces)


def _trajectory_rows():
    def record(identifier, trajectory, split, groups, messages, target):
        return {'id': identifier, 'teacher_trajectory_id': trajectory,
                'source_ref': {'trajectory_id': trajectory, 'invocation_id': 'invocation-' + trajectory},
                'split': split, 'source_groups': groups, 'training_admission': {'approved': True},
                'messages': messages, 'target': {'role': 'assistant', 'content': target}, 'tools': []}

    a_context = [{'role': 'system', 'content': 'program system'}, {'role': 'user', 'content': 'input A'}]
    b_context = [{'role': 'system', 'content': 'program system'}, {'role': 'user', 'content': 'input B'}]
    a0 = record('a0', 'trajectory-a', 'train', ['g-a'], a_context, 'answer A0')
    block_id = 'nz1_' + 'a' * 64
    a0['decision'] = {'training_approved': True, 'failed_action': False}
    a0['target']['tool_calls'] = [{
        'id': 'call-a0', 'type': 'function',
        'function': {'name': 'write', 'arguments': json.dumps({'value': {
            '$write': {'name': 'soft-state:' + block_id, 'block_id': block_id,
                       'type': 'Neuralese<string>', 'source': 'shared producer payload'}}})},
    }]
    a0['target']['reasoning_content'] = 'writer reasoning retained in history'
    b0 = record('b0', 'trajectory-b', 'test', ['g-b'], b_context, 'answer B0')
    a1_messages = [*a_context, a0['target'], {'role': 'user', 'content': 'follow up A'}]
    b1_messages = [*b_context, b0['target'], {'role': 'user', 'content': 'follow up B'}]
    a1 = record('a1', 'trajectory-a', 'train', ['g-a'], a1_messages, 'answer A1')
    b1 = record('b1', 'trajectory-b', 'test', ['g-b'], b1_messages, 'answer B1')
    reader_context = [{'role': 'system', 'content': 'program system'},
                      {'role': 'user', 'content': [{'type': 'neuralese', 'id': block_id}]}]
    c0 = record('c0', 'trajectory-c', 'train', ['g-a'], reader_context, 'answer C0')
    duplicate = record('a0-copy', 'trajectory-copy', 'train', ['g-a'], a_context, 'answer A0')
    return [a0, b0, a1, b1, c0, duplicate]


def test_real_materializer_and_streaming_wrapper_are_byte_equivalent(tmp_path, monkeypatch):
    from scripts.neuralese_data.strip_generic_system_prompt import strip_row
    tokenizers = {'lfm25': _CharacterChatTokenizer(), 'mellum21': _CharacterChatTokenizer()}
    renderer._G.clear()
    renderer._G.update({'tok': tokenizers, 'pieces': {},
                        'strip_row': strip_row, 'cross_trajectory_writer_sources': {}})
    selected = tmp_path / 'actual-selected.jsonl'
    with write_jsonl_text(selected) as stream:
        rows = _trajectory_rows()
        for row in rows:
            stream.write(renderer.canonical(row) + '\n')
    modes = ('full', 'program-header', 'whole-trajectory')
    blocked = {renderer.split_chain_key(rows[2]): len(rows[2]['messages'])}
    sql_traces = []
    sqlite_paths = []
    real_connect = renderer.sqlite3.connect

    def traced_connect(*args, **kwargs):
        sqlite_paths.append(args[0])
        connection = real_connect(*args, **kwargs)
        statements = []
        connection.set_trace_callback(statements.append)
        sql_traces.append(statements)
        return connection

    monkeypatch.setattr(renderer.sqlite3, 'connect', traced_connect)
    for tokenizer_name in ('lfm25', 'mellum21'):
        ref_dir, stream_dir = tmp_path / f'ref-{tokenizer_name}', tmp_path / f'stream-{tokenizer_name}'
        job_ref = (str(selected), str(ref_dir), tokenizer_name, blocked, 'fixture', set(), {'b0'}, False,
                   None, modes, False, set())
        ref_receipt = renderer._render_shard_materialized(job_ref)
        renderer._G['cross_trajectory_writer_sources'] = {}
        sqlite_scratch = tmp_path / f'sqlite-scratch-{tokenizer_name}'
        job_stream = (str(selected), str(stream_dir), tokenizer_name, blocked, 'fixture', set(), {'b0'}, False,
                      None, modes, False, set(), None, None, sqlite_scratch)
        stream_receipt = renderer.render_shard(job_stream)
        assert Path(sqlite_paths[-1]).is_relative_to(sqlite_scratch)
        assert not Path(sqlite_paths[-1]).exists()
        for mode in modes:
            reference_rows = _read(ref_dir / mode / tokenizer_name / 'actual-selected.jsonl')
            streamed_rows = _read(stream_dir / mode / tokenizer_name / 'actual-selected.jsonl')
            assert reference_rows == streamed_rows
            if mode == 'full':
                reader = next(row for row in reference_rows if row['id'] == 'c0')
                assert 'shared producer payload' in reader['text']
                assert reader['neuralese_context_attestations'][0]['writer_record_id'] == 'a0'
        assert ref_receipt['records'] == stream_receipt['records']
        for key in ('full', 'program-header', 'whole-trajectory', 'decontaminated', 'decision_excluded'):
            assert ref_receipt.get(key, 0) == stream_receipt.get(key, 0)
        assert ref_receipt['parity'] == stream_receipt['parity']
        assert _read(ref_dir / 'omissions' / tokenizer_name / 'actual-selected.jsonl') == \
               _read(stream_dir / 'omissions' / tokenizer_name / 'actual-selected.jsonl')

    assert len(sql_traces) == 2
    assert len(sqlite_paths) == 2
    for statements in sql_traces:
        normalized = [statement.strip().upper() for statement in statements]
        assert 'PRAGMA CACHE_SIZE=-65536' in normalized
        last_insert = max(i for i, statement in enumerate(normalized)
                         if statement.startswith('INSERT INTO ROWS'))
        trajectory_index = normalized.index('CREATE INDEX ROWS_TRAJECTORY_SEQ ON ROWS(TRAJECTORY, SEQ)')
        id_index = normalized.index('CREATE INDEX ROWS_ID_SEQ ON ROWS(ROW_ID, SEQ)')
        assert last_insert < trajectory_index < id_index


def test_streamed_trajectory_groups_are_exact_and_resume_after_interruption(tmp_path, monkeypatch):
    selected = tmp_path / 'selected.jsonl.zst'
    source_rows = _source(selected)
    calls = []
    connections = []
    sqlite_paths = []
    real_connect = renderer.sqlite3.connect

    def traced_connect(*args, **kwargs):
        sqlite_paths.append(Path(args[0]))
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(renderer.sqlite3, 'connect', traced_connect)

    def materialized(job):
        shard_path, out_dir, tokenizer, _blocked, _corpus, _held, _decision, _header, _full, modes, compressed, *_storage = job
        group_source = Path(shard_path)
        group_root = Path(out_dir)
        group_id = group_source.name.removesuffix('.jsonl')
        rows = _read(group_source)
        calls.append([row['id'] for row in rows])
        if len(calls) == 2:
            raise RuntimeError('simulated worker interruption before trajectory commit')
        result = {'shard': group_id, 'tokenizer': tokenizer, 'records': len(rows), 'whole-trajectory': len(rows),
                  'parity': {'equal': len(rows)}}
        for mode in modes:
            output = group_root / mode / tokenizer / (f'{group_id}.jsonl.zst' if compressed else f'{group_id}.jsonl')
            output.parent.mkdir(parents=True, exist_ok=True)
            with write_jsonl_text(output) as stream:
                for row in rows:
                    stream.write(renderer.canonical({'id': row['id'], 'split': row['split'],
                                                     'source_groups': row['source_groups']}) + '\n')
        omission = group_root / 'omissions' / tokenizer / f'{group_id}.jsonl'
        omission.parent.mkdir(parents=True, exist_ok=True)
        omission.write_text('')
        return result

    monkeypatch.setattr(renderer, '_render_shard_materialized', materialized)
    work = tmp_path / 'work'
    sqlite_scratch = tmp_path / 'sqlite-scratch'
    job = (str(selected), str(work), 'lfm25', {}, 'fixture', set(), set(), False, None,
           ('whole-trajectory',), True, set(), None, None, sqlite_scratch)
    import pytest
    with pytest.raises(RuntimeError, match='simulated worker interruption'):
        renderer.render_shard(job)
    assert len(list((work / 'trajectory-stream' / 'lfm25' / 'selected').glob('receipts/*.json'))) == 1
    assert sqlite_paths and not sqlite_paths[0].exists()
    with pytest.raises(renderer.sqlite3.ProgrammingError, match='closed database'):
        connections[0].execute('SELECT 1')

    calls.clear()
    result = renderer.render_shard(job)
    assert result['records'] == len(source_rows)
    assert result['trajectory_groups'] == 2
    # The first trajectory was committed before the interruption and must be
    # reused; the other group is rendered exactly once after restart.
    assert len(calls) == 1
    output = work / 'whole-trajectory' / 'lfm25' / 'selected.jsonl.zst'
    assert [row['id'] for row in _read(output)] == ['a0', 'b0', 'a1']
    assert not list(output.parent.glob('*.tmp'))
    receipt_path = next((work / 'trajectory-stream' / 'lfm25' / 'selected' / 'receipts').glob('*.json'))
    receipt = json.loads(receipt_path.read_text())
    receipt['outputs'] = {}
    receipt_path.write_text(json.dumps(receipt) + '\n')
    with __import__('pytest').raises(ValueError, match='output inventory mismatch'):
        renderer.render_shard(job)


def test_streamed_trajectory_binding_rejects_changed_source(tmp_path, monkeypatch):
    selected = tmp_path / 'selected.jsonl'
    rows = _source(selected)

    def materialized(job):
        group_id = Path(job[0]).stem
        root = Path(job[1])
        path = root / 'whole-trajectory' / job[2] / f'{group_id}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('')
        omission = root / 'omissions' / job[2] / f'{group_id}.jsonl'
        omission.parent.mkdir(parents=True, exist_ok=True)
        omission.write_text('')
        return {'shard': group_id, 'tokenizer': job[2], 'records': 1, 'whole-trajectory': 0}

    monkeypatch.setattr(renderer, '_render_shard_materialized', materialized)
    job = (str(selected), str(tmp_path / 'work'), 'lfm25', {}, 'fixture', set(), set(), False, None,
           ('whole-trajectory',), False, set())
    renderer.render_shard(job)
    changed = rows[0]
    changed['target']['content'] = 'changed target'
    with write_jsonl_text(selected) as stream:
        for row in rows:
            stream.write(renderer.canonical(row) + '\n')
    with __import__('pytest').raises(ValueError, match='committed trajectory binding changed'):
        renderer.render_shard(job)


def _reviewed_legacy_dependency(monkeypatch):
    # These tests exercise the approved historical transition, not an automatic
    # waiver for whichever text renderer happens to be in the current checkout.
    # New assistant-text receipts intentionally changed that renderer's bytes.
    actual_file_sha = renderer.file_sha
    dependency = renderer.ROOT / 'training/neuralese/natlang_neuralese/data/text_corpus.py'
    monkeypatch.setattr(renderer, 'file_sha', lambda path:
        renderer.LEGACY_SEARCH_TEXT_CORPUS_DEFAULT_PROFILE_TRANSITION['current_sha256']
        if Path(path) == dependency else actual_file_sha(path))


@__import__('pytest').mark.parametrize('legacy_text', [None, '41b011ebc425839d62c50f9f9118b46c3749b1dfdbbf4e8979e4df92cbe356b3'])
def test_streamed_resume_preserves_old_group_receipt_and_records_mixed_renderer_ids(tmp_path, monkeypatch, legacy_text):
    _reviewed_legacy_dependency(monkeypatch)
    from scripts.neuralese_data.strip_generic_system_prompt import strip_row
    tokenizer = _CharacterChatTokenizer()
    renderer._G.clear()
    renderer._G.update({'tok': {'lfm25': tokenizer}, 'pieces': {},
                        'strip_row': strip_row, 'cross_trajectory_writer_sources': {}})
    selected = tmp_path / 'selected.jsonl'
    rows = _trajectory_rows()
    with write_jsonl_text(selected) as stream:
        for row in rows:
            stream.write(renderer.canonical(row) + '\n')
    out = tmp_path / 'work'
    renderings = ('whole-trajectory',)
    job = (str(selected), str(out), 'lfm25', {}, 'fixture', set(), set(), False,
           None, renderings, False, set())
    renderer.render_shard(job)

    source_digest = renderer.file_sha(selected)
    old_renderer_sha = renderer.LEGACY_SEARCH_RENDERER_SHA256
    old_binding = renderer._legacy_render_job_binding(job, source_digest, old_renderer_sha, historical_text_sha256=legacy_text)
    # The one text-corpus dependency transition is allowed only for the
    # unchanged default profile. A KEEP render must never reuse old default
    # receipts even when the renderer bytes match this reviewed transition.
    with __import__('pytest').raises(ValueError, match='dependency differs'):
        renderer._legacy_render_job_binding(
            (*job, {'history_reasoning_policies': {'lfm25': 'keep'}}), source_digest, old_renderer_sha)
    code_binding = renderer.LEGACY_SEARCH_CODE_BINDING
    assert set(code_binding) == set(renderer.LEGACY_SEARCH_CODE_BINDING)
    current_code_binding = renderer._render_code_binding()
    lease_path = str(renderer.ROOT / 'scripts/neuralese_data/conversion_output_lease.py')
    assert current_code_binding[lease_path] == renderer.file_sha(Path(lease_path))
    assert current_code_binding[lease_path] != code_binding.get(lease_path)
    current_code_binding = renderer._render_code_binding()
    lease_path = str(renderer.ROOT / 'scripts/neuralese_data/conversion_output_lease.py')
    assert current_code_binding[lease_path] == renderer.file_sha(Path(lease_path))
    assert current_code_binding[lease_path] != code_binding.get(lease_path)
    trajectory_id = 'trajectory-a'
    group_id = renderer.sha(trajectory_id.encode())
    receipt_path = out / 'trajectory-stream' / 'lfm25' / 'selected' / 'receipts' / f'{group_id}.json'
    old_receipt = json.loads(receipt_path.read_text())
    trajectory_rows = []
    source_hashes = []
    with open_jsonl_text(selected) as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['teacher_trajectory_id'] == trajectory_id:
                trajectory_rows.append(row)
                source_hashes.append(renderer.sha(line.rstrip('\n').encode('utf-8')))
    source_ids = [row['id'] for row in trajectory_rows]
    old_group_binding = renderer._trajectory_render_binding(old_binding, trajectory_id,
                                                           source_ids, source_hashes)
    old_receipt['binding'] = old_group_binding
    old_receipt.pop('renderer_code_sha256')
    # The b054 output inventory includes its per-trajectory done sidecar.
    done_relative = f'done/lfm25/{group_id}.json'
    done_path = receipt_path.parent.parent / 'groups' / group_id / done_relative
    done_path.parent.mkdir(parents=True, exist_ok=True)
    done_bytes = renderer.canonical({'shard': group_id, 'tokenizer': 'lfm25',
                                     'records': len(trajectory_rows)}).encode() + b'\n'
    done_path.write_bytes(done_bytes)
    old_receipt['outputs'][done_relative] = renderer.sha(done_bytes)
    receipt_path.write_text(json.dumps(old_receipt, sort_keys=True) + '\n')

    original_output = _read(out / 'whole-trajectory' / 'lfm25' / 'selected.jsonl')
    result = renderer.render_shard((*job, old_renderer_sha))
    assert json.loads(receipt_path.read_text())['binding'] == old_group_binding
    assert 'renderer_code_sha256' not in json.loads(receipt_path.read_text())
    assert result['source_sha256'] == source_digest
    current_renderer_sha = renderer.file_sha(Path(renderer.__file__).resolve())
    assert result['trajectory_bindings'][group_id] == old_group_binding
    assert result['trajectory_renderers'][group_id] == old_renderer_sha
    assert result['trajectory_legacy_text_dependencies'][group_id] == (legacy_text or renderer.LEGACY_SEARCH_TEXT_CORPUS_DEFAULT_PROFILE_TRANSITION['historical_sha256'])
    bad = {**result, 'trajectory_legacy_text_dependencies': {group_id: 'f' * 64}}
    with __import__('pytest').raises(ValueError, match='historical text dependency'):
        renderer._verify_trajectory_receipts(job, bad, {old_renderer_sha, current_renderer_sha})
    renderer._verify_trajectory_receipts(job, result, {old_renderer_sha, current_renderer_sha})
    assert result['group_renderer_code_sha256s'] == sorted({
        old_renderer_sha, current_renderer_sha})
    assert _read(out / 'whole-trajectory' / 'lfm25' / 'selected.jsonl') == original_output
    done_original = done_path.read_bytes()
    done_path.write_bytes(done_original + b'corrupt')
    with __import__('pytest').raises(ValueError, match='trajectory output checksum mismatch'):
        renderer._verify_trajectory_receipts(job, result, {old_renderer_sha, current_renderer_sha})
    done_path.write_bytes(done_original)


def test_legacy_done_marker_requires_exact_source_and_old_renderer_witness(tmp_path):
    from scripts.neuralese_data.strip_generic_system_prompt import strip_row
    renderer._G.clear()
    renderer._G.update({'tok': {'lfm25': _CharacterChatTokenizer()}, 'pieces': {},
                        'tokenizer_paths': {'lfm25': tmp_path}, 'strip_row': strip_row,
                        'cross_trajectory_writer_sources': {}})
    selected = tmp_path / 'selected.jsonl'
    with write_jsonl_text(selected) as stream:
        for row in _trajectory_rows():
            stream.write(renderer.canonical(row) + '\n')
    out = tmp_path / 'work'
    input_binding = renderer._render_input_binding({'lfm25': tmp_path}, {})
    job = (str(selected), str(out), 'lfm25', {}, 'fixture', set(), set(), False,
           None, ('whole-trajectory',), False, set(), None, input_binding)
    result = renderer.render_shard(job)
    old_sha = renderer.LEGACY_SEARCH_RENDERER_SHA256
    old_done = {key: value for key, value in result.items()
                if key not in {'source_sha256', 'renderer_code_sha256', 'group_renderer_code_sha256s',
                               'trajectories_committed', 'aggregate_outputs',
                               'trajectory_bindings', 'trajectory_renderers',
                               'render_job_binding'}}
    renderer._verify_legacy_done_binding(job, old_done, old_sha, renderer.file_sha(selected))
    with __import__('pytest').raises(ValueError, match='approved historical migration witness'):
        renderer._verify_legacy_done_binding(job, old_done, 'c' * 64, renderer.file_sha(selected))
    # The prior Search renderer emitted aggregate-only markers. Those are
    # accepted solely as rerender triggers; current/incomplete receipts cannot
    # omit the trajectory count and bypass group validation.
    incomplete_current = dict(old_done, render_job_binding='0' * 64)
    with __import__('pytest').raises(ValueError, match='incomplete current receipt'):
        renderer._verify_legacy_done_binding(job, incomplete_current, old_sha,
                                              renderer.file_sha(selected))
    with __import__('pytest').raises(ValueError, match='selected-record count'):
        renderer._verify_legacy_done_binding(job, {key: value for key, value in old_done.items()
                                                    if key != 'records'}, old_sha,
                                              renderer.file_sha(selected))


def test_render_subset_selects_only_named_shards():
    jobs = [
        ('/selected/0000.jsonl.zst', '/work', 'lfm25'),
        ('/selected/0001.jsonl.zst', '/work', 'lfm25'),
        ('/selected/0001.jsonl.zst', '/work', 'mellum21'),
    ]
    assert renderer._filter_render_jobs(jobs, ['0001']) == jobs[1:]
    assert renderer._filter_render_jobs(jobs, None) == jobs
    with __import__('pytest').raises(ValueError, match='unknown selected shard stems'):
        renderer._filter_render_jobs(jobs, ['0002'])


def test_history_reasoning_opt_in_is_explicit_and_part_of_render_binding(tmp_path):
    default = renderer._render_input_binding({'lfm25': tmp_path}, {})
    keep = renderer._render_input_binding({'lfm25': tmp_path}, {}, {'lfm25': 'keep'})
    assert set(default) == {'tokenizers', 'pieces_sha256'}
    assert 'history_reasoning_policies' not in default
    assert keep['history_reasoning_policies'] == {'lfm25': 'keep'}
    assert default != keep
    assert renderer._parse_history_reasoning_policies(['lfm25=keep'], {'lfm25'}) == {'lfm25': 'keep'}
    for values, names in [(['lfm25=keep', 'lfm25=last_turn_only'], {'lfm25'}),
                          (['other=keep'], {'lfm25'}), (['lfm25=unknown'], {'lfm25'}),
                          (['keep'], {'lfm25'})]:
        with __import__('pytest').raises(ValueError):
            renderer._parse_history_reasoning_policies(values, names)


def test_renderer_worker_binds_explicit_profile_and_keeps_default(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    (tmp_path / 'config.json').write_text('{"model_type":"lfm2"}')

    class Switchable:
        def __init__(self):
            self.name_or_path = str(tmp_path)
            self.chat_template = '{{ preserve_thinking }}'

        def apply_chat_template(self, messages, **options):
            assistants = [m.get('reasoning_content', '') for m in messages if m.get('role') == 'assistant']
            if not options.get('preserve_thinking', False):
                assistants = assistants[-1:]
            return ' '.join(assistants)

    gold = sys.modules.get('build_neuralese_gold_text_delta')
    if gold is None:
        import importlib
        gold = importlib.import_module('build_neuralese_gold_text_delta')
    monkeypatch.setattr(gold, 'load_pinned_tokenizer', lambda _path: Switchable())
    renderer._G.clear()
    renderer._init({'lfm25': str(tmp_path)}, {})
    assert renderer._G['tok']['lfm25'].natlang_history_reasoning['policy'] == 'last_turn_only'
    renderer._G.clear()
    renderer._init({'lfm25': str(tmp_path)}, {}, {'lfm25': 'keep'})
    assert renderer._G['tok']['lfm25'].natlang_history_reasoning['policy'] == 'keep'


def test_done_receipts_verify_output_checksums_without_rewriting_them(tmp_path):
    from scripts.neuralese_data.strip_generic_system_prompt import strip_row
    renderer._G.clear()
    renderer._G.update({'tok': {'lfm25': _CharacterChatTokenizer()}, 'pieces': {},
                        'tokenizer_paths': {'lfm25': tmp_path}, 'strip_row': strip_row,
                        'cross_trajectory_writer_sources': {}})
    selected = tmp_path / 'selected.jsonl'
    with write_jsonl_text(selected) as stream:
        for row in _trajectory_rows():
            stream.write(renderer.canonical(row) + '\n')
    out = tmp_path / 'work'
    input_binding = renderer._render_input_binding({'lfm25': tmp_path}, {})
    job = (str(selected), str(out), 'lfm25', {}, 'fixture', set(), set(), False,
           None, ('whole-trajectory',), False, set(), None, input_binding)
    result = renderer.render_shard(job)
    old_sha = renderer.LEGACY_SEARCH_RENDERER_SHA256
    receipts_root = out / 'trajectory-stream' / 'lfm25' / 'selected' / 'receipts'
    before = {path.name: path.read_bytes() for path in receipts_root.glob('*.json')}
    renderer._verify_trajectory_receipts(job, result, {
        renderer.file_sha(Path(renderer.__file__).resolve()), old_sha})
    changed_policy_job = (*job[:7], True, *job[8:])
    with __import__('pytest').raises(ValueError, match='current render job policy'):
        renderer._verify_trajectory_receipts(changed_policy_job, result, {
            renderer.file_sha(Path(renderer.__file__).resolve()), old_sha})
    changed_inputs = {**input_binding, 'pieces_sha256': 'd' * 64}
    changed_input_job = (*job[:13], changed_inputs)
    with __import__('pytest').raises(ValueError, match='current render job policy'):
        renderer._verify_trajectory_receipts(changed_input_job, result, {
            renderer.file_sha(Path(renderer.__file__).resolve()), old_sha})
    assert {path.name: path.read_bytes() for path in receipts_root.glob('*.json')} == before
    aggregate_path = out / 'whole-trajectory' / 'lfm25' / 'selected.jsonl'
    aggregate_original = aggregate_path.read_bytes()
    aggregate_path.write_bytes(aggregate_original + b'corrupt')
    with __import__('pytest').raises(ValueError, match='aggregate output checksum mismatch'):
        renderer._verify_trajectory_receipts(job, result, {
            renderer.file_sha(Path(renderer.__file__).resolve()), old_sha})
    aggregate_path.write_bytes(aggregate_original)
    with __import__('pytest').raises(ValueError, match='approved historical migration witness'):
        renderer._legacy_render_job_binding(job, renderer.file_sha(selected), 'd' * 64)
    receipt_path = next(receipts_root.glob('*.json'))
    receipt = json.loads(receipt_path.read_text())
    relative = next(iter(receipt['outputs']))
    output_path = out / 'trajectory-stream' / 'lfm25' / 'selected' / 'groups' / receipt_path.stem / relative
    original = output_path.read_bytes()
    output_path.write_bytes(original + b'corrupt')
    with __import__('pytest').raises(ValueError, match='trajectory output checksum mismatch'):
        renderer._verify_trajectory_receipts(job, result, {
            renderer.file_sha(Path(renderer.__file__).resolve()), old_sha})


@__import__('pytest').mark.parametrize('parent_sha', sorted(renderer.SEARCH_REVIEWED_RESOURCE_PARENT_CHAIN))
def test_resource_only_renderer_parent_continues_exact_trajectory_receipts(tmp_path, monkeypatch, parent_sha):
    _reviewed_legacy_dependency(monkeypatch)
    from scripts.neuralese_data.strip_generic_system_prompt import strip_row
    renderer._G.clear()
    renderer._G.update({'tok': {'lfm25': _CharacterChatTokenizer()}, 'pieces': {},
                        'tokenizer_paths': {'lfm25': tmp_path}, 'strip_row': strip_row,
                        'cross_trajectory_writer_sources': {}})
    selected = tmp_path / 'selected.jsonl'
    rows = _trajectory_rows()
    with write_jsonl_text(selected) as stream:
        for row in rows:
            stream.write(renderer.canonical(row) + '\n')
    out = tmp_path / 'work'
    input_binding = renderer._render_input_binding({'lfm25': tmp_path}, {})
    job = (str(selected), str(out), 'lfm25', {}, 'fixture', set(), set(), False,
           None, ('whole-trajectory',), False, set(), None, input_binding, None, parent_sha)
    parent_code = renderer._resource_parent_code_binding(parent_sha, job)
    text_corpus_path = str(renderer._LEGACY_REPO / 'training/neuralese/natlang_neuralese/data/text_corpus.py')
    expected_text_dependency = ('current_sha256' if parent_sha == renderer.SEARCH_PROFILE_AWARE_RENDERER_SHA256 else 'historical_sha256')
    assert parent_code[text_corpus_path] == renderer.LEGACY_SEARCH_TEXT_CORPUS_DEFAULT_PROFILE_TRANSITION[expected_text_dependency]
    explicit_inputs = dict(input_binding)
    explicit_inputs['history_reasoning_policies'] = {'lfm25': 'keep'}
    explicit_job = (*job[:13], explicit_inputs, *job[14:])
    explicit_parent_code = renderer._resource_parent_code_binding(parent_sha, explicit_job)
    assert explicit_parent_code[text_corpus_path] == renderer.LEGACY_SEARCH_TEXT_CORPUS_DEFAULT_PROFILE_TRANSITION[
        'current_sha256']
    assert renderer._render_job_binding_from_job(
        explicit_job, renderer.file_sha(selected), explicit_parent_code) != renderer._render_job_binding_from_job(
            job, renderer.file_sha(selected), parent_code)
    source_digest = renderer.file_sha(selected)
    parent_job_binding = renderer._render_job_binding_from_job(job, source_digest, parent_code)
    rows_by_trajectory = {}
    for row in rows:
        rows_by_trajectory.setdefault(renderer._trajectory_key(row), []).append(row)
    renderer.render_shard(job)
    receipts_root = out / 'trajectory-stream' / 'lfm25' / 'selected' / 'receipts'
    for path in receipts_root.glob('*.json'):
        receipt = json.loads(path.read_text())
        trajectory = next(key for key in rows_by_trajectory
                          if renderer.sha(key.encode('utf-8')) == path.stem)
        source_hashes = [renderer.sha(renderer.canonical(row).encode('utf-8'))
                         for row in rows_by_trajectory[trajectory]]
        source_ids = [row['id'] for row in rows_by_trajectory[trajectory]]
        receipt['binding'] = renderer._trajectory_render_binding(
            parent_job_binding, trajectory, source_ids, source_hashes)
        receipt['renderer_code_sha256'] = parent_sha
        path.write_text(json.dumps(receipt, sort_keys=True) + '\n')

    # The real resume path validates every old output, then writes a current
    # shard marker that retains the old per-trajectory renderer identity.
    calls = []
    original = renderer._render_shard_materialized
    def traced(job_to_render):
        calls.append(job_to_render)
        return original(job_to_render)
    monkeypatch.setattr(renderer, '_render_shard_materialized', traced)
    resumed = renderer.render_shard(job)
    assert calls == []
    assert resumed['renderer_code_sha256'] == renderer.file_sha(Path(renderer.__file__).resolve())
    assert set(resumed['trajectory_renderers'].values()) == {parent_sha}
    renderer._verify_trajectory_receipts(job, resumed, {
        renderer.file_sha(Path(renderer.__file__).resolve()), parent_sha},
        expected_job_binding=renderer._render_job_binding_from_job(
            job, renderer.file_sha(selected), renderer._render_code_binding()))
    with __import__('pytest').raises(ValueError, match='exact reviewed resource-only continuation'):
        renderer._resource_parent_code_binding('f' * 64, job)


def test_legacy_search_binding_matches_historical_equation_without_null_render_inputs(tmp_path, monkeypatch):
    _reviewed_legacy_dependency(monkeypatch)
    selected = tmp_path / 'selected.jsonl'
    selected.write_text('{}\n')
    source_sha = renderer.file_sha(selected)
    job = (str(selected), str(tmp_path / 'out'), 'lfm25', {'blocked-chain': 2}, 'fixture',
           {'held-group'}, {'excluded-decision'}, False, ['program-b', 'program-a'],
           ('program-header', 'whole-trajectory'), True, {'excluded-target'},
           renderer.LEGACY_SEARCH_RENDERER_SHA256, {'tokenizers': {}, 'pieces_sha256': None})

    actual = renderer._legacy_render_job_binding(
        job, source_sha, renderer.LEGACY_SEARCH_RENDERER_SHA256)
    historical = renderer.sha(renderer.canonical({
        'source_sha256': source_sha,
        'tokenizer': 'lfm25',
        'corpus': 'fixture',
        'blocked': {'blocked-chain': 2},
        'held_groups': ['held-group'],
        'decision_exclude': ['excluded-decision'],
        'target_exclude': ['excluded-target'],
        'header_as_full': False,
        'full_programs': ['program-a', 'program-b'],
        'renderings': ['program-header', 'whole-trajectory'],
        'compress_text': True,
        'code': renderer.LEGACY_SEARCH_CODE_BINDING,
    }).encode())
    assert actual == historical

    # ``render_inputs: None`` is not the historical equation: it was added by
    # a later renderer and must not be confused with the key being absent.
    current_equation = renderer._render_job_binding(
        source_sha, 'lfm25', {'blocked-chain': 2}, 'fixture', {'held-group'},
        {'excluded-decision'}, {'excluded-target'}, False,
        ['program-b', 'program-a'], ('program-header', 'whole-trajectory'), True,
        renderer.LEGACY_SEARCH_CODE_BINDING, None)
    assert actual != current_equation


def test_legacy_search_binding_rejects_unreviewed_text_dependency(tmp_path, monkeypatch):
    actual_file_sha = renderer.file_sha
    dependency = renderer.ROOT / 'training/neuralese/natlang_neuralese/data/text_corpus.py'
    monkeypatch.setattr(renderer, 'file_sha', lambda path:
        'f' * 64 if Path(path) == dependency else actual_file_sha(path))
    job = (str(tmp_path / 'selected.jsonl'), str(tmp_path / 'out'), 'lfm25', {},
           'fixture', set(), set(), False, None, ('whole-trajectory',), False, set())
    with __import__('pytest').raises(ValueError, match='dependency differs.*text_corpus'):
        renderer._legacy_render_job_binding(job, 'a' * 64, renderer.LEGACY_SEARCH_RENDERER_SHA256)
    binding = renderer._resource_parent_code_binding(
        renderer.SEARCH_RESOURCE_ONLY_PARENT_RENDERER_SHA256, job)
    assert binding[str(renderer._LEGACY_REPO / dependency.relative_to(renderer.ROOT))] == 'f' * 64


def test_bounded_render_jobs_cancel_outstanding_and_do_not_queue_entire_corpus():
    from concurrent.futures import Future
    import pytest

    class ImmediateFailurePool:
        def __init__(self):
            self.futures = []
            self.submitted_jobs = []

        def submit(self, _function, job):
            future = Future()
            self.submitted_jobs.append(job)
            self.futures.append(future)
            if len(self.futures) == 1:
                future.set_exception(RuntimeError('first shard failed'))
            return future

    pool = ImmediateFailurePool()
    with pytest.raises(RuntimeError, match='first shard failed'):
        renderer._run_bounded_render_jobs(pool, list(range(1000)), 4, lambda *_: None)
    assert pool.submitted_jobs == [0, 1, 2, 3]
    assert all(future.cancelled() for future in pool.futures[1:])


def test_explicit_keep_native_turn_ids_are_distinct_and_preserve_source_identity():
    row = {'id': 'text:task-1', 'source_record_ids': ['task-1'], 'text': 'changed KEEP bytes'}
    default = dict(row)
    renderer.apply_explicit_history_profile_ids([default], None)
    keep = dict(row)
    renderer.apply_explicit_history_profile_ids([keep], 'keep')
    assert default == row
    assert keep['id'] == 'text:task-1:history-reasoning:keep'
    assert keep['source_row_id'] == row['id']
    assert keep['source_record_ids'] == row['source_record_ids']
    assert keep['history_reasoning_policy'] == 'keep'


def test_explicit_mellum_last_turn_only_ids_are_distinct_from_default_keep():
    row = {'id': 'text:task-1', 'source_record_ids': ['task-1'], 'text': 'changed last-turn-only bytes'}
    default_keep = dict(row)
    renderer.apply_explicit_history_profile_ids([default_keep], None)
    explicit_last_turn = dict(row)
    renderer.apply_explicit_history_profile_ids([explicit_last_turn], 'last_turn_only')
    assert default_keep == row
    assert explicit_last_turn['id'] == 'text:task-1:history-reasoning:last_turn_only'
    assert explicit_last_turn['id'] != default_keep['id']
    assert explicit_last_turn['source_row_id'] == row['id']
    assert explicit_last_turn['source_record_ids'] == row['source_record_ids']
    assert explicit_last_turn['history_reasoning_policy'] == 'last_turn_only'


def test_actual_render_input_profile_map_covers_all_configured_tokenizers():
    assert renderer._history_profiles_for_tokenizers({'lfm': 'a', 'mellum': 'b'}, {'lfm': 'keep'}) == {
        'lfm': 'keep', 'mellum': 'backbone-default'}


def test_completed_search_parent_chain_requires_explicit_exact_ancestry():
    import pytest
    for parent in renderer.SEARCH_REVIEWED_RESOURCE_PARENT_CHAIN:
        assert renderer._completed_renderer_parent(parent, 'actual-current',
            renderer.SEARCH_RESOURCE_ONLY_PARENT_RENDERER_SHA256) == parent
        with pytest.raises(ValueError, match='neither current nor explicitly continued'):
            renderer._completed_renderer_parent(parent, 'actual-current', None)
    for requested, actual in [('f'*64, renderer.SEARCH_LEGACY_EQUATION_RENDERER_SHA256),
                              (renderer.SEARCH_RESOURCE_ONLY_PARENT_RENDERER_SHA256, 'f'*64)]:
        with pytest.raises(ValueError, match='neither current nor explicitly continued'):
            renderer._completed_renderer_parent(actual, 'actual-current', requested)
    assert renderer._completed_renderer_parent('actual-current','actual-current',None) is None


def test_search_parent_chain_preserves_each_actual_text_dependency(monkeypatch):
    _reviewed_legacy_dependency(monkeypatch)
    path=str(renderer._LEGACY_REPO/'training/neuralese/natlang_neuralese/data/text_corpus.py')
    transition=renderer.LEGACY_SEARCH_TEXT_CORPUS_DEFAULT_PROFILE_TRANSITION
    for parent in (renderer.SEARCH_RESOURCE_ONLY_PARENT_RENDERER_SHA256,
                   renderer.SEARCH_LEGACY_EQUATION_RENDERER_SHA256):
        assert renderer._resource_parent_code_binding(parent)[path]==transition['historical_sha256']
    assert renderer._resource_parent_code_binding(renderer.SEARCH_PROFILE_AWARE_RENDERER_SHA256)[path]==transition['current_sha256']


def test_legacy_search_rejects_unknown_historical_text_witness(tmp_path, monkeypatch):
    _reviewed_legacy_dependency(monkeypatch)
    job = (str(tmp_path/'source'), str(tmp_path/'out'), 'lfm25', {}, 'fixture', set(), set(), False, None, ('whole-trajectory',), False, set())
    with __import__('pytest').raises(ValueError, match='witness is not approved'):
        renderer._legacy_render_job_binding(job, 'a'*64, renderer.LEGACY_SEARCH_RENDERER_SHA256, historical_text_sha256='f'*64)
