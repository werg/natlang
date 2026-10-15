"""Completed zero-edge conversions must still account for their source rows."""
import importlib.util
import json
from pathlib import Path

import pytest


def test_assemble_requires_restoration_of_archived_sources_before_creating_output(tmp_path):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('e2e_archived_source', Path(__file__).parents[1] / 'scripts/neuralese_data/e2e_mine.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    work = tmp_path / 'work'; source = work / 'large-source'; source.mkdir(parents=True)
    (source / 'archived-source.json').write_text(json.dumps({'archive': '/external/source.tar.zst'}))
    with pytest.raises(ValueError, match='Restore archived mined sources'):
        module.cmd_assemble(SimpleNamespace(work=str(work)))
    assert not (tmp_path / 'out').exists()


def test_review_filter_preserves_good_and_held_graphs_and_omits_dependent_train_context(tmp_path, monkeypatch):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('e2e_review_filter', Path(__file__).parents[1] / 'scripts/neuralese_data/e2e_mine.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'pending_unpinned_source_groups', lambda: {'musique:bad'})
    monkeypatch.setattr(module, 'registry_sha256', lambda: 'a' * 64)
    source, out = tmp_path / 'source', tmp_path / 'filtered'; source.mkdir()
    chains, edges, raw = [], [], []
    for name, split in [('bad', 'train'), ('good', 'train'), ('held', 'test')]:
        ids = [f'{name}:{i}' for i in range(2)]
        chains.append({'source': 'fixture', 'trajectory': name, 'family': 'f', 'split': split, 'records': ids, 'edges': 1})
        edges.append({'source': 'fixture', 'trajectory': name, 'split': split, 'writer_split': split,
                      'writer_record_id': ids[0], 'reader_record_id': ids[1]})
        for i, rid in enumerate(ids):
            row = {'id': rid, 'split': split, 'source_ref': {'trajectory_id': name},
                   'source_groups': ['musique:bad'] if name != 'good' and i == 0 else ['musique:seed:shared'],
                   'target': {'role': 'assistant', 'content': 'keep\u2028literal'}}
            raw.append(json.dumps(row, ensure_ascii=False).encode() + b'\r\n')
    (source / 'records.jsonl').write_bytes(b''.join(raw))
    for name, rows in [('chains', chains), ('edges', edges)]:
        (source / f'{name}.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    (source / 'pieces.jsonl').write_text('{"name":"p","text":"shared"}\n')
    (source / 'receipt.json').write_text('{"id":"source"}\n')
    module.cmd_review_filter(SimpleNamespace(source=str(source), out=str(out), id='filtered'))
    assert (out / 'records.jsonl').read_bytes() == b''.join(raw[2:])
    assert [json.loads(line)['trajectory'] for line in (out / 'edges.jsonl').open()] == ['good', 'held']
    assert {json.loads(line)['id'] for line in (out / 'omissions.jsonl').open()} == {'bad:0', 'bad:1'}
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['totals']['records'] == 4 and receipt['by_split']['test']['records'] == 2
    assert receipt['held_source_review_flags'] == [{'id': 'held:0', 'source_groups': ['musique:bad']}]
    assert (out / 'pieces.jsonl').read_bytes() == (source / 'pieces.jsonl').read_bytes()
    with pytest.raises(SystemExit, match='immutable'):
        module.cmd_review_filter(SimpleNamespace(source=str(source), out=str(out), id='filtered'))
    # A retained edge pointing outside its chain must never become a corpus.
    edges[1]['reader_record_id'] = 'not-in-the-chain'
    (source / 'edges.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in edges))
    rejected = tmp_path / 'broken-filter'
    with pytest.raises(ValueError, match='not closed'):
        module.cmd_review_filter(SimpleNamespace(source=str(source), out=str(rejected), id='broken'))
    assert not rejected.exists()


@pytest.mark.parametrize('prior_failure', [False, True])
def test_zero_edge_trajectory_has_row_level_omissions(tmp_path, prior_failure):
    path = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/e2e_mine.py'
    spec = importlib.util.spec_from_file_location('e2e_mine_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{'id': f'run:decision:{i:04d}', 'split': 'train',
             'source_ref': {'trajectory_id': 'run'},
             'messages': [{'role': 'user', 'content': 'Return the supplied text.'}],
             'target': {'role': 'assistant', 'content': 'plain text'}} for i in range(2)]
    omissions = ([{'id': rows[0]['id'], 'stage': 'convert', 'reason': 'specific_source_failure'}]
                 if prior_failure else [])

    def unused_tokenizer(*args, **kwargs):
        raise AssertionError('a zero-edge trajectory has no kept targets to tokenize')

    stem = tmp_path / 'shard-0000'
    module.derive(stem, rows, {}, {'sites': {}}, omissions, unused_tokenizer)
    recorded = [json.loads(line) for line in stem.with_suffix('.omissions.jsonl').read_text().splitlines()]
    assert {row['id'] for row in recorded} == {row['id'] for row in rows}
    assert len(recorded) == len(rows)
    if prior_failure:
        assert recorded[0]['reason'] == 'specific_source_failure'
    assert recorded[-1]['reason'] == 'trajectory_has_no_authenticated_producer_consumer_edge'
    stats = json.loads(stem.with_suffix('.stats.json').read_text())
    assert (stats['trajectories_without_edges'], stats['records_without_edges']) == (1, 2)
    assert stats['edges'] == stats['kept_records'] == 0
    assert stem.with_suffix('.records.jsonl').read_text() == ''


@pytest.mark.parametrize('keep_valid', [False, True])
def test_unreadable_converter_input_is_accounted_and_empty_conversion_fails(monkeypatch, tmp_path, keep_valid):
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/e2e_mine.py'
    spec = importlib.util.spec_from_file_location('e2e_mine_unreadable_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    valid = {'id': 'valid', 'source_ref': {'trajectory_id': 'run'},
             'messages': [{'role': 'user', 'content': 'Return text.'}],
             'target': {'role': 'assistant', 'content': 'plain text'}, 'split': 'train'}
    shard = tmp_path / 'shard-0001.jsonl'
    # Python accepts this ancillary NaN literal; the Node JSON converter rejects it.
    shard.write_text((json.dumps(valid) + '\n' if keep_valid else '') + '{"id":"bad","metric":NaN}\n')

    def converter(command, **kwargs):
        Path(command[command.index('--out') + 1]).write_text(json.dumps(valid) + '\n' if keep_valid else '')
        Path(command[command.index('--pieces') + 1]).write_text('')
        Path(command[command.index('--summary') + 1]).write_text(json.dumps({'records': 1 + keep_valid, 'unreadable': 1}))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(module.subprocess, 'run', converter)
    module.convert_shard(shard, lambda *args, **kwargs: pytest.fail('no kept targets to tokenize'))
    stem = shard.with_suffix('')
    omitted = [json.loads(line) for line in stem.with_suffix('.omissions.jsonl').read_text().splitlines()]
    assert {r['id'] for r in omitted} == ({'bad', 'valid'} if keep_valid else {'bad'})
    assert next(r for r in omitted if r['id'] == 'bad')['reason'] == 'conversion_unreadable_source_json'
    stats = json.loads(stem.with_suffix('.stats.json').read_text())
    assert stats['input_records'] == 1 + keep_valid and stats['unreadable_input_records'] == 1
    assert stats['converted'] is keep_valid
    if keep_valid:
        assert stats['rows'] == 1 and stats['records_without_edges'] == 1
    else:
        assert stats['error'] == 'conversion_produced_no_records_from_nonempty_source'


def test_held_union_cache_tracks_compressed_relocated_views_and_exact_bytes(tmp_path):
    import subprocess
    path = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/e2e_mine.py'
    spec = importlib.util.spec_from_file_location('e2e_held_cache_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    work = tmp_path / 'work'; work.mkdir()
    (work / 'held-groups.json').write_text('["obsolete"]')
    union = tmp_path / 'union'
    paths = []
    for mode in ('full', 'program-header', 'whole-trajectory'):
        directory = union / mode / 'lfm25'; directory.mkdir(parents=True)
        physical = tmp_path / (mode + '.zst')
        payload = json.dumps({'split': 'test', 'source_groups': [mode]}).encode() + b'\n'
        physical.write_bytes(subprocess.run(['/usr/bin/zstd', '-q', '-c'], input=payload,
                                           capture_output=True, check=True).stdout)
        logical = directory / 'text-data.jsonl.zst'; logical.symlink_to(physical); paths.append(physical)
    expected = {'full', 'program-header', 'whole-trajectory'}
    groups, bindings = module.held_union_groups(work, union)
    assert groups == expected and len(bindings) == 3
    assert module.held_union_groups(work, union) == (groups, bindings)
    payload = json.dumps({'split': 'test', 'source_groups': ['replacement']}).encode() + b'\n'
    paths[0].write_bytes(subprocess.run(['/usr/bin/zstd', '-q', '-c'], input=payload,
                                     capture_output=True, check=True).stdout)
    refreshed, new_bindings = module.held_union_groups(work, union)
    assert refreshed == {'replacement', 'program-header', 'whole-trajectory'}
    assert new_bindings != bindings
    with pytest.raises(ValueError, match='no full text view'):
        module.held_union_groups(work, tmp_path / 'missing-union')


@pytest.mark.parametrize('hold_kind', ['quality', 'decision'])
def test_assemble_quarantine_removes_whole_train_chain_and_preserves_held(tmp_path, monkeypatch, hold_kind):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('e2e_assemble_quality_test', Path(__file__).parents[1] / 'scripts/neuralese_data/e2e_mine.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'pending_unpinned_source_groups', lambda: set())
    monkeypatch.setattr(module, 'registry_sha256', lambda: 'a' * 64)
    monkeypatch.setattr(module, 'held_union_groups', lambda *args: (set(), [{'sha256': 'b' * 64}]))
    work = tmp_path / 'work'; source = work / 'fixture'; source.mkdir(parents=True)
    records, edges = [], []
    for name, split in [('bad', 'train'), ('good', 'train'), ('held', 'test')]:
        ids = [name + ':0', name + ':1']
        for i, rid in enumerate(ids):
            records.append({'id': rid, 'split': split, 'source_ref': {'trajectory_id': name},
                            'source_groups': ['quality:bad'] if name != 'good' and i == 0 else [name],
                            'target': {'role': 'assistant', 'content': rid}, 'family': 'fixture'})
        edges.append({'trajectory': name, 'split': split, 'writer_split': split,
                      'writer_record_id': ids[0], 'reader_record_id': ids[1],
                      'write_name': name, 'body_sha256': 'c' * 64, 'kind': 'result'})
    stem = source / 'shard-0000'
    for suffix, rows in [('records', records), ('edges', edges), ('pieces', []), ('omissions', [])]:
        Path(str(stem) + '.' + suffix + '.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    Path(str(stem) + '.stats.json').write_text('{}')
    (source / 'sharded.json').write_text('{}')
    witness = tmp_path / 'quality.jsonl'; witness.write_text('{"source_groups":["quality:bad"]}\n')
    decision_witness = tmp_path / 'decisions.jsonl'
    decision_witness.write_text(''.join(json.dumps({'id': rid, 'reason': 'reviewed non-success'}) + '\n'
                                        for rid in ['bad:0', 'held:0']))
    module.cmd_assemble(SimpleNamespace(work=str(work), out_dir=str(tmp_path), id='assembled',
                                       held_union='bound-union',
                                       quarantine_reference=[witness] if hold_kind == 'quality' else [],
                                       decision_hold_ids=[decision_witness] if hold_kind == 'decision' else []))
    out = tmp_path / 'assembled'
    assert {json.loads(l)['id'] for l in (out / 'records.jsonl').open()} == {'good:0', 'good:1', 'held:0', 'held:1'}
    omissions = [json.loads(l) for l in (out / 'omissions.jsonl').open()]
    assert {r['id'] for r in omissions} == {'bad:0', 'bad:1'}
    reason = 'source_quality_quarantine_chain' if hold_kind == 'quality' else 'native_decision_hold_chain'
    assert all(r['reason'] == reason for r in omissions)
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['totals']['edges'] == 2
    if hold_kind == 'quality':
        assert receipt['source_quality_quarantine_bindings'][0]['sha256'] == module.sha256_file_hex(witness)
    else:
        assert all(r['decision_ids'] == ['bad:0'] for r in omissions)
        assert receipt['native_decision_hold_bindings'][0]['sha256'] == module.sha256_file_hex(decision_witness)
        assert receipt['held_native_decision_flags'] == [{'id': 'held:0', 'source': 'fixture', 'split': 'test'}]


def test_native_decision_hold_files_reject_unreviewed_ids(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
    from neuralese_data.decision_holds import load_decision_holds
    witness = tmp_path / 'bad.jsonl'; witness.write_text('{"id":"unreviewed"}\n')
    with pytest.raises(ValueError, match='invalid native decision hold'):
        load_decision_holds([witness])
