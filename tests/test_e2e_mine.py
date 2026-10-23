"""Completed zero-edge conversions retain ordinary source trajectories."""
import importlib.util
import hashlib
import json
from pathlib import Path

import pytest


def native_row(row):
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    row.setdefault('source_groups', ['fixture-source'])
    row.setdefault('provenance', {})
    row.setdefault('training_admission', {'approved': True})
    source_ref = row.setdefault('source_ref', {})
    source_ref.setdefault('trajectory_id', 'run')
    source_ref.setdefault('invocation_id', row['id'])
    source_record = dict(row)
    row['native_source_conversion'] = {
        'schema': 'natlang.native-source-conversion/1', 'row_id': row['id'],
        'source_record_sha256': hashlib.sha256(canonical(source_record).encode()).hexdigest(),
        'source_target_sha256': hashlib.sha256(canonical(row.get('target')).encode()).hexdigest(),
        'source_target': row.get('target'),
        'source_target_present': 'target' in row,
        'source_neuralese_conversion_present': 'neuralese_conversion' in row,
        'source_neuralese_conversion': row.get('neuralese_conversion'),
        'native_target_sha256': hashlib.sha256(canonical(row.get('target')).encode()).hexdigest(),
        'source_messages': row.get('messages', []),
        'source_messages_sha256': hashlib.sha256(canonical(row.get('messages', [])).encode()).hexdigest(),
        'trajectory_id': source_ref['trajectory_id'], 'invocation_id': source_ref['invocation_id'],
        **({'source_trajectory_row_sha256': source_ref['source_row_sha256']}
           if source_ref.get('source_row_sha256') else {}),
        'source_ref': dict(source_ref), 'source_provenance': row['provenance'],
        'training_admission': row['training_admission'], 'split': row.get('split'),
        'source_groups': row['source_groups'], 'occurrences': [], 'unresolved_references': []}
    return row


def test_native_occurrence_reader_binds_exact_paths_and_builds_per_occurrence_edges():
    import sys
    sys.path.insert(0, str(Path(__file__).parents[1] / 'training/neuralese'))
    from natlang_neuralese.data.trajectory_text import authenticated_recurrence_context_view
    body = 'one exact retained value'
    body_sha = hashlib.sha256(body.encode()).hexdigest()
    event = 'teacher-event:42'
    writer = native_row({'id': 'writer', 'split': 'train',
        'source_ref': {'trajectory_id': 'run', 'invocation_id': 'writer', 'source_row_sha256': 'c' * 64},
        'messages': [], 'target': {'role': 'assistant', 'content': body}})
    writer['native_source_conversion']['occurrences'] = [{
        'schema': 'natlang.native-source-occurrence/1', 'id': 'nso_' + '1' * 64,
        'role': 'producer', 'reference': event, 'path': ['target', 'content'], 'body': body,
        'body_sha256': body_sha, 'source_kind': 'typed-result', 'trajectory_id': 'run',
        'invocation_id': 'writer'}]
    reader = native_row({'id': 'reader', 'split': 'train',
        'source_ref': {'trajectory_id': 'run', 'invocation_id': 'reader', 'source_row_sha256': 'c' * 64},
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': body}]}],
        'target': {'role': 'assistant', 'content': 'used'}})
    reader['native_source_conversion']['occurrences'] = [{
        'schema': 'natlang.native-source-occurrence/1', 'id': 'nso_' + '2' * 64,
        'role': 'consumer', 'reference': event, 'path': ['messages', '0', 'content', '0'],
        'body': body, 'body_sha256': body_sha, 'source_kind': 'authenticated-static-read',
        'trajectory_id': 'run', 'invocation_id': 'reader'}]
    _, review = authenticated_recurrence_context_view([writer, reader], {})
    edge = review['decisions'][1]['references'][0]
    assert edge['gradient_edge'] is True
    assert edge['writer_occurrence_id'] == 'nso_' + '1' * 64
    assert edge['reader_occurrence_id'] == 'nso_' + '2' * 64
    changed = json.loads(json.dumps(reader))
    changed['messages'][0]['content'][0]['text'] = 'altered'
    with pytest.raises(ValueError, match='source path/body mismatch'):
        authenticated_recurrence_context_view([writer, changed], {})
    legacy = json.loads(json.dumps(reader))
    legacy.pop('native_source_conversion')
    with pytest.raises(ValueError, match='lacks the current native occurrence conversion'):
        authenticated_recurrence_context_view([legacy], {})
    unbound = json.loads(json.dumps(reader))
    unbound['native_source_conversion']['unresolved_references'] = [{
        'id': 'unavailable', 'reason': 'source bytes not present'}]
    with pytest.raises(ValueError, match='unresolved exact source references'):
        authenticated_recurrence_context_view([unbound], {})


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
    source_rows = [native_row({'id': f'run:decision:{i:04d}', 'split': 'train',
             'source_ref': {'trajectory_id': 'run'},
             'messages': [{'role': 'user', 'content': 'Return the supplied text.'}],
             'target': {'role': 'assistant', 'content': 'plain text'}}) for i in range(2)]
    omissions = ([{'id': source_rows[0]['id'], 'stage': 'convert', 'reason': 'specific_source_failure'}]
                 if prior_failure else [])
    rows = source_rows[1:] if prior_failure else source_rows

    def tokenizer(text, **kwargs):
        return {'input_ids': list(text)}

    stem = tmp_path / 'shard-0000'
    module.derive(stem, rows, {}, {'sites': {}}, omissions, tokenizer)
    recorded = [json.loads(line) for line in stem.with_suffix('.omissions.jsonl').read_text().splitlines()]
    assert {row['id'] for row in recorded} == ({source_rows[0]['id']} if prior_failure else set())
    if prior_failure:
        assert recorded[0]['reason'] == 'specific_source_failure'
    stats = json.loads(stem.with_suffix('.stats.json').read_text())
    assert (stats['trajectories_without_edges'], stats['records_without_edges']) == (1, len(rows))
    assert stats['edges'] == 0 and stats['kept_records'] == len(rows)
    assert {json.loads(line)['id'] for line in stem.with_suffix('.records.jsonl').read_text().splitlines()} == {
        row['id'] for row in rows}


@pytest.mark.parametrize('keep_valid', [False, True])
def test_unreadable_converter_input_is_accounted_and_empty_conversion_fails(monkeypatch, tmp_path, keep_valid):
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/e2e_mine.py'
    spec = importlib.util.spec_from_file_location('e2e_mine_unreadable_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    valid = native_row({'id': 'valid', 'source_ref': {'trajectory_id': 'run'},
             'messages': [{'role': 'user', 'content': 'Return text.'}],
             'target': {'role': 'assistant', 'content': 'plain text'}, 'split': 'train'})
    shard = tmp_path / 'shard-0001.jsonl'
    # Python accepts this ancillary NaN literal; the Node JSON converter rejects it.
    shard.write_text((json.dumps(valid) + '\n' if keep_valid else '') + '{"id":"bad","metric":NaN}\n')

    def converter(command, **kwargs):
        Path(command[command.index('--out') + 1]).write_text(json.dumps(valid) + '\n' if keep_valid else '')
        Path(command[command.index('--pieces') + 1]).write_text('')
        Path(command[command.index('--summary') + 1]).write_text(json.dumps({'records': 1 + keep_valid, 'unreadable': 1}))
        return SimpleNamespace(returncode=0, stdout='', stderr='')

    monkeypatch.setattr(module.subprocess, 'run', converter)
    module.convert_shard(shard, lambda text, **kwargs: {'input_ids': list(text)})
    stem = shard.with_suffix('')
    omitted = [json.loads(line) for line in stem.with_suffix('.omissions.jsonl').read_text().splitlines()]
    assert {r['id'] for r in omitted} == {'bad'}
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
        physical.write_bytes(subprocess.run(['zstd', '-q', '-c'], input=payload,
                                           capture_output=True, check=True).stdout)
        logical = directory / 'text-data.jsonl.zst'; logical.symlink_to(physical); paths.append(physical)
    expected = {'full', 'program-header', 'whole-trajectory'}
    groups, bindings = module.held_union_groups(work, union)
    assert groups == expected and len(bindings) == 3
    assert module.held_union_groups(work, union) == (groups, bindings)
    payload = json.dumps({'split': 'test', 'source_groups': ['replacement']}).encode() + b'\n'
    paths[0].write_bytes(subprocess.run(['zstd', '-q', '-c'], input=payload,
                                     capture_output=True, check=True).stdout)
    refreshed, new_bindings = module.held_union_groups(work, union)
    assert refreshed == {'replacement', 'program-header', 'whole-trajectory'}
    assert new_bindings != bindings
    with pytest.raises(ValueError, match='no full text view'):
        module.held_union_groups(work, tmp_path / 'missing-union')


def test_extract_chains_preserves_exact_complete_source_rows_and_proves_membership(tmp_path):
    import hashlib
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/e2e_mine.py'
    spec = importlib.util.spec_from_file_location('e2e_extract_chains_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source, out = tmp_path / 'source', tmp_path / 'slice'; source.mkdir()
    chains = [
        {'source': 'fixture-a', 'trajectory': 'train-a', 'family': 'fixture', 'split': 'train',
         'records': ['a:0', 'a:1', 'a:2'], 'edges': 2},
        {'source': 'fixture-b', 'trajectory': 'held-b', 'family': 'fixture', 'split': 'test',
         'records': ['b:0', 'b:1'], 'edges': 1},
        {'source': 'fixture-c', 'trajectory': 'unused', 'family': 'fixture', 'split': 'train',
         'records': ['c:0', 'c:1'], 'edges': 1},
    ]
    records = []
    for chain in chains:
        for index, rid in enumerate(chain['records']):
            row = {'id': rid, 'split': chain['split'], 'source_groups': [chain['trajectory']],
                   'source_ref': {'trajectory_id': chain['trajectory']},
                   'messages': [{'role': 'user', 'content': 'keep\u2028exact'}],
                   'target': {'role': 'assistant', 'content': f'answer {index}'} }
            records.append(json.dumps(row, ensure_ascii=False).encode() + b'\r\n')
    edge_rows = []
    for chain in chains:
        ids = chain['records']
        for left, right in zip(ids, ids[1:]):
            edge_rows.append(json.dumps({'id': f'{left}->{right}', 'source': chain['source'],
                'trajectory': chain['trajectory'], 'split': chain['split'], 'writer_split': chain['split'],
                'writer_record_id': left, 'reader_record_id': right}).encode() + b'\n')
    (source / 'records.jsonl').write_bytes(b''.join(records))
    (source / 'edges.jsonl').write_bytes(b''.join(edge_rows))
    chain_bytes = [json.dumps(c).encode() + b'\n' for c in chains]
    (source / 'chains.jsonl').write_bytes(b''.join(chain_bytes))
    (source / 'pieces.jsonl').write_bytes(b'{"name":"p","text":"shared"}\r\n')
    (source / 'omissions.jsonl').write_bytes(
        b'{"source":"fixture-a","trajectory":"train-a","reason":"retained provenance"}\n'
        b'{"source":"fixture-c","trajectory":"unused","reason":"outside selection"}\n')
    required = ('records.jsonl', 'edges.jsonl', 'chains.jsonl', 'pieces.jsonl')
    receipt = {'id': 'parent-fixture', 'files': {name: {'sha256': hashlib.sha256((source/name).read_bytes()).hexdigest(),
                 'bytes': (source/name).stat().st_size} for name in required}}
    (source / 'receipt.json').write_text(json.dumps(receipt) + '\n')
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'schema': 'natlang.e2e-chain-selection/1', 'chains': [
        {'source': 'fixture-a', 'trajectory': 'train-a'},
        {'source': 'fixture-b', 'trajectory': 'held-b'}]}) + '\n')
    module.cmd_extract_chains(SimpleNamespace(source=str(source), out=str(out), selection=str(selection), id='slice'))
    expected_records = b''.join(records[:5])
    expected_edges = b''.join(edge_rows[:3])
    assert (out / 'records.jsonl').read_bytes() == expected_records
    assert (out / 'edges.jsonl').read_bytes() == expected_edges
    assert (out / 'chains.jsonl').read_bytes() == b''.join(chain_bytes[:2])
    assert (out / 'pieces.jsonl').read_bytes() == (source / 'pieces.jsonl').read_bytes()
    assert (out / 'omissions.jsonl').read_bytes().startswith(b'{"source":"fixture-a"')
    extracted = json.loads((out / 'extract-receipt.json').read_text())
    assert extracted['parent']['id'] == 'parent-fixture'
    assert extracted['totals'] == {'chains': 2, 'records': 5, 'edges': 3,
        'by_split': {'train': {'chains': 1, 'records': 3, 'edges': 2, 'source_group_count': 1},
                     'test': {'chains': 1, 'records': 2, 'edges': 1, 'source_group_count': 1}}}
    assert len(extracted['membership']['records']) == 5
    assert len(extracted['membership']['edges']) == 3
    with pytest.raises(SystemExit, match='immutable'):
        module.cmd_extract_chains(SimpleNamespace(source=str(source), out=str(out), selection=str(selection), id='slice'))


def test_extract_chains_rejects_incomplete_chain_membership_and_split_leaks(tmp_path):
    import hashlib
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/e2e_mine.py'
    spec = importlib.util.spec_from_file_location('e2e_extract_chains_bad_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'source'; source.mkdir()
    (source / 'chains.jsonl').write_text(json.dumps({'source':'s','trajectory':'t','split':'train','records':['r0','r1'],'edges':1})+'\n')
    (source / 'records.jsonl').write_text(json.dumps({'id':'r0','split':'train','source_groups':['g'],'source_ref':{'trajectory_id':'t'}})+'\n')
    (source / 'edges.jsonl').write_text('')
    (source / 'pieces.jsonl').write_text('')
    required=('records.jsonl','edges.jsonl','chains.jsonl','pieces.jsonl')
    (source / 'receipt.json').write_text(json.dumps({'id':'parent','files':{
        name:{'sha256':hashlib.sha256((source/name).read_bytes()).hexdigest(),'bytes':(source/name).stat().st_size}
        for name in required}}))
    selection=tmp_path/'selection.json'; selection.write_text(json.dumps({'schema':'natlang.e2e-chain-selection/1','chains':[{'source':'s','trajectory':'t'}]}))
    with pytest.raises(ValueError,match='records absent'):
        module.cmd_extract_chains(SimpleNamespace(source=str(source),out=str(tmp_path/'bad'),selection=str(selection),id='bad'))
    assert not (tmp_path/'bad').exists()


def test_extract_chains_verifies_parent_hash_even_when_file_size_is_unchanged(tmp_path):
    import hashlib
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/e2e_mine.py'
    spec = importlib.util.spec_from_file_location('e2e_extract_chains_hash_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = tmp_path / 'source'; source.mkdir()
    payloads = {
        'records.jsonl': b'{"id":"r","split":"train","source_groups":["g"],"source_ref":{"trajectory_id":"t"}}\n',
        'edges.jsonl': b'',
        'chains.jsonl': b'{"source":"s","trajectory":"t","split":"train","records":["r"],"edges":0}\n',
        'pieces.jsonl': b'',
    }
    for name, payload in payloads.items():
        (source / name).write_bytes(payload)
    required = tuple(payloads)
    (source / 'receipt.json').write_text(json.dumps({'id':'parent','files':{
        name:{'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
        for name,payload in payloads.items()}}))
    selection = tmp_path / 'selection.json'
    selection.write_text(json.dumps({'schema':'natlang.e2e-chain-selection/1','chains':[{'source':'s','trajectory':'t'}]}))
    changed = payloads['records.jsonl'].replace(b'"g"', b'"x"')
    assert len(changed) == len(payloads['records.jsonl'])
    (source / 'records.jsonl').write_bytes(changed)
    with pytest.raises(ValueError, match='records.jsonl differs'):
        module.cmd_extract_chains(SimpleNamespace(source=str(source),out=str(tmp_path/'bad'),selection=str(selection),id='bad'))
    assert not (tmp_path/'bad').exists()


@pytest.mark.parametrize('hold_kind', ['quality', 'decision', 'held-group', 'authentication'])
def test_assemble_quarantine_removes_whole_train_chain_and_preserves_held(tmp_path, monkeypatch, hold_kind):
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('e2e_assemble_quality_test', Path(__file__).parents[1] / 'scripts/neuralese_data/e2e_mine.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'pending_unpinned_source_groups', lambda: set())
    monkeypatch.setattr(module, 'registry_sha256', lambda: 'a' * 64)
    monkeypatch.setattr(module, 'held_union_groups', lambda *args:
                        ({'quality:bad'} if hold_kind == 'held-group' else set(), [{'sha256': 'b' * 64}]))
    work = tmp_path / 'work'; source = work / 'fixture'; source.mkdir(parents=True)
    records, edges = [], []
    for name, split in [('bad', 'train'), ('good', 'train'), ('held', 'test')]:
        ids = [name + ':0', name + ':1', name + ':2']
        for i, rid in enumerate(ids):
            records.append({'id': rid, 'split': split, 'source_ref': {'trajectory_id': name},
                            # Only a non-endpoint context record carries the held source.
                            'source_groups': ['quality:bad'] if name != 'good' and i == 2 else [name],
                            'target': {'role': 'assistant', 'content': rid}, 'family': 'fixture'})
        edges.append({'trajectory': name, 'split': split, 'writer_split': split,
                      'writer_record_id': ids[0], 'reader_record_id': ids[1],
                      'write_name': name, 'body_sha256': 'c' * 64, 'kind': 'result'})
    records.append({'id': 'zero:0', 'split': 'train', 'source_ref': {'trajectory_id': 'zero'},
                    'source_groups': ['zero'], 'target': {'role': 'assistant', 'content': 'ordinary native SFT'},
                    'family': 'fixture'})
    stem = source / 'shard-0000'
    for suffix, rows in [('records', records), ('edges', edges), ('pieces', []), ('omissions', ([{'id': 'bad:2', 'stage': 'authenticate', 'reason': 'missing producer receipt'}]
                                                           if hold_kind == 'authentication' else []))]:
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
    assert {json.loads(l)['id'] for l in (out / 'records.jsonl').open()} == {
        'good:0', 'good:1', 'good:2', 'held:0', 'held:1', 'held:2', 'zero:0'}
    zero_chain = next(json.loads(l) for l in (out / 'chains.jsonl').open()
                      if json.loads(l)['trajectory'] == 'zero')
    assert zero_chain['records'] == ['zero:0'] and zero_chain['edges'] == 0
    omissions = [json.loads(l) for l in (out / 'omissions.jsonl').open()]
    assert {r['id'] for r in omissions} == {'bad:0', 'bad:1', 'bad:2'}
    reason = {'quality': 'source_quality_quarantine_chain', 'decision': 'native_decision_hold_chain',
              'held-group': 'train_trajectory_shares_held_source_group',
              'authentication': 'unresolved_authentication_chain'}[hold_kind]
    assert all(r['reason'] == reason for r in omissions if r['stage'] == 'assemble')
    if hold_kind == 'authentication':
        assert len(omissions) == 4
        assert all(r['authentication_failure_ids'] == ['bad:2'] for r in omissions if r['stage'] == 'assemble')
    receipt = json.loads((out / 'receipt.json').read_text())
    assert receipt['totals']['edges'] == 2
    if hold_kind == 'quality':
        assert receipt['source_quality_quarantine_bindings'][0]['sha256'] == module.sha256_file_hex(witness)
    elif hold_kind == 'decision':
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


def test_neuralese_reference_metadata_retains_two_edges_between_same_records(monkeypatch):
    spec = importlib.util.spec_from_file_location('e2e_edge_identity', Path(__file__).parents[1] / 'scripts/neuralese_data/e2e_mine.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    rows = [native_row({'id': rid, 'split': 'train', 'source_ref': {'trajectory_id': 'chain'}})
            for rid in ['writer', 'reader']]
    refs = [{'kind': 'native-positional-state', 'writer_record_id': 'writer', 'name': 'result:a',
             'body_sha256': 'a' * 64, 'writer_occurrence_id': 'nso_' + '1' * 64,
             'reader_occurrence_id': 'nso_' + '2' * 64, 'gradient_edge': True},
            {'kind': 'native-positional-state', 'writer_record_id': 'writer', 'name': 'result:b',
             'body_sha256': 'b' * 64, 'writer_occurrence_id': 'nso_' + '3' * 64,
             'reader_occurrence_id': 'nso_' + '4' * 64, 'gradient_edge': True},
            {'kind': 'native-positional-state', 'writer_record_id': 'writer', 'name': 'result:a',
             'body_sha256': 'a' * 64, 'writer_occurrence_id': 'nso_' + '1' * 64,
             'reader_occurrence_id': 'nso_' + '5' * 64, 'gradient_edge': True}]
    review = {'decisions': [{'id': 'reader', 'references': refs}], 'excluded': []}
    actual = module.edges_from_authenticated_decisions(rows, review['decisions'])
    assert [(edge['write_name'], edge['body_sha256']) for edge in actual] == [
        ('result:a', 'a' * 64), ('result:b', 'b' * 64), ('result:a', 'a' * 64)]
    from natlang_neuralese.data import trajectory_text as trajectories
    calls = []
    def authenticate(records, pieces):
        calls.append((records, pieces))
        return records, review
    monkeypatch.setattr(trajectories, 'authenticated_recurrence_context_view', authenticate)
    legacy = [{**actual[0], 'write_name': None, 'body_sha256': None}]
    repaired, receipt = module.repair_legacy_edge_metadata(rows, {}, legacy)
    assert repaired == actual and len(calls) == 1
    assert receipt['trajectories'] == 1 and receipt['prior_edges'] == 1 and receipt['reconstructed_edges'] == 3
    # Complete sidecars do not need a second authentication pass.
    assert module.repair_legacy_edge_metadata(rows, {}, actual) == (actual, None)
    assert len(calls) == 1
    # A concrete historical identity must not be silently replaced by a different body.
    with pytest.raises(ValueError, match='contradicts prior sidecar'):
        module.repair_legacy_edge_metadata(rows, {}, legacy + [{**actual[0], 'body_sha256': 'c' * 64}])
    with pytest.raises(ValueError, match='lacks exact producer/import occurrence'):
        module.edges_from_authenticated_decisions(rows, [{'id': 'reader', 'references': [
            {**refs[0], 'writer_record_id': 'missing'}]}])
    review['excluded'] = [{'id': 'reader', 'reason': 'body authentication failed'}]
    with pytest.raises(ValueError, match='failed native authentication'):
        module.repair_legacy_edge_metadata(rows, {}, legacy)


def test_derive_does_not_keep_a_known_unresolved_writer_because_it_has_a_valid_outgoing_edge(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('e2e_unresolved_writer', Path(__file__).parents[1] / 'scripts/neuralese_data/e2e_mine.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from natlang_neuralese.data import trajectory_text as trajectories
    rows = [native_row({'id': rid, 'split': 'train', 'source_ref': {'trajectory_id': 'chain'},
             'target': {'role': 'assistant', 'content': rid}}) for rid in ['writer', 'reader']]
    review = {'excluded': [{'id': 'writer', 'reason': 'missing producer receipt'}],
              'decisions': [{'id': 'reader', 'references': [{'kind': 'native-positional-state',
                 'writer_record_id': 'writer', 'name': 'result:body', 'body_sha256': 'a' * 64,
                 'writer_occurrence_id': 'nso_' + '1' * 64, 'reader_occurrence_id': 'nso_' + '2' * 64,
                 'gradient_edge': True}]}]}
    monkeypatch.setattr(trajectories, 'authenticated_recurrence_context_view', lambda *args: ([], review))
    stem = tmp_path / 'shard-0000'
    module.derive(stem, rows, {}, {}, [], lambda *args, **kwargs: pytest.fail('unresolved chain is not trained'))
    assert Path(str(stem) + '.records.jsonl').read_text() == ''
    assert Path(str(stem) + '.edges.jsonl').read_text() == ''
    omissions = [json.loads(l) for l in Path(str(stem) + '.omissions.jsonl').open()]
    assert {r['id'] for r in omissions} == {'writer', 'reader'}
    assert next(r for r in omissions if r['id'] == 'reader')['reason'] == 'trajectory_requires_authentication_repair'
