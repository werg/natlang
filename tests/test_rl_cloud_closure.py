"""Cloud SFT launches ship a complete input closure by construction (rl_cloud.py seed/launch, rl_cloud_sandbox.py
preflight): fail closed on the DGX, observed reads by default, relaunch = seed closure + export, row-index entries,
sandbox preflight before any controller call."""
import hashlib
import json
import pickle
from pathlib import Path

import pytest

from scripts.coreweave import rl_cloud_autopilot as autopilot
from scripts.coreweave import rl_cloud_sandbox as sandbox

rl_cloud = autopilot.rl_cloud


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_preflight_lists_every_missing_or_changed_path_once(tmp_path):
    L = tmp_path / 'root' / 'lin'
    L.mkdir(parents=True)
    good = tmp_path / 'data' / 'good.json'
    good.parent.mkdir()
    good.write_text('{"a": 1}')
    changed = tmp_path / 'data' / 'changed.json'
    changed.write_text('{"b": 2}')
    big = tmp_path / 'data' / 'big.jsonl'
    big.write_bytes(b'x' * 64)
    entries = [
        {'path': str(good), 'bytes': good.stat().st_size, 'sha256': sha(good), 'kind': 'file', 'check': 'sha256'},
        {'path': str(changed), 'bytes': changed.stat().st_size, 'sha256': '0' * 64, 'kind': 'file', 'check': 'sha256'},
        {'path': str(big), 'bytes': 65, 'sha256': sha(big), 'kind': 'file', 'check': 'sha256'},
        {'path': str(tmp_path / 'data/absent-1.json'), 'bytes': 1, 'sha256': '1' * 64, 'kind': 'file', 'check': 'sha256'},
        {'path': str(tmp_path / 'data/absent-2.pt'), 'bytes': 9, 'sha256': '2' * 64, 'kind': 'checkpoint', 'check': 'sha256'},
        {'path': str(tmp_path / 'data/link.pt'), 'kind': 'link', 'check': 'exists'},
        {'path': str(L / 'w000000/config.json'), 'bytes': 3, 'sha256': '3' * 64, 'kind': 'file', 'check': 'exists'}]
    assert sandbox.preflight(L) == 0  # no manifest: an older DGX launch
    manifest = sandbox.manifest_path(L)
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'schema': sandbox.MANIFEST_SCHEMA, 'entries': entries}))
    report = tmp_path / 'report.json'
    assert sandbox.preflight(L, report) == 1
    problems = {Path(p['path']).name: p['problem'] for p in json.loads(report.read_text())['problems']}
    assert problems == {'changed.json': 'sha256', 'big.jsonl': 'size', 'absent-1.json': 'missing',
                        'absent-2.pt': 'missing', 'link.pt': 'missing', 'config.json': 'missing'}
    (L / 'w000000').mkdir()
    (L / 'w000000/config.json').write_text('{"other": "bytes"}')  # lineage files: existence only
    (tmp_path / 'data/link.pt').symlink_to(tmp_path / 'data/nowhere.pt')
    manifest.write_text(json.dumps({'schema': sandbox.MANIFEST_SCHEMA, 'entries': [entries[0], entries[5], entries[6]]}))
    assert sandbox.preflight(L, report) == 0 and json.loads(report.read_text())['ok']


def test_fetch_keeps_the_launch_manifest_and_drops_a_stale_one(tmp_path):
    L = tmp_path / 'root' / 'lin'
    window = L / 'w000000'
    window.mkdir(parents=True)
    (window / 'config.json').write_text(json.dumps({'out': str(window), 'checkpoint': str(L / 'parent/checkpoint.pt'),
                                                    'mandatory_sft_only': True}))
    (L / 'lineage.json').write_text(json.dumps({'schema': sandbox.LINEAGE_SCHEMA, 'windows': [
        {'index': 0, 'out': str(window), 'config': str(window / 'config.json')}]}))
    entry = {'path': '/x', 'kind': 'link', 'check': 'exists'}
    capability = {'schema': 'natlang.rl-cloud-parent/1', 'lineage_dir': str(L), 'seq': 3, 'blobs': [], 'inputs': [],
                  'links': [], 'manifest': [entry]}
    parent = tmp_path / 'rl-parent'
    parent.write_text(json.dumps(capability))
    assert sandbox.fetch(parent, L) == 0
    assert json.loads(sandbox.manifest_path(L).read_text())['entries'] == [entry]
    capability.pop('manifest')
    parent.write_text(json.dumps(capability))
    sandbox.fetch(parent, L)
    assert not sandbox.manifest_path(L).exists()


def seed_fixture(monkeypatch, tmp_path, closure, *, skipped=(), spec=None, trace=True):
    """A seed_closure call with every DGX side effect stubbed; returns (result or exception, calls)."""
    calls = {'trace': 0, 'closure_args': []}
    monkeypatch.setattr(rl_cloud, 'hub', lambda run: tmp_path / 'hub')
    monkeypatch.setattr(rl_cloud, 'note', lambda *a: None)

    def tool(runtime, *args):
        calls['closure_args'].append(args)
        rl_cloud.store_json(args[args.index('-o') + 1], closure)
    monkeypatch.setattr(rl_cloud, 'runtime_tool', tool)

    def trace_start(run, name, spec, L, config, parent, predicted):
        calls['trace'] += 1
        path = tmp_path / 'start.strace'
        path.write_text('')
        return path, {'status': 'traced'}
    monkeypatch.setattr(rl_cloud, 'trace_seed_start', trace_start)
    monkeypatch.setattr(rl_cloud, 'origin_referenced_files', lambda *a: [])
    monkeypatch.setattr(rl_cloud, 'row_index_files', lambda run, files: [])
    monkeypatch.setattr(rl_cloud, 'lineage_inputs', lambda run, name, closure: [{'kind': 'file', **f} for f in closure['files']])
    monkeypatch.setattr(rl_cloud, 'put_checkpoint', lambda run, src, dst: {'kind': 'checkpoint', 'encoding': 'raw',
                                                                            'sha256': 'p', 'bytes': 1, 'path': str(dst)})
    monkeypatch.setattr(rl_cloud, 'historical_checkpoint_blobs', lambda run, closure, parent: ([], list(skipped), []))
    monkeypatch.setattr(rl_cloud, '_tar_zst_put', lambda *a: ('t', 1))
    L = tmp_path / 'root' / 'lin'
    window = L / 'w000000'
    window.mkdir(parents=True)
    (window / 'config.json').write_text('{}')
    boundary = {'runtime': '/rt', 'parent': '/dgx/parent.pt'}
    return rl_cloud.seed_closure({}, 'lin', dict(spec or {}), boundary, L, window, L / 'parent/checkpoint.pt'), calls


def closure_of(files, excluded=(), missing=()):
    return {'runtime': '/rt', 'files': files, 'bytes': 2, 'excluded': list(excluded), 'missing': list(missing),
            'historical_checkpoints': []}


def test_seed_traces_by_default_and_records_reads_the_prediction_missed(monkeypatch, tmp_path):
    files = [{'path': '/data/a.json', 'sha256': 'a', 'bytes': 1, 'from': ['control']},
             {'path': '/data/b.jsonl', 'sha256': 'b', 'bytes': 1, 'from': ['observed-read']},
             {'path': '/data/c.json', 'sha256': 'c', 'bytes': 1, 'from': ['observed-read', 'recipe']}]
    result, calls = seed_fixture(monkeypatch, tmp_path, closure_of(files))
    assert calls['trace'] == 1
    assert calls['closure_args'][-1][-2:] == ('--observed-reads', str(tmp_path / 'start.strace'))
    assert result['closure']['observed_only'] == ['/data/b.jsonl']  # a closure bug, still shipped
    assert {i['path'] for i in result['inputs']} == {'/data/a.json', '/data/b.jsonl', '/data/c.json'}
    assert json.loads((tmp_path / 'hub/closures/lin-closure-bugs.json').read_text())['observed_only'] == ['/data/b.jsonl']
    manifest = {e['path']: e for e in result['closure_manifest']}
    assert manifest['/data/b.jsonl']['sha256'] == 'b' and manifest['/data/b.jsonl']['check'] == 'sha256'
    assert result['closure']['config_sha256']


def test_seed_observed_reads_opt_out_and_declared_trace(monkeypatch, tmp_path):
    files = [{'path': '/data/a.json', 'sha256': 'a', 'bytes': 1, 'from': ['control']}]
    monkeypatch.setattr(rl_cloud, 'trace_seed_start', lambda *a: pytest.fail('traced'))
    trace = tmp_path / 'declared.strace'
    trace.write_text('')
    original = rl_cloud.trace_seed_start
    result, calls = seed_fixture(monkeypatch, tmp_path, closure_of(files), spec={'observed_reads': str(trace)})
    assert calls['trace'] == 0 and calls['closure_args'] == [('closure', '--config', calls['closure_args'][0][2], '-o',
                                                              calls['closure_args'][0][4], '--observed-reads', str(trace))]
    assert result['closure']['observed_reads']['status'] == 'declared'
    assert original is not None


def test_seed_fails_closed_on_unshipped_checkpoints_and_large_files(monkeypatch, tmp_path):
    files = [{'path': '/data/a.json', 'sha256': 'a', 'bytes': 1, 'from': ['control']}]
    skipped = [{'path': '/dgx/proof/checkpoint.pt', 'sha256': 'h'}]
    with pytest.raises(SystemExit, match='historical checkpoints have no exact blob'):
        seed_fixture(monkeypatch, tmp_path, closure_of(files), skipped=skipped)
    with pytest.raises(SystemExit, match='need a reason'):
        seed_fixture(monkeypatch, tmp_path / 'b', closure_of(files), skipped=skipped,
                     spec={'allow_unshipped': {'/dgx/proof/checkpoint.pt': ''}})
    result, _ = seed_fixture(monkeypatch, tmp_path / 'c', closure_of(files), skipped=skipped,
                             spec={'allow_unshipped': [{'path': '/dgx/proof/checkpoint.pt',
                                                        'reason': 'evidence only: never read at start'}]})
    assert result['closure']['allowed_unshipped'] == {'/dgx/proof/checkpoint.pt': 'evidence only: never read at start'}
    large = [{'path': '/raw/records.jsonl', 'reason': 'large', 'bytes': 5 << 30, 'from': ['control']}]
    with pytest.raises(SystemExit, match='above the closure size limit'):
        seed_fixture(monkeypatch, tmp_path / 'd', closure_of(files, excluded=large))


def test_trace_command_sees_the_host_read_only_and_builds_row_index_entries(tmp_path):
    argv = rl_cloud.trace_argv('u1', '/root/lin', tmp_path / 'copy', '/dgx/parent.pt', '/root/lin/w000000/config.json',
                               '/rt/v45', tmp_path, ['/home', '/mnt'])
    text = ' '.join(argv)
    assert '-v /home:/home:ro' in text and '-v /mnt:/mnt:ro' in text
    assert f'-v {rl_cloud.ROW_INDEX}:{rl_cloud.ROW_INDEX} ' in text
    assert f'-v {tmp_path / "copy"}:/root/lin ' in text and '-v /dgx/parent.pt:/root/lin/parent/checkpoint.pt:ro' in text
    assert '--network none' in text and 'CUDA_VISIBLE_DEVICES=' in text and '--cap-add SYS_PTRACE' in text
    assert argv[argv.index('/usr/bin/strace'):][:6] == ['/usr/bin/strace', '-f', '-qq', '-e', 'trace=%file', '-o']
    assert argv[-4:] == ['--config', '/root/lin/w000000/config.json', '--rounds', '1']
    assert rl_cloud.trace_visible_roots(['/data2/x/y.json', '/usr/lib/z', '/home/werg/a']) == \
        sorted({'/data2'} | {r for r in ('/home', '/mnt') if Path(r).is_dir()})


def test_row_index_entries_of_every_closure_source_ship(monkeypatch, tmp_path):
    cache = tmp_path / 'row-index'
    cache.mkdir()
    for name, source in (('native-graphs-1.pkl', 'a'), ('native-graphs-2.pkl', 'zz'), ('native-graphs-3.pkl', 'a')):
        with open(cache / name, 'wb') as handle:
            pickle.dump({'schema': 'natlang.row-index/1', 'key': {'source_sha256': source}, 'entries': []}, handle)
    (cache / 'native-graphs-4.pkl').write_bytes(b'not a pickle')
    monkeypatch.setattr(rl_cloud, 'ROW_INDEX', cache)
    monkeypatch.setattr(rl_cloud, 'hub', lambda run: tmp_path / 'hub')
    out = rl_cloud.row_index_files({}, [{'path': '/data/a.jsonl', 'sha256': 'a'}])
    assert [Path(f['path']).name for f in out] == ['native-graphs-1.pkl', 'native-graphs-3.pkl']
    assert all(f['from'] == ['row-index-cache'] and f['sha256'] == sha(f['path']) for f in out)
    assert json.loads((tmp_path / 'hub/row-index-map.json').read_text())['native-graphs-2.pkl']['source_sha256'] == 'zz'
    manifest = rl_cloud.manifest_entries(out, [], [], [], tmp_path / 'root/lin')
    assert {e['check'] for e in manifest} == {'exists'}  # a cache entry may be rewritten by the trainer


def test_launch_fails_closed_before_any_sandbox_on_unshipped_bound_paths(monkeypatch, tmp_path):
    monkeypatch.setattr(rl_cloud, 'hub', lambda run: tmp_path)
    L = '/root/lin'
    seed = {'lineage_dir': L, 'links': [], 'inputs': [
        {'kind': 'archive', 'path': '/root/.inputs/lin-closure', 'sha256': 'a',
         'files': [{'path': '/data/a.json', 'sha256': '1'}]},
        {'kind': 'runtime', 'path': '/rt/v45', 'sha256': 'r', 'closure_sha256': 'c'}],
        'blobs': [{'kind': 'checkpoint', 'path': '/dgx/proof/checkpoint.pt', 'sha256': 'h', 'encoding': 'raw'}],
        'closure_manifest': [
            {'path': '/data/a.json', 'sha256': '1', 'bytes': 1, 'kind': 'file', 'check': 'sha256'},
            {'path': '/dgx/proof/checkpoint.pt', 'sha256': 'h', 'bytes': 1, 'kind': 'checkpoint', 'check': 'sha256'},
            {'path': '/dgx/other/checkpoint.pt', 'sha256': 'o', 'bytes': 1, 'kind': 'checkpoint', 'check': 'sha256'},
            {'path': '/root/lin/diloco-s2/session.json', 'sha256': 's', 'bytes': 1, 'kind': 'file', 'check': 'exists'}]}
    rl_cloud.store_json(tmp_path / 'exports/lin/seed.json', seed)
    record = {'lineage': 'lin', 'lineage_dir': L, 'seq': 4, 'links': [],
              'blobs': [{'kind': 'checkpoint', 'path': '/root/lin/w000001/learner/checkpoint.pt', 'sha256': 'w',
                         'encoding': 'raw'}]}
    with pytest.raises(SystemExit, match="1 bound paths are not shipped.*other/checkpoint.pt"):
        rl_cloud.launch_closure({'lineages': {'lin': {}}}, 'lin', record)
    inputs, blobs, links, manifest = rl_cloud.launch_closure(
        {'lineages': {'lin': {'allow_unshipped': {'/dgx/other/checkpoint.pt': 'evidence only'}}}}, 'lin', record)
    paths = {e['path'] for e in manifest}
    assert {'/rt/v45/runtime-input-closure.json', '/root/lin/w000001/learner/checkpoint.pt'} <= paths


def test_relaunch_ships_later_control_checkpoints_and_missing_session_trees(monkeypatch, tmp_path):
    monkeypatch.setattr(rl_cloud, 'hub', lambda run: tmp_path / 'hub')
    monkeypatch.setattr(rl_cloud, 'note', lambda *a: None)
    L = tmp_path / 'root' / 'lin'
    for window in ('w000000', 'w000001'):
        (L / window).mkdir(parents=True)
        (L / window / 'config.json').write_text(json.dumps({'out': str(L / window), 'mandatory_sft_only': True,
                                                            'checkpoint': str(L / window / 'parent.pt')}))
    (L / 'w000001/state.json').write_text('{"phase": "x"}')
    (L / 'lineage.json').write_text(json.dumps({'schema': sandbox.LINEAGE_SCHEMA, 'windows': [
        {'index': i, 'out': str(L / w), 'config': str(L / w / 'config.json')} for i, w in enumerate(('w000000', 'w000001'))]}))
    (L / 'diloco-s2' / 'workers').mkdir(parents=True)
    (L / 'diloco-s2/session.json').write_text('{}')
    (L / 'diloco-s2/workers/w.json').write_text('{}')
    (L / 'diloco-s2/workers/ckpt.pt').write_text('not shipped')
    (L / 'parent').mkdir()
    closure = tmp_path / 'hub/closures/lin.json'
    rl_cloud.store_json(closure, {'config': str(L / 'w000000/config.json'), 'files': []})
    rl_cloud.store_json(tmp_path / 'hub/exports/lin/seed.json', {'lineage_dir': str(L), 'inputs': [], 'blobs': [],
                                                                 'closure': {'path': str(closure)}})
    added = []
    monkeypatch.setattr(rl_cloud, 'add_closure', lambda run, name, config: added.append(str(config)) or
                        {'files': 0, 'checkpoints': 0})
    monkeypatch.setattr(rl_cloud, 'top_members', lambda run, record: {'lineage.json', 'w000001'})
    packed = []
    monkeypatch.setattr(rl_cloud, 'file_inputs', lambda run, name, files: packed.extend(files) or
                        [{'kind': 'archive', 'path': f'/inputs/{name}', 'sha256': 's', 'files': files}])
    record = {'lineage': 'lin', 'lineage_dir': str(L), 'seq': 7, 'state_sha256': sha(L / 'w000001/state.json'),
              'blobs': [], 'links': []}
    run = {'lineages': {'lin': {'controller': '/dgx/control.json'}}}
    extra = rl_cloud.prepare_launch(run, 'lin', record)
    assert added == [str(L / 'w000001/config.json')]
    assert extra['session_dirs'] == [str(L / 'diloco-s2')]
    assert sorted(Path(f['path']).name for f in packed) == ['session.json', 'w.json']
    assert rl_cloud.record_digests(run, record, 'lin') >= {'s'}
    # an export whose top tar carries the session trees needs nothing from the mirror
    packed.clear()
    monkeypatch.setattr(rl_cloud, 'top_members', lambda run, record: {'lineage.json', 'w000001', 'diloco-s2'})
    assert rl_cloud.prepare_launch(run, 'lin', record)['inputs'] == [] and packed == []
    # later-control checkpoints recorded by add_closure travel with every launch, and their links too
    rl_cloud._add_inputs(run, 'lin', [], checkpoints=[{'kind': 'checkpoint', 'path': '/dgx/p2.pt', 'sha256': 'q',
                                                      'encoding': 'raw'}],
                         links=[{'path': '/dgx/l.pt', 'symlink_to': '/dgx/p2.pt'}])
    assert [b['path'] for b in rl_cloud.launch_blobs(run, record, 'lin')] == ['/dgx/p2.pt']
    assert rl_cloud.launch_links(run, record, 'lin') == [{'path': '/dgx/l.pt', 'symlink_to': '/dgx/p2.pt'}]
