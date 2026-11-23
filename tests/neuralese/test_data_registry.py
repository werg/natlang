import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO / 'scripts')]

from natlang_neuralese.common import data_registry as dr  # noqa: E402


@pytest.fixture
def roots(tmp_path, monkeypatch):
    for name in ('data_nvme', 'data_hdd', 'repo'):
        (tmp_path / name).mkdir()
        monkeypatch.setenv(f'NATLANG_{name.upper()}', str(tmp_path / name))
    monkeypatch.setenv('NATLANG_DATA_REGISTRY', str(tmp_path / 'registry'))
    monkeypatch.delenv(dr.ALLOW_ENV, raising=False)
    monkeypatch.setattr(dr, '_DIGESTS', None)
    monkeypatch.setattr(dr, '_LOADED', None)
    repo = tmp_path / 'repo'
    (repo / 'training' / 'corpora').mkdir(parents=True)
    return tmp_path


def git_entries(repo, entries):
    from natlang_neuralese.common import corpus_registry
    corpus_registry.save(repo, {'schema': corpus_registry.SCHEMA, 'corpora': entries})


def pool(path, ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps({'id': i}) + '\n' for i in ids))
    return path


def test_portable_paths_round_trip(roots):
    p = roots / 'data_hdd' / 'pool-a' / 'cases.jsonl'
    assert dr.portable(p) == 'data_hdd:pool-a/cases.jsonl'
    assert dr.absolute('data_hdd:pool-a/cases.jsonl') == p
    assert dr.portable(roots / 'repo' / 'runs' / 'x') == 'runs/x'
    with pytest.raises(ValueError):
        dr.absolute('/abs/path')


def test_record_dataset_writes_manifest_rows_and_provenance(roots):
    d = roots / 'data_hdd' / 'pool-a'
    pool(d / 'cases.jsonl', ['a', 'b', 'c'])
    (d / 'build.json').write_text('{}')
    rec = dr.record_dataset('pool-a', d, kind='rl-tasks', format='natlang.program/2 JSONL', include=['*.jsonl'],
                            license='project-authored', inputs=[])
    assert rec['rows'] == 3 and rec['files'] == 1 and rec['path'] == 'data_hdd:pool-a'
    assert rec['producer']['code']['schema'] == 'natlang.code-provenance/1'
    manifest = dr.stored_manifest(rec['manifest_sha256'])
    assert [f['path'] for f in manifest['files']] == ['cases.jsonl'] and manifest['files'][0]['lines'] == 3
    reg = dr.load(repo=roots / 'repo', fresh=True)
    assert reg.find(d / 'cases.jsonl')['id'] == 'pool-a'
    assert reg.datasets['pool-a']['layer'] == 'journal'


def test_record_dataset_of_a_single_file_includes_only_it(roots):
    d = roots / 'data_nvme' / 'x'
    pool(d / 'a.jsonl', ['a'])
    pool(d / 'b.jsonl', ['b'])
    rec = dr.record_dataset('x-a', d / 'a.jsonl', kind='rl-tasks', format='f')
    assert rec['include'] == ['a.jsonl'] and rec['files'] == 1


def test_consume_refuses_unregistered_and_records_reads(roots, monkeypatch):
    d = roots / 'data_hdd' / 'pool-a'
    pool(d / 'cases.jsonl', ['a'])
    stray = pool(roots / 'data_hdd' / 'stray' / 'cases.jsonl', ['z'])
    dr.record_dataset('pool-a', d, kind='rl-tasks', format='f')
    with pytest.raises(dr.UnregisteredData):
        dr.consume([d / 'cases.jsonl', stray], run='grpo:test', how='GRPO task pool')
    refs = dr.consume([d / 'cases.jsonl'], run='grpo:test', how='GRPO task pool', amount=1)
    assert refs[0]['id'] == 'pool-a' and len(refs[0]['sha256']) == 64
    monkeypatch.setenv(dr.ALLOW_ENV, '1')
    refs = dr.consume([stray], run='grpo:test2', how='pool')
    assert refs[0]['unregistered']
    reg = dr.load(repo=roots / 'repo', fresh=True)
    assert [c['run'] for c in reg.consumers('pool-a')] == ['grpo:test']
    assert reg.consumption[-1]['allowed_unregistered'] is True


def test_growing_dataset_files_and_snapshots(roots):
    d = roots / 'data_hdd' / 'grpo-x' / 'rollouts'
    dr.declare_dataset('grpo-rollouts-x', d, kind='rollouts', format='zst JSONL per group', inputs=[])
    for i in range(3):
        (d / f'g{i}.zst').write_bytes(b'x' * i)
        dr.record_files('grpo-rollouts-x', [d / f'g{i}.zst'], d)
    reg = dr.load(repo=roots / 'repo', fresh=True)
    assert [f['path'] for f in reg.files('grpo-rollouts-x')] == ['g0.zst', 'g1.zst', 'g2.zst']
    assert reg.find(d / 'g1.zst')['id'] == 'grpo-rollouts-x'


def test_file_digest_cache_detects_change(roots):
    p = pool(roots / 'data_nvme' / 'f.jsonl', ['a'])
    first = dr.file_digest(p)
    pool(p, ['a', 'b'])
    second = dr.file_digest(p)
    assert first['sha256'] != second['sha256'] and second['lines'] == 2


def test_roles_come_from_the_git_layer_in_order(roots):
    repo = roots / 'repo'
    pool(roots / 'data_hdd' / 'p1' / 'cases.jsonl', ['a'])
    pool(roots / 'data_hdd' / 'p2' / 'cases.jsonl', ['b'])
    entries = [{'id': 'p2', 'owner': 'dgx', 'path': 'data_hdd:p2', 'roles': {'grpo-pool': {'order': 2, 'file': 'cases.jsonl'}}},
               {'id': 'p1', 'owner': 'dgx', 'path': 'data_hdd:p1', 'roles': {'grpo-pool': {'order': 1, 'file': 'cases.jsonl'}}},
               {'id': 'p3', 'owner': 'dgx', 'path': 'data_hdd:p3', 'retired': True,
                'roles': {'grpo-pool': {'order': 0, 'file': 'cases.jsonl'}}}]
    git_entries(repo, entries)
    assert dr.role_paths('grpo-pool', repo=repo) == [str(roots / 'data_hdd' / 'p1' / 'cases.jsonl'),
                                                      str(roots / 'data_hdd' / 'p2' / 'cases.jsonl')]
    # a journal snapshot of a git entry adds facts but never moves its pinned path or roles
    dr.record_dataset('p1', roots / 'data_hdd' / 'p1', kind='rl-tasks', format='f')
    reg = dr.load(repo=repo, fresh=True)
    assert reg.datasets['p1']['layer'] == 'git+journal' and reg.datasets['p1']['rows'] == 1
    assert reg.datasets['p1']['roles']['grpo-pool']['order'] == 1


def test_git_entry_carries_producer_and_license(roots):
    d = roots / 'data_hdd' / 'pool-a'
    pool(d / 'cases.jsonl', ['a'])
    rec = dr.record_dataset('pool-a', d, kind='rl-tasks', format='f', license='MIT', roles={'grpo-pool': {'order': 9}})
    entry = dr.git_entry(rec)
    assert entry['path'] == 'data_hdd:pool-a' and entry['license'] == 'MIT' and entry['roles'] == {'grpo-pool': {'order': 9}}
    assert set(entry['producer']) == {'script', 'argv', 'git_commit', 'dirty', 'host'}


def test_sync_publish_resolves_root_qualified_paths(roots):
    import sync_training_corpora as sync
    d = roots / 'data_hdd' / 'pool-a'
    pool(d / 'cases.jsonl', ['a', 'b'])
    (d / 'skip.tmp').write_text('x')
    entry = {'id': 'pool-a', 'owner': 'dgx', 'path': 'data_hdd:pool-a', 'exclude_globs': ['*.tmp']}
    out = roots / 'm.json'
    value = sync.publish(roots / 'repo', entry, manifest_output=out)
    assert [f['path'] for f in value['files']] == ['cases.jsonl'] and value['files'][0]['lines'] == 2
    # the registry's own snapshot of the same directory lists the same files with the same digests
    snap = dr.snapshot('pool-a', d, exclude=['*.tmp'])
    assert [(f['path'], f['sha256']) for f in snap['files']] == [(f['path'], f['sha256']) for f in value['files']]


def test_a_pin_as_natlang_repo_keeps_runs_paths_in_the_canonical_checkout(roots, monkeypatch):
    pin = roots / 'pin-abc'
    pin.mkdir()
    (pin / '.natlang-pin.json').write_text(json.dumps({'commit': 'abc', 'source': str(roots / 'repo')}))
    monkeypatch.setenv('NATLANG_REPO', str(pin))
    out = roots / 'repo' / 'runs' / 'mix-v2'
    assert dr.canonical_repo() == roots / 'repo'
    assert dr.portable(out) == 'runs/mix-v2' and dr.dataset_id(out, 'sft-rows') == 'mix-v2'


def test_cli_check_reports_unregistered(roots, capsys):
    import data_registry as cli
    pool(roots / 'data_hdd' / 'p' / 'cases.jsonl', ['a'])
    assert cli.main(['check', str(roots / 'data_hdd' / 'p')]) == 1
    assert cli.main(['record', '--id', 'p', '--path', str(roots / 'data_hdd' / 'p'), '--kind', 'rl-tasks',
                     '--format', 'f']) == 0
    assert cli.main(['check', str(roots / 'data_hdd' / 'p' / 'cases.jsonl')]) == 0


def test_inventory_is_discoverable_but_not_admitted(roots):
    d = roots / 'data_hdd' / 'old-archive' / 'sub'
    pool(d / 'cases.jsonl', ['a', 'b'])
    rec = dr.record_inventory('archive-old', roots / 'data_hdd' / 'old-archive', note='x')
    assert rec['admission'] == dr.INVENTORY and rec['files'] == 1
    tree = dr.stored_manifest(rec['manifest_sha256'])['tree']
    assert {t['path'] for t in tree} == {'.', 'sub'}
    reg = dr.load(repo=roots / 'repo', fresh=True)
    assert reg.find(d / 'cases.jsonl')['id'] == 'archive-old'
    with pytest.raises(dr.UnregisteredData):
        dr.consume([d / 'cases.jsonl'], run='r', how='h')
    with pytest.raises(dr.UnregisteredData):
        dr.require_registered([d / 'cases.jsonl'])
    # recorded as a dataset of its own, the inner directory is admitted (the longest path wins)
    dr.record_dataset('inner', d, kind='rl-tasks', format='f')
    assert dr.consume([d / 'cases.jsonl'], run='r', how='h')[0]['id'] == 'inner'


def test_dataset_ids_follow_where_data_lives(roots):
    assert dr.dataset_id(roots / 'data_hdd' / 'natlang-rl-task-supply-x', 'rl-tasks') == 'natlang-rl-task-supply-x'
    assert dr.dataset_id(roots / 'repo' / 'runs' / 'sft-mix-x', 'sft-rows') == 'sft-mix-x'
    assert dr.dataset_id(roots / 'data_hdd' / 'grpo' / 'run1' / 'rollouts', 'grpo-rollouts') == 'grpo-rollouts-grpo-run1-rollouts'


def test_promote_role_appends_after_current_members(roots, monkeypatch, capsys):
    import data_registry as cli
    repo = roots / 'repo'
    pool(roots / 'data_hdd' / 'p1' / 'cases.jsonl', ['a'])
    git_entries(repo, [
        {'id': 'p1', 'owner': 'dgx', 'path': 'data_hdd:p1', 'roles': {'grpo-pool': {'order': 4, 'file': 'cases.jsonl'}}}])
    pool(roots / 'data_hdd' / 'new' / 'cases.jsonl', ['b'])
    dr.record_dataset('new', roots / 'data_hdd' / 'new', kind='rl-tasks', format='f', license='L')
    monkeypatch.setattr(dr, 'code_repo', lambda: repo)
    assert cli.main(['promote', 'new', '--role', 'grpo-pool', '--dry-run']) == 0
    entry = json.loads(capsys.readouterr().out)[0]
    assert entry['roles'] == {'grpo-pool': {'order': 5, 'file': 'cases.jsonl'}}
    assert entry['path'] == 'data_hdd:new' and entry['license'] == 'L' and entry['kind'] == 'rl-tasks'


def test_build_supply_records_the_source(roots, monkeypatch):
    import importlib.util
    spec = importlib.util.spec_from_file_location('build_supply', REPO / 'scripts/rl_tasks/build_supply.py')
    bs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bs)
    authoring = roots / 'data_hdd' / 'natlang-rl-authoring'
    rows = [{'id': f't{i}', 'family': 'f', 'split': 'train'} for i in range(3)]
    pool(authoring / 'admitted' / 'cases.jsonl', [])
    (authoring / 'admitted' / 'cases.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (authoring / 'admitted' / 'admissions.jsonl').write_text(json.dumps({'batch': 'w300/b1', 'admitted_case_ids': ['t0', 't1']}) + '\n')
    dr.record_dataset('natlang-rl-authoring-admitted', authoring / 'admitted', kind='rl-tasks', format='f', closed=False)
    existing = pool(roots / 'data_hdd' / 'old-pool' / 'cases.jsonl', ['t1'])
    dr.record_dataset('old-pool', existing.parent, kind='rl-tasks', format='f')
    monkeypatch.setattr(bs.grpo, 'default_pools', lambda: [str(existing)])
    out = roots / 'data_hdd' / 'natlang-rl-task-supply-x'
    bs.main([str(out), '--min-worker', '300', '--authoring', str(authoring)])
    assert [json.loads(l)['id'] for l in open(out / 'cases.jsonl')] == ['t0']
    reg = dr.load(repo=roots / 'repo', fresh=True)
    ds = reg.datasets['natlang-rl-task-supply-x']
    assert ds['kind'] == 'rl-tasks' and ds['rows'] == 1 and ds['license'].startswith('project-authored')
    assert {i.get('id') for i in ds['inputs']} >= {'natlang-rl-authoring-admitted', 'old-pool'}


def test_single_file_datasets_share_a_directory(roots):
    d = roots / 'data_nvme' / 'ops'
    pool(d / 'probe-v1.json', ['a'])
    pool(d / 'probe-v2.json', ['b'])
    pool(d / 'other' / 'x.jsonl', ['c'])
    dr.record_dataset('p1', d / 'probe-v1.json', kind='probe', format='f')
    dr.record_dataset('p2', d / 'probe-v2.json', kind='probe', format='f')
    reg = dr.load(repo=roots / 'repo', fresh=True)
    assert reg.find(d / 'probe-v1.json')['id'] == 'p1' and reg.find(d / 'probe-v2.json')['id'] == 'p2'
    assert reg.find(d / 'other' / 'x.jsonl') is None


def test_consumers_include_runs_and_derived_datasets(roots):
    src = pool(roots / 'data_hdd' / 'src' / 'cases.jsonl', ['a'])
    dr.record_dataset('src', src.parent, kind='rl-tasks', format='f')
    out = pool(roots / 'data_hdd' / 'derived' / 'cases.jsonl', ['a'])
    dr.record_dataset('derived', out.parent, kind='rl-tasks', format='f', inputs=[str(src)])
    dr.consume([src], run='grpo:x', how='GRPO task pool')
    reg = dr.load(repo=roots / 'repo', fresh=True)
    assert [c['run'] for c in reg.consumers('src')] == ['grpo:x', 'dataset:derived']


def test_named_file_dataset_directory_names_it(roots):
    d = roots / 'data_nvme' / 'held'
    pool(d / 'cases.jsonl', ['a'])
    pool(d / 'runs' / 'r.jsonl', ['b'])
    dr.record_dataset('held', d, kind='held', format='f', include=['cases.jsonl'])
    reg = dr.load(repo=roots / 'repo', fresh=True)
    assert reg.find(d)['id'] == 'held'
    assert reg.find(d / 'runs' / 'r.jsonl') is None


def test_glob_includes_claim_only_their_fixed_prefix(roots):
    repo = roots / 'repo'
    pool(repo / 'runs' / 'a-1' / 'cases.jsonl', ['a'])
    pool(repo / 'runs' / 'b-1' / 'cases.jsonl', ['b'])
    git_entries(repo, [
        {'id': 'legacy', 'owner': 'dgx', 'path': 'runs', 'include': ['a-1/*']},
        {'id': 'b', 'owner': 'dgx', 'path': 'runs/b-1', 'include': ['cases.jsonl']}])
    reg = dr.load(repo=repo, fresh=True)
    assert reg.find(repo / 'runs' / 'a-1' / 'cases.jsonl')['id'] == 'legacy'
    assert reg.find(repo / 'runs' / 'b-1' / 'cases.jsonl')['id'] == 'b'
    pool(repo / 'runs' / 'c-1' / 'x.jsonl', ['c'])
    assert reg.find(repo / 'runs' / 'c-1' / 'x.jsonl') is None


def test_leading_wildcard_include_claims_only_matching_entries(roots):
    repo = roots / 'repo'
    pool(repo / 'runs' / 'succ-v1' / 'x.jsonl', ['a'])
    pool(repo / 'runs' / 'other' / 'cases.jsonl', ['b'])
    git_entries(repo, [
        {'id': 'legacy', 'owner': 'dgx', 'path': 'runs', 'include': ['succ-v*/**']}])
    reg = dr.load(repo=repo, fresh=True)
    assert reg.find(repo / 'runs' / 'succ-v1' / 'x.jsonl')['id'] == 'legacy'
    assert reg.find(repo / 'runs' / 'other' / 'cases.jsonl') is None


def test_find_takes_the_lexical_fast_path_and_agrees_with_the_full_index(roots, monkeypatch):
    work = roots / 'data_nvme' / 'work'
    pool(work / 'episodes.jsonl', ['a'])
    pool(work / 'sub' / 'inner.jsonl', ['b'])
    literal = roots / 'data_nvme' / 'lit'
    pool(literal / 'cases.jsonl', ['c'])
    dr.record_dataset('work', work, kind='trajectories', format='f', include=['*.jsonl'])
    dr.record_dataset('lit', literal, kind='rl-tasks', format='f')
    dr.record_dataset('inner', work / 'sub', kind='rl-tasks', format='f', include=['*.jsonl'])
    reg = dr.load(repo=roots / 'repo', fresh=True)
    inner = work / 'sub' / 'inner.jsonl'
    full = {p: (reg._find_full(p) or {}).get('id') for p in (literal / 'cases.jsonl', literal, work, work / 'sub', inner)}
    calls = []
    monkeypatch.setattr(reg, '_find_full', lambda p: calls.append(p) or {'id': full.get(p)})
    # literal datasets and a glob dataset's own directory: no filesystem index
    assert reg.find(literal / 'cases.jsonl')['id'] == 'lit' and reg.find(literal)['id'] == 'lit'
    assert reg.find(work)['id'] == 'work' and reg.find(work / 'sub')['id'] == full[work / 'sub'] == 'inner'
    assert not calls
    # Glob-owned files now have the same answer without building the global filesystem index.
    assert reg.find(inner)['id'] == full[inner] == 'inner' and not calls


def test_find_of_a_temporary_path_indexes_only_temporary_datasets(roots, monkeypatch):
    """A temporary path is looked up among datasets recorded in temporary directories only: the full index lists and
    resolves every registered directory (minutes on the busy HDD), none of which can hold a temporary path."""
    data = roots / 'data_nvme' / 'ds'
    data.mkdir()
    (data / 'a.jsonl').write_text('{}\n')
    dr.record_dataset('tmp-ds', data, kind='trajectories', format='x', include=['*.jsonl'])
    reg = dr.load(fresh=True)
    reg.datasets['far'] = {'id': 'far', 'path': '/nonexistent-hdd/far', 'include': ['x-*/y']}
    listed = []
    real_listdir = dr.os.listdir
    monkeypatch.setattr(dr.os, 'listdir', lambda d: listed.append(str(d)) or real_listdir(d))
    assert reg.find(data / 'a.jsonl')['id'] == 'tmp-ds'
    assert reg.find(roots / 'elsewhere' / 'b') is None
    assert 'far' not in reg._temp.datasets and not any('nonexistent-hdd' in d for d in listed)


def test_promoted_role_file_follows_the_roles_members():
    """grpo-sd-teacher members are sft_sources WORK directories: a promotion adds no role file; pool roles keep
    cases.jsonl; an explicit --role-file wins."""
    import types
    import data_registry as cli
    reg = types.SimpleNamespace(git={'w': {'roles': {'grpo-sd-teacher': {'order': 1}}},
                                     'p': {'roles': {'grpo-pool': {'order': 0, 'file': 'cases.jsonl'}}}})
    assert cli.role_spec(reg, 'grpo-sd-teacher', 2) == {'order': 2}
    assert cli.role_spec(reg, 'grpo-pool', 1) == {'order': 1, 'file': 'cases.jsonl'}
    assert cli.role_spec(reg, 'new-role', 0) == {'order': 0, 'file': 'cases.jsonl'}
    assert cli.role_spec(reg, 'grpo-sd-teacher', 2, 'x.json') == {'order': 2, 'file': 'x.json'}


def test_a_producer_record_wins_over_an_earlier_inventory_of_the_same_directory(roots):
    """The 2026-11-21 backfill registered the then-empty ei-sources WORK as an inventory; sft_sources' later record of
    the same directory must make it a valid build input (sft-v7's build was refused)."""
    work = roots / 'data_nvme' / 'ei-sources'
    pool(work / 'replay.jsonl', ['a'])
    dr.record_inventory('archive-ei-sources', work)
    with pytest.raises(dr.UnregisteredData):
        dr.require_registered([work])
    dr.record_dataset('ei-sources', work, kind='trajectories', include=['*.jsonl'], format='f', closed=False)
    dr.require_registered([work])
    assert dr.load(repo=roots / 'repo', fresh=True).find(work)['id'] == 'ei-sources'


def test_an_inventory_of_a_run_directory_does_not_shadow_literal_include_datasets_below_it(roots):
    """grpo-v4's run directory is archived whole (inventory); its ``run2/held/step-N`` held-eval results are datasets with
    literal includes. The lexical fast path walked up to the archive and returned it (check-main's catalog coverage failed
    for 12 step directories), where the full index names the step's own dataset."""
    run = roots / 'data_nvme' / 'ops' / 'grpo-v4'
    step = run / 'run2' / 'held' / 'step-00004'
    pool(step / 'rewards.jsonl', ['a'])
    pool(step / 'result.json', ['b'])
    pool(run / 'other.jsonl', ['c'])
    dr.record_inventory('archive-grpo-v4', run)
    dr.record_dataset('held-step', step, kind='held-eval', protected=True, format='f',
                      include=['result.json', 'rewards.jsonl'])
    reg = dr.load(repo=roots / 'repo', fresh=True)
    assert reg.find(step)['id'] == reg._find_full(step)['id'] == 'held-step'
    assert reg.find(step / 'result.json')['id'] == 'held-step'
    assert reg.find(run / 'other.jsonl')['id'] == 'archive-grpo-v4'


def test_snapshot_skips_a_file_removed_after_the_walk(roots, monkeypatch):
    d = roots / 'data_nvme' / 'live'
    d.mkdir(parents=True)
    (d / 'kept.json').write_text('{}')
    (d / 'gone.partial.json').write_text('{}')
    walk = dr._walk

    def racing_walk(*args):
        listed = walk(*args)
        (d / 'gone.partial.json').unlink()   # a live writer renames its partial file away mid-snapshot
        return listed

    monkeypatch.setattr(dr, '_walk', racing_walk)
    assert [f['path'] for f in dr.snapshot('live-x', d)['files']] == ['kept.json']


@pytest.mark.parametrize('include,relative', [
    (['capture/**'], 'capture/run-a/events.jsonl'),
    (['run-*/**'], 'run-a/events.jsonl'),
    (['run-?.jsonl'], 'run-a.jsonl'),
    (['capture/*/events.jsonl'], 'capture/run-a/events.jsonl'),
    (['capture/**', 'run-*/**'], 'run-a/events.jsonl'),
])
def test_glob_owner_does_not_enumerate_or_resolve_unrelated_datasets(roots, monkeypatch, include, relative):
    data = roots / 'data_nvme' / 'vibe'
    query = pool(data / relative, ['a'])
    entries = [
        {'id': 'vibe', 'path': 'data_nvme:vibe', 'include': include},
        {'id': 'unrelated', 'path': 'data_hdd:acquired-agent', 'include': ['run-*/**']},
    ]
    expected = dr.Registry(entries, [])._find_full(query)['id']
    reg = dr.Registry(entries, [])
    listdir, realpath = dr.os.listdir, dr.os.path.realpath

    def forbid_unrelated(fn):
        def checked(path, *args, **kwargs):
            assert 'acquired-agent' not in str(path), 'lookup touched the unrelated HDD dataset'
            return fn(path, *args, **kwargs)
        return checked

    monkeypatch.setattr(dr.os, 'listdir', forbid_unrelated(listdir))
    monkeypatch.setattr(dr.os.path, 'realpath', forbid_unrelated(realpath))
    monkeypatch.setattr(dr, 'load', lambda: reg)
    assert reg.find(query)['id'] == expected == 'vibe'
    dr.require_registered([query])
    assert dr.consume([query], run='test:vibe', how='admission', record=False)[0]['id'] == 'vibe'
    assert reg._index is None
    monkeypatch.setenv('NATLANG_REGISTRY_SCOPED_GLOBS', '0')
    with pytest.raises(AssertionError, match='unrelated HDD'):
        reg.find(query)


def test_glob_lookup_preserves_deepest_protected_owner_and_inventory_rejection(roots, monkeypatch):
    base = roots / 'data_nvme' / 'campaign'
    query = pool(base / 'capture' / 'held' / 'result.jsonl', ['a'])
    inventory = pool(base / 'archived' / 'run.jsonl', ['b'])
    entries = [
        {'id': 'inventory', 'path': 'data_nvme:campaign', 'admission': dr.INVENTORY},
        {'id': 'vibe', 'path': 'data_nvme:campaign', 'include': ['capture/**']},
        {'id': 'held', 'path': 'data_nvme:campaign/capture/held', 'include': ['*.jsonl'], 'protected': True},
    ]
    reg = dr.Registry(entries, [])
    for path in [base, query.parent, query, base / 'capture' / 'missing' / 'future.jsonl', inventory]:
        assert reg.find(path)['id'] == reg._find_full(path)['id']
    assert reg.find(query)['protected'] is True
    monkeypatch.setattr(dr, 'load', lambda: reg)
    with pytest.raises(dr.UnregisteredData):
        dr.require_registered([inventory])
    with pytest.raises(dr.UnregisteredData):
        dr.consume([inventory], run='test:inventory', how='admission', record=False)


def test_manifest_prefix_and_glob_ties_keep_full_index_ownership(roots):
    base = roots / 'data_nvme' / 'campaign'
    query = pool(base / 'capture' / 'run-a' / 'events.jsonl', ['a'])
    entries = [
        {'id': 'manifest', 'path': 'data_nvme:campaign', 'include_manifest': 'manifest.json',
         'include_prefixes': ['capture']},
        {'id': 'literal', 'path': 'data_nvme:campaign/capture/run-a'},
        {'id': 'inventory', 'path': 'data_nvme:campaign', 'include': ['capture/*'],
         'admission': dr.INVENTORY},
    ]
    for ordering in [entries, entries[::-1]]:
        reg = dr.Registry(ordering, [])
        assert reg.find(query)['id'] == reg._find_full(query)['id']
        assert reg.find(query.parent)['id'] == reg._find_full(query.parent)['id']


def test_glob_lookup_preserves_symlink_alias_ownership(roots):
    base = roots / 'data_nvme' / 'campaign'
    query = pool(base / 'capture' / 'run-a' / 'events.jsonl', ['a'])
    alias = roots / 'repo' / 'alias'
    alias.symlink_to(base, target_is_directory=True)
    for location in ['data_nvme:campaign', 'alias']:
        reg = dr.Registry([{'id': 'vibe', 'path': location, 'include': ['capture/**']}], [])
        for path in [query, alias / 'capture' / 'run-a' / 'events.jsonl']:
            assert reg.find(path)['id'] == reg._find_full(path)['id'] == 'vibe'


def test_inventory_dataset_id_cannot_bypass_path_admission(roots, monkeypatch):
    reg = dr.Registry([
        {'id': 'archive', 'path': 'data_nvme:archive', 'admission': dr.INVENTORY},
        {'id': 'admitted', 'path': 'data_nvme:admitted'},
    ], [])
    monkeypatch.setattr(dr, 'load', lambda: reg)
    with pytest.raises(dr.UnregisteredData):
        dr.require_registered(['archive'])
    with pytest.raises(dr.UnregisteredData):
        dr.consume(['archive'], run='test:inventory', how='admission', record=False)
    dr.require_registered(['admitted'])
    assert dr.consume(['admitted'], run='test:admitted', how='admission', record=False)[0]['id'] == 'admitted'
    ref = dr.consume(['archive'], run='test:inventory', how='explicit opt-out',
                     allow_unregistered=True, record=False)[0]
    assert ref == {'id': 'archive', 'unregistered': True, 'inventory_only': True}
