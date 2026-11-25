"""The standing live-campaign ingester (scripts/neuralese_data/live_campaign_ingest.py): committed directories only,
held and registration drops, duplicates, idempotent reruns, the catalog copy, and the round driver's --live-work."""
import json
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / 'training/neuralese', REPO, REPO / 'scripts/neuralese_data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import live_campaign_ingest as ing  # noqa: E402
import sft_sources  # noqa: E402


def test_committed_partial_manifests_are_ingested_without_claiming_completed_goals(tmp_path):
    (tmp_path/'receipt.json').write_text(json.dumps({
        'full': {'status': 'verified', 'manifest': 'full/manifest.json'},
        'partial': {'status': 'partial', 'manifest': 'partial/manifest.json'},
        'interrupted': {'status': 'interrupted', 'raw': 'unfinished.json.gz'}}))
    (tmp_path/'exported.json').write_text(json.dumps({'verified': 2, 'completed_goals': 1, 'partial_trajectories': 1, 'dataset': 'accepted'}))
    manifests, dataset = ing.verified_manifests(tmp_path)
    assert manifests == [tmp_path/'capture/full/manifest.json', tmp_path/'capture/partial/manifest.json']
    assert dataset == 'accepted'


def wave(root, name, cases, exported=True):
    d = root / name
    (d / 'capture').mkdir(parents=True)
    receipt = {}
    for case, status in cases.items():
        receipt[case] = {'status': status, **({'manifest': f'cases/{case}/verified/manifest.json'}
                                               if status == 'verified' else {'raw': 'x'})}
    (d / 'receipt.json').write_text(json.dumps(receipt))
    if exported:
        n = sum(s == 'verified' for s in cases.values())
        (d / 'exported.json').write_text(json.dumps({'verified': n, 'dataset': f'ds-{name}' if n else None}))
    return d


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    root = tmp_path / 'campaign'
    root.mkdir()
    (root / 'catalog.jsonl').write_text(json.dumps({'taskId': 'vibestudio:fixture', 'interaction': 'open-user-goal',
        'split': 'train', 'sourceGroups': ['fixture'], 'goalSha256': 'a' * 64}) + '\n')
    wave(root, '0000', {'a': 'verified', 'b': 'test-failed', 'held': 'verified'})
    wave(root, '0001', {'c': 'verified', 'again': 'verified', 'unregistered': 'verified'})
    wave(root, '0002', {'d': 'verified'}, exported=False)        # still being written
    wave(root, 'seed-0000', {'a2': 'verified'})
    goals = {'a': 'Fix the notice', 'held': 'Held goal', 'c': 'Answer', 'again': 'fix THE notice', 'unregistered': 'U',
             'a2': 'Fix the notice', 'd': 'Late goal'}

    def load(path):
        case = Path(path).parts[-3] if 'cases' in Path(path).parts else 'a2'
        return {'key': f'live:vibestudio:{case}:{case}sha', 'task_id': f'vibestudio:{case}', 'turns': [1, 2],
                'task': {'id': f'vibestudio:{case}', 'source_groups': [f'g-{case}'],
                         'interaction': {'goal': goals[case]}}}
    monkeypatch.setattr('live_teacher_sources.load_episode', load)
    monkeypatch.setattr(sft_sources, 'held_closure', lambda m: (set(), set(), set()))
    monkeypatch.setattr(sft_sources, 'live_holdout_registry', lambda path, held: {'registered': True})
    monkeypatch.setattr(sft_sources, 'held_excluded_ids',
                        lambda m, tasks, held: {t for t in tasks if t == 'vibestudio:held'})

    def registered(candidate, training):
        if candidate['task_id'] == 'vibestudio:unregistered':
            raise ValueError('differs')
    monkeypatch.setattr(sft_sources, 'require_registered_open_goal', registered)
    records = []
    return root, records


def test_committed_directories_are_ingested_once_with_held_and_repeat_drops(campaign, tmp_path):
    root, records = campaign
    work = tmp_path / 'work'
    extracted = []
    new = ing.ingest_once(root, work, 'held.json', extract=lambda a: extracted.append((a.out_name, a.live_run)))
    assert [r['directory'] for r in new] == ['0000', '0001', 'seed-0000']     # 0002 has no exported.json yet
    by = {r['directory']: r for r in new}
    assert by['0000']['kept'] == 1 and by['0000']['dropped'] == {'held': 1}
    assert by['0001']['kept'] == 1 and by['0001']['dropped'] == {'repeated_open_goal': 1, 'unregistered_open_goal': 1}
    assert by['seed-0000']['kept'] == 0 and by['seed-0000']['dropped'] == {'repeated_open_goal': 1}
    # one extraction per scan, named after its first directory with episodes
    assert extracted == [('episodes-0000', [str(root / '0000/capture/cases/a/verified/manifest.json'),
                                            str(root / '0001/capture/cases/c/verified/manifest.json')])]
    assert [r['extraction'] for r in new] == ['episodes-0000.jsonl.zst'] * 2 + [None]
    assert (work / ing.REGISTRY).read_text() == (root / 'catalog.jsonl').read_text()
    # a rerun reads nothing new; a newly committed directory is taken and a duplicate key dropped
    assert ing.ingest_once(root, work, 'held.json', extract=lambda a: extracted.append(a.out_name)) == []
    (root / '0002' / 'exported.json').write_text(json.dumps({'verified': 1, 'dataset': 'ds'}))
    [late] = ing.ingest_once(root, work, 'held.json', extract=lambda a: extracted.append(a.out_name))
    assert late['directory'] == '0002' and late['kept'] == 1 and extracted[-1] == 'episodes-0002'
    assert ing.summary(work) == {'directories': 4, 'verified': 7, 'episodes': 3,
                                 'dropped': {'held': 1, 'repeated_open_goal': 2, 'unregistered_open_goal': 1}}
    # the catalog the WORK was built against must not change under it
    (root / 'catalog.jsonl').write_text('{"catalog": 2}\n')
    with pytest.raises(SystemExit, match='changed'):
        ing.ingest_once(root, work, 'held.json', extract=lambda a: None)


def test_a_receipt_that_disagrees_with_its_export_marker_is_refused(campaign):
    root, _ = campaign
    (root / '0000' / 'exported.json').write_text(json.dumps({'verified': 3, 'dataset': 'x'}))
    with pytest.raises(ValueError, match='receipt lists 2 verified'):
        ing.verified_manifests(root / '0000')


def test_rounds_link_the_live_works_build_against_their_registry_and_fill_them_first(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / 'tests/neuralese'))
    from test_async_train import _driver
    live = tmp_path / 'live-work'
    live.mkdir()
    (live / 'episodes-0000.jsonl.zst').write_bytes(b'x')
    (live / ing.REGISTRY).write_text('{}\n')
    ei, d = _driver(tmp_path, monkeypatch, argv=['--live-work', str(live)])
    work = Path(d.c.work)
    work.mkdir(parents=True, exist_ok=True)
    assert ei.link_live_works(work, [live]) == 1 and ei.link_live_works(work, [live]) == 0
    assert (work / 'episodes-0000-live-work.jsonl.zst').resolve() == (live / 'episodes-0000.jsonl.zst').resolve()
    (live / 'episodes-0001.jsonl.zst').write_bytes(b'y')                 # the ingester's next wave
    assert ei.link_live_works(work, [live]) == 1
    cur = d.controller.current
    d.collect_dir(cur).mkdir(parents=True)
    (d.collect_dir(cur) / 'runs.jsonl').write_text(json.dumps({'started': 0}) + '\n')
    _, _, build = d.prepare_plan(d.root / 'r0001', [cur])
    assert build[build.index('--live-holdout-registry') + 1] == str(live / ing.REGISTRY)
    monkeypatch.setattr('natlang_neuralese.common.data_registry.role_paths', lambda role, **kw: ['/role/work'])
    fill = d.c.fill_command({'sglang': 's', 'url': 'u', 'export': 'e'})
    works = [fill[i + 1] for i, x in enumerate(fill) if x == '--work']
    assert works[0] == str(live) and fill[fill.index('--prefer-work') + 1] == str(live)
    assert fill[fill.index('--rescan-minutes') + 1] == '30'
    assert fill[fill.index('--registry-repo') + 1] == str(d.c.live_data_owner())
    assert '--registry-ref' not in fill  # standing generation follows later publications
    assert build[build.index('--registry-repo') + 1] == str(d.c.live_data_owner())
    (tmp_path / 'other').mkdir()
    (tmp_path / 'other' / ing.REGISTRY).write_text('{"different": 1}\n')
    with pytest.raises(SystemExit, match='different holdout registries'):
        ei.live_holdout_registry([live, tmp_path / 'other'])
    assert ei.live_holdout_registry([tmp_path]) is None


def test_partial_decisions_never_reserve_a_whole_goal_or_displace_completed_data(monkeypatch):
    monkeypatch.setattr(sft_sources, 'held_excluded_ids', lambda *args: set())
    monkeypatch.setattr(sft_sources, 'require_registered_open_goal', lambda *args: None)
    def load(path):
        return {'key': str(path), 'task_id': 'registered-task', 'turns': [1],
                'task': {'source_groups': ['registered-source'], 'interaction': {'goal': 'Same user goal'}},
                'trajectory_ir': {'outcome': {'task_completed': str(path) == 'complete'}}}
    seen_keys, seen_goals = set(), set()
    kept, records, dropped = ing.select(['partial-before', 'complete', 'partial-after', 'complete'],
        'held', set(), {}, seen_keys, seen_goals, load=load)
    assert kept == ['partial-before', 'complete', 'partial-after']
    assert [row['task_completed'] for row in records] == [False, True, False]
    assert dropped == {'duplicate_episode': 1}
    assert seen_goals == {sft_sources.open_goal_key('Same user goal')}


def test_evidence_only_completed_trajectory_does_not_reserve_goal(monkeypatch):
    monkeypatch.setattr(sft_sources, 'held_excluded_ids', lambda *args: set())
    monkeypatch.setattr(sft_sources, 'require_registered_open_goal', lambda *args: None)
    def load(path):
        return {'key': str(path), 'task_id': 'registered-task', 'turns': [] if path == 'evidence' else [1],
                'task': {'source_groups': ['registered-source'], 'interaction': {'goal': 'Same user goal'}},
                'trajectory_ir': {'outcome': {'task_completed': True}}}
    kept, records, dropped = ing.select(['evidence', 'usable'], 'held', set(), {}, set(), set(), load=load)
    assert kept == ['evidence', 'usable']
    assert [row['decisions'] for row in records] == [0, 1]
    assert dropped == {}


def test_ingest_forwards_one_published_owner_snapshot_to_extraction(campaign, tmp_path):
    root, _ = campaign
    owner = tmp_path / 'registry-owner'
    owner.mkdir()
    extracted = []
    ing.ingest_once(root, tmp_path / 'work-owner', 'held.json', extract=extracted.append,
                    registry_repo=owner, registry_ref='published-snapshot')
    assert len(extracted) == 1 and len(extracted[0].live_run) == 2
    assert extracted[0].registry_repo == owner and extracted[0].registry_ref == 'published-snapshot'
    assert extracted[0].held_manifest == 'held.json'


def test_retirement_runs_after_durable_ingest_and_retries_without_reextracting(campaign, tmp_path, monkeypatch):
    root, _ = campaign
    work = tmp_path / 'work'
    calls = []
    def retire(directory, record, **kwargs):
        committed = [json.loads(line) for line in (work / 'ingest.jsonl').read_text().splitlines()]
        assert record in committed
        assert kwargs['work'] == work
        calls.append(record['directory'])
        raise OSError('diagnostic state temporarily busy')
    monkeypatch.setattr('retained_state_archive.retire_ingested_native_state', retire)
    # Housekeeping failure must not invalidate a committed useful partial episode
    # or hold subsequent extraction/training. The next pass retries old records.
    new = ing.ingest_once(root, work, 'held.json', extract=lambda args: None)
    assert len(new) == 3 and len(calls) == 3
    assert ing.ingest_once(root, work, 'held.json', extract=lambda args: pytest.fail('no reextraction')) == []
    assert len(calls) == 6


def identity_catalog(tmp_path):
    from live_teacher_sources import open_goal_fingerprint
    goal = 'Explain the café change.\nKeep "approved" exact.'
    rows = [{'taskId': f'vibestudio:{split}', 'interaction': 'open-user-goal', 'split': split,
             'sourceGroups': [f'{split}-world', f'{split}-source'], 'scenario': f'case-{split}',
             'goalSha256': open_goal_fingerprint(goal + split)[1], 'requiresReview': False,
             'entry': {'program': 'old.nl', 'callable': 'old', 'resultType': 'OldReport', 'skill': 'research',
                       'arguments': {'goal': goal + split, 'scope': 'owned'},
                       'argumentJson': {'goal': json.dumps(goal + split), 'scope': '"owned"'},
                       'parameterNames': ['scope', 'goal'], 'inputValues': ['owned', goal + split]}}
            for split in ('train', 'validation', 'test')]
    root, work = tmp_path / 'catalog-campaign', tmp_path / 'catalog-work'
    root.mkdir(); work.mkdir()
    source = root / 'catalog.jsonl'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    snapshot = ing.holdout_registry(root, work)
    return rows, root, work, source, snapshot


def test_catalog_metadata_updates_preserve_immutable_held_snapshot_and_registration(tmp_path):
    import copy
    import hashlib
    from live_catalog_identity import compatible_catalogs
    rows, root, work, source, snapshot = identity_catalog(tmp_path)
    original = snapshot.read_bytes()
    before_held = (set(), set(), set())
    training = sft_sources.live_holdout_registry(snapshot, before_held)
    updated = copy.deepcopy(rows)
    for row in updated:
        row['requiresReview'] = True
        row['entry'].update(program='semantic-entry.nl', callable='semanticEntry', resultType='Findings')
        row['devices'] = ['desktop']
        row['sourceGroups'].reverse()  # group membership is a set, not an execution ordering
    source.write_text(''.join(json.dumps(row, sort_keys=True) + '\n' for row in reversed(updated)))
    assert ing.holdout_registry(root, work) == snapshot
    assert snapshot.read_bytes() == original
    after_held = (set(), set(), set())
    assert sft_sources.live_holdout_registry(source, after_held) == training
    assert after_held == before_held
    sft_sources.require_registered_open_goal({'task_id': rows[0]['taskId'], 'source_groups': rows[0]['sourceGroups'],
        'open_goal': rows[0]['entry']['arguments']['goal']}, training)
    receipt = json.loads((work / 'catalog-compatibility.json').read_text())
    assert receipt == compatible_catalogs(snapshot, source)
    assert receipt['snapshot_sha256'] == hashlib.sha256(original).hexdigest()
    assert receipt['current_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt['snapshot_sha256'] != receipt['current_sha256']
    assert len(receipt['execution_metadata_changes']) == 3
    assert 'entry.resultType' in receipt['execution_metadata_changes'][0]['fields']
    assert receipt['validator_sha256'] == hashlib.sha256((REPO/'scripts/neuralese_data/live_catalog_identity.py').read_bytes()).hexdigest()


@pytest.mark.parametrize('mutation', ['add', 'drop', 'id', 'split', 'groups', 'interaction', 'digest',
    'actual-goal', 'encoded-goal', 'bound-goal', 'non-goal-input', 'unknown', 'unknown-entry', 'duplicate', 'metadata-type'])
def test_catalog_identity_changes_fail_even_with_unchanged_digest_or_review_flags(tmp_path, mutation):
    rows, root, work, source, snapshot = identity_catalog(tmp_path)
    original, receipt = snapshot.read_bytes(), (work/'catalog-compatibility.json').read_bytes()
    row = rows[-1]  # exercise actual held rows, not only training registration
    if mutation == 'add': rows.append({**row, 'taskId': 'vibestudio:added'})
    elif mutation == 'drop': rows.pop()
    elif mutation == 'id': row['taskId'] = 'vibestudio:renamed'
    elif mutation == 'split': row['split'] = 'train'
    elif mutation == 'groups': row['sourceGroups'] = ['different-world']
    elif mutation == 'interaction': row['interaction'] = 'assigned-workflow'
    elif mutation == 'digest': row['goalSha256'] = 'b' * 64
    elif mutation == 'actual-goal': row['entry']['arguments']['goal'] += ' changed'
    elif mutation == 'encoded-goal': row['entry']['argumentJson']['goal'] = '"Other goal"'
    elif mutation == 'bound-goal': row['entry']['inputValues'][1] = 'Other goal'
    elif mutation == 'non-goal-input': row['entry']['arguments']['scope'] = 'foreign'
    elif mutation == 'unknown': row['futureAuthority'] = 'foreign'
    elif mutation == 'unknown-entry': row['entry']['futureInput'] = 'foreign'
    elif mutation == 'duplicate': rows.append(row)
    elif mutation == 'metadata-type': row['requiresReview'] = 'false'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(SystemExit, match='changed'):
        ing.holdout_registry(root, work)
    assert snapshot.read_bytes() == original
    assert (work/'catalog-compatibility.json').read_bytes() == receipt


def test_registry_validates_actual_goal_even_without_comparison(tmp_path):
    rows, _, _, source, _ = identity_catalog(tmp_path)
    rows[0]['entry']['arguments']['goal'] = 'Forged goal with retained hash'
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(ValueError, match='actual goal differs'):
        sft_sources.live_holdout_registry(source, (set(), set(), set()))


def test_round_driver_accepts_metadata_only_registry_differences(tmp_path):
    import expert_iteration as ei
    rows, _, work, source, snapshot = identity_catalog(tmp_path)
    other = tmp_path/'other-work'; other.mkdir()
    rows[0]['requiresReview'] = True
    rows[0]['entry']['resultType'] = 'Findings'
    (other/ing.REGISTRY).write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)))
    assert ei.live_holdout_registry([work, other]) == snapshot
    rows[-1]['sourceGroups'] = ['unregistered-held-group']
    (other/ing.REGISTRY).write_text(''.join(json.dumps(row) + '\n' for row in rows))
    with pytest.raises(SystemExit, match='different holdout registries'):
        ei.live_holdout_registry([work, other])
