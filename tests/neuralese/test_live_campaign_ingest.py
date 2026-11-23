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
    (root / 'catalog.jsonl').write_text('{"catalog": 1}\n')
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
