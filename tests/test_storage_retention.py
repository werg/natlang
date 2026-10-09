import importlib.util
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
SPEC = importlib.util.spec_from_file_location('storage_retention', ROOT / 'scripts/storage_retention.py')
retention = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(retention)
DAY = 86400


def policy(tmp_path):
    p = json.loads((ROOT / 'training/storage-policy.json').read_text())
    p['min_report_bytes'] = 10
    p['archive']['root'] = str(tmp_path / 'archive')
    return p


NO_ACTIVITY = {'held': set(), 'claims': [], 'busy': set(), 'excludes': [], 'globs': ()}


def entry(path, size=100, age_days=10, category='checkpoint', run=None):
    path = Path(path)
    return {'path': str(path), 'logical': str(path), 'bytes': size, 'mtime': time.time() - age_days * DAY,
            'category': category, 'run': str(run or path.parent)}


def tiers(entries, p, text='', active=NO_ACTIVITY, registered=()):
    out = retention.classify(entries, p, registered=set(registered), text=text, active=active)
    return {Path(e['path']).name: (e['tier'], e['reason']) for e in out}


def test_intermediate_step_checkpoints_of_a_finished_run_are_auto_pruned_latest_and_best_kept(tmp_path):
    run = tmp_path / 'run-a'
    got = tiers([entry(run / 'step_100.pt'), entry(run / 'step_200.pt'), entry(run / 'step_300.pt'),
                 entry(run / 'best.pt', category='weights-snapshot')], policy(tmp_path))
    assert got['step_100.pt'][0] == got['step_200.pt'][0] == 'PRUNE-AUTO'
    assert got['step_300.pt'][0] == 'PRUNE-ASK'  # the latest: kept unless the owner approves
    assert got['best.pt'][0] == 'PRUNE-ASK'


def test_recent_runs_open_files_and_claims_are_active_and_never_pruned(tmp_path):
    run = tmp_path / 'run-b'
    p = policy(tmp_path)
    recent = [entry(run / 'step_1.pt', age_days=10), entry(run / 'step_2.pt', age_days=1)]
    assert {t for t, _ in tiers(recent, p).values()} == {'ACTIVE'}  # the run changed a day ago
    old = [entry(run / 'step_1.pt'), entry(run / 'step_2.pt')]
    held = dict(NO_ACTIVITY, held={str(run / 'step_1.pt')})
    assert tiers(old, p, active=held)['step_1.pt'][0] == 'ACTIVE'
    claimed = dict(NO_ACTIVITY, claims=[run])
    assert {t for t, _ in tiers(old, p, active=claimed).values()} == {'ACTIVE'}


def test_registered_referenced_and_log_files_are_kept(tmp_path):
    run = tmp_path / 'run-c'
    p = policy(tmp_path)
    es = [entry(run / 'step_1.pt'), entry(run / 'step_2.pt'), entry(run / 'train.log', category='logs-metrics'),
          entry(run / 'corpus.jsonl', category='corpus-registered')]
    got = tiers(es, p, text=f'--continue-from {run / "step_1.pt"}', registered={str(run / 'corpus.jsonl')})
    assert got['step_1.pt'][0] == 'KEEP' and got['train.log'][0] == 'KEEP' and got['corpus.jsonl'][0] == 'KEEP'


def test_aborted_runs_and_leftovers_are_auto_pruned_unless_their_directory_is_named(tmp_path):
    p = policy(tmp_path)
    aborted = tmp_path / 'run-d.aborted-1'
    es = [entry(aborted / 'checkpoint.pt'), entry(tmp_path / 'run-e' / 'checkpoint.pending', size=5)]
    got = tiers(es, p)
    assert got['checkpoint.pt'][0] == 'PRUNE-AUTO' and got['checkpoint.pending'][0] == 'PRUNE-AUTO'
    assert tiers(es[:1], p, text=f'see {aborted}')['checkpoint.pt'][0] == 'PRUNE-ASK'


def test_delete_rechecks_the_file_and_removes_the_symlink_to_an_archived_copy(tmp_path):
    archive = tmp_path / 'archive' / 'runs' / 'r'
    archive.mkdir(parents=True)
    copy = archive / 'step_1.pt'
    copy.write_bytes(b'x' * 100)
    link = tmp_path / 'nvme' / 'step_1.pt'
    link.parent.mkdir()
    os.symlink(copy, link)
    st = copy.stat()
    e = {'path': str(copy), 'logical': str(link), 'bytes': 100, 'mtime': st.st_mtime, 'category': 'checkpoint',
         'tier': 'PRUNE-AUTO', 'reason': 'test', 'run': str(archive)}
    manifest = tmp_path / 'deletions.jsonl'
    assert retention.delete(dict(e, bytes=99), manifest, set()) is None and copy.exists()  # changed: left alone
    assert retention.delete(e, manifest, {str(copy)}) is None and copy.exists()  # open: left alone
    record = retention.delete(e, manifest, set())
    assert record['link'] == str(link) and not copy.exists() and not os.path.lexists(link)
    assert json.loads(manifest.read_text())['bytes'] == 100


def test_scan_never_follows_symlinks_and_maps_archived_files_to_their_source(tmp_path):
    p = policy(tmp_path)
    archived = tmp_path / 'archive' / 'runs' / 'r1' / 'step_5.pt'
    archived.parent.mkdir(parents=True)
    archived.write_bytes(b'y' * (65 << 20))
    os.symlink(archived, tmp_path / 'link.pt')
    entries, totals = retention.scan(p, [tmp_path], registered=set())
    assert [e['logical'] for e in entries] == [str(retention.LOGICAL_ROOTS['runs'] / 'r1' / 'step_5.pt')]
    assert totals[str(tmp_path)]['symlink']['files'] == 1


def test_sync_revisions_are_ask_tier_and_an_approved_revision_directory_is_removed_whole(tmp_path):
    import argparse
    revision = tmp_path / '.sync-history' / '2026-10-03T15-30-16'
    (revision / 'runs' / 'x').mkdir(parents=True)
    (revision / 'runs' / 'x' / 'old.jsonl').write_bytes(b'z' * 1000)
    p = policy(tmp_path)
    revisions = {}
    entries, totals = retention.scan(p, [tmp_path], registered=set(), revisions=revisions)
    assert revisions == {str(revision): 1000} and totals[str(tmp_path)]['sync-revision']['bytes'] == 1000
    approvals = tmp_path / 'approved.txt'
    approvals.write_text(str(revision) + '\n')
    pol = tmp_path / 'policy.json'
    pol.write_text(json.dumps(p))
    args = argparse.Namespace(apply=True, policy=str(pol), roots=[str(tmp_path)], approvals=str(approvals),
                              manifest=str(tmp_path / 'deletions.jsonl'), no_docker=True)
    report = retention.run(args)
    assert not revision.exists() and report['deleted'][0]['bytes'] == 1000


def test_reference_expansion_never_descends_through_symlinked_directories(tmp_path):
    real = tmp_path / 'runs' / 'r1'
    real.mkdir(parents=True)
    (real / 'a-certificate.json').write_text('{}')
    elsewhere = tmp_path / 'hdd' / 'r2'
    elsewhere.mkdir(parents=True)
    (elsewhere / 'b-certificate.json').write_text('{}')
    os.symlink(elsewhere, tmp_path / 'runs' / 'r2')
    found = retention.expand(tmp_path, 'runs/*/*certificate*.json')
    assert found == [str(real / 'a-certificate.json')]
    assert retention.expand(tmp_path, 'runs/**/*.json') == [str(real / 'a-certificate.json')]
