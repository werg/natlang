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


# Pluggable classifier (applications/ops-retention/DECOMPOSITION.md) --------------------------------------------------
def facts_and_crisp(entries, p, text='', active=NO_ACTIVITY, registered=()):
    pairs = retention.retention_facts(entries, p, registered=set(registered), text=text, active=active)
    return pairs, [dict(e, tier=t, reason=r) for e, f in pairs for t, r in [retention.crisp_verdict(f, p)]]


def test_facts_carry_the_field_names_of_the_natlang_type_and_the_crisp_verdict_reads_only_them(tmp_path):
    import re
    block = (ROOT / 'applications/ops-retention/types.ts').read_text().split('export type RetentionFacts = {')[1].split('\n};')[0]
    declared = set(re.findall(r'^\s+(\w+):', block, re.M))
    run = tmp_path / 'run-f.failed'
    p = policy(tmp_path)
    es = [entry(run / 'step_1.pt'), entry(run / 'step_2.pt'), entry(run / 'x.tmp', size=5, category='other')]
    pairs, classified = facts_and_crisp(es, p, text=f'see {run}')
    assert {frozenset(f) for _, f in pairs} == {frozenset(declared)}
    assert [(e['tier'], e['reason']) for e in classified] == \
        [(e['tier'], e['reason']) for e in retention.classify(es, p, registered=set(), text=f'see {run}', active=NO_ACTIVITY)]
    by_name = {Path(e['path']).name: f for e, f in pairs}
    assert by_name['step_1.pt']['step'] == 1 and by_name['step_1.pt']['latest_step_in_directory'] == 2
    assert by_name['step_1.pt']['run_marked_stopped'] and by_name['step_1.pt']['directory_named_by_reference']
    assert by_name['x.tmp']['partial_write'] and by_name['x.tmp']['step'] is None
    assert {Path(e['path']).name: e['tier'] for e in classified} == \
        {'step_1.pt': 'PRUNE-ASK', 'step_2.pt': 'PRUNE-ASK', 'x.tmp': 'PRUNE-AUTO'}


def fake_bridge(nl_tiers, calls, status=0):
    """Stands in for ts-host/scripts/retention-classify.mjs: serves crisp in shadow, the given tiers in nl."""
    def bridge(argv):
        calls.append(argv)
        args = dict(zip(argv[::2], argv[1::2]))
        items = json.loads(Path(args['--input']).read_text())['items']
        mode = args['--mode']
        classified = [{'id': i['id'], 'tier': nl_tiers.get(i['id'], i['crisp']['tier']) if mode == 'nl' else i['crisp']['tier'],
                       'reason': 'nl reason' if mode == 'nl' else i['crisp']['reason'], 'served': 'nl' if mode == 'nl' else 'crisp'}
                      for i in items]
        disagree = sum(1 for i in items if nl_tiers.get(i['id'], i['crisp']['tier']) != i['crisp']['tier'])
        Path(args['--out']).write_text(json.dumps({'mode': mode, 'classifier': 'natlang@test', 'classified': classified,
                                                    'summary': {'mode': mode, 'items': len(items), 'agree': len(items) - disagree,
                                                                'disagree': disagree, 'refused': 0, 'errors': 0}}))
        return status
    return bridge


def test_shadow_keeps_the_crisp_tiers_and_nl_serves_the_natural_language_ones(tmp_path):
    p = policy(tmp_path)
    run = tmp_path / 'run-g'
    pairs, classified = facts_and_crisp([entry(run / 'step_1.pt'), entry(run / 'step_2.pt')], p)
    before = [(e['tier'], e['reason']) for e in classified]
    setting = {'mode': 'shadow', 'server': 'http://x', 'model': 'm', 'shadow_log': str(tmp_path / 'shadow.jsonl')}
    calls = []
    summary = retention.nl_classify(classified, [f for _, f in pairs], p, setting, bridge=fake_bridge({'0': 'KEEP'}, calls))
    assert [(e['tier'], e['reason']) for e in classified] == before
    assert summary['disagree'] == 1 and summary['classifier'] == 'natlang@test'
    assert calls[0][calls[0].index('--mode') + 1] == 'shadow' and '--model' in calls[0]
    summary = retention.nl_classify(classified, [f for _, f in pairs], p, dict(setting, mode='nl'),
                                    bridge=fake_bridge({'0': 'KEEP'}, calls))
    assert classified[0]['tier'] == 'KEEP' and classified[0]['crisp_tier'] == before[0][0] and classified[0]['served'] == 'nl'


def test_a_broken_bridge_leaves_crisp_in_shadow_and_stops_nl_before_anything_is_deleted(tmp_path):
    import pytest
    p = policy(tmp_path)
    pairs, classified = facts_and_crisp([entry(tmp_path / 'r' / 'step_1.pt')], p)
    before = classified[0]['tier']
    setting = {'mode': 'shadow', 'shadow_log': str(tmp_path / 's.jsonl')}
    summary = retention.nl_classify(classified, [f for _, f in pairs], p, setting, bridge=fake_bridge({}, [], status=1))
    assert 'error' in summary and classified[0]['tier'] == before
    with pytest.raises(RuntimeError):
        retention.nl_classify(classified, [f for _, f in pairs], p, dict(setting, mode='nl'), bridge=fake_bridge({}, [], status=1))


def test_classifier_setting_flag_over_environment_over_policy(tmp_path, monkeypatch):
    import argparse
    import pytest
    p = dict(policy(tmp_path), classifier={'mode': 'shadow', 'model': 'policy-model'})
    args = argparse.Namespace(classifier=None, nl_server=None, nl_model=None, nl_shadow_log=None, nl_concurrency=None)
    monkeypatch.delenv('NATLANG_RETENTION_CLASSIFIER', raising=False)
    assert retention.classifier_setting(args, p)['mode'] == 'shadow'
    assert retention.classifier_setting(argparse.Namespace(), {})['mode'] == 'crisp'
    monkeypatch.setenv('NATLANG_RETENTION_CLASSIFIER', 'natlang')
    assert retention.classifier_setting(args, p)['mode'] == 'nl'
    got = retention.classifier_setting(argparse.Namespace(**dict(vars(args), classifier='crisp', nl_model='m2')), p)
    assert got['mode'] == 'crisp' and got['model'] == 'm2'
    monkeypatch.setenv('NATLANG_RETENTION_CLASSIFIER', 'bogus')
    with pytest.raises(SystemExit):
        retention.classifier_setting(args, p)
