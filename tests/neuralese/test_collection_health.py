"""Collection health report (scripts/neuralese_data/collection_health.py) and the async driver's held-v4 cadence
(expert_iteration.parse_held_v4_cadence, held_v4_reasons, HeldV4Queue, AsyncRoundDriver.health_report)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / 'training/neuralese', REPO, REPO / 'scripts/neuralese_data'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import collection_health as ch  # noqa: E402


def _group(work, index, task, family, outcomes, *, constructs=(), diag=None, runtime=None, guided=(), job_code=None):
    """One scored fixture group: ``outcomes`` per sample (rewards.jsonl), and per failed sample a results.jsonl with a
    rejected action carrying ``diag`` and/or a runtime scope failure ``runtime``."""
    g = work / 'groups' / f'g{index:06d}'
    g.mkdir(parents=True)
    (g / 'cases.jsonl').write_text(json.dumps({'id': task, 'family': family, 'curriculum': {
        'external': {'constructs': list(constructs)}}}) + '\n')
    rows = []
    for i, outcome in enumerate(outcomes):
        rows.append({'task_id': task, 'family': family, 'sample': i, 'outcome': outcome,
                     'finish_status': 'complete', **({'jolt_role': 'teacher'} if i in guided else {})})
        sample = g / 'row-000' / f'sample-{i:02d}'
        sample.mkdir(parents=True)
        ledger, failures = [], []
        if outcome != 'task_success' and diag:
            ledger.append({'kind': 'action', 'outcome': 'rejected', 'diagnostics': [diag], 'result_text': 'x'})
        if outcome != 'task_success' and runtime:
            failures.append({'kind': 'scope_failure', 'failure_kind': 'runtime', 'message': runtime})
        (sample / 'results.jsonl').write_text(json.dumps({'outcome': {
            'status': 'done', 'action_ledger': ledger, 'scope_failures': failures, 'host_events': []}}) + '\n')
        if job_code:
            (sample / 'jobs').mkdir()
            (sample / 'jobs' / '000000.error.json').write_text(json.dumps({'code': job_code}))
    (g / 'rewards.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return g


def _collection(path, groups, log=()):
    path.mkdir(parents=True)
    (path / 'runs.jsonl').write_text(json.dumps({'started': 1.0}) + '\n')
    for i, spec in enumerate(groups):
        _group(path, i, **spec)
    (path / 'collect.log').write_text(''.join(json.dumps(e) + '\n' for e in log))
    return path


SFP = frozenset({'semantic_merge', 'inline_nl'})
OK = ['task_success', 'semantic_mismatch', 'task_success', 'task_success']


def _healthy(n=3, family='auth_merges', constructs=('semantic_merge',)):
    return [dict(task=f'{family}:{i}', family=family, outcomes=OK, constructs=constructs) for i in range(n)]


def test_a_family_that_collapses_to_zero_is_flagged_with_its_errors_and_marked_sfp(tmp_path):
    before = _collection(tmp_path / 'v0', [*_healthy(), *_healthy(family='ext_qa', constructs=())])
    broken = ['semantic_mismatch'] * 4
    now = _collection(tmp_path / 'v1', [
        *[dict(task=f'auth_merges:{i}', family='auth_merges', outcomes=broken, constructs=('semantic_merge',),
               runtime='Error: No natlang task is active here') for i in range(3)],
        *_healthy(family='ext_qa', constructs=())])
    health = ch.report([now], [before], [], cache_dir=tmp_path / 'cache', sfp=SFP)
    fam = health['current']['families']['auth_merges']
    assert (fam['scored'], fam['successes'], fam['pass_rate']) == (12, 0, 0.0)
    assert fam['categories'] == {'runtime_error': 12}
    assert fam['reasons']['runtime:Error: No natlang task is active'] == 12
    kinds = {(f['flag'], f.get('family') or f.get('construct') or f.get('reason')): f for f in health['flags']}
    collapse = kinds[('family_collapse', 'auth_merges')]
    assert collapse['sfp'] and collapse['before']['pass_rate'] == 0.75
    assert kinds[('construct_collapse', 'semantic_merge')]['sfp']
    assert ('family_collapse', 'ext_qa') not in kinds            # the healthy family is not flagged
    assert health['sfp_flagged']
    # the runtime reason did not exist before: its share jumps from 0
    assert kinds[('error_share_jump', 'runtime:Error: No natlang task is active')]['share'] == 0.5
    assert health['compared']['families']['auth_merges'] == {'pass_rate': 0.0, 'scored': 12, 'previous': 0.75,
                                                             'earlier': None}
    line = ch.summary_line(health)
    assert 'family_collapse(auth_merges)[sfp]' in line and '24 own episodes' in line
    assert all(' COLLECTION HEALTH aei r0002: ' in a for a in ch.alert_lines(health, 'aei r0002'))


def test_a_new_family_that_never_passes_and_tasks_without_episodes_are_flagged(tmp_path):
    before = _collection(tmp_path / 'v0', _healthy(family='ext_qa', constructs=()))
    now = _collection(tmp_path / 'v1', [
        *_healthy(family='ext_qa', constructs=()),
        *[dict(task=f'auth_std:{i}', family='auth_std', outcomes=['invalid_action'] * 4, diag='typescript-syntax')
          for i in range(2)],
        # memetic: every episode unscored (a refused binding, a wall clock): the group yields no episode
        dict(task='memetic:a', family='memetic_improve', outcomes=['unresolved'] * 4),
        # an abandoned group (no rewards) and a still-running one
    ], log=[{'event': 'group abandoned', 'group': 6, 'episode': 'k', 'returncode': -9}])
    for index in (6, 7):
        g = now / 'groups' / f'g{index:06d}'
        g.mkdir()
        (g / 'cases.jsonl').write_text(json.dumps({'id': f'pend:{index}', 'family': 'ext_qa'}) + '\n')
    health = ch.report([now], [], [before], sfp=SFP)
    kinds = {(f['flag'], f.get('family') or f.get('construct')) for f in health['flags']}
    assert ('family_never_passes', 'auth_std') in kinds
    assert ('family_never_passes', 'memetic_improve') not in kinds    # nothing scored: no pass rate to judge
    no_eps = {t['task_id']: t for t in health['current']['no_episodes']}
    assert set(no_eps) == {'memetic:a', 'pend:6'}
    assert no_eps['memetic:a']['reasons'] == ['outcome:unresolved'] and no_eps['pend:6']['state'] == 'abandoned'
    assert health['current']['pending_groups'] == 1
    infra = health['current']['infrastructure']
    assert infra['unscored_episodes'] == 4 and infra['abandoned_groups'] == 1 and infra['rate'] == round(5 / 25, 4)
    flags = {f['flag'] for f in health['flags']}
    assert {'infrastructure', 'no_episodes'} <= flags
    assert not health['sfp_flagged']


def test_guided_attempts_and_budget_stops_are_counted_apart(tmp_path):
    now = _collection(tmp_path / 'v1', [
        dict(task='t', family='f', outcomes=['budget_exhaustion', 'task_success', 'task_success', 'task_success'],
             guided=(2, 3), job_code='NATLANG_MODEL_REQUEST_BUDGET')])
    agg = ch.aggregate([ch.collection_summary(now)])
    fam = agg['families']['f']
    assert (fam['episodes'], fam['scored'], fam['successes'], fam['guided'], fam['guided_successes']) == (2, 2, 1, 2, 2)
    assert fam['categories'] == {'budget_stop': 1, 'success': 1}


def test_scored_groups_are_cached_and_pending_ones_are_read_again(tmp_path):
    work = _collection(tmp_path / 'v1', _healthy(n=1))
    cache = tmp_path / 'cache'
    first = ch.collection_summary(work, cache)
    (work / 'groups' / 'g000000' / 'rewards.jsonl').write_text('')        # a cached group is not read again
    assert ch.collection_summary(work, cache)['groups'] == first['groups']


def test_without_history_only_failing_families_with_errors_are_flagged(tmp_path):
    now = _collection(tmp_path / 'v1', [
        *[dict(task=f'hard:{i}', family='hard', outcomes=['semantic_mismatch'] * 4) for i in range(3)],
        *[dict(task=f'std:{i}', family='std', outcomes=['semantic_mismatch'] * 4,
               runtime='TypeError: x is not a function') for i in range(3)]])
    health = ch.report([now], [], [], sfp=SFP)
    assert [(f['flag'], f.get('family')) for f in health['flags']] == [('family_never_passes', 'std')]


# ---- held-v4 cadence ------------------------------------------------------------------------------------------------
def test_held_v4_cadence_default_every_third_round_significant_verdicts_and_sfp_health():
    import expert_iteration as ei
    c = ei.parse_held_v4_cadence(ei.HELD_V4_CADENCE)
    assert c == {'every': 3, 'triggers': {'significant', 'health-sfp'}}
    assert [n for n in range(1, 10) if ei.held_v4_reasons(c, round_number=n)] == [3, 6, 9]
    assert ei.held_v4_reasons(c, verdict={'verdict': 'success', 'vs_start': {'p': 0.01}}) == ['significant:success']
    assert ei.held_v4_reasons(c, verdict={'verdict': 'failure', 'vs_start': {'p': 0.04}}) == ['significant:failure']
    assert ei.held_v4_reasons(c, verdict={'verdict': 'neutral', 'vs_start': {'p': 0.3}}) == []
    assert ei.held_v4_reasons(c, health={'sfp_flagged': True}) == ['health-sfp']
    assert ei.held_v4_reasons(c, health={'sfp_flagged': False}) == []
    off = ei.parse_held_v4_cadence('off')
    assert ei.held_v4_reasons(off, round_number=3, verdict={'vs_start': {'p': 0.0}}, health={'sfp_flagged': True}) == []
    assert ei.parse_held_v4_cadence('every=2') == {'every': 2, 'triggers': set()}
    import pytest
    with pytest.raises(ValueError):
        ei.parse_held_v4_cadence('weekly')


def test_held_v4_queue_merges_reasons_and_measures_the_newest_first(tmp_path):
    import expert_iteration as ei
    q = ei.HeldV4Queue(tmp_path / 'q.json')
    q.add({'sha256': '1' * 64, 'checkpoint': '/a.pt'}, ['every-3:round-3'])
    q.add({'sha256': '2' * 64, 'checkpoint': '/b.pt'}, ['health-sfp'])
    q.add({'sha256': '1' * 64, 'checkpoint': '/a.pt'}, ['significant:success'])
    assert q.add({'sha256': '3' * 64, 'checkpoint': '/c.pt'}, []) is None
    assert q.next()['sha256'] == '2' * 64
    q.finish('2' * 64, 'measured')
    again = ei.HeldV4Queue(tmp_path / 'q.json')         # persisted
    assert again.next() == {**again.next(), 'sha256': '1' * 64,
                            'reasons': ['every-3:round-3', 'significant:success']}


def _driver(tmp_path, monkeypatch, *argv):
    sys.path.insert(0, str(REPO / 'tests/neuralese'))
    from test_async_train import _driver as make
    return make(tmp_path, monkeypatch, argv=list(argv))


def test_the_driver_measures_held_v3_on_pop_by_default(tmp_path, monkeypatch):
    ei, d = _driver(tmp_path, monkeypatch)
    assert d.c.held_host == 'pop' and d.c.a.held_samples == 2
    ei, d = _driver(tmp_path / 'b', monkeypatch, '--held-profile', 'dgx')
    assert d.c.held_host == 'dgx' and d.c.a.held_samples == 4


def test_held_v4_runs_only_while_no_held_v3_job_waits_and_yields_to_one(tmp_path, monkeypatch):
    ei, d = _driver(tmp_path, monkeypatch)
    calls = []

    def measure(out, checkpoint, *, pin, samples, manifest=None, preempt=None, poll=30):
        calls.append((Path(out).name, samples, preempt()))
        if preempt():
            return 'preempted'
        Path(out).mkdir(parents=True, exist_ok=True)
        (Path(out) / 'result.json').write_text(json.dumps({'complete': True, 'success_rate': 0.4}))
        return 'complete'
    monkeypatch.setattr(ei, 'pop_held_measure', measure)
    monkeypatch.setattr(d, 'held_v4_manifest', lambda: Path('/held-v4/manifest.json'))
    cur = d.controller.current
    d.queue_held_v4(cur, ['every-3:round-3'])
    # a held-v3 job is due: held-v4 yields (the measurement stops and stays queued)
    d.controller.state['measuring'] = {'candidate': {'sha256': '9' * 64}, 'reference': {}}
    d.held_v4_step()
    assert calls == [(f'hv4-{cur["sha256"][:12]}', 2, True)] and d.held_v4.next()['sha256'] == cur['sha256']
    d.held_done.add('9' * 64)                     # the held-v3 job finished
    d.held_v4_step()
    assert d.held_v4.next() is None
    record = json.loads((d.root / 'held-v4' / f'hv4-{cur["sha256"][:12]}' / 'record.json').read_text())
    assert record['result']['success_rate'] == 0.4 and record['reasons'] == ['every-3:round-3']


def _published(d, n):
    """``n`` published rounds on top of the driver's lineage: [(sha, checkpoint)]."""
    out = []
    for i in range(1, n + 1):
        sha = f'{i:x}' * 64
        rid = d.controller.begin_round(d.controller.current['sha256'])
        d.controller.record_round_published(rid, f'/r{i}.pt', sha[:64], base_sha256=d.controller.current['sha256'])
        out.append({'sha256': sha[:64], 'checkpoint': f'/r{i}.pt'})
    return out


def test_a_rollback_drops_held_work_of_policies_that_left_the_lineage(tmp_path, monkeypatch):
    """2026-11-24 19:12: a neutral verdict rolled the lineage back from round 7 to round 1; held-v4 entries of rounds
    3 and 6 stayed pending and round 6's held-v4 kept measuring on Pop for 1.5 h. A lineage change closes pending
    held-v4 entries of dropped policies, cancels a held-v3 measurement of a dropped candidate (its units included),
    and a dropped policy is not queued again."""
    import pop_rl_actor
    ei, d = _driver(tmp_path, monkeypatch)
    rounds = _published(d, 3)
    d.queue_held_v4(rounds[0], ['every-1:round-1'])
    d.queue_held_v4(rounds[2], ['every-1:round-3'])
    d.controller.state['measuring'] = {'candidate': rounds[2], 'reference': d.controller.state['reference_policy']}
    stopped, actors = [], []
    monkeypatch.setattr(d.c, 'stop_unit', lambda unit: stopped.append(unit))
    monkeypatch.setattr(ei, 'stop_pop_held_units', lambda out: stopped.append(f'pop:{Path(out).name}'))
    monkeypatch.setattr(pop_rl_actor, 'stop_pop_actor_serving', lambda sha: actors.append(sha) or True)
    # the lineage returns to round 1 (a rollback to the reference adopted there)
    d.controller.state['lineage'] = d.controller.state['lineage'][:2]
    d.drop_off_lineage()
    assert d.held_v4.next()['sha256'] == rounds[0]['sha256']
    dropped = [e for e in d.held_v4.entries if e['state'] == 'dropped']
    assert [e['sha256'] for e in dropped] == [rounds[2]['sha256']]
    assert d.controller.state['measuring'] is None and d.held_v3_pending() is None
    name = f'policy-{rounds[2]["sha256"][:12]}'
    assert name in d.c.held_cancelled and f'aei-held-{name}' in stopped and f'pop:{name}' in stopped
    assert actors == [rounds[2]['sha256']]
    d.queue_held_v4(rounds[2], ['significant:failure'])          # a dropped policy is not measured
    assert not d.held_v4.pending(rounds[2]['sha256'])
    # a cancelled DGX held measurement is not launched (or relaunched)
    launched = []
    monkeypatch.setattr(d.c, 'ledger', lambda *a, **kw: launched.append(a))
    monkeypatch.setattr(d.c, 'held_measuring', lambda out: False)
    monkeypatch.setattr(d.c, 'reuse_held', lambda out, ckpt: False)
    monkeypatch.setattr(ei, 'held_budget_gb', lambda slots: 1)
    monkeypatch.setattr(d.c, 'record_cost', lambda *a, **kw: None)
    import natlang_neuralese.common.hashing as hashing
    monkeypatch.setattr(hashing, 'sha256_file_hex_cached', lambda p: rounds[2]['sha256'])
    d.c.dgx_held(name, '/r3.pt')
    assert launched == []


def test_a_running_held_v4_measurement_stops_when_its_policy_leaves_the_lineage(tmp_path, monkeypatch):
    import pop_rl_actor
    ei, d = _driver(tmp_path, monkeypatch)
    (r1,) = _published(d, 1)
    d.queue_held_v4(r1, ['every-1:round-1'])
    actors = []
    monkeypatch.setattr(pop_rl_actor, 'stop_pop_actor_serving', lambda sha: actors.append(sha) or True)
    monkeypatch.setattr(d, 'held_v4_manifest', lambda: Path('/held-v4/manifest.json'))

    def measure(out, checkpoint, *, pin, samples, manifest=None, preempt=None, poll=30):
        assert not preempt()
        d.controller.state['lineage'] = d.controller.state['lineage'][:1]     # a rollback meanwhile
        d.drop_off_lineage()
        assert preempt()                                                      # the poll stops it
        return 'preempted'
    monkeypatch.setattr(ei, 'pop_held_measure', measure)
    d.held_v4_step()
    assert actors == [r1['sha256']] and d.held_v4.next() is None
    log = [json.loads(line) for line in (d.root / 'drive.log').read_text().splitlines()]
    assert any(e.get('held_v4_stopped') == r1['sha256'] and e.get('pop_actor_stopped') for e in log)


def test_a_rounds_health_report_is_written_logged_alerted_and_queues_held_v4_on_sfp(tmp_path, monkeypatch):
    alerts = tmp_path / 'alerts.log'
    ei, d = _driver(tmp_path, monkeypatch, '--alerts-log', str(alerts))
    cur = d.controller.current
    work = d.collect_dir(cur)
    _collection(work, [dict(task=f'm:{i}', family='auth_merges', outcomes=['semantic_mismatch'] * 4,
                            constructs=('semantic_merge',), runtime='Error: No natlang task is active here')
                       for i in range(3)])
    rd = d.root / 'r0001'
    rd.mkdir()
    health = d.health_report(rd, d.lineage_policies())
    assert (rd / 'health.json').exists() and health['sfp_flagged'] and health['policy'] == cur['sha256']
    log = [json.loads(line) for line in (d.root / 'drive.log').read_text().splitlines()]
    assert any('health' in e and 'family_never_passes(auth_merges)[sfp]' in e['health'] for e in log)
    assert 'COLLECTION HEALTH aei r0001: family_never_passes' in alerts.read_text()
    assert d.held_v4.next()['reasons'] == ['health-sfp'] and d.held_v4.next()['sha256'] == cur['sha256']
    assert d.health_report(rd, d.lineage_policies()) is None           # once per round


def test_a_memetic_group_its_preparation_drops_is_not_admitted_or_a_failure(tmp_path):
    now = _collection(tmp_path / 'v1', _healthy(n=1), log=[
        {'event': 'memetic group dropped', 'group': 1, 'task_id': 'memetic:a', 'reason': 'not admitted'},
        {'event': 'memetic group dropped', 'group': 2, 'task_id': 'memetic:b', 'reason': 'preparation failed'}])
    for index, task in ((1, 'memetic:a'), (2, 'memetic:b')):
        g = now / 'groups' / f'g{index:06d}'
        g.mkdir()
        (g / 'cases.jsonl').write_text(json.dumps({'id': task, 'family': 'memetic_improve'}) + '\n')
    agg = ch.aggregate([ch.collection_summary(now)])
    assert agg['memetic_not_admitted'] == 1 and agg['pending_groups'] == 0
    assert [(t['task_id'], t['state']) for t in agg['no_episodes']] == [('memetic:b', 'memetic_preparation_failed')]


def test_resumed_groups_and_rerun_episodes_are_telemetry_never_infrastructure_losses(tmp_path):
    """grpo Collector.resume_unfinished: a resumed group scores its full k; its stopped and lost attempts are set aside
    (``sample-NN.stopped-N``, ``.lost-N``) and never enter the rate. A group a restart left as superseded (its task was
    measured in another group) is no loss; one left because its policy changed is abandoned."""
    now = _collection(tmp_path / 'v1', _healthy(n=2))
    g = now / 'groups' / 'g000000'
    (g / 'row-000' / 'sample-01.stopped-1').mkdir()
    (g / 'row-000' / 'sample-02.lost-1').mkdir()
    (g / 'resumed.jsonl').write_text(json.dumps({'kept': ['a', 'b'], 'continued': ['c'], 'rerun': ['d']}) + '\n')
    for index, reason in ((2, 'superseded: task measured in another group'), (3, 'policy changed')):
        d = now / 'groups' / f'g{index:06d}'
        d.mkdir()
        (d / 'cases.jsonl').write_text(json.dumps({'id': f'auth_merges:{index}', 'family': 'auth_merges'}) + '\n')
        (d / 'lost.json').write_text(json.dumps({'reason': reason}))
    agg = ch.aggregate([ch.collection_summary(now)])
    infra = agg['infrastructure']
    assert infra['resumed_groups'] == 1 and infra['resumed_kept_episodes'] == 2 and infra['resumed_continued_episodes'] == 1
    assert infra['resumed_restarted_episodes'] == 1 and infra['rerun_lost_episodes'] == 1
    assert infra['superseded_groups'] == 1 and infra['abandoned_groups'] == 1 and agg['pending_groups'] == 0
    assert infra['rate'] == round(1 / 9, 4)               # 8 scored episodes + the abandoned group; nothing else
    assert [t['state'] for t in agg['no_episodes']] == ['abandoned']


def test_earlier_collections_come_only_from_caches_and_a_finished_collection_closes_its_cache(tmp_path):
    """2026-11-24: the report read every earlier collection's groups, v0000's 45k files tiered to the HDD at ~350 ms,
    and held the driver 25 min. Earlier collections now come from their caches only (a closed whole summary, else the
    scored groups an older cache holds); one without a cache is listed as uncached and never read. The previous
    collection is finished: its cache closes while it is still on the NVMe, and a closed cache is not read again."""
    cache = tmp_path / 'cache'
    now = _collection(tmp_path / 'v3', _healthy(n=2))
    prev = _collection(tmp_path / 'v2', _healthy(n=2))
    old = _collection(tmp_path / 'v1', _healthy(n=2))
    ch.collection_summary(old, cache)                    # an older (open) group cache of v1
    gone = _collection(tmp_path / 'v0', _healthy(n=2))   # never cached
    health = ch.report([now], [prev], [old, gone], cache_dir=cache, sfp=SFP, close_previous=True,
                       earlier_cached_only=True)
    assert health['collections']['uncached'] == [str(gone)]
    assert health['earlier']['overall']['scored'] == 8
    import shutil
    shutil.rmtree(old / 'groups')                        # the cache answers for v1 without its groups
    shutil.rmtree(prev / 'groups')                       # v2's cache was closed: a later report reads none of it
    again = ch.report([now], [], [prev, old], cache_dir=cache, sfp=SFP, earlier_cached_only=True)
    assert again['earlier']['overall']['scored'] == 16 and again['collections']['uncached'] == []


def test_an_incomplete_held_v4_attempt_of_a_policy_off_the_lineage_is_never_retried(tmp_path, monkeypatch):
    """2026-11-24 20:53: r6's entry, removed from queue.json on disk, was still pending in the running driver's memory;
    it was measured again and, stopped by hand, logged attempt 2 and stayed due. The lineage decides: an entry whose
    policy left it is dropped before an attempt, and after an incomplete one, never retried."""
    ei, d = _driver(tmp_path, monkeypatch)
    (r1,) = _published(d, 1)
    d.queue_held_v4(r1, ['every-1:round-1'])
    monkeypatch.setattr(d, 'held_v4_manifest', lambda: Path('/held-v4/manifest.json'))
    calls = []

    def measure(out, checkpoint, *, pin, samples, manifest=None, preempt=None, poll=30):
        calls.append(1)
        d.controller.state['lineage'] = d.controller.state['lineage'][:1]     # rolled back, nothing pruned yet
        return 'incomplete'
    monkeypatch.setattr(ei, 'pop_held_measure', measure)
    d.held_v4_step()
    assert calls == [1] and d.held_v4.next() is None
    assert [e['state'] for e in d.held_v4.entries] == ['dropped'] and d.held_v4.entries[0]['attempts'] == 0
    # an entry in memory for a policy off the lineage (a hand-edited file does not reach it) is not measured
    d.held_v4.entries.append({**d.held_v4.entries[0], 'state': 'pending'})
    d.held_v4_step()
    assert calls == [1] and d.held_v4.next() is None
