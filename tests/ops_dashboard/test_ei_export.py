import importlib.util
import json
import time
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'ei_export', Path(__file__).resolve().parents[2] / 'scripts' / 'ops_dashboard' / 'ei_export.py')
ei = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ei)

SHA = {v: f'{v}' * 64 for v in 'abcd'}       # a = reference (v0), b = v1, c = v2


def lines(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(''.join((r if isinstance(r, str) else json.dumps(r)) + '\n' for r in recs))


def phase():
    return {'lineage': [{'sha256': SHA['a'], 'version': 0, 'round_id': None, 'checkpoint': '/x/sft-v9/sft/checkpoint.pt'},
                        {'sha256': SHA['b'], 'version': 1, 'round_id': 'r0001'}, {'sha256': SHA['c'], 'version': 2, 'round_id': 'r0002'}],
            'reference_policy': {'sha256': SHA['a'], 'checkpoint': '/x/sft-v9/sft/checkpoint.pt'},
            'measuring': {'candidate': {'sha256': SHA['b'], 'version': 1}, 'reference': {'sha256': SHA['a']}},
            'active_round': {'round_id': 'r0003', 'phase': 'sft'}, 'held_every': 1, 'since_held': 2}


def test_unit_roles():
    assert ei.unit_role('aei1-collect-v0003.service') == {'role': 'collect', 'where': 'dgx', 'title': 'Collection of v3 on the DGX'}
    assert ei.unit_role('aei1-collect-v0002-remote0.service')['where'] == 'colab'
    assert ei.unit_role('aei1-collect-v0002-pop.service')['where'] == 'pop'
    assert ei.unit_role('aei1-collect-v0003-fill.service')['role'] == 'fill'
    assert ei.unit_role('aei1-r0005-data.service')['title'] == 'Round r5: data and build'
    assert ei.unit_role('popheld-policy-e15b-17955.service')['role'] == 'held'
    assert ei.unit_role('progdev-ext_x-base-1.service')['role'] == 'tuning'
    assert ei.unit_role('something-else')['role'] == 'other'


def test_lineage_labels():
    lin = ei.Lineage(phase())
    assert lin.ver(SHA['b']) == 1 and lin.latest == 2 and lin.rounds == {'r0001': 1, 'r0002': 2}
    assert lin.label(SHA['a']) == 'v0 (sft-v9)' and lin.label(SHA['c']) == 'v2'


def test_drive_log_events_collapse_repeats_and_stay_incremental(tmp_path):
    log = tmp_path / 'drive.log'
    held_fail = {'time': '2026-11-24T06:05:15', 'stage': 'async', 'held_failed': SHA['b'], 'error': "CalledProcessError(1, ['x/held_eval', 'compare', 'a'])"}
    lines(log, [{'time': '2026-11-24T01:00:00', 'stage': 'pin', 'done': True, 'pin': '/x/pin-d29451b846ca'},
                {'time': '2026-11-24T01:42:52', 'stage': 'async', 'round': 'r0001', 'published': SHA['b'], 'version': 1},
                {'time': '2026-11-24T02:00:00', 'stage': 'async', 'round': 'r0002', 'phase': 'sft', 'policy': SHA['b'], 'version': 1},
                {'time': '2026-11-24T02:00:05', 'stage': 'async', 'round': 'r0002', 'phase': 'sft', 'policy': SHA['b'], 'version': 1},
                held_fail, held_fail, {**held_fail, 'time': '2026-11-24T06:15:15'},
                {'time': '2026-11-24T02:00:09', 'stage': 'async', 'reader_sources': {}}, 'not json'])
    st = {}
    ei.process_drive_log(st, str(log), ei.Lineage(phase()), time.time())
    kinds = [e['kind'] for e in st['events']]
    assert kinds == ['driver', 'publish', 'round', 'held']                       # a repeated phase line and the repeated failures are one event each
    assert st['gate_fail']['count'] == 3 and 'not comparable' in st['gate_fail']['error']
    assert st['events'][1]['text'] == 'Published v1 from r0001' and 'v1 failed' in st['events'][3]['text']
    lines(log, [{'time': '2026-11-24T07:00:00', 'stage': 'async', 'held_failed': SHA['b'], 'error': 'other failure'}])
    ei.process_drive_log(st, str(log), ei.Lineage(phase()), time.time())
    assert st['gate_fail']['count'] == 1 and len(st['events']) == 5               # only the new line was read


def test_sft_tail_eta_and_downsampling(tmp_path):
    p = tmp_path / 'train.jsonl'
    recs = [{'step': 0, 'val_loss:v': 0.3}]
    recs += [{'step': i, 'of': 100, 'loss': 0.5 - i * 0.001, 'loss_kind': 'sft', 'elapsed_s': 10.0 * i, 'chain_tokens': 1000, 'lr': 1e-5} for i in range(1, 61)]
    recs += [{'step': 61, 'of': 100, 'loss': 6.5, 'loss_kind': 'tm', 'elapsed_s': 610.0, 'chain_tokens': 1000}]
    lines(p, recs)
    st = {}
    ei.tail_sft(st, str(p))
    now = 1_000_000.0
    s = ei.sft_summary(st, now - 700, True, now)
    assert s['step'] == 61 and s['of'] == 100 and s['s_per_step'] == 10.0 and s['tm'] == [[61, 6.5]] and s['val'] == [[0, 0.3]]
    assert s['eta_ts'] == int((now - 700 + 610 + 39 * 10) * 1000)               # launch + elapsed at the last step + remaining steps at the recent pace
    assert s['loss_first'] > s['loss_last']
    lines(p, [{'step': 62, 'of': 100, 'loss': 0.4, 'loss_kind': 'sft', 'elapsed_s': 620.0, 'chain_tokens': 1000}])
    ei.tail_sft(st, str(p))
    assert st['step'] == 62 and len(st['loss']) == 61
    assert len(ei.bucket([[i, float(i)] for i in range(1000)], 70)) == 70
    assert ei.sft_summary(st, now - 700, False, now)['eta_ts'] is None


def test_collection_counts_scored_episodes_and_hourly_buckets(tmp_path):
    run = tmp_path / 'aei1'
    now = time.time()
    for name, host_dir in (('v0003-' + 'd' * 12, 'dgx'), ('v0003-' + 'd' * 12 + '-pop', 'pop')):
        for g, (rollouts, scored, succ) in enumerate([(4, 4, 3), (4, 2, 1)]):
            lines(run / 'collect' / name / 'groups' / f'g{g:06d}' / 'score.json', [json.dumps({'rollouts': rollouts, 'scored': scored, 'successes': succ})])
    lin = ei.Lineage(phase())
    out = ei.collection_section({}, str(run), lin, {'aei1-collect-v0003.service'}, now)
    pol = out['policies'][0]
    assert pol['version'] == 3 and pol['episodes'] == 16 and pol['scored'] == 12 and pol['successes'] == 8 and pol['pass_rate'] == 0.667
    assert pol['hosts']['pop']['episodes'] == 8
    assert [a['host'] for a in out['actors']] == ['dgx'] and out['actors'][0]['rate_h'] == 8        # only the DGX collector is running
    assert sum(b['dgx'] + b['pop'] for b in out['hourly']) == 16


def held_result(path, sha, series, rates, finished):
    samples = []
    for i, r in enumerate(rates):
        for s in range(2):
            samples.append({'task_id': f't{i}', 'sample': s, 'ran': True, 'outcome': 'task_success' if s < r else 'semantic_mismatch', 'reward': 1.0 if s < r else 0.0})
    ok = sum(1 for x in samples if x['outcome'] == 'task_success')
    lines(path, [json.dumps({'policy_id': sha, 'series': {'key': series}, 'success_rate': ok / len(samples), 'successes': ok, 'scored': len(samples), 'ran': len(samples),
                             'tasks': len(rates), 'samples_per_task': 2, 'finished_at': finished, 'outcome_counts': {'task_success': ok}, 'engine': {'dtype': 'bfloat16'}, 'samples': samples})])


def test_held_gate_pairs_only_one_series(tmp_path, monkeypatch):
    run = tmp_path / 'ei' / 'aei1'
    monkeypatch.setattr(ei, 'EI_ROOT', str(tmp_path / 'ei'))
    base = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    held_result(run / 'held' / 'policy-bbbbbbbbbbbb' / 'result.json', SHA['b'], 'S2', [min(2, r + 1) for r in base], 2000.0)
    held_result(tmp_path / 'ei' / 'sft-v9' / 'held' / 'candidate' / 'result.json', SHA['a'], 'S1', base, 1000.0)
    lin = ei.Lineage(phase())
    log = {'gate_fail': {'error': 'x', 'count': 4, 'first_ts': 1.0, 'last_ts': 2.0}}
    out = ei.held_section({}, str(run), lin, log, phase(), time.time())
    assert [r['label'] for r in out['results']] == ['v0 (sft-v9)', 'v1']
    g = out['gate']
    assert g['candidate'] == 'v1' and g['reference_same_series'] is False and g['pair'] is None and g['failing']['count'] == 4
    # the reference re-measured on the candidate's series pairs them
    held_result(run / 'held' / 'policy-aaaaaaaaaaaa' / 'result.json', SHA['a'], 'S2', base, 3000.0)
    out = ei.held_section({}, str(run), lin, log, phase(), time.time())
    pair = out['gate']['pair']
    assert out['gate']['reference_same_series'] is True and pair['tasks'] == 12 and pair['delta_pts'] > 0 and pair['up'] > pair['down']


def test_held_in_progress_reads_pop_log(tmp_path, monkeypatch):
    run = tmp_path / 'ei' / 'aei1'
    monkeypatch.setattr(ei, 'EI_ROOT', str(tmp_path / 'ei'))
    lines(run / 'held' / 'policy-aaaaaaaaaaaa' / 'pop-held.log', ['noise', json.dumps({'held_eval_progress': 128, 'of': 512, 'minutes': 60.0})])
    out = ei.held_section({}, str(run), ei.Lineage(phase()), {}, phase(), time.time())
    p = out['in_progress'][0]
    assert p['label'] == 'v0 (sft-v9)' and p['done'] == 128 and p['eta_s'] == 10800


def test_teacher_campaign_summary(tmp_path):
    pool = tmp_path / 'teacher-r0005'
    mk = lambda task, model, outcomes, **kw: {'task_id': task, 'family': 'fam', 'teacher': {'model_id': model, 'provider': 'p', 'reasoning': 'summary'}, 'seconds': 10.0,
                                              'episodes': [{'outcome': o, 'completion_tokens': 5} for o in outcomes], **kw}
    lines(pool / 'manifest.jsonl', [mk('t1', 'm1', ['task_success', 'semantic_mismatch']), mk('t2', 'm1', ['semantic_mismatch']), mk('t1', 'm2', ['task_success'])])
    lines(pool / 'deferred.jsonl', [mk('t3', 'm3', ['unresolved'])])
    lines(pool / 'driver-launch.json', [json.dumps({'at': 1.0, 'tasks': 5})])
    out = ei.summarize_campaign(str(pool), {}, time.time())
    assert out['tasks_done'] == 2 and out['tasks_won'] == 1 and out['success'] == 2 and out['eligible'] == 5 and out['deferred'] == 1
    assert [t['id'] for t in out['teachers']][:2] == ['m1', 'm2'] and out['teachers'][0]['success_rate'] == 0.333
    assert next(t for t in out['teachers'] if t['id'] == 'm3')['deferred'] == {'unresolved': 1}


def test_build_snapshot_and_alerts(tmp_path, monkeypatch):
    root = tmp_path / 'ops' / 'ei'
    run = root / 'aei1'
    for name in ('EI_ROOT',):
        monkeypatch.setattr(ei, name, str(root))
    monkeypatch.setattr(ei, 'OPS', str(tmp_path / 'ops'))
    monkeypatch.setattr(ei, 'REPO', str(tmp_path / 'repo'))
    monkeypatch.setattr(ei, 'WATCHDOG_STATE', str(tmp_path / 'wd.json'))
    monkeypatch.setattr(ei, 'colab_section', lambda st, run, now: {'balance': 30.0, 'rate': 10.0, 'runway_h': 3.0, 'at': 1})
    (run).mkdir(parents=True)
    (run / 'phase-state.json').write_text(json.dumps(phase()))
    lines(run / 'drive.log', [{'time': '2026-11-24T01:42:52', 'stage': 'async', 'round': 'r0001', 'published': SHA['b'], 'version': 1}])
    (run / 'r0001').mkdir()
    (run / 'r0001' / 'size.json').write_text(json.dumps({'hours': 2.0, 'max_target_tokens': 1000, 'decided_at': 5.0, 'gpu': 'A100'}))
    snap = ei.build({}, {'dgx': {'mem_used_gb': 1, 'gpu_util_percent': 3, 'cpu_percent': 4, 'ledger': {'headroom_gb': 20}, 'disks': [{'mount': '/', 'free_gb': 100}]}}, [], time.time())
    assert snap['run'] == 'aei1' and snap['phase']['latest_version'] == 2 and [r['id'] for r in snap['rounds']] == ['r0001'] and snap['rounds'][0]['state'] == 'published'
    assert snap['errors'] == [] and snap['driver']['running'] is False and snap['events'][0]['text'] == 'Published v1 from r0001'
    ids = {a['id']: a['level'] for a in ei.derived_alerts(snap, {}, {}, time.time())}
    assert ids['ei:driver'] == 'crit' and ids['ei:credits'] == 'crit'


def test_disk_fall_alert_uses_the_exporters_own_history():
    snap = {'driver': {'running': True}, 'disks': {'dgx:/mnt': {'rate_gb_h': 20.0}}, 'rounds': []}
    machines = {'dgx': {'disks': [{'mount': '/mnt', 'free_gb': 110}]}}
    alerts = ei.derived_alerts(snap, {'dgx': {'/mnt': 50}}, machines, time.time())
    assert [a['id'] for a in alerts] == ['ei:diskfall:dgx:/mnt'] and alerts[0]['level'] == 'warn' and '3.0 h' in alerts[0]['text']


def test_watchdog_alerts_are_listed_once_per_condition(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setattr(ei, 'OPS', str(tmp_path))
    monkeypatch.setattr(ei, 'WATCHDOG_STATE', str(tmp_path / 'wd.json'))
    (tmp_path / 'wd.json').write_text(json.dumps({'alerts': {'colab:tunnel:t': {'last': now, 'once': False}, 'units:storm:t': {'last': now, 'once': False},
                                                  'colab:balance': {'last': now, 'once': False}, 'health:x': {'last': now - 9 * 3600, 'once': True}},
                                                  'gpu': {'colab-g4a': {'util': 0, 'idle_s': 900.0, 'checked': now}}, 'last_run': now}))
    lines(tmp_path / 'alerts.jsonl', [{'key': k, 'check': k.split(':')[0], 'severity': 'critical', 'message': k, 'ts': now} for k in ('colab:tunnel:t', 'units:storm:t', 'colab:balance', 'health:x')])
    wd = ei.watchdog_section(now)
    assert sorted(a['key'] for a in wd['alerts']) == ['colab:balance', 'colab:tunnel:t', 'units:storm:t'] and wd['gpu']['colab-g4a']['idle_s'] == 900.0
    ids = [a['id'] for a in ei.derived_alerts({'driver': {'running': True}, 'rounds': [], 'watchdog': wd}, {}, {}, now)]
    assert ids == ['wd:colab:tunnel:t']          # the storm of the same unit and the exporter's own credit alert replace the rest


def test_stalled_collector_and_teacher_alerts():
    now = time.time()
    snap = {'driver': {'running': True}, 'rounds': [],
            'collection': {'actors': [{'host': 'dgx', 'unit': 'aei1-collect-v0003.service', 'version': 3, 'last_group_ts': int((now - 3000) * 1000)}]},
            'teacher': {'pool': 'teacher-r0005', 'eligible': 41, 'tasks_done': 20, 'last_activity_ts': int((now - 3 * 3600) * 1000)}}
    ids = [a['id'] for a in ei.derived_alerts(snap, {}, {}, now)]
    assert ids == ['ei:collect-stall:aei1-collect-v0003.service', 'ei:teacher-stall']
    snap['teacher']['tasks_done'] = 41          # a finished campaign is not stalled
    assert 'ei:teacher-stall' not in [a['id'] for a in ei.derived_alerts(snap, {}, {}, now)]


def test_open_items_list_unqualified_free_teacher_candidates(tmp_path, monkeypatch):
    monkeypatch.setattr(ei, 'OPS', str(tmp_path / 'ops'))
    monkeypatch.setattr(ei, 'REPO', str(tmp_path / 'repo'))
    lines(tmp_path / 'repo' / 'plans' / 'semantic-fp' / 'OPEN_ITEMS.md',
          ['## In flight', '| Item | Owner |', '|---|---|', '| fix x | agent |'])
    lines(tmp_path / 'ops' / 'teacher-free-models' / 'candidates.jsonl',
          [{'provider': 'openrouter', 'model': 'a/b:free', 'first_seen': '2026-11-25T08:00:00+0100', 'status': 'new'},
           {'provider': 'kilo', 'model': 'c-free', 'first_seen': '2026-11-25', 'status': 'qualified'}])
    rows = ei.open_items_section({})
    assert rows[0] == {'item': 'fix x', 'owner': 'agent'}
    assert len(rows) == 2 and 'a/b:free on openrouter (first seen 2026-11-25)' in rows[1]['item']
