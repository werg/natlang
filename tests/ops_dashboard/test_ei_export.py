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
    assert ei.unit_role('grpo-v12-1.service') == {'role': 'grpo', 'where': 'dgx', 'title': 'GRPO run (grpo-v12)'}
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
    lines(log, [{'time': '2026-11-24T08:00:00', 'stage': 'async', 'verdict': 'neutral', 'policy': SHA['b'], 'action': 'adopt'}])
    ei.process_drive_log(st, str(log), ei.Lineage(phase()), time.time())
    assert 'gate_fail' not in st and st['events'][-1]['text'] == 'Held verdict for v1: neutral, adopt'     # a verdict ends the failure streak


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


def test_held_gate_pairs_against_reference_and_fixed_anchor(tmp_path, monkeypatch):
    """held_eval.compare pairs across scoring series (flagged same_series False) and prefers the reference's measurement
    on the candidate's own series; every policy also carries its delta against the fixed anchor (the start policy)."""
    run = tmp_path / 'ei' / 'aei1'
    monkeypatch.setattr(ei, 'EI_ROOT', str(tmp_path / 'ei'))
    base = [1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1]
    held_result(run / 'held' / 'policy-bbbbbbbbbbbb' / 'result.json', SHA['b'], 'S2', [min(2, r + 1) if i % 3 else r for i, r in enumerate(base)], 2000.0)
    held_result(tmp_path / 'ei' / 'sft-v9' / 'held' / 'candidate' / 'result.json', SHA['a'], 'S1', base, 1000.0)
    lin = ei.Lineage(phase())
    log = {'gate_fail': {'error': 'x', 'count': 4, 'first_ts': 1.0, 'last_ts': 2.0}}
    out = ei.held_section({}, str(run), lin, log, {**phase(), 'consecutive_neutral': 3}, time.time())
    assert [r['label'] for r in out['results']] == ['v0 (sft-v9)', 'v1']
    g = out['gate']
    assert g['candidate'] == 'v1' and g['anchor'] == 'v0 (sft-v9)' and g['reference_same_series'] is False and g['failing']['count'] == 4
    assert g['pair']['same_series'] is False and g['pair']['delta_pts'] > 0 and g['neutral'] == 3
    v1 = out['results'][1]
    assert v1['vs_anchor']['anchor_series'] == 'S1' and v1['vs_anchor']['se_pts'] > 0 and out['results'][0]['vs_anchor'] is None
    assert out['no_gain']['best']['label'] == 'v1' and out['no_gain']['consecutive_neutral'] == 3
    # the reference re-measured on the candidate's series is the one paired
    held_result(run / 'held' / 'policy-aaaaaaaaaaaa' / 'result.json', SHA['a'], 'S2', base, 3000.0)
    out = ei.held_section({}, str(run), lin, log, phase(), time.time())
    pair = out['gate']['pair']
    assert out['gate']['reference_same_series'] is True and pair['same_series'] and pair['tasks'] == 12 and pair['up'] > pair['down']
    assert next(r for r in out['results'] if r['label'] == 'v1')['vs_anchor']['anchor_series'] == 'S2'
    # a partial measurement (incomplete) is listed but never paired
    held_result(run / 'held' / 'policy-cccccccccccc' / 'result.json', SHA['c'], 'S2', base[:4], 4000.0)
    rec = json.loads((run / 'held' / 'policy-cccccccccccc' / 'result.json').read_text())
    (run / 'held' / 'policy-cccccccccccc' / 'result.json').write_text(json.dumps({**rec, 'complete': False}))
    out = ei.held_section({}, str(run), lin, log, phase(), time.time())
    v2 = next(r for r in out['results'] if r['label'] == 'v2')
    assert v2['complete'] is False and v2['vs_anchor'] is None and all(p['b'] != 'v2' for p in out['pairs'])


def test_held_rate_counts_unscored_samples_as_failures(tmp_path):
    """8e634a0f: the stored success_rate (293/460) dropped 52 unscored samples; held_eval counts the unresolved ones as
    failures and leaves only infrastructure losses out (293/494); rate_all divides by every planned sample (293/512)."""
    samples = []
    for i in range(256):
        for k in range(2):
            n = 2 * i + k
            if n < 18:
                samples.append({'task_id': f't{i}', 'sample': k, 'ran': True, 'outcome': 'infrastructure_error', 'reward': None})
            elif n < 52:
                samples.append({'task_id': f't{i}', 'sample': k, 'ran': True, 'outcome': 'unresolved', 'reward': None})
            else:
                ok = n < 52 + 293
                samples.append({'task_id': f't{i}', 'sample': k, 'ran': True, 'outcome': 'task_success' if ok else 'semantic_mismatch', 'reward': 1.0 if ok else 0.0})
    p = tmp_path / 'result.json'
    p.write_text(json.dumps({'policy_id': 'x', 'series': {'key': 'S'}, 'success_rate': 0.637, 'successes': 293, 'scored': 460, 'tasks': 256,
                             'samples_per_task': 2, 'complete': True, 'manifest_sha256': '4547106069df', 'samples': samples}))
    r = ei.held_record(str(p))
    assert (r['successes'], r['counted'], r['scored'], r['unscored'], r['infra'], r['planned']) == (293, 494, 460, 34, 18, 512)
    assert r['rate'] == round(293 / 494, 4) and r['rate_all'] == round(293 / 512, 4) and r['stored_rate'] == 0.637 and r['set'] == 'held-v3'


def test_pairing_refuses_other_held_sets_and_engines(tmp_path):
    a = {'set': 'held-v3', 'law': 'L', 'engine_key': 'E'}
    assert ei.pairable(a, {**a}) is None and ei.pairable(a, {**a, 'engine_key': None}) is None
    assert ei.pairable(a, {**a, 'set': 'held-v4'}) == 'different held sets'
    assert ei.pairable(a, {**a, 'law': 'M'}) and ei.pairable(a, {**a, 'engine_key': 'F'}) == 'different engine settings'
    assert ei.held_set('/x/aei1/held-v4/hv4-abc/result.json', '4547106069') == 'held-v4'


def test_no_gain_alert():
    now = time.time()
    held = {'no_gain': {'consecutive_neutral': 3, 'anchor': 'v0 (sft-v9)', 'set': 'held-v3', 'beats_anchor': False,
                        'best': {'label': 'v8', 'rate': 0.61, 'vs_anchor': {'delta_pts': 3.5, 'ci_pts': [-0.8, 7.8], 'tasks': 256}}}}
    alerts = ei.derived_alerts({'driver': {'running': True}, 'rounds': [], 'held': held}, {}, {}, now)
    assert [a['id'] for a in alerts] == ['ei:no-gain'] and '3 consecutive neutral rounds' in alerts[0]['text'] and 'v8' in alerts[0]['text']
    held['no_gain']['beats_anchor'] = True
    assert ei.derived_alerts({'driver': {'running': True}, 'rounds': [], 'held': held}, {}, {}, now) == []


def test_supply_wait_is_what_the_driver_waits_on(tmp_path):
    log = tmp_path / 'drive.log'
    lines(log, [{'time': '2026-11-27T08:42:08', 'stage': 'async', 'round': 'r0016', 'decision': 'wait', 'minimum': 5000000, 'waited_hours': 11.55,
                 'supply': {'new_weighted_target_tokens': 2449476, 'weighted_target_tokens': 4073562, 'parts': {'teacher': 1}}},
                {'time': '2026-11-27T08:42:08', 'stage': 'async', 'round': 'r0016', 'supply_next_build_in_minutes': 45},
                {'time': '2026-11-27T08:42:09', 'stage': 'async', 'round': 'r0016', 'supply_next_build_in_minutes': 45}])
    st = {}
    ei.process_drive_log(st, str(log), ei.Lineage(phase()), time.time())
    w = ei.waiting_on(st, str(tmp_path), 'r0016', True, time.time())
    assert w['waiting_on'].startswith('r0016 waits for training data: 2.45 M of 5.0 M new weighted target tokens (waited 11.6 h); next build check')
    assert w['supply']['new_tokens'] == 2449476 and w['supply']['next_ts']
    assert [e['kind'] for e in st['events']] == ['round']
    assert ei.waiting_on(st, str(tmp_path), 'r0016', False, time.time())['waiting_on'] == 'the round driver is not running'
    (tmp_path / 'r0016' / 'sft').mkdir(parents=True)
    (tmp_path / 'r0016' / 'sft' / 'launch.json').write_text('{}')
    assert ei.waiting_on(st, str(tmp_path), 'r0016', True, time.time())['waiting_on'] is None


def test_held_record_leaves_infrastructure_losses_out_of_both_sides(tmp_path):
    """compare's rule in the held view: a lost episode drops its pair on both sides, another unscored one fails."""
    def result(path, lost, unresolved=()):
        samples = []
        for i in range(10):
            for s in range(2):
                ok = s == 0
                sample = {'task_id': f't{i}', 'sample': s, 'ran': True, 'outcome': 'task_success' if ok else 'semantic_mismatch',
                          'reward': 1.0 if ok else 0.0, 'infra_lost': None}
                if (i, s) in lost or (i, s) in unresolved:
                    sample.update(outcome='unresolved', reward=None, infra_lost='infrastructure_error' if (i, s) in lost else None)
                samples.append(sample)
        lines(path, [json.dumps({'policy_id': 'x', 'series': {'key': 'S'}, 'success_rate': 0.5, 'samples': samples})])
        return ei.held_record(str(path))
    ref = result(tmp_path / 'a.json', ())
    cand = result(tmp_path / 'b.json', {(i, 1) for i in range(10)})      # every failure of the candidate lost: 10/10
    assert cand['infra'] == 10 and cand['counted'] == 10
    json.dumps(cand)                                                           # cached in the JSON exporter state
    assert ei.paired(ref, cand)['delta_pts'] == 0
    model = result(tmp_path / 'c.json', (), {(i, 0) for i in range(10)})    # unscored, not lost: failures
    assert model['infra'] == 0 and ei.paired(ref, model)['delta_pts'] == -50.0


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


def test_teacher_section_marks_removed_and_paid_teachers(tmp_path, monkeypatch):
    """teachers.json is what runs: Luna/Sol rows stay in a standing campaign's history but are 'removed'; Vertex is paid;
    since_launch counts only rows after the launch's scored_at_launch."""
    root = tmp_path / 'ei'
    monkeypatch.setattr(ei, 'EI_ROOT', str(root))
    (root).mkdir()
    (root / 'teachers.json').write_text(json.dumps([{'model_id': 'nem', 'endpoint': 'https://integrate.api.nvidia.com/v1'},
                                                     {'model_id': 'google/gemini-4-argon', 'endpoint': 'https://aiplatform.googleapis.com/v1/x', 'api_key_env': 'VERTEX_ACCESS_TOKEN'}]))
    pool = root / 'aei1' / 'teacher-r0016'
    mk = lambda model, o: {'task_id': model + o, 'family': 'f', 'teacher': {'model_id': model}, 'episodes': [{'outcome': o}]}
    lines(pool / 'manifest.jsonl', [mk('gpt-6-luna', 'task_success'), mk('gpt-6-luna', 'semantic_mismatch'), mk('nem', 'task_success')])
    (pool / 'driver-launch.json').write_text(json.dumps({'at': 1.0, 'tasks': 44, 'scored_at_launch': 2}))
    out = ei.teacher_section({}, str(root / 'aei1'), time.time())
    by = {t['id']: t for t in out['teachers']}
    assert by['gpt-6-luna']['status'] == 'removed' and by['gpt-6-luna']['since_launch'] == {'episodes': 0, 'success': 0}
    assert by['nem']['status'] == 'configured' and by['nem']['since_launch'] == {'episodes': 1, 'success': 1}
    assert by['google/gemini-4-argon']['paid'] and by['google/gemini-4-argon']['episodes'] == 0


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
    # the roster names only another experiment: aei1 is paused and its stopped driver raises nothing
    (tmp_path / 'ops' / 'watchdog.json').write_text(json.dumps({'experiments': [{'name': 'grpo-v12-1'}]}))
    snap = ei.build({}, {}, [], time.time())
    assert snap['driver']['expected'] is False and 'paused' in snap['phase']['waiting_on']
    assert 'ei:driver' not in {a['id'] for a in ei.derived_alerts(snap, {}, {}, time.time())}


def test_disk_fall_alert_uses_the_exporters_own_history():
    snap = {'driver': {'running': True}, 'disks': {'dgx:/mnt': {'rate_gb_h': 20.0}}, 'rounds': []}
    machines = {'dgx': {'disks': [{'mount': '/mnt', 'free_gb': 110}]}}
    alerts = ei.derived_alerts(snap, {'dgx': {'/mnt': 50}}, machines, time.time())
    assert [a['id'] for a in alerts] == ['ei:diskfall:dgx:/mnt'] and alerts[0]['level'] == 'warn' and '3.0 h' in alerts[0]['text']


def test_watchdog_alerts_are_listed_once_per_condition(tmp_path, monkeypatch):
    now = time.time()
    monkeypatch.setattr(ei, 'OPS', str(tmp_path))
    monkeypatch.setattr(ei, 'WATCHDOG_STATE', str(tmp_path / 'wd.json'))
    (tmp_path / 'watchdog.json').write_text(json.dumps({'colab': {'sessions': [{'session': 's', 'gpu': 'colab-g4a', 'tunnel_unit': 't'}]}}))
    (tmp_path / 'wd.json').write_text(json.dumps({'alerts': {'colab:tunnel:t': {'last': now, 'once': False}, 'units:storm:t': {'last': now, 'once': False},
                                                  'colab:balance': {'last': now, 'once': False}, 'health:x': {'last': now - 9 * 3600, 'once': True}},
                                                  'gpu': {'colab-g4a': {'util': 0, 'idle_s': 900.0, 'checked': now}}, 'last_run': now}))
    lines(tmp_path / 'alerts.jsonl', [{'key': k, 'check': k.split(':')[0], 'severity': 'critical', 'message': k, 'ts': now} for k in ('colab:tunnel:t', 'units:storm:t', 'colab:balance', 'health:x')])
    wd = ei.watchdog_section(now)
    assert sorted(a['key'] for a in wd['alerts']) == ['colab:balance', 'colab:tunnel:t', 'units:storm:t'] and wd['gpu']['colab-g4a']['idle_s'] == 900.0
    ids = [a['id'] for a in ei.derived_alerts({'driver': {'running': True}, 'rounds': [], 'watchdog': wd}, {}, {}, now)]
    assert ids == ['wd:colab:tunnel:t']          # the storm of the same unit and the exporter's own credit alert replace the rest


def test_watchdog_alerts_of_resources_out_of_the_roster_are_suppressed(tmp_path, monkeypatch):
    """Colab stopped by the owner (roster: no sessions, GPUs disabled): its lingering session, tunnel, idle-burn and GPU
    alerts and CPU stray assignments are listed as suppressed, not raised; a stray GPU assignment still alerts."""
    now = time.time()
    monkeypatch.setattr(ei, 'OPS', str(tmp_path))
    monkeypatch.setattr(ei, 'WATCHDOG_STATE', str(tmp_path / 'wd.json'))
    (tmp_path / 'watchdog.json').write_text(json.dumps({'gpus': {'colab-g4a': {'kind': 'colab', 'disabled': True}}, 'colab': {'sessions': []},
                                                       '_doc_colab': 'stopped by owner'}))
    keys = {'colab:missing:g4a': {}, 'colab:tunnel:colab-g4a-tunnel': {}, 'colab:burn_idle': {}, 'gpu:colab-g4a': {},
            'colab:stray:m-cpu': {'hardware': 'CPU'}, 'colab:stray:m-gpu': {'hardware': 'A100'}, 'disk:/': {}}
    (tmp_path / 'wd.json').write_text(json.dumps({'alerts': {k: {'last': now, 'once': False} for k in keys},
                                                  'gpu': {'colab-g4a': {'util': None, 'idle_s': 2400.0, 'checked': now}}, 'last_run': now}))
    lines(tmp_path / 'alerts.jsonl', [{'key': k, 'check': k.split(':')[0], 'severity': 'critical', 'message': k, 'ts': now, 'detail': d} for k, d in keys.items()])
    wd = ei.watchdog_section(now)
    assert sorted(a['key'] for a in wd['alerts']) == ['colab:stray:m-gpu', 'disk:/']
    assert len(wd['suppressed']) == 5 and wd['colab_stopped'] and wd['gpu']['colab-g4a']['disabled']
    ro = ei.roster()
    assert ei.colab_stopped(ro) and ei.alert_unexpected('colab:missing:g4a', {}, {}) is None      # no roster: the watchdog defaults expect g4a


def test_stalled_collector_and_teacher_alerts():
    now = time.time()
    snap = {'driver': {'running': True}, 'rounds': [],
            'collection': {'actors': [{'host': 'dgx', 'unit': 'aei1-collect-v0003.service', 'version': 3, 'last_group_ts': int((now - 3000) * 1000)}]},
            'teacher': {'pool': 'teacher-r0005', 'eligible': 41, 'tasks_done': 20, 'last_activity_ts': int((now - 3 * 3600) * 1000)}}
    ids = [a['id'] for a in ei.derived_alerts(snap, {}, {}, now)]
    assert ids == ['ei:collect-stall:aei1-collect-v0003.service', 'ei:teacher-stall']
    snap['teacher']['tasks_done'] = 41          # a finished campaign is not stalled
    assert 'ei:teacher-stall' not in [a['id'] for a in ei.derived_alerts(snap, {}, {}, now)]


def test_open_items_lead_with_the_handoff_and_list_qualified_free_teachers(tmp_path, monkeypatch):
    monkeypatch.setattr(ei, 'OPS', str(tmp_path / 'ops'))
    monkeypatch.setattr(ei, 'REPO', str(tmp_path / 'repo'))
    lines(tmp_path / 'repo' / 'plans' / 'semantic-fp' / 'OPEN_ITEMS.md',
          ['## In flight', '| Item | Owner |', '|---|---|', '| fix x | agent |'])
    lines(tmp_path / 'repo' / 'plans' / 'neuralese' / 'HANDOFF_20261127.md',
          ['# Handoff', '## 8. Next steps, in order', '', '1. Read investigators A and B and build the fix', '   (GRPO on the frontier).',
           '2. Ask the owner about `Vertex`.', '## 9. Other', '3. not a step'])
    lines(tmp_path / 'ops' / 'teacher-free-models' / 'candidates.jsonl',
          [{'provider': 'openrouter', 'model': 'a/b:free', 'status': 'new'},
           {'provider': 'kilo', 'model': 'c-free', 'status': 'qualified', 'note': 'yield 11/20'}])
    rows = ei.open_items_section({})
    assert [r['item'] for r in rows[:2]] == ['Read investigators A and B and build the fix (GRPO on the frontier).', 'Ask the owner about Vertex.']
    assert rows[0]['source'] == 'handoff' and rows[2] == {'item': 'fix x', 'owner': 'agent'}
    assert len(rows) == 4 and 'c-free on kilo qualified (yield 11/20)' in rows[3]['item']


def test_decisions_take_the_newest_entries_with_their_cause_and_next_step():
    text = '\n'.join(['## 2026-11-26 15:30 Older entry', '', 'First paragraph.', '',
                      '### 2026-11-26 — minor entry', 'Body.', '',
                      '## 2026-11-27 Why SFT rounds give no held gain (investigator A)', '', 'Question: why.', '',
                      '**Cause.** The training signal does not raise pass rates.', 'Second line.', '', '**Fix (most direct).**', '1. Train with GRPO.'])
    e = ei.decision_entries(text)
    assert [x['when'] for x in e] == ['2026-11-26 15:30', '2026-11-26', '2026-11-27']
    assert e[2]['title'] == 'Why SFT rounds give no held gain (investigator A)' and e[2]['summary'].startswith('Cause. The training signal')
    assert e[2]['next'] == 'Fix (most direct). 1. Train with GRPO.' and e[1]['title'] == 'minor entry' and e[0]['next'] is None
