import importlib.util
import json
import os
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    'ops_export', Path(__file__).resolve().parents[2] / 'scripts' / 'ops_dashboard' / 'export.py')
export = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(export)

STEP = {'step': 2, 'groups': 24, 'reward_mean': 0.5, 'elapsed_s': 200, 'cuda_peak_gb': 39.1, 'behaviour_shift': True,
        'grad_spike': False, 'kl_mean': 0.01, 'name': 'text', 'none': None, 'meminfo_gb': {'MemFree': 3.0},
        'behaviour': {'close_ratio': 1.31, 'entropy': 0.6}, 'phase_s': {'wait_groups': 12.5},
        'gain': {'gain': -0.003, 'gain_per_episode': -5e-05, 'seconds': 85.3}, 'grad_mass_share': {'think': 0.98}}


def write_lines(path, recs):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r) + '\n' for r in recs))


def test_flatten_and_select_metrics():
    flat = export.flatten(STEP)
    assert flat['behaviour.close_ratio'] == 1.31 and flat['behaviour_shift'] == 1 and flat['grad_spike'] == 0
    assert 'name' not in flat and 'none' not in flat
    m = export.select_metrics(STEP)
    assert 'step' not in m and 'meminfo_gb.MemFree' not in m
    for k in ('reward_mean', 'behaviour.entropy', 'phase_s.wait_groups', 'grad_mass_share.think', 'gain.gain',
              'gain.gain_per_episode', 'gain.seconds'):
        assert k in m
    assert all(f in m for f in export.FLAGS) and m['behaviour_shift'] == 1


def test_select_metrics_cap():
    rec = {'step': 1, 'cuda_peak_gb': 1, **{f'x{i}': i for i in range(300)}}
    assert len(export.select_metrics(rec)) <= 60 + len(export.FLAGS)


def test_bench_runs_keep_only_chart_metrics():
    rec = {**STEP, 'learner_tokens_per_s': 1600, 'cuda_peak_gb': 40, 'decision_diagnostics': {'n': 3}}
    m = export.select_metrics(rec, 'bench')
    assert 'behaviour.close_ratio' in m and 'gain.gain' in m and 'kl_mean' in m
    assert 'learner_tokens_per_s' not in m and 'phase_s.wait_groups' not in m and 'grad_mass_share.think' not in m
    assert 'decision_diagnostics.n' not in export.select_metrics(rec)


def test_pop_held_unit_is_a_held_eval():
    assert export.kind_of('popheld-abc-1.service', '/usr/bin/python3 driver.py job.json') == 'held-eval'


def test_run_ids_and_discovery(tmp_path):
    ops = tmp_path / 'ops'
    write_lines(ops / 'grpo-v2' / 'run2' / 'train.jsonl', [STEP])
    write_lines(ops / 'grpo-v2' / 'hunt' / 'bisA' / 'train.jsonl', [STEP])
    write_lines(ops / 'a' / 'b' / 'c' / 'd' / 'train.jsonl', [STEP])          # depth 4: ignored
    write_lines(ops / 'grpo-v2' / 'run2' / 'pin-abc' / 'train.jsonl', [STEP])  # pinned code copy: ignored
    runs = dict(export.discover_runs(str(ops)))
    assert set(runs) == {'grpo-v2.run2', 'grpo-v2.hunt.bisA'}
    assert export.run_id(str(ops), str(ops / 'grpo-v2' / 'hunt' / 'bisA')) == 'grpo-v2.hunt.bisA'
    assert export.run_id(str(ops), '/elsewhere/x') is None


def test_tail_incremental_and_records(tmp_path):
    ops = tmp_path / 'ops'
    path = ops / 'grpo-v2' / 'hunt' / 'bisA' / 'train.jsonl'
    write_lines(path, [{'event': 'start'}, {'step': 1, 'event': 'learner progress'}, STEP])
    state = {}
    m, e = export.scan_runs(str(ops), state, __import__('time').time())
    assert len(m) == 1 and m[0]['type'] == 'train'
    d = m[0]['data']
    assert (d['run_id'], d['group'], d['kind'], d['step'], d['wall_s']) == ('grpo-v2.hunt.bisA', 'grpo-v2.hunt', 'bench', 2, 200)
    assert e and e[0]['data']['flags'] == ['behaviour_shift']
    assert export.scan_runs(str(ops), state, 0)[0] == []                         # nothing new
    with open(path, 'a') as f:
        f.write(json.dumps({**STEP, 'step': 3}))                                  # partial line (no newline): not read
    assert export.scan_runs(str(ops), state, 0)[0] == []
    with open(path, 'a') as f:
        f.write('\n')
    m, _ = export.scan_runs(str(ops), state, 0)
    assert [r['data']['step'] for r in m] == [3]


def test_held_record(tmp_path):
    ops = tmp_path / 'ops'
    held = ops / 'grpo-v2' / 'run2' / 'held' / 'step-00003'
    held.mkdir(parents=True)
    write_lines(ops / 'grpo-v2' / 'run2' / 'train.jsonl', [])
    cmp = {'first': {'policy': 'aaa', 'ran': 512}, 'second': {'policy': 'bbb', 'ran': 336, 'complete': False},
           'primary': {'tasks': 224, 'first_rate': 0.529, 'second_rate': 0.433, 'delta': -0.096, 'permutation_p': 0.001,
                       'ci95_task_bootstrap': [-0.147, -0.04]},
           'paired': {'mcnemar_exact_p': 0.0055}, 'loop_rate_delta': 0.0958}
    (held / 'compare-vs-step0.json').write_text(json.dumps(cmp))
    state = {}
    m, e = export.scan_runs(str(ops), state, __import__('time').time())
    d = m[0]['data']
    assert m[0]['type'] == 'held' and d['run_id'] == 'grpo-v2.run2' and d['step'] == 3 and d['policy'] == 'bbb'
    assert d['delta_pts'] == -9.6 and d['p'] == 0.001 and d['ran'] == 336 and d['of'] == 512 and d['complete'] is False
    assert e[0]['data']['kind'] == 'held'
    assert export.scan_runs(str(ops), state, 0)[0] == []                         # unchanged: no repeat


FALLOW = """# Title
intro
## Paused

| # | Piece | Code now | Fix before it returns | Next action |
|---|---|---|---|---|
| P1 | Variants | live, `a | b` | none | first |
| P2 | Gradients | deleted | rebuild | later |

## Register

### RL task supply

| # | Capability | Built where | Status | Blocker | Impact | Next action |
|---|---|---|---|---|---|---|
| K1 | bgkit | `scripts/x.py` | Qualified | wiring | H | wire it |

text after
"""


def test_parse_fallow():
    rows = export.parse_fallow(FALLOW)
    assert [r['id'] for r in rows] == ['P1', 'P2', 'K1']
    assert rows[0]['section'] == 'Paused' and rows[0]['piece'] == 'Variants' and rows[0]['code_now'] == 'live, `a | b`'
    assert rows[0]['blocker'] == 'none' and rows[0]['next_action'] == 'first'
    assert rows[2]['section'] == 'RL task supply' and rows[2]['blocker'] == 'wiring'
    assert rows[2]['fields'] == {'Status': 'Qualified', 'Impact': 'H'}


def test_snapshot_record_fits_one_stream_record():
    snap = {'ts': 1, 'machines': {}, 'queue': [], 'investigations': [], 'alerts': [], 'runs': [],
            'processes': [{'unit': f'u{i}', 'state': 'running' if i < 5 else 'finished', 'description': 'x' * 400, 'rationale': 'y' * 400}
                          for i in range(400)],
            'fallow': [{'id': f'F{i}', 'piece': 'p' * 300, 'fields': {'a': 'b' * 200}} for i in range(80)],
            'teachers': {'history': [[i, i, i] for i in range(300)]}}
    rec = export.snapshot_record(snap)
    assert export.line_size(rec) <= export.SNAP_LIMIT and rec['type'] == 'snapshot'
    assert sum(p['state'] == 'running' for p in rec['data']['processes']) == 5          # running units survive the shrink
    small = {'ts': 1, 'processes': [{'unit': 'a', 'state': 'running', 'description': 'd', 'rationale': 'r'}], 'fallow': [], 'teachers': None}
    assert export.snapshot_record(small)['data']['processes'][0]['rationale'] == 'r'    # nothing is cut when it already fits


def test_runs_window_and_summary(tmp_path):
    ops = tmp_path / 'ops'
    old = ops / 'old' / 'train.jsonl'
    write_lines(old, [STEP])
    write_lines(ops / 'grpo-v3' / 'run1' / 'train.jsonl', [STEP, {**STEP, 'step': 3, 'reward_mean': 0.7}])
    now = os.path.getmtime(old) + export.WINDOW_S + 10
    os.utime(ops / 'grpo-v3' / 'run1' / 'train.jsonl', (now, now))
    state = {}
    m, _ = export.scan_runs(str(ops), state, now)
    assert {r['data']['run_id'] for r in m} == {'grpo-v3.run1'}                          # the stale run was never exported
    rows = export.runs_summary(state, {'grpo-v3.run1'}, now)
    assert rows[0]['id'] == 'grpo-v3.run1' and rows[0]['steps'] == 3 and rows[0]['active'] and rows[0]['kind'] == 'grpo'
    assert rows[0]['spark']['reward_mean'] == [0.5, 0.7]


def test_progress_and_helpers():
    assert export.held_progress(['x', '{"held_eval_progress": 336, "of": 512, "minutes": 153.1}', 'tail']) == (336, 512, 153.1)
    assert export.held_progress(['nothing']) is None
    cmd = '/usr/bin/env A=1 /usr/bin/python3 /x/grpo.py train --work /o/grpo-v2/hunt/a --max-steps 1 --max-steps 4'
    assert export.last_arg(cmd, '--max-steps') == '4' and export.last_arg(cmd, '--work') == '/o/grpo-v2/hunt/a'
    assert export.summarize(cmd).startswith('grpo.py train --work')
    assert export.kind_of('hunt-x.service', cmd) == 'bench'
    assert export.parse_systemd_time('Sun 2026-11-15 16:31:46 CET') is not None
    assert export.parse_systemd_time('') is None


def test_unit_lifecycle_and_alerts(tmp_path):
    import time
    now = time.time()
    state = {'runs': {'grpo-v2.hunt.a': {'max_step': 2, 'recent': [[1, 100.0], [2, 200.0]]}}}
    cmd = f'/usr/bin/python3 /x/grpo.py train --work {export.OPS}/grpo-v2/hunt/a --max-steps 5'
    unit = {'loaded': True, 'active': 'active', 'result': 'success', 'pid': 1, 'cmd': cmd, 'start': now - 600, 'end': None, 'cpu_ns': None}
    data = {'units': {'hunt-a.service': unit}, 'ledger': {'claims': {'hunt-a.service': {'budget_gb': 60, 'used_gb': 47, 'gpu': True}}},
            'journal': {'hunt-a.service': ['one', 'last line']}}
    ann = {'units': [{'match': 'hunt-*', 'title': 'Bench arm'}]}
    items, events, alerts = export.update_units(state, 'dgx', data, now, ann)
    it = items[0]
    assert it['state'] == 'running' and it['title'] == 'Bench arm' and it['kind'] == 'bench' and it['run'] == 'grpo-v2.hunt.a'
    assert it['progress'] == {'done': 2, 'total': 5, 'unit': 'steps'} and it['eta_s'] == 300 and it['last_log'] == 'last line'
    assert it['resources']['budget_gb'] == 60 and events[0]['data']['kind'] == 'unit_started'
    # the unit fails
    data['units']['hunt-a.service'] = {**unit, 'active': 'failed', 'result': 'timeout', 'end': now}
    items, events, alerts = export.update_units(state, 'dgx', data, now + 30, ann)
    assert items[0]['state'] == 'failed' and events[0]['data']['kind'] == 'unit_failed' and alerts[0]['level'] == 'warn' and alerts[0]['unit'] == 'hunt-a.service'
    # it disappears much later: dropped after 6 h
    items, events, alerts = export.update_units(state, 'dgx', {'units': {}, 'ledger': None, 'journal': {}}, now + 7 * 3600, ann)
    assert items == [] and state['units'] == {}


def test_alerts_disk_and_ledger(monkeypatch):
    monkeypatch.setattr(export, 'FLOORS', {'dgx': {'/': 40, '/big': 50}, 'pop': {'/': None}})
    host = {'dgx': {'disks': [{'mount': '/', 'size_gb': 900, 'free_gb': 50, 'floor_gb': 40},
                              {'mount': '/big', 'size_gb': 3600, 'free_gb': 40}]},
            'pop': {'disks': [{'mount': '/', 'size_gb': 900, 'free_gb': 5}]}}
    alerts = export.build_alerts(host, {'dgx': {'headroom_gb': 9.0}}, [], ['Pop unreachable: x'])
    texts = [(a['level'], a['text']) for a in alerts]
    assert ('warn', '/ has 50 GB free (floor 40 GB)') in texts
    assert ('crit', '/big has 40 GB free (floor 50 GB)') in texts
    assert any('headroom 9.0' in t for _, t in texts) and any('Pop unreachable' in t for _, t in texts)
    assert not any(a['machine'] == 'pop' for a in alerts)                        # Pop's / has no floor
    assert {a['id'] for a in alerts} >= {'disk:dgx:/', 'disk:dgx:/big', 'ledger:dgx', 'error:0'}


def test_run_without_finished_step_is_listed_while_fresh(tmp_path):
    path = tmp_path / 'train.jsonl'
    write_lines(path, [{'event': 'start'}, {'event': 'learner progress', 'step': 1}])
    st = {}
    steps, mtime = export.tail_run(st, str(path))
    assert steps == [] and st['phase'] == {'event': 'learner progress', 'step': 1}
    state = {'runs': {'grpo-v9.run1': {**st, 'mtime': mtime}, 'grpo-v1.old': {'mtime': mtime - 7 * 3600}}}
    rows = export.runs_summary(state, set(), mtime + 60)
    assert [r['id'] for r in rows] == ['grpo-v9.run1'] and rows[0]['steps'] == 0 and rows[0]['kind'] == 'grpo'


def test_held_unit_progress_reads_its_measurement_log(tmp_path, monkeypatch):
    """A popheld-policy-<sha12> unit reports the progress line of ops/ei/<run>/held/policy-<sha12>/pop-held.log."""
    monkeypatch.setattr(export, 'OPS', str(tmp_path))
    write_lines(tmp_path / 'ei' / 'aei1' / 'held' / 'policy-919472c49381' / 'pop-held.log',
                [{'held_eval_progress': 128, 'of': 512, 'minutes': 60.0}])
    import time
    rec = {'unit': 'popheld-policy-919472c49381-1795721331.service', 'cmd': '/usr/bin/python3 driver.py job.json', 'state': 'running'}
    prog, eta = export.progress_of(rec, {}, [], time.time())
    assert prog == {'done': 128, 'total': 512, 'unit': 'episodes'} and eta == 10800
    write_lines(tmp_path / 'ei' / 'aei1' / 'held-v4' / 'hv4-919472c49381' / 'pop-held.log', [{'held_eval_progress': 4, 'of': 504, 'minutes': 1.0}])
    assert export.progress_of({**rec, 'unit': 'popheld-hv4-919472c49381-1.service'}, {}, [], time.time())[0]['done'] == 4


def test_queue_lists_ledger_waiting_units():
    q = export.build_queue({'dgx': {'waiting': [{'unit': 'aei1-r0016-sft.service', 'budget_gb': 30}], 'headroom_gb': 5}})
    assert [(i['title'], i['source'], i['blocked_on']) for i in q] == [
        ('Round r16: SFT', 'ledger-waiting', 'memory ledger: needs 30 GB, headroom 5 GB')]


def test_retired_readers_are_gone():
    """The annotations' investigations and queue, the bench queue and the teacher-gen fallback are removed (dead producers)."""
    for name in ('teacher_summary', 'active_teacher_pool', 'bench_queue'):
        assert not hasattr(export, name)
