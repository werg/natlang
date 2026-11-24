"""held_eval dispatch order (longest expected episode first within a sample) and the attested running-sequence cap."""
import json

import pytest

from natlang_neuralese.rl import held_eval


def entries(tasks, samples):
    return [{'key': f'{task}-{sample}', 'task_id': task, 'sample': sample, 'index': i}
            for sample in range(samples) for i, task in enumerate(tasks)]


def test_sample_major_and_prior_orders_the_first_sample():
    dispatch = held_eval.Dispatch(entries(['a', 'b', 'c'], 2), prior={'b': 900., 'c': 50.})
    first = [dispatch.take(now=0.)['key'] for _ in range(3)]
    assert first == ['b-0', 'c-0', 'a-0']          # longest prior first; a task without a prior (0) last
    assert dispatch.take(now=1.)['sample'] == 1     # sample 1 starts only once every sample-0 entry started


def test_ties_keep_the_sealed_order_and_sealed_opts_out():
    dispatch = held_eval.Dispatch(entries(['a', 'b', 'c'], 1))
    assert [dispatch.take(now=0.)['key'] for _ in range(3)] == ['a-0', 'b-0', 'c-0']
    sealed = held_eval.Dispatch(entries(['a', 'b'], 2), prior={'b': 10.}, order='sealed')
    assert [sealed.take(now=0.)['key'] for _ in range(4)] == ['a-0', 'b-0', 'a-1', 'b-1']
    assert sealed.take() is None
    with pytest.raises(ValueError):
        held_eval.Dispatch([], order='random')


def test_mean_of_every_known_time_orders_and_running_episodes_count_their_time_so_far():
    # a: one long earlier episode and a short one here (heavy tail: still expected long); c: short before, longer here
    dispatch = held_eval.Dispatch(entries(['a', 'b', 'c'], 2), prior={'a': [1000., 2000.], 'b': [10.], 'c': [5.]})
    a, b, c = (dispatch.take(now=0.) for _ in range(3))
    assert [a['key'], b['key'], c['key']] == ['a-0', 'b-0', 'c-0']
    dispatch.done(a, 30.)          # a: mean(1000, 2000, 30) = 1010
    dispatch.done(c, 400.)         # c: mean(5, 400) = 202.5
    # b still runs at t=500: its 500 s so far beat its 10 s prior, not a's mean
    assert [dispatch.take(now=500.)['key'] for _ in range(3)] == ['a-1', 'b-1', 'c-1']


def test_longest_first_max_keeps_the_previous_estimate():
    dispatch = held_eval.Dispatch(entries(['a', 'b', 'c'], 2), prior={'a': 1000., 'b': 10., 'c': 5.},
                                  order='longest-first-max')
    a, b, c = (dispatch.take(now=0.) for _ in range(3))
    assert [a['key'], b['key'], c['key']] == ['a-0', 'b-0', 'c-0']
    dispatch.done(a, 30.)          # a was short in this run: its prior no longer counts
    dispatch.done(c, 200.)
    # b still runs at t=400: its 400 s so far make it the longest known task
    assert dispatch.take(now=400.)['key'] == 'b-1'
    assert dispatch.take(now=400.)['key'] == 'c-1'
    assert dispatch.take(now=400.)['key'] == 'a-1'


def test_resumed_finished_episodes_seed_the_estimates():
    dispatch = held_eval.Dispatch([e for e in entries(['a', 'b'], 2) if e['sample'] == 1], observed={'a': 500., 'b': 20.})
    assert dispatch.take(now=0.)['key'] == 'a-1'


def test_duration_prior_keeps_every_ran_episode(tmp_path):
    for name, seconds in (('one', 100.), ('two', 300.)):
        (tmp_path / f'{name}.json').write_text(json.dumps({'samples': [
            {'task_id': 't', 'ran': True, 'elapsed_seconds': seconds},
            {'task_id': 'u', 'ran': False, 'elapsed_seconds': None}]}))
    assert held_eval.duration_prior([tmp_path / 'one.json', tmp_path / 'two.json']) == {'t': [100., 300.]}


def test_collect_runs_every_entry_longest_first_and_skips_finished(tmp_path, monkeypatch):
    work = tmp_path / 'workers'
    work.mkdir()
    (work / 'b-0.journal.jsonl').write_text(json.dumps({'event': 'finish', 'elapsed_seconds': 700.}) + '\n')
    started = []

    class Child:
        def __init__(self, command, **_):
            journal = command[3]
            started.append(journal.split('/')[-1].removesuffix('.journal.jsonl'))
            with open(journal, 'w') as handle:
                handle.write(json.dumps({'event': 'finish', 'elapsed_seconds': 1.}) + '\n')

        def wait(self):
            return 0

        def poll(self):
            return 0

    monkeypatch.setattr(held_eval.subprocess, 'Popen', Child)
    monkeypatch.setattr('natlang_neuralese.rl.actor_client.NativeActorClient.get_bytes', lambda *_a, **_k: b'{}')
    import threading
    held_eval.collect(tmp_path, entries(['a', 'b'], 2), [('http://x', 1)], repo=tmp_path, runtime=tmp_path, env={},
                      stop=threading.Event())
    # b-0 finished before (700 s): sample 1 runs b first; a-0 ran in this run (1 s)
    assert started == ['a-0', 'b-1', 'a-1']


def test_engine_attests_the_running_cap_and_compare_refuses_another():
    binding = {'serving_engine': {'engine': 'sglang', 'max_running': 20, 'fork': {'commit': 'f'}},
               'actor_export': {'format': 'x'}}
    assert held_eval.engine_of(binding)['max_running'] == 20
    assert held_eval.engine_of({**binding, 'serving_engine': {'engine': 'sglang'}})['max_running'] is None
    speculative = {'serving_engine': {**binding['serving_engine'], 'speculative': {'algorithm': 'ngram', 'draft_tokens': 4}}}
    assert held_eval.engine_of(speculative)['speculative'] == {'algorithm': 'ngram', 'draft_tokens': 4}
    assert held_eval.engine_of(binding)['speculative'] is None   # plain decode pairs only with plain decode
    result = {'samples': [{'task_id': 't', 'sample': 0, 'ran': True, 'reward': 1.0, 'outcome': 'task_success'}],
              'policy_id': 'p', 'series': {'key': 's'}, 'ran': 1, 'scored': 1, 'successes': 1, 'loop_rate': 0}
    sixteen = {**result, 'engine': held_eval.engine_of({**binding, 'serving_engine': {'engine': 'sglang', 'max_running': 16}})}
    twenty = {**result, 'engine': held_eval.engine_of(binding)}
    with pytest.raises(ValueError, match='max_running'):
        held_eval.compare(sixteen, twenty, strict_series=True)
    report = held_eval.compare(sixteen, twenty)   # the running cap alone pairs, with a warning
    assert report['engine_differences']['max_running'] == [16, 20] and report['warnings']


def test_compare_pairs_other_series_and_fork_commits_with_warnings_but_not_other_precision_or_law():
    def result(tree, fork, *, kv='fp8_e4m3', top_p=0.95, cap=None, commit='c1'):
        engine = {'engine': 'sglang', 'dtype': 'bfloat16', 'kv_cache_dtype': kv, 'fork_commit': fork,
                  'export_format': f'x-kv-{kv}', **({'max_running': cap} if cap else {})}
        return {'samples': [{'task_id': 't', 'sample': 0, 'ran': True, 'reward': 1.0, 'outcome': 'task_success'}],
                'policy_id': 'p', 'ran': 1, 'scored': 1, 'successes': 1, 'loop_rate': 0, 'engine': engine,
                'complete': True, 'sampling': {'temperature': 1.0, 'top_p': top_p}, 'code': {'git_commit': commit},
                'series': {'key': tree, 'ts_host_tree': tree, 'model_facing': 'effect-receipt',
                           'scoring_sources': {'scripts/run_bonsai_queue.py': 'a'}}}
    base = result('tree-old', 'fork-old')                       # e.g. grpo-v3's Pop base: no attested running cap
    new = result('tree-new', 'fork-new', cap=20, commit='c2')
    new['series']['scoring_sources'] = {'scripts/run_bonsai_queue.py': 'b'}
    report = held_eval.compare(base, new)
    assert report['primary']['delta'] == 0 and not report['same_series']
    assert set(report['series_differences']) == {'ts_host_tree', 'scripts/run_bonsai_queue.py', 'git_commit'}
    assert report['engine_differences'] == {'fork_commit': ['fork-old', 'fork-new'], 'max_running': [None, 20]}
    assert len(report['warnings']) == 2
    assert held_eval.compare(base, base)['warnings'] == []
    with pytest.raises(ValueError, match='scoring series'):
        held_eval.compare(base, result('tree-new', 'fork-old'), strict_series=True)
    with pytest.raises(ValueError, match='kv_cache_dtype'):
        held_eval.compare(base, result('tree-new', 'fork-new', kv='auto'))
    with pytest.raises(ValueError, match='different laws'):
        held_eval.compare(base, result('tree-new', 'fork-new', top_p=1.0))


def test_pop_actor_start_reuses_a_served_policy_only_with_the_same_engine_settings(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/neuralese_data'))
    import pop_rl_actor
    from types import SimpleNamespace
    settings, _ = pop_rl_actor.pop_sglang('fp8_e4m3', 20)
    previous = {'policy_id': 'p', 'endpoint': 'http://x', 'engine': 'sglang', 'sglang': settings, 'unit': 'u'}
    monkeypatch.setattr(pop_rl_actor, 'read_json', lambda *_a, **_k: dict(previous))
    monkeypatch.setattr(pop_rl_actor, 'attested', lambda *_a: True)

    class Replaced(Exception):
        pass

    def stop_unit(_state):
        raise Replaced
    monkeypatch.setattr(pop_rl_actor, 'stop_unit', stop_unit)
    # never the host's real units: stop_pop_fills stops every live Pop fill and Pop collector on this machine (this test
    # stopped aei1's Pop collector on 2026-11-24)
    monkeypatch.setattr(pop_rl_actor, 'stop_pop_fills', lambda: None)
    args = SimpleNamespace(policy_id='p', engine='sglang', kv_cache_dtype='fp8_e4m3', max_running=20)
    assert pop_rl_actor.start(args)['endpoint'] == 'http://x'
    with pytest.raises(Replaced):
        pop_rl_actor.start(SimpleNamespace(**{**vars(args), 'max_running': 16}))


def test_pop_actor_helper_import_preserves_pinned_checkout_and_pythonpath():
    import os
    import subprocess
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    pythonpath = os.pathsep.join([str(repo / 'training/neuralese'), str(repo)])
    environment = {**os.environ, 'NATLANG_REPO': str(repo), 'PYTHONPATH': pythonpath}
    script = (
        'import json, os; '
        'from scripts.neuralese_data.pop_rl_actor import pop_sglang; '
        'print(json.dumps({key: os.environ.get(key) for key in ("NATLANG_REPO", "PYTHONPATH")}))'
    )
    result = subprocess.run([sys.executable, '-c', script], cwd=repo, env=environment,
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {'NATLANG_REPO': str(repo), 'PYTHONPATH': pythonpath}


def _fake_repo(tmp_path, body):
    (tmp_path / 'scripts').mkdir()
    (tmp_path / 'scripts/run_bonsai_queue.py').write_text(body)
    return tmp_path


def test_a_stop_returns_within_the_drain_even_when_episodes_ignore_sigterm(tmp_path, monkeypatch):
    import threading
    import time as clock
    repo = _fake_repo(tmp_path, 'import signal, time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(600)\n')
    monkeypatch.setattr('natlang_neuralese.rl.actor_client.NativeActorClient.get_bytes', lambda *_a, **_k: b'{}')
    (tmp_path / 'run').mkdir()
    stop, lines = threading.Event(), []
    threading.Timer(1.0, stop.set).start()
    started = clock.monotonic()
    held_eval.collect(tmp_path / 'run', entries(['a', 'b', 'c'], 1), [('http://x', 3)], repo=repo, runtime=tmp_path,
                      env={}, stop=stop, log=lines.append, drain_seconds=4)
    assert clock.monotonic() - started < 8
    assert any('"killed_episodes": 3' in line for line in lines)   # SIGKILL at half the drain


def test_a_stop_returns_and_dumps_stacks_when_a_slot_thread_is_stuck(tmp_path, monkeypatch):
    import threading
    import time as clock
    repo = _fake_repo(tmp_path, 'import time\ntime.sleep(0.1)\n')
    monkeypatch.setattr('natlang_neuralese.rl.actor_client.NativeActorClient.get_bytes', lambda *_a, **_k: b'{}')
    stuck = threading.Event()
    real = held_eval.Dispatch.done

    def done(self, entry, seconds):   # a slot thread that never comes back after its episode
        stuck.wait(600)
        real(self, entry, seconds)
    monkeypatch.setattr(held_eval.Dispatch, 'done', done)
    (tmp_path / 'run').mkdir()
    stop, lines = threading.Event(), []
    threading.Timer(1.0, stop.set).start()
    started = clock.monotonic()
    held_eval.collect(tmp_path / 'run', entries(['a'], 1), [('http://x', 1)], repo=repo, runtime=tmp_path, env={},
                      stop=stop, log=lines.append, drain_seconds=2)
    stuck.set()
    assert clock.monotonic() - started < 6
    assert any('"stuck_slots": 1' in line and 'stuck.wait' in line for line in lines)
