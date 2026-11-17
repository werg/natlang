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


def test_observed_episodes_override_the_prior_and_running_ones_count_their_time_so_far():
    dispatch = held_eval.Dispatch(entries(['a', 'b', 'c'], 2), prior={'a': 1000., 'b': 10., 'c': 5.})
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


def test_duration_prior_averages_ran_episodes(tmp_path):
    for name, seconds in (('one', 100.), ('two', 300.)):
        (tmp_path / f'{name}.json').write_text(json.dumps({'samples': [
            {'task_id': 't', 'ran': True, 'elapsed_seconds': seconds},
            {'task_id': 'u', 'ran': False, 'elapsed_seconds': None}]}))
    assert held_eval.duration_prior([tmp_path / 'one.json', tmp_path / 'two.json']) == {'t': 200.}


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
    result = {'samples': [{'task_id': 't', 'sample': 0, 'ran': True, 'reward': 1.0, 'outcome': 'task_success'}],
              'policy_id': 'p', 'series': {'key': 's'}, 'ran': 1, 'scored': 1, 'successes': 1, 'loop_rate': 0}
    sixteen = {**result, 'engine': held_eval.engine_of({**binding, 'serving_engine': {'engine': 'sglang', 'max_running': 16}})}
    twenty = {**result, 'engine': held_eval.engine_of(binding)}
    with pytest.raises(ValueError, match='max_running'):
        held_eval.compare(sixteen, twenty)


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
    args = SimpleNamespace(policy_id='p', engine='sglang', kv_cache_dtype='fp8_e4m3', max_running=20)
    assert pop_rl_actor.start(args)['endpoint'] == 'http://x'
    with pytest.raises(Replaced):
        pop_rl_actor.start(SimpleNamespace(**{**vars(args), 'max_running': 16}))
