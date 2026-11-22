"""GRPO across machines (plans/neuralese/async-train.md §7.1-§7.2): the scored-group exchange carries exactly what the
learner reads, every ``--extra-actor`` gets each published version and its episodes are stamped with the version that
actor served, and rho per staleness covers the students' own episodes only."""
import collections
import json
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/neuralese_data'))
import grpo  # noqa: E402

from natlang_neuralese.common.paths import resolve  # noqa: E402

REAL_RUN = Path(resolve('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'grpo-v5', 'run1'))


def same(a, b, where='payload'):
    """Byte-identical learner inputs: arrays by dtype, shape and bytes, everything else by type and value."""
    assert type(a) is type(b), f'{where}: {type(a)} vs {type(b)}'
    if isinstance(a, np.ndarray):
        assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), where
    elif isinstance(a, dict):
        assert list(a) == list(b), f'{where}: keys {list(a)} vs {list(b)}'
        for key in a:
            same(a[key], b[key], f'{where}.{key}')
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f'{where}: {len(a)} vs {len(b)} items'
        for i, (x, y) in enumerate(zip(a, b)):
            same(x, y, f'{where}[{i}]')
    else:
        assert a == b, f'{where}: {a!r} vs {b!r}'


def _capture(prompt, samples=((7, -0.5), (8, -0.25))):
    return {'prompt': prompt, 'policy': {'checkpoint_sha256': 'p'},
            'sampling_law': {'temperature': 1, 'top_p': 1.0, 'probabilities': 'post-processing-normalized'},
            'samples': [{'token_id': t, 'logp': lp, 'input_role': 'ordinary'} for t, lp in samples]}


def _collector(directory, versions=(0, 1), states=None, reward='graded'):
    return SimpleNamespace(groups={'g0': {'dir': directory, 'task_id': 't', 'versions': list(versions)}}, generated=0,
                           episodes=0, args=SimpleNamespace(top_p=1.0, reward=reward), states=states)


def test_compact_exchange_round_trips_the_lowered_learner_inputs(tmp_path):
    graded = {'task_id': 't', 'reward': {'quality': 0.0, 'graded': 0.375, 'memetic': {'anchor': 0.5}},
              'outcome': 'wrong_result', 'turns': [{'invocation_id': 'root',
                                                    'sampling': _capture([{'kind': 'tokens', 'token_ids': [1, 2, 3]}],
                                                                         ((7, -0.123456789), (8, -2.5)))}]}
    unscored = {**graded, 'unscored': True}
    guided = [{'kind': 'tokens', 'token_ids': [1, 2]}, {'kind': 'tokens', 'token_ids': [90, 91], 'privileged': True},
              {'kind': 'tokens', 'token_ids': [3]}]
    teacher = {'task_id': 't', 'jolt_role': 'teacher', 'reward': {'quality': 1.0}, 'outcome': 'task_success',
               'turns': [{'invocation_id': 'root', 'sampling': {**_capture(guided), 'privileged': {'mode': 'text'}}}]}
    (tmp_path / 'rollouts.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in (graded, unscored, teacher)))
    stats = collections.Counter()
    expected = grpo.load_group(_collector(tmp_path), 'g0', stats)
    archive = tmp_path / 'g0.zst'
    info = grpo.pack_scored_group(_collector(tmp_path), 'g0', archive)
    payload = grpo.unpack_scored_group(archive)
    same((payload['task_id'], payload['episodes'], payload['versions']), expected)
    assert payload['stats'] == dict(stats) and info['episodes'] == 2 and payload['episodes_read'] == 3
    student, guided_episode = payload['episodes']
    assert student['reward'] == 0.375 and student['anchor'] == 0.5 and student['memetic'] == {'anchor': 0.5}
    assert student['rows'][0]['sample'].tolist() == [7, 8] and student['rows'][0]['logp'].dtype == np.float32
    assert guided_episode['role'] == 'teacher'
    assert guided_episode['rows'][0]['prompt'].tolist() == [1, 2, 3]       # guidance stripped, as the learner trains it


def test_compact_exchange_ships_the_stored_states_calls_import(tmp_path, monkeypatch):
    from lower_rollout_turns import NativeStates
    block = 'nzs1_abcdefgh'
    store = NativeStates(tmp_path / 'store')
    store.path(block).parent.mkdir(parents=True)
    store.path(block).write_bytes(b'state-bytes')
    episode = {'reward': 1.0, 'quality': 1.0, 'task_id': 't', 'round': 'g0', 'outcome': 'ok', 'role': 'student',
               'rows': [{'prompt': np.arange(4, dtype=np.int32), 'imports': [(2, block)], 'sources': {}}],
               'decisions': []}
    monkeypatch.setattr(grpo, 'load_group', lambda c, g, s: ('t', [episode], [0]))
    grpo.pack_scored_group(_collector(tmp_path, states=store), 'g0', tmp_path / 'g0.zst')
    with pytest.raises(ValueError, match='Neuralese states'):
        grpo.unpack_scored_group(tmp_path / 'g0.zst')
    payload = grpo.unpack_scored_group(tmp_path / 'g0.zst', tmp_path / 'learner-store')
    assert NativeStates(tmp_path / 'learner-store').path(block).read_bytes() == b'state-bytes'
    same(payload['episodes'], [episode])


@pytest.mark.skipif(not (REAL_RUN / 'groups/g000121/rollouts.jsonl').exists(), reason='grpo-v5 run1 groups are on the DGX')
@pytest.mark.parametrize('group', ['g000121', 'g000090', 'g000105', 'g000138', 'g000172'])
def test_compact_exchange_is_lossless_on_real_groups(tmp_path, group):
    from lower_rollout_turns import NativeStates
    directory = REAL_RUN / 'groups' / group
    if not (directory / 'rollouts.jsonl').exists():
        pytest.skip(f'{group} pruned')
    args = grpo.build_parser().parse_args(['train', '--work', str(tmp_path), '--checkpoint', 'x'])
    versions = json.loads((directory / 'versions.json').read_text()) if (directory / 'versions.json').exists() else []

    def collector():
        return SimpleNamespace(groups={group: {'dir': directory, 'task_id': None, 'versions': versions}}, args=args,
                               states=NativeStates(REAL_RUN / 'actor'), generated=0, episodes=0)
    stats = collections.Counter()
    expected = grpo.load_group(collector(), group, stats)
    grpo.pack_scored_group(collector(), group, tmp_path / 'g.zst')
    payload = grpo.unpack_scored_group(tmp_path / 'g.zst', tmp_path / 'store')
    same((payload['task_id'], payload['episodes'], payload['versions']), expected)
    assert payload['stats'] == dict(stats)
    if group == 'g000121':     # 263 calls; natlang.compact-scored-group/1 shipped 0 sampled ids of them
        assert sum(len(r['sample']) for e in payload['episodes'] for r in e['rows']) > 10_000


def test_extra_actor_specs_name_their_sglang_server_and_host():
    local = grpo.parse_extra_actor('http://dgx:8470=16')
    remote = grpo.parse_extra_actor('http://100.64.0.9:18531=32@colab-g4-1')
    assert local == {'front': 'http://dgx:8470', 'slots': 16, 'host': None, 'sglang': 'http://dgx:9470'}
    assert remote == {'front': 'http://100.64.0.9:18531', 'slots': 32, 'host': 'colab-g4-1',
                      'sglang': 'http://127.0.0.1:19531'}
    specs = grpo.parse_actor_specs('http://dgx:8460', ['http://g4a:18531=32@g4a', 'http://g4b:18531=32'], default_slots=96)
    assert specs == [('http://dgx:8460', 32), ('http://g4a:18531', 32), ('http://g4b:18531', 32)]


def test_episodes_are_stamped_with_the_version_their_actor_serves():
    c = grpo.Collector.__new__(grpo.Collector)
    c.args = SimpleNamespace(actor='http://dgx:8460', max_staleness=2)
    c.version, c.actor_versions = 5, {'http://g4:18531': 4}
    assert c.served_version('http://dgx:8460') == 5 and c.served_version('http://g4:18531') == 4
    assert not c.actor_lagging('http://g4:18531')
    c.actor_versions['http://g4:18531'] = 2                      # its publications failed: 3 behind
    assert c.actor_lagging('http://g4:18531') and not c.actor_lagging('http://dgx:8460')
    info = {'rl_policy_binding': {'checkpoint_sha256': 'start'}}
    actors = [{'front': 'http://g4:18531'}]
    assert grpo.extra_actor_start_versions(actors, 'start', 0, info=lambda f: info) == {'http://g4:18531': 0}
    assert grpo.extra_actor_start_versions(actors, 'other', 0, info=lambda f: info)['http://g4:18531'] < -1000
    assert grpo.extra_actor_start_versions(actors, 'start', 7, info=lambda f: info)['http://g4:18531'] < -1000


def test_publication_reaches_every_extra_actor_after_the_primary(monkeypatch):
    sync = grpo.WeightSync.__new__(grpo.WeightSync)
    events = []
    sync.extra_actors = [{'front': 'http://local:8470', 'sglang': 'http://local:9470', 'host': None},
                         {'front': 'http://g4:18531', 'sglang': 'http://127.0.0.1:19531', 'host': 'g4'},
                         {'front': 'http://dead:1', 'sglang': 'http://dead:1001', 'host': None}]
    sync.on_actor = lambda front, version: events.append(('served', front, version))
    sync.sglang_url, sync.front_url, sync.pending, sync.publishing, sync.thread = 'http://primary:9460', 'x', None, None, None
    sync.released, sync.last = None, {}
    target = {'dir': '/dev/shm/slot-a', 'slot': 0, 'version': 3, 'policy_id': 'p3', 'weights_sha256': 'w3',
              'tensor_identity': {'t': 1}}

    def bind(target, timing, sglang_url=None, front_url=None):
        if sglang_url == 'http://dead:1001':
            raise OSError('connection refused')
        events.append(('bind', sglang_url or 'primary', target['version']))
    monkeypatch.setattr(sync, '_bind', bind)
    monkeypatch.setattr(sync, '_sentinel', lambda *a: None)
    monkeypatch.setattr(sync, '_identity', lambda slot: {'weights_sha256': 'w3', 'tensor_identity': {'t': 1}})
    remote = []
    import pop_rl_actor
    monkeypatch.setattr(pop_rl_actor, 'rebind_actor', lambda **kw: remote.append(kw) or {'export_dir': '/remote/x'})
    sync.slots, sync.staging = [{'dir': Path('/dev/shm/slot-a'), 'views': {}}], 0
    sync.pending = {'kind': 'native', 'path': '/x.pt'}
    monkeypatch.setattr('natlang_neuralese.common.hashing.sha256_file_hex_cached', lambda p: 'p3')
    done = threading.Event()
    sync.publish(3, lambda version, ok: (events.append(('primary done', version, ok)), done.set()))
    sync.wait()
    assert events[0] == ('bind', 'primary', 3) and events[1] == ('primary done', 3, True)
    assert ('bind', 'http://local:9470', 3) in events and ('served', 'http://local:8470', 3) in events
    assert ('served', 'http://g4:18531', 3) in events and not any(e[1] == 'http://dead:1' for e in events if e[0] == 'served')
    assert remote[0]['host'] == 'g4' and remote[0]['export_dir'] == '/dev/shm/slot-a' and remote[0]['version'] == 3
    report = sync.last['extra_actors']
    assert report['http://dead:1']['ok'] is False and report['http://g4:18531']['ok'] is True


def test_rho_by_staleness_counts_only_the_students_own_episodes():
    assert grpo.rho_stale('student', 2) == 2
    assert grpo.rho_stale('teacher', 2) is None
