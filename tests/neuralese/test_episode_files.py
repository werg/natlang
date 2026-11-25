"""natlang_neuralese.rl.episode_files: an episode's job files once nothing reads them raw (DECISIONS 2026-11-25 "episode
files compacted"), and every place that runs it: the RL collector at a group's last scoring, the expert-iteration
collection's start (a stopped run's leftovers) and native-state cache, the remote pull, held measurement, the
collection trim."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / 'training/neuralese', REPO, REPO / 'scripts/neuralese_data', REPO / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from natlang_neuralese.rl import episode_files as ef  # noqa: E402

UUID = '499edaea-19ea-421f-bc53-868153b886a8'


def dead_pid():
    process = subprocess.Popen(['true'])
    process.wait()
    return process.pid


def sample(root, result=b'{"row": 1}\n'):
    s = root / 'groups' / 'g000001' / 'row-000' / 'sample-00'
    jobs = s / 'jobs'
    jobs.mkdir(parents=True)
    (s / 'results.jsonl').write_bytes(result)
    (jobs / '000000-abc.result.json').write_bytes(result)                       # the byte copy
    partial = jobs / '000000-abc.partial.json'
    partial.write_bytes(json.dumps({'turns': [{'ids': list(range(500))}] * 20}).encode())
    evidence = jobs / f'000000-abc.partial.json.evidence-{UUID}.jsonl'
    evidence.write_bytes(b'{"payload": {"kind": "collector_provider_request_retry_scheduled"}}\n' * 500)
    (jobs / f'000000-abc.partial.json.tmp-{dead_pid()}-{UUID}').write_bytes(b'half')
    (jobs / f'000000-abc.partial.json.tmp-{os.getpid()}-{UUID}').write_bytes(b'live writer')
    (jobs / '000000.error.json').write_text('{"error": "budget"}')
    os.utime(partial, (1000, 1000))
    return s


def test_a_finished_sample_keeps_every_byte_compressed_and_drops_only_copies(tmp_path):
    s = sample(tmp_path)
    jobs = s / 'jobs'
    partial_bytes = (jobs / '000000-abc.partial.json').read_bytes()
    evidence_bytes = (jobs / f'000000-abc.partial.json.evidence-{UUID}.jsonl').read_bytes()
    stats = ef.compact_sample(s)
    names = sorted(p.name for p in jobs.iterdir())
    assert names == sorted(['000000-abc.partial.json.zst', f'000000-abc.partial.json.evidence-{UUID}.jsonl.zst',
                            f'000000-abc.partial.json.tmp-{os.getpid()}-{UUID}', '000000.error.json'])
    assert stats['compressed_files'] == 2 and stats['duplicate_result_files'] == 1 and stats['tmp_files'] == 1
    assert ef.read_bytes(jobs / '000000-abc.partial.json') == partial_bytes         # by the raw name
    assert ef.read_bytes(jobs / f'000000-abc.partial.json.evidence-{UUID}.jsonl.zst') == evidence_bytes
    assert (jobs / '000000-abc.partial.json.zst').stat().st_mtime == 1000              # mtime kept
    assert ef.freed(stats) > 0 and (s / 'results.jsonl').exists()
    assert [p.name for p in ef.evidence_files(jobs)] == [f'000000-abc.partial.json.evidence-{UUID}.jsonl.zst']
    assert ef.compact_sample(s) == {}                                                  # idempotent


def test_a_result_that_differs_from_results_stays_and_the_opt_out_touches_nothing(tmp_path, monkeypatch):
    s = sample(tmp_path)
    (s / 'results.jsonl').write_bytes(b'{"row": 2}\n')
    monkeypatch.setenv(ef.ENV, '0')
    before = sorted(p.name for p in (s / 'jobs').iterdir())
    assert ef.compact_sample(s) == {} and sorted(p.name for p in (s / 'jobs').iterdir()) == before
    monkeypatch.delenv(ef.ENV)
    ef.compact_sample(s)
    assert (s / 'jobs' / '000000-abc.result.json').exists()


def test_an_interrupted_pass_finishes_and_a_conflicting_copy_is_an_error_not_a_loss(tmp_path):
    s = sample(tmp_path)
    partial = s / 'jobs' / '000000-abc.partial.json'
    data = partial.read_bytes()
    (s / 'jobs' / '000000-abc.partial.json.zst').write_bytes(ef._compress(data))      # renamed, raw not yet removed
    ef.compact_sample(s)
    assert not partial.exists() and ef.read_bytes(partial) == data
    other = s / 'jobs' / f'000001-def.partial.json'
    other.write_bytes(b'new' * 4096)
    other.with_name(other.name + '.zst').write_bytes(ef._compress(b'old'))
    assert ef.compact_sample(s)['errors'] == 1 and other.read_bytes() == b'new' * 4096
    small = s / 'jobs' / '000002-fed.partial.json'
    small.write_bytes(b'{}')
    ef.compact_sample(s)
    assert small.exists()                                                           # too small to gain


def test_the_cli_leaves_open_and_recent_samples(tmp_path):
    s = sample(tmp_path)
    quiet = sample(tmp_path / 'b')
    old = time.time() - 7200
    for path in [quiet, *quiet.rglob('*')]:
        os.utime(path, (old, old), follow_symlinks=False)
    ef.main(['compact', str(tmp_path / 'groups'), str(tmp_path / 'b'), '--apply', '--quiet-minutes', '30'])
    assert (s / 'jobs' / '000000-abc.partial.json').exists()                       # written just now: live
    assert not (quiet / 'jobs' / '000000-abc.partial.json').exists()
    assert ef.quiet_sample(quiet, time.time(), 1800, {str((quiet / 'jobs' / '000000.error.json').resolve())}) is False


def test_held_measurement_reads_compacted_evidence(tmp_path):
    from natlang_neuralese.rl import held_eval
    s = sample(tmp_path)
    ef.compact_sample(s)
    out = s / 'results.jsonl'
    out.write_text('')
    entry = {'jobs': str(s / 'jobs'), 'output': str(out)}
    finish = {'resource_limit_reason': 'model_request_budget', 'status': 'failed'}
    import natlang_neuralese.eval.task_packet as tp
    original = tp.classify_case
    tp.classify_case = lambda row, finish: 'budget_exhaustion'
    try:
        assert held_eval.infra_loss(entry, finish) == 'request_budget_spent_on_resends'
    finally:
        tp.classify_case = original


def test_the_collector_compacts_a_group_at_its_last_scoring_only(tmp_path, monkeypatch):
    import grpo
    calls = []
    monkeypatch.setattr(grpo, 'compact_group_files', lambda d: calls.append(d))
    from natlang_neuralese.rl import task_rollouts
    monkeypatch.setattr(task_rollouts, 'score', lambda *a, **k: None)
    directory = tmp_path / 'g000001'
    (directory / 'workers').mkdir(parents=True)
    c = grpo.Collector.__new__(grpo.Collector)
    import threading
    import queue
    c.lock, c.ready, c.groups = threading.Lock(), queue.Queue(), {1: {'dir': directory, 'versions': []}}
    c._record_outcome = lambda d: None
    c._teachers = lambda g: 2                  # teacher attempts queued: scored again later, nothing compacted
    c._score(1)
    assert calls == [] and c.ready.empty()
    c._teachers = lambda g: 0
    c._score(1)
    assert calls == [directory] and c.ready.get_nowait() == 1


def test_expert_iteration_compacts_leftovers_and_collects_old_native_states(tmp_path, monkeypatch):
    import expert_iteration as ei
    monkeypatch.setattr(ei, 'log', lambda record: None)
    s = sample(tmp_path)
    old = time.time() - 3600
    for path in [s, *s.rglob('*')]:
        os.utime(path, (old, old), follow_symlinks=False)
    ei.compact_leftovers(tmp_path, [1])
    assert (s / 'jobs' / '000000-abc.partial.json.zst').exists()
    store = tmp_path / 'actor'
    (store / 'blocks' / 'ab').mkdir(parents=True)
    stale, fresh = store / 'blocks' / 'ab' / 'nzs1_abold.safetensors', store / 'blocks' / 'ab' / 'nzs1_abnew.safetensors'
    stale.write_bytes(b'x')
    fresh.write_bytes(b'y')
    os.utime(stale, (old - 7200, old - 7200))
    import threading
    collector = SimpleNamespace(lock=threading.Lock(), groups={
        1: {'created': old, 'done': 4, 'size': 4},                 # scored: no longer pins the cutoff
        2: {'created': old - 600, 'done': 1, 'size': 4},           # running
        3: {'created': old - 9000, 'done': 0, 'size': 4, 'abandoned': True}})
    ei.collect_actor_blocks(collector, store, 600)
    assert not stale.exists() and fresh.exists()
    ei.collect_actor_blocks(collector, store, -1)                  # opt-out
    assert fresh.exists()


def test_the_remote_pull_never_copies_raw_job_files(tmp_path, monkeypatch):
    import remote_collect as rc
    vm, work = tmp_path / 'vm', tmp_path / 'W'

    def fake(*args, timeout=3600, check=True):
        argv = [str(vm) + str(a)[len('vm:'):] if str(a).startswith('vm:') else str(a) for a in args]
        return subprocess.run(['rsync', '-a', *argv], check=check, capture_output=True, text=True)
    monkeypatch.setattr(rc, 'rsync', fake)
    remote = Path(str(vm) + str(work))
    s = sample(remote)
    compacted = s / 'jobs' / '000000-def.partial.json.zst'
    compacted.write_bytes(b'z')
    work.mkdir(parents=True)
    rc.pull('vm', work)
    local = work / s.relative_to(remote)
    assert sorted(p.name for p in (local / 'jobs').iterdir()) == ['000000-def.partial.json.zst', '000000.error.json']
    assert (local / 'results.jsonl').exists()
    monkeypatch.setenv('NATLANG_REMOTE_PULL_RAW_JOBS', '1')
    rc.pull('vm', work)
    assert (local / 'jobs' / '000000-abc.partial.json').exists()


def test_the_collection_trim_compacts_episode_files(tmp_path, monkeypatch):
    import ei_round_tier as tier
    s = sample(tmp_path)
    old = time.time() - 7200
    for path in [s, *s.rglob('*')]:
        os.utime(path, (old, old), follow_symlinks=False)
    planned = tier.trim(tmp_path, apply=False)
    assert planned.get('episode-raw (compacts ~30x)', 0) > 0
    removed = tier.trim(tmp_path, apply=True)
    assert removed['episode-compaction'] > 0 and (s / 'jobs' / '000000-abc.partial.json.zst').exists()


BIG_RESULT = (json.dumps({'trajectory': [{'ids': list(range(800))}] * 8, 'outcome': {'status': 'done'}}) + '\n').encode()


def scored_group(root, age=7200):
    s = sample(root, BIG_RESULT)
    group = s.parent.parent
    (group / 'score.json').write_text('{}\n')
    old = time.time() - age
    for path in [group, *group.rglob('*')]:
        os.utime(path, (old, old), follow_symlinks=False)
    return group, s


def test_compact_results_keeps_every_byte_and_readers_resolve_it(tmp_path, monkeypatch):
    from natlang_neuralese.common.jsonio import read_jsonl
    _, s = scored_group(tmp_path)
    mtime = (s / 'results.jsonl').stat().st_mtime
    assert ef.compact_sample(s)['duplicate_result_files'] == 1
    stats = ef.compact_results(s)
    assert stats['results_files'] == 1 and stats['compressed_bytes'] < len(BIG_RESULT)
    packed = s / 'results.jsonl.zst'
    assert not (s / 'results.jsonl').exists() and packed.stat().st_mtime == mtime
    assert ef.read_bytes(s / 'results.jsonl') == BIG_RESULT
    assert ef.present(s / 'results.jsonl') == packed and read_jsonl(packed)[0]['outcome'] == {'status': 'done'}
    # the duplicate check compares with the compacted results too
    (s / 'jobs' / '000000-abc.result.json').write_bytes(BIG_RESULT)
    assert ef.compact_sample(s)['duplicate_result_files'] == 1
    import collection_health
    assert collection_health._last_jsonl(s / 'results.jsonl')['outcome'] == {'status': 'done'}
    from natlang_neuralese.rl import task_rollouts
    seen = []
    monkeypatch.setattr(task_rollouts, 'classify_case', lambda result, finish: seen.append(result) or 'task_success')
    monkeypatch.setattr(task_rollouts, 'reward_of', lambda outcome, finish: 1.0)
    monkeypatch.setattr(task_rollouts, 'graded_reward', lambda *a: 1.0)
    task_rollouts.entry_outcome({'output': str(s / 'results.jsonl')}, {}, None)
    assert seen[0]['outcome'] == {'status': 'done'}
    other = tmp_path / 'other'
    _, t = scored_group(other)
    monkeypatch.setenv('NATLANG_RESULTS_COMPACT', '0')                  # opt-out
    assert not ef.compact_results(t) and (t / 'results.jsonl').exists()


def test_the_retention_pass_compacts_scored_groups_of_open_collections(tmp_path, monkeypatch):
    import ei_round_tier as tier
    collection = tmp_path / 'collect' / 'v0009-cde55352685b'
    group, s = scored_group(collection)
    live_group, live = scored_group(collection / 'live')               # a group still running: no score.json
    live_group.rename(collection / 'groups' / 'g000002')
    (collection / 'groups' / 'g000002' / 'score.json').unlink()
    fresh = collection / 'groups' / 'g000003'
    fresh.mkdir()
    (fresh / 'score.json').write_text('{}\n')
    (fresh / 'row-000' / 'sample-00').mkdir(parents=True)
    (fresh / 'row-000' / 'sample-00' / 'results.jsonl').write_bytes(BIG_RESULT)   # scored, not yet quiet
    actions = tier.plan(tmp_path, {'lineage': []}, None)
    assert {'op': 'compact-groups', 'path': str(collection)} in [{k: a[k] for k in ('op', 'path')} for a in actions]
    assert not any(a['op'] == 'compact-groups' for a in tier.plan(tmp_path, {'lineage': []}, None,
                                                                  group_compaction=False))
    stats = tier.compact_groups(collection)
    assert stats['freed'] > 0 and stats['skipped'] == 1
    assert (s / 'results.jsonl.zst').exists() and (group / tier.GROUP_COMPACTED).exists()
    assert (collection / 'groups' / 'g000002' / 'row-000' / 'sample-00' / 'results.jsonl').exists()
    assert (fresh / 'row-000' / 'sample-00' / 'results.jsonl').exists() and not (fresh / tier.GROUP_COMPACTED).exists()
    # a puller whose remote_collect.py lacks the compacted-results excludes blocks the collection; a current one not
    old = tmp_path / 'old' / 'remote_collect.py'
    old.parent.mkdir()
    old.write_text('# no excludes\n')
    procs = [('1', 'cmdline', f'python {old} run --host colab-g4 --work {collection} -- python collect')]
    assert tier.remote_puller_blocks(collection, procs, [])
    current = REPO / 'scripts' / 'neuralese_data' / 'remote_collect.py'
    assert tier.remote_puller_blocks(collection, [('1', 'cmdline', f'python {current} run --work {collection} ')],
                                     [('u', f'python {current} run --work {collection} -- x')]) is None
    assert tier.remote_puller_blocks(collection, [('1', 'cmdline', f'python {old} run --work {collection}-remote0 ')],
                                     []) is None


def test_the_remote_pull_never_pulls_a_compacted_result_again(tmp_path, monkeypatch):
    import remote_collect as rc
    vm, work = tmp_path / 'vm', tmp_path / 'W'

    def fake(*args, timeout=3600, check=True):
        argv = [str(vm) + str(a)[len('vm:'):] if str(a).startswith('vm:') else str(a) for a in args]
        return subprocess.run(['rsync', '-a', *argv], check=check, capture_output=True, text=True)
    monkeypatch.setattr(rc, 'rsync', fake)
    remote = Path(str(vm) + str(work))
    s = sample(remote, BIG_RESULT)
    work.mkdir(parents=True)
    rc.pull('vm', work)
    local = work / s.relative_to(remote)
    assert (local / 'results.jsonl').exists()
    ef.compact_results(local)
    rc.pull('vm', work)
    assert not (local / 'results.jsonl').exists() and ef.read_bytes(local / 'results.jsonl') == BIG_RESULT


def test_actor_block_gc_cuts_at_the_oldest_running_episode(tmp_path, monkeypatch):
    import threading
    import expert_iteration as ei
    monkeypatch.setattr(ei, 'log', lambda record: None)
    store = tmp_path / 'actor'
    (store / 'blocks' / 'ab').mkdir(parents=True)
    now = time.time()
    before, during = store / 'blocks' / 'ab' / 'nzs1_abbefore.safetensors', store / 'blocks' / 'ab' / 'nzs1_abduring.safetensors'
    before.write_bytes(b'x')
    during.write_bytes(b'y')
    os.utime(before, (now - 5000, now - 5000))
    os.utime(during, (now - 1500, now - 1500))
    memetic = tmp_path / 'g000009'
    memetic.mkdir()
    collector = SimpleNamespace(lock=threading.Lock(), slot_starts={0: now - 2000, 1: now - 1000}, groups={
        1: {'created': now - 30000, 'done': 1, 'size': 4, 'dir': tmp_path / 'g000001'},   # resumed, still running
        2: {'created': now - 30000, 'done': 4, 'size': 4, 'dir': tmp_path / 'g000002'}})
    import grpo
    collector.oldest_running_start = lambda: grpo.Collector.oldest_running_start(collector)
    monkeypatch.setenv('NATLANG_BLOCK_GC_BY_GROUP', '1')               # opt-out: the old group rule keeps both
    ei.collect_actor_blocks(collector, store, 600)
    assert before.exists() and during.exists()
    monkeypatch.delenv('NATLANG_BLOCK_GC_BY_GROUP')
    collector.groups[3] = {'created': now - 6000, 'done': 0, 'size': 4, 'dir': memetic}
    (memetic / 'memetic-spec.json').write_text('{}')                   # a live memetic group holds to its creation
    ei.collect_actor_blocks(collector, store, 600)
    assert before.exists() and during.exists()
    del collector.groups[3]
    ei.collect_actor_blocks(collector, store, 600)                     # cutoff: oldest running episode - grace
    assert not before.exists() and during.exists()
