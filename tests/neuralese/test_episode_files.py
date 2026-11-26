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


# ---- a scored group's rollouts.jsonl compacted (DECISIONS 2026-11-26 "rollouts compacted") ---------------------------

def rollouts_group(collection, n=1, age=7200, policy='p' * 64):
    """A scored, quiet group of ``collection`` with a compressible rollouts.jsonl (two episodes)."""
    group, s = scored_group(collection / f'tmp{n}', age)
    target = collection / 'groups' / f'g{n:06d}'
    target.parent.mkdir(parents=True, exist_ok=True)
    group.rename(target)
    turn = {'invocation_id': 'root', 'messages': [{'role': 'user', 'content': 'x' * 3000}], 'tools': [],
            'sampling': {'prompt': [{'token_ids': list(range(300))}]}}
    rows = [{'task_id': 't1', 'sample': i, 'policy': {'checkpoint_sha256': policy}, 'outcome': 'task_failure',
             'reward': {'quality': 0.0}, 'turns': [turn]} for i in range(2)]
    data = ''.join(json.dumps(r) + '\n' for r in rows).encode()
    (target / 'rollouts.jsonl').write_bytes(data)
    (target / 'rewards.jsonl').write_text(json.dumps({'task_id': 't1', 'reward': 0.0}) + '\n')
    (target / 'versions.json').write_text('[3, 3]\n')
    (target / 'cases.jsonl').write_text(json.dumps({'id': 't1'}) + '\n')
    old = time.time() - age
    for path in [target, *target.rglob('*')]:
        os.utime(path, (old, old), follow_symlinks=False)
    return target, data, rows


def test_the_retention_pass_compacts_a_quiet_scored_groups_rollouts_and_records_it(tmp_path, monkeypatch):
    import ei_round_tier as tier
    collection = tmp_path / 'collect' / 'v0011-401be1c55949'
    group, data, _ = rollouts_group(collection)
    recent, _, _ = rollouts_group(collection, 2)
    os.utime(recent / 'rewards.jsonl')                       # its scoring wrote a moment ago: not quiet
    recorded = []
    monkeypatch.setattr(tier, 'record_packed', lambda c, packed: recorded.append((c, packed)))
    stats = tier.compact_groups(collection)
    packed = group / 'rollouts.jsonl.zst'
    assert stats['rollouts_files'] == 1 and stats['freed'] > 0
    assert not (group / 'rollouts.jsonl').exists() and ef.read_bytes(packed) == data
    assert (group / ef.ROLLOUTS_MARKER).exists() and (group / tier.GROUP_COMPACTED).exists()
    assert (recent / 'rollouts.jsonl').exists() and not (recent / ef.ROLLOUTS_MARKER).exists()
    import hashlib
    assert recorded == [(collection, [(str(packed), hashlib.sha256(data).hexdigest(), len(data))])]
    # a later pass skips the done group; once the other one is quiet it follows
    os.utime(recent / 'rewards.jsonl', (1000, 1000))
    assert tier.compact_groups(collection)['rollouts_files'] == 1 and ef.group_rollouts(recent).name.endswith('.zst')
    # a rescore (JOLT+ attempts, an old pin) writes rollouts.jsonl again: the stale .zst and the marker go
    from natlang_neuralese.rl.task_rollouts import score  # noqa: F401  (the writer calls drop_stale_packed)
    (group / 'rollouts.jsonl').write_bytes(data + data)
    ef.drop_stale_packed(group / 'rollouts.jsonl')
    assert not packed.exists() and not (group / ef.ROLLOUTS_MARKER).exists()
    assert ef.group_rollouts(group) == group / 'rollouts.jsonl'
    # opt-outs: NATLANG_ROLLOUTS_COMPACT=0, or a puller without the compacted-rollouts excludes (rollouts=False)
    other = tmp_path / 'other' / 'v0012-401be1c55949'
    g3, _, _ = rollouts_group(other, 3)
    monkeypatch.setenv('NATLANG_ROLLOUTS_COMPACT', '0')
    assert not tier.compact_groups(other).get('rollouts_files') and (g3 / 'rollouts.jsonl').exists()
    monkeypatch.delenv('NATLANG_ROLLOUTS_COMPACT')
    assert not tier.compact_groups(other, rollouts=False).get('rollouts_files') and (g3 / 'rollouts.jsonl').exists()
    old = tmp_path / 'old' / 'remote_collect.py'
    old.parent.mkdir()
    old.write_text('COMPACTED_RESULTS_EXCLUDED = True\n')
    procs = [('1', 'cmdline', f'python {old} run --work {other} -- python collect')]
    assert tier.remote_puller_blocks(other, procs, []) is None
    assert tier.remote_puller_blocks(other, procs, [], 'COMPACTED_ROLLOUTS_EXCLUDED')
    current = REPO / 'scripts' / 'neuralese_data' / 'remote_collect.py'
    assert tier.remote_puller_blocks(other, [('1', 'cmdline', f'python {current} run --work {other} ')], [],
                                     'COMPACTED_ROLLOUTS_EXCLUDED') is None


def test_the_registry_records_a_compacted_rollouts_file_with_its_raw_digest(tmp_path, monkeypatch):
    for name in ('data_nvme', 'data_hdd', 'data_drive', 'repo_runs'):
        monkeypatch.setenv(f'NATLANG_{name.upper()}', str(tmp_path / name))
    monkeypatch.setenv('NATLANG_DATA_REGISTRY', str(tmp_path / 'registry'))
    monkeypatch.setenv('NATLANG_HASH_CACHE', str(tmp_path / 'hash-cache.jsonl'))
    import ei_round_tier as tier
    import expert_iteration
    from natlang_neuralese.common import data_registry as dr
    collection = tmp_path / 'data_nvme' / 'collect' / 'v0011-401be1c55949'
    group, data, _ = rollouts_group(collection)
    (collection / 'collect.json').write_text('{}\n')
    expert_iteration.record_collection(collection, closed=False)
    identity = dr.dataset_id(collection, 'grpo-rollouts')
    assert [f['path'] for f in dr.load(fresh=True).files(identity)].count('groups/g000001/rollouts.jsonl') == 1
    tier.compact_groups(collection)
    import hashlib
    extra = [f for f in dr.load(fresh=True).files(identity) if f.get('compacted_from')]
    assert extra and extra[0]['path'] == 'groups/g000001/rollouts.jsonl.zst'
    assert extra[0]['compacted_from'] == {'path': 'groups/g000001/rollouts.jsonl',
                                          'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}
    assert hashlib.sha256(ef.read_bytes(collection / extra[0]['path'])).hexdigest() == extra[0]['compacted_from']['sha256']
    assert dr.load(fresh=True).find(group / 'rollouts.jsonl.zst')['id'] == identity     # owned: readers may read it
    expert_iteration.record_collection(collection, closed=False)                         # the next snapshot names it
    assert 'groups/g000001/rollouts.jsonl.zst' in [f['path'] for f in dr.load(fresh=True).files(identity)]


def test_every_reader_class_reads_a_compacted_group(tmp_path, monkeypatch):
    import queue
    import threading
    import grpo
    import prompt_cache_map
    import sft_sources
    collection = tmp_path / 'collect' / 'v0011-401be1c55949'
    group, data, rows = rollouts_group(collection)
    ef.compact_rollouts(group)
    assert not (group / 'rollouts.jsonl').exists() and ef.group_rollouts(group) == group / 'rollouts.jsonl.zst'
    # the shared opener: text, bytes and rows from the raw name
    with ef.open_lines(group / 'rollouts.jsonl') as handle:
        assert [json.loads(line) for line in handle] == rows
    with ef.open_lines(group / 'rollouts.jsonl', binary=True) as handle:
        assert b''.join(handle) == data
    assert ef.read_rows(group / 'rollouts.jsonl.zst') == rows
    # existence as the "group scored" marker: a resumed collector reuses it
    c = grpo.Collector.__new__(grpo.Collector)
    c.work, c.lock, c.groups, c.outstanding, c.ready = collection, threading.Lock(), {}, 0, queue.Queue()
    assert c.reuse_scored(3, 2) == (['g000001'], [])
    # the learner's content reader (load_group) takes its rows through the shared opener
    read = []
    real = ef.read_rows
    monkeypatch.setattr(ef, 'read_rows', lambda path: read.append(real(path)) or read[-1])
    try:
        grpo.load_group(SimpleNamespace(groups={1: {'dir': group, 'versions': [3, 3]}}, args=SimpleNamespace(),
                                        generated=0, episodes=0, states=None), 1, __import__('collections').Counter())
    except Exception:       # the minimal rows lack the learner's capture fields; what it read is what counts here
        pass
    assert read == [rows]
    monkeypatch.setattr(ef, 'read_rows', real)
    # the SFT extraction (listing and its race fallback) and the call-level engine's units
    listed = list(sft_sources.rl_rollout_files(collection, tmp_path / 'no-archive'))
    assert listed == [('g000001', group / 'rollouts.jsonl.zst', group)]
    _, _, counts, _ = sft_sources._extract_rl_group('rl', 'g000001', group / 'rollouts.jsonl', group,
                                                    tmp_path / 'no-archive')
    assert counts['episodes'] == 2
    # the prompt cache map's glob
    assert len(list(prompt_cache_map.collect_calls(collection, 0))) == 2
    # the step verifier's and twin builder's group walk
    from natlang_neuralese.rl import held_eval
    from natlang_neuralese.train import step_verifier
    monkeypatch.setattr(held_eval, 'training_exclusion_ids', lambda cases, manifest=None: set())
    funnel = step_verifier.build_t0_sequences([collection], tmp_path / 't0.jsonl.zst', policy_sha256='q' * 64)
    assert funnel['groups'] == 1 and funnel['rollouts'] == 2 and funnel['other_policy'] == 2
    # an archived (pruned) group: the compacted file is copied as it is
    archive = tmp_path / 'archive'
    monkeypatch.setattr(grpo, 'registry_files', lambda *a: None)
    stats = __import__('collections').Counter()
    c.groups[1] = {'dir': group}
    grpo.prune_group(c, 1, keep=False, archive=archive, stats=stats)
    assert ef.read_bytes(archive / 'g000001.rollouts.jsonl.zst') == data and stats['rollouts_archived'] == 1
    assert not (group / 'rollouts.jsonl.zst').exists()


def test_the_remote_pull_never_pulls_a_compacted_rollouts_file_again(tmp_path, monkeypatch):
    import remote_collect as rc
    vm, work = tmp_path / 'vm', tmp_path / 'W'

    def fake(*args, timeout=3600, check=True):
        argv = [str(vm) + str(a)[len('vm:'):] if str(a).startswith('vm:') else str(a) for a in args]
        return subprocess.run(['rsync', '-a', *argv], check=check, capture_output=True, text=True)
    monkeypatch.setattr(rc, 'rsync', fake)
    remote = Path(str(vm) + str(work))
    group, data, _ = rollouts_group(remote)
    work.mkdir(parents=True)
    rc.pull('vm', work)
    local = work / group.relative_to(remote)
    assert (local / 'rollouts.jsonl').read_bytes() == data
    ef.compact_rollouts(local)
    os.utime(group / 'score.json')                  # a changed marker on the VM: the marker pass runs again
    assert rc.pull('vm', work) >= 1
    assert not (local / 'rollouts.jsonl').exists() and ef.read_bytes(local / 'rollouts.jsonl') == data
    assert 'rollouts.jsonl.zst' in rc.PUSH_EXCLUDE and rc.COMPACTED_ROLLOUTS_EXCLUDED
