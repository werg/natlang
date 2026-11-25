"""remote_collect.py: a Colab actor's collector runs on its VM (DECISIONS 2026-11-25 "collectors run beside their remote
actors"): the data mirror (symlinks kept), the staged pull (scores never before their episodes), the VM launch, the audit
of missing reads, and the async driver's targets (every --colab-hosts session, Pop kept on the DGX, --collect-on-dgx)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / 'training/neuralese', REPO, REPO / 'scripts/neuralese_data', REPO / 'scripts'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import remote_collect as rc  # noqa: E402


def test_the_mirror_plan_keeps_every_symlink_on_the_way(tmp_path):
    real = tmp_path / 'hdd' / 'pool'
    real.mkdir(parents=True)
    (real / 'cases.jsonl').write_text('{}\n')
    (tmp_path / 'nvme').mkdir()
    (tmp_path / 'nvme' / 'pool').symlink_to(real)                    # data_nvme:pool -> the HDD
    (tmp_path / 'checkout').mkdir()
    (tmp_path / 'checkout' / 'runs').symlink_to(tmp_path / 'nvme')   # a pin's runs -> the checkout's
    lexical = tmp_path / 'checkout' / 'runs' / 'pool' / 'cases.jsonl'
    links, files = rc.mirror_plan([str(lexical), str(tmp_path / 'absent')])
    assert links == sorted([(str(tmp_path / 'checkout' / 'runs'), str(tmp_path / 'nvme')),
                            (str(tmp_path / 'nvme' / 'pool'), str(real))])
    assert files == [str(real / 'cases.jsonl')]


def test_scanned_paths_are_existing_files_and_deep_directories(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, 'PATH_RE', __import__('re').compile(rb'(?:' + str(tmp_path).encode() + rb')/[A-Za-z0-9._+\-/]+'))
    monkeypatch.setattr(rc, 'MIN_DIR_DEPTH', len(tmp_path.parts) + 1)
    assets = tmp_path / 'a' / 'assets'
    assets.mkdir(parents=True)
    (assets / 'x.db').write_text('x')
    (tmp_path / 'f.json').write_text('{}')
    pool = tmp_path / 'pool.jsonl'
    pool.write_text(json.dumps({'code': f'const root = "{assets}"; read("{assets}/x.db"); "{tmp_path}/f.json" '
                                        f'"{tmp_path}/missing/y" "{tmp_path}/a"'}) + '\n')
    cache = tmp_path / 'cache.json'
    found = rc.scan_paths([str(pool)], cache)
    assert found == [str(assets), str(tmp_path / 'f.json')]          # x.db inside assets; a too shallow; missing gone
    assert rc.scan_paths([str(pool)], cache) == found and cache.exists()


def test_pull_end_to_end_with_a_local_vm(tmp_path, monkeypatch):
    vm = tmp_path / 'vm'
    work = tmp_path / 'data' / 'W'
    # the VM mirrors absolute paths: emulate with a VM root and rewrite host:PATH to VMROOT/PATH
    def fake(*args, timeout=3600, check=True):
        argv = [str(vm) + str(a)[len('vm:'):] if str(a).startswith('vm:') else str(a) for a in args]
        return subprocess.run(['rsync', '-a', *argv], check=check, capture_output=True, text=True)
    monkeypatch.setattr(rc, 'rsync', fake)
    remote = Path(str(vm) + str(work))
    group = remote / 'groups' / 'g000001'
    (group / 'workers').mkdir(parents=True)
    (group / 'workers' / 's7-task-000-00.journal.jsonl').write_text('{}\n')
    (group / 'rewards.jsonl').write_text('{"reward": 1}\n')
    (group / 'rollouts.jsonl').write_text('{"episode": 1}\n')
    (remote / '.remote-vm').mkdir()
    (remote / '.remote-vm' / 'remote_collect.py').write_text('#')
    work.mkdir(parents=True)
    (work / 'collector.json').write_text('{"unit": "dgx"}')
    calls = []

    def spy(*args, **kw):
        calls.append(args)
        if len(calls) == 3:   # the scores are staged, the content pulled again: none in place yet
            assert (work / 'groups' / 'g000001' / 'workers' / 's7-task-000-00.journal.jsonl').exists()
            assert not (work / 'groups' / 'g000001' / 'rewards.jsonl').exists()
            assert (work / '.remote-stage' / 'groups' / 'g000001' / 'rewards.jsonl').exists()
        return fake(*args, **kw)
    monkeypatch.setattr(rc, 'rsync', spy)
    assert rc.pull('vm', work) == 2 and len(calls) == 3
    assert (work / 'groups' / 'g000001' / 'rewards.jsonl').read_text() == '{"reward": 1}\n'
    assert (work / 'groups' / 'g000001' / 'workers' / 's7-task-000-00.journal.jsonl').exists()
    assert not (work / '.remote-vm').exists() and json.loads((work / 'collector.json').read_text())['unit'] == 'dgx'
    assert not list((work / '.remote-stage').glob('groups/*/*'))          # moved, never left behind
    assert rc.pull('vm', work) == 0                                        # unchanged: nothing staged
    (group / 'rewards.jsonl').write_text('{"reward": 0, "teacher": 2}\n')  # rescored after teacher attempts
    os.utime(group / 'rewards.jsonl', (1, 1))
    assert rc.pull('vm', work) == 1
    assert 'teacher' in (work / 'groups' / 'g000001' / 'rewards.jsonl').read_text()


def test_the_vm_command_is_detached_admitted_and_records_its_exit():
    cfg = {'python': '/c/natlang/.venv-neuralese/bin/python', 'admission': '/c/natlang/scripts/colab/admission.py',
           'env': '/c/natlang-env.sh'}
    cmd = rc.vm_command(cfg, '/p/pin-abc', '/d/W', ['/p/pin-abc/.venv-neuralese/bin/python',
                                                    '/p/pin-abc/scripts/neuralese_data/expert_iteration.py', 'collect',
                                                    '--work', '/d/W', '--slots', '64'], 64)
    assert cmd.startswith('. /c/natlang-env.sh || exit 1;') and 'setsid nohup bash -c' in cmd and cmd.endswith('& echo $!')
    assert '--vram-gib 0 --host-ram-gib 18.8' in cmd and '/d/W/.remote-vm/remote_collect.py vm --work /d/W --' in cmd
    assert 'remote-collect.exit' in cmd and 'NATLANG_REPO=/p/pin-abc' in cmd


def test_the_audit_counts_reads_only():
    assert rc.reads('r', 0) and rc.reads('rb', 0) and not rc.reads('w', 0) and not rc.reads('a+', 0)
    assert rc.reads(None, os.O_RDONLY) and not rc.reads(None, os.O_WRONLY | os.O_CREAT)


def test_missing_paths_are_the_ones_the_dgx_has_and_did_not_mirror(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, 'MIRROR_ROOTS', (str(tmp_path) + '/',))
    (tmp_path / 'have.json').write_text('{}')
    (tmp_path / 'W').mkdir()
    (tmp_path / 'W' / 'missing-paths.jsonl').write_text(''.join(json.dumps({'path': str(p)}) + '\n' for p in (
        tmp_path / 'have.json', tmp_path / 'nowhere.json', '/etc/elsewhere')))
    assert rc.missing_on_dgx(tmp_path / 'W', set()) == [str(tmp_path / 'have.json')]
    assert rc.missing_on_dgx(tmp_path / 'W', {str(tmp_path / 'have.json')}) == []


def test_options_read_both_spellings():
    argv = ['collect', '--tasks', 'T', '--pool=A', '--pool', 'B', '--slots', '64']
    assert rc.option(argv, '--tasks') == 'T' and rc.option(argv, '--absent', 'd') == 'd'
    assert sorted(rc.options(argv, '--pool')) == ['A', 'B']


# ---- the async driver -------------------------------------------------------------------------------------------------
def _driver(tmp_path, monkeypatch, *argv):
    from test_async_train import _remote_driver
    ei, d = _remote_driver(tmp_path, monkeypatch, *argv)
    return ei, d


def test_every_colab_session_is_a_target_and_colab_collectors_run_on_their_vm(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / 'tests' / 'neuralese'))
    ei, d = _driver(tmp_path, monkeypatch, '--remote-slots', '64')
    import pop_rl_actor
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path / 'pop-eval')
    (tmp_path / 'pop-eval').mkdir()
    pop_rl_actor.write_json(pop_rl_actor.colab_state_path('colab-g4b'), {'sglang': {'max_running': 48}})
    d.colab_hosts = ['colab-g4', 'colab-g4b']
    t = d.remote_targets()
    assert [(x['host'], x['suffix'], x['slots'], x['session'], x['on_host']) for x in t] == [
        ('colab-g4', '-remote0', 64, False, True), ('colab-g4b', '-remote-colab-g4b', 48, True, True)]
    assert t[1]['front'] == 'http://127.0.0.1:8570'
    cur = d.controller.current
    unit = d.collection_unit(cur)
    assert d.remote_units(unit) == [f'{unit}-remote0', f'{unit}-remote-colab-g4b']
    _u, work, argv, budget = d.remote_collect_argv(cur, t[1], 'http://127.0.0.1:20731')
    assert budget == rc.REMOTE_SYNC_GB and argv[argv.index('--host') + 1] == 'colab-g4b'
    assert argv[argv.index('--work') + 1] == str(work) and work.name.endswith('-remote-colab-g4b')
    inner = argv[argv.index('--') + 1:]
    assert inner[2] == 'collect' and inner[inner.index('--actor') + 1] == 'http://127.0.0.1:8570=48'
    assert inner[inner.index('--work') + 1] == str(work)
    # opt-out: the collector on the DGX through the forwarded endpoint, the old claim
    _e, d2 = _driver(tmp_path / 'off', monkeypatch, '--collect-on-dgx')
    (t0,) = d2.remote_targets()
    assert not t0['on_host']
    _u, _w, argv, budget = d2.remote_collect_argv(d2.controller.current, t0, 'http://127.0.0.1:18731')
    assert argv[argv.index('--actor') + 1] == 'http://127.0.0.1:18731=64'
    assert budget == max(8, round(64 * ei.REMOTE_COLLECTOR_SLOT_GB) + 4)


def test_a_session_without_an_actor_is_served_the_head_and_the_sft_session_is_left_out(tmp_path, monkeypatch):
    sys.path.insert(0, str(REPO / 'tests' / 'neuralese'))
    ei, d = _driver(tmp_path, monkeypatch)
    import pop_rl_actor
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path / 'pop-eval')
    d.colab_hosts = ['colab-g4', 'colab-g4b']
    cur = d.controller.current
    target = d.remote_targets()[1]
    runs = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: runs.append(argv) or
                        __import__('types').SimpleNamespace(returncode=0))
    monkeypatch.setattr(ei, 'endpoint_alive', lambda url, sha, timeout=10: False)

    def nothing_answers(url, timeout=10):
        raise OSError('connection refused')
    monkeypatch.setattr(ei.urllib.request, 'urlopen', nothing_answers)

    class Thread:
        def __init__(self, target, daemon):
            self.target = target

        def start(self):
            self.target()
    monkeypatch.setattr(ei.threading, 'Thread', Thread)
    d.colab_sft_hosts.add('colab-g4b')
    d.ensure_colab_actor(cur, target)
    assert not runs                                        # the round's SFT session: never served for collection
    d.colab_sft_hosts.clear()
    d.ensure_colab_actor(cur, target)
    (argv,) = runs
    assert argv[2:6] == ['serve-checkpoint', '--host-profile', 'colab', '--host'] and argv[6] == 'colab-g4b'
    assert argv[argv.index('--policy-id') + 1] == cur['sha256'] and argv[argv.index('--weight-dtype') + 1] == 'fp8'
    d.ensure_colab_actor(cur, target)
    assert len(runs) == 1                                  # retried only after COLAB_SERVE_RETRY_SECONDS
    _e, d2 = _driver(tmp_path / 'no', monkeypatch, '--no-colab-collect-serve')
    d2.colab_hosts = ['colab-g4', 'colab-g4b']
    d2.ensure_colab_actor(d2.controller.current, d2.remote_targets()[1])
    assert len(runs) == 1


def test_two_colab_sessions_never_share_a_dgx_port():
    import pop_rl_actor
    ports = {}
    for host in ('colab-g4', 'colab-g4b', 'colab-g4c'):
        off = pop_rl_actor.colab_forward_offset(host)
        for vm_port in (8570, 8571, 9570, 9571):
            assert off + vm_port not in ports, (host, vm_port, ports.get(off + vm_port))
            ports[off + vm_port] = host


def test_descendants_follow_parent_links(tmp_path):
    proc = tmp_path / 'proc'
    for pid, ppid in ((10, 1), (11, 10), (12, 11), (13, 1), (14, 10)):
        (proc / str(pid)).mkdir(parents=True)
        (proc / str(pid) / 'stat').write_text(f'{pid} (python (x)) S {ppid} 0 0')
    (proc / 'self').mkdir()
    assert sorted(rc.descendants(10, proc)) == [11, 12, 14]
