"""Execution profiles, Colab G4 admission and bootstrap, the remote actor and SFT, sharded collection and cost telemetry
(plans/neuralese/lfm3opti.md, async-train.md). The Cycle stages run with a Colab profile against faked remote calls."""
import collections
import json
import os
import shlex
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO), str(REPO / 'scripts/colab')]

from natlang_neuralese.common.execution_profile import (  # noqa: E402
    CostTelemetry,
    resolve_profile,
    select_stage_profile,
)
import admission as colab_admission  # noqa: E402
import bootstrap as colab_bootstrap  # noqa: E402
import expert_iteration as ei  # noqa: E402
import grpo  # noqa: E402
import pop_rl_actor  # noqa: E402


def _cycle_args(argv):
    captured = []
    old_argv, old_cycle = sys.argv, ei.cycle
    sys.argv, ei.cycle = ['expert_iteration.py', 'cycle', *argv], captured.append
    try:
        ei.main()
    finally:
        sys.argv, ei.cycle = old_argv, old_cycle
    return captured[0]


def _cycle(tmp_path, *extra):
    argv = ['--dir', str(tmp_path / 'sft-v10'), '--name', 'sft-v10', '--start', '/v9.pt', '--start-sha', 's' * 64,
            '--commit', 'c' * 40, '--pin', str(tmp_path / 'pin'), *extra]
    return ei.Cycle(_cycle_args(argv))


# ---- profiles and fallback ---------------------------------------------------------------------------------------------
def test_profiles_put_the_held_series_on_the_dgx_and_reject_l4():
    dgx, g4, pop = resolve_profile('dgx'), resolve_profile('colab_g4', hosts='g4-a,g4-b'), resolve_profile('pop')
    assert dgx.held_series_host == 'dgx' and dgx.held_samples == 4 and not dgx.is_discrete_gpu
    assert g4.held_series_host is None and g4.hosts == ('g4-a', 'g4-b') and g4.fallback_profile == 'dgx'
    assert pop.held_series_host is None and pop.held_samples == 2 and pop.sglang_max_running == 20
    for bad in ('l4', 'colab_l4', 'colab-l4', 'L4'):
        with pytest.raises(ValueError, match='excluded'):
            resolve_profile(bad)


def test_select_stage_profile_falls_back_to_dgx_after_the_wait():
    now = {'t': 0.0}

    def sleep(dt):
        now['t'] += dt
    prof, hosts, reason = select_stage_profile('colab_g4', hosts=['g4-1'], reachable_fn=lambda h: False,
                                               wait_minutes=30.0, poll_seconds=300.0, sleep_fn=sleep,
                                               clock_fn=lambda: now['t'])
    assert prof.name == 'dgx' and hosts == ['local'] and 'unreachable' in reason and now['t'] == 1800.0


def test_choose_profile_takes_a_name_and_falls_back_without_hosts(tmp_path):
    """The Colab collect fallback called ``choose_profile('dgx', ...)`` with a string: AttributeError ended the driver."""
    cycle = _cycle(tmp_path)
    assert cycle.choose_profile('dgx', 'collect')[0].name == 'dgx'
    profile, hosts = cycle.choose_profile('colab_g4', 'collect')     # no Colab host configured
    assert profile.name == 'dgx' and hosts == ['local']


def test_held_profile_choices_exclude_colab_and_default_to_dgx(tmp_path):
    cycle = _cycle(tmp_path)
    assert cycle.held_host == 'dgx' and cycle.a.held_samples == 4
    assert cycle.held_env['NATLANG_HELD_SERIES_HOST'] == 'dgx'
    with pytest.raises(SystemExit):
        _cycle_args(['--dir', 'd', '--name', 'n', '--start', 's', '--commit', 'c', '--held-profile', 'colab_g4'])


# ---- cost ----------------------------------------------------------------------------------------------------------------
def test_cost_charges_sessions_from_start_to_stop_idle_included(tmp_path):
    cost = CostTelemetry('sft-v10')
    cost.open_session('g4-1', 'colab_g4', at=0.0)
    cost.record_stage('collect', 'colab_g4', wall_seconds=1800.0, session_host='g4-1')   # busy half the session
    cost.close_session('g4-1', at=3600.0)
    cost.record_stage('sft', 'colab_g4', wall_seconds=1800.0, canonical_chain_tokens=50_000_000)
    cost.record_stage('data', 'dgx', wall_seconds=600.0)
    summary = cost.save(tmp_path / 'cycle-cost.json')
    assert summary['session_cu_spent'] == pytest.approx(13.0)          # the whole hour, not the 30 busy minutes
    assert summary['total_cu_spent'] == pytest.approx(13.0 + 6.5)
    assert summary['by_stage']['collect']['cu_spent'] == 0               # attribution only: the session charged it
    again = CostTelemetry.load(tmp_path / 'cycle-cost.json', 'sft-v10')
    assert len(again.entries) == 3 and again.sessions[0].stopped == 3600.0
    again.open_session('g4-2', 'colab_g4', at=time.time() - 7200)       # left running: charged up to now
    assert again.summary()['open_sessions'] == ['g4-2'] and again.summary()['session_cu_spent'] >= 13.0 + 26.0 - 0.1


# ---- admission -----------------------------------------------------------------------------------------------------------
def _probes(total_vram=94.0, total_host=160.0):
    return {'vram_probe': lambda: 10.0, 'host_probe': lambda: 100.0, 'total_vram_probe': lambda: total_vram,
            'total_host_probe': lambda: total_host}


def test_admission_does_not_hold_its_lock_while_a_long_lived_actor_runs(tmp_path):
    """The shim held its flock for the child's lifetime: every command after a nohup'd actor timed out (exit 75)."""
    state = tmp_path / 'adm'
    release = tmp_path / 'release'
    actor = threading.Thread(target=colab_admission.wait_and_run, kwargs=dict(
        command=['/bin/sh', '-c', f'while [ ! -e {release} ]; do sleep 0.1; done'], vram_gib=34.0, host_ram_gib=24.0,
        label='actor', wait_seconds=5, state_dir=state, probes={**_probes(), 'vram_probe': lambda: 90.0}))
    actor.start()
    for _ in range(100):
        if colab_admission.read_claims(state):
            break
        time.sleep(0.05)
    assert list(colab_admission.read_claims(state)) and actor.is_alive()
    started = time.monotonic()
    rc = colab_admission.wait_and_run(['/bin/true'], vram_gib=59.0, host_ram_gib=40.0, label='sft', wait_seconds=5,
                                      state_dir=state, probes=_probes())    # the learner beside the capped actor
    assert rc == 0 and time.monotonic() - started < 3
    rc = colab_admission.wait_and_run(['/bin/true'], vram_gib=60.0, host_ram_gib=1.0, wait_seconds=0.5, poll_seconds=0.1,
                                      state_dir=state, probes=_probes())    # 34 + 60 > 93: does not fit
    assert rc == colab_admission.EXIT_TIMEOUT
    release.touch()
    actor.join(timeout=10)
    assert colab_admission.read_claims(state) == {}


def test_admission_without_claims_needs_the_probed_free_memory():
    ok, info = colab_admission.decide(80.0, 32.0, {}, total_vram=94.0, total_host=160.0, free_vram=40.0, free_host=120.0)
    assert not ok and 'unadmitted' in info['reasons'][0]
    ok, _ = colab_admission.decide(80.0, 32.0, {}, total_vram=94.0, total_host=160.0, free_vram=90.0, free_host=120.0)
    assert ok
    ok, info = colab_admission.decide(1.0, 157.0, {}, total_vram=94.0, total_host=160.0, free_vram=90.0, free_host=160.0)
    assert not ok and info['reasons'][0].startswith('host_ram')


def test_g4_actor_beside_the_learner_and_the_sft_claim_fit_one_gpu():
    cfg = pop_rl_actor.host_config('colab_g4', host='g4-1')
    serve, claim, sglang, fmt = pop_rl_actor.colab_serve_command(cfg, checkpoint='/c.pt', sha='a' * 64, port=18531,
                                                                  work='/w', beside_learner=True,
                                                                  kv_cache_dtype='fp8_e4m3')
    assert serve[serve.index('--sglang-kv-gb') + 1] == '24' and serve[serve.index('--sglang-max-running') + 1] == '64'
    assert '--sglang-mem-fraction' not in serve and serve[serve.index('--sglang-python') + 1] == cfg['sglang_python']
    sft_claim = pop_rl_actor.G4_VRAM_GIB - pop_rl_actor.G4_ACTOR_BESIDE_LEARNER['vram_gib'] - 1.0   # run-sft default
    ok, _ = colab_admission.decide(sft_claim, 40.0, {'actor': {'vram_gib': claim['vram_gib'], 'host_ram_gib': 24.0}},
                                   total_vram=94.0, total_host=160.0, free_vram=50.0, free_host=100.0)
    assert ok
    alone = pop_rl_actor.colab_serve_command(cfg, checkpoint='/c.pt', sha='a' * 64, port=18531, work='/w',
                                             beside_learner=False, kv_cache_dtype='fp8_e4m3')
    assert alone[0][alone[0].index('--sglang-max-running') + 1] == '128' and '--sglang-mem-fraction' in alone[0]
    assert fmt == 'sglang-colab-g4-bf16-vendor-projection-kv-fp8_e4m3'


# ---- bootstrap -----------------------------------------------------------------------------------------------------------
def test_tailscaled_start_command_starts_the_daemon_as_it_runs(tmp_path):
    """``pgrep -f tailscaled`` matched its own ``sh -c`` line, so the daemon never started (reproduced)."""
    bin_dir = tmp_path / 'bin'
    bin_dir.mkdir()
    marker = tmp_path / 'started'
    fake = bin_dir / 'tsdfake4711'       # not the host's own tailscaled (the DGX runs one)
    fake.write_text('#!/bin/sh\n'
                    f'touch {marker}\n'
                    'for a in "$@"; do case "$a" in --socket=*) sock="${a#--socket=}";; esac; done\n'
                    f'exec {sys.executable} -c "import socket,sys,time; s=socket.socket(socket.AF_UNIX); s.bind(sys.argv[1]); '
                    'time.sleep(3)" "$sock"\n')
    fake.chmod(0o755)
    state = tmp_path / 'ts'
    state.mkdir()
    command = colab_bootstrap.tailscaled_start_command(state, tmp_path, daemon='tsdfake4711')
    assert 'pgrep -x tsdfake4711' in command and 'pgrep -x tailscaled' in colab_bootstrap.tailscaled_start_command(state, tmp_path)
    env = {**os.environ, 'PATH': f'{bin_dir}:{os.environ["PATH"]}'}
    result = subprocess.run(command, shell=True, env=env, timeout=60)
    assert result.returncode == 0 and marker.exists() and (state / 'tailscaled.sock').exists()


def test_bootstrap_never_prints_the_tailscale_key_and_sets_up_the_dgx_environment(tmp_path, capsys):
    secret = 'tskey-auth-kSECRET123456-abcdefABCDEF'
    ready = colab_bootstrap.bootstrap(tailscale_authkey=secret, ssh_authorized_key='ssh-ed25519 AAAA',
                                      hostname='colab-g4-1', commit='abc1234', root=tmp_path / 'colab', dry_run=True)
    out = capsys.readouterr().out
    assert secret not in out and secret not in json.dumps(ready) and 'auth-key=file:' in out
    commands = [json.loads(line)['cmd'] for line in out.splitlines() if '"exec"' in line]
    assert not any(' -e ' in c and c.rstrip().endswith('training/neuralese') for c in commands)    # not a package
    assert any('training/neuralese/docker/requirements.txt' in c for c in commands)
    fork = json.loads((REPO / 'training/neuralese/sglang-fork.json').read_text())
    assert any('checkout' in c and fork['commit'] in c for c in commands)              # the pinned SGLang fork
    assert any(fork['repository'] in c for c in commands) and not any('sglang[all]' in c for c in commands)
    assert any('nvidia-cuda-mps-control -d' in c for c in commands) and any('nodejs.org' in c for c in commands)
    assert ready['plan']['steps'][-2:] == ['start_mps', 'probe_hardware_and_write_ready']
    env = colab_bootstrap.env_script(Path('/r/natlang'), Path('/r/natlang/.venv-neuralese'), Path('/r/node/bin'), Path('/r'))
    assert 'PYTHONPATH=/r/natlang/training/neuralese:/r/natlang' in env and 'CUDA_MPS_PIPE_DIRECTORY' in env


def test_bootstrap_failure_messages_are_redacted(monkeypatch):
    colab_bootstrap._SECRETS[:] = ['tskey-auth-abc123XYZ']
    monkeypatch.setattr(colab_bootstrap.subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(
        a, 1, stdout='', stderr='bad key tskey-auth-abc123XYZ'))
    with pytest.raises(RuntimeError) as error:
        colab_bootstrap._run(['tailscale', 'up', '--authkey=tskey-auth-abc123XYZ'])
    assert 'tskey-auth-abc123XYZ' not in str(error.value) and '<redacted>' in str(error.value)


@pytest.mark.parametrize('version,ok', [('v22.12.0', False), ('v22.13.0', True), ('v24.0.1\n', True), ('', False),
                                        ('v20.18.1', False)])
def test_node_version_floor(version, ok):
    assert colab_bootstrap.node_ok(version) is ok


# ---- remote SFT ----------------------------------------------------------------------------------------------------------
@pytest.fixture
def remote(tmp_path, monkeypatch):
    """A fake Colab VM: files under ``root``; ssh_host runs the few commands the code sends."""
    root = tmp_path / 'vm'
    root.mkdir()
    state = {'ssh': [], 'run': [], 'truncate': False}

    def to_local(path):
        return root / str(path).lstrip('/')

    def ssh_host(host, command, **kw):
        state['ssh'].append(command)
        words = shlex.split(command.split('&&')[0]) if not command.startswith('.') else []
        if command.startswith('stat -c %s'):
            path = to_local(shlex.split(command)[3])
            import hashlib
            return f'{path.stat().st_size}\n{hashlib.sha256(path.read_bytes()).hexdigest()}  {path}\n'
        if words[:1] == ['test']:
            return 'ok' if to_local(words[2]).exists() else ''
        if words[:1] == ['mv']:
            os.replace(to_local(words[1]), to_local(words[2]))
        if words[:2] == ['mkdir', '-p']:
            for path in words[2:]:
                to_local(path).mkdir(parents=True, exist_ok=True)
        return ''

    def sync_to(local, remote_path, host, **kw):
        dest = to_local(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if Path(local).is_file():
            dest.write_bytes(Path(local).read_bytes())

    def sync_from(remote_path, local, host, **kw):
        src = to_local(remote_path.rstrip('/'))
        if src.is_file():
            data = src.read_bytes()
            Path(local).write_bytes(data[:len(data) // 2] if state['truncate'] else data)

    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh_host)
    monkeypatch.setattr(pop_rl_actor, 'sync_to_remote', sync_to)
    monkeypatch.setattr(pop_rl_actor, 'sync_from_remote', sync_from)
    state['to_local'] = to_local
    return state


def test_pull_verified_never_leaves_a_truncated_checkpoint(tmp_path, remote):
    src = remote['to_local']('/content/natlang-work/runs/sft-x/checkpoint.pt')
    src.parent.mkdir(parents=True)
    src.write_bytes(b'w' * 1000)
    local = tmp_path / 'sft' / 'checkpoint.pt'
    local.parent.mkdir()
    remote['truncate'] = True
    with pytest.raises(RuntimeError, match='discarded'):
        pop_rl_actor.pull_verified('/content/natlang-work/runs/sft-x/checkpoint.pt', local, 'g4')
    assert not local.exists() and not list(local.parent.iterdir())
    remote['truncate'] = False
    import hashlib
    assert pop_rl_actor.pull_verified('/content/natlang-work/runs/sft-x/checkpoint.pt', local, 'g4') == \
        hashlib.sha256(b'w' * 1000).hexdigest() and local.read_bytes() == b'w' * 1000


def _launch_json(tmp_path):
    chains, ckpt, val = tmp_path / 'mix/chains.pkl', tmp_path / 'start.pt', tmp_path / 'mix/val-own.pkl'
    chains.parent.mkdir()
    chains.write_bytes(b'chains')
    ckpt.write_bytes(b'start')
    val.write_bytes(b'val')
    import hashlib
    base = '/hf/snapshots/40ebe0aa'
    work = tmp_path / 'sft'
    args = (f'train --chains {chains} --checkpoint {ckpt} --base {base} --out-dir {work} --epochs 1 --lr 5e-5 '
            f'--warmup 8 --min-lr-fraction 0.1 --clip 1.0 --batch-loss-tokens 32000 --activation-gb 44.0 '
            f'--val-chains {val} --val-every 20')
    work.mkdir()
    (work / 'launch.json').write_text(json.dumps({
        'exp0_sft_args': args, 'chains_sha256': hashlib.sha256(b'chains').hexdigest(),
        'start_checkpoint_sha256': hashlib.sha256(b'start').hexdigest()}))
    return work


def test_run_remote_sft_runs_the_dgx_recipe_and_pulls_only_a_verified_checkpoint(tmp_path, remote, monkeypatch):
    """The Colab SFT ran exp0_sft without --epochs 1 (default 3), without the preflight and --val-every; and pulled
    checkpoint.pt with rsync --partial to its final name, where a truncated copy was adopted."""
    work = _launch_json(tmp_path)
    ran = []

    def run(argv, **kw):
        ran.append(argv)
        remote_cmd = argv[-1]
        out_dir = shlex.split(remote_cmd.split('&&', 1)[1])
        out_dir = out_dir[out_dir.index('--out-dir') + 1]
        dest = remote['to_local'](out_dir) / 'checkpoint.pt'
        dest.write_bytes(b'trained' * 100)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(pop_rl_actor.subprocess, 'run', run)
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    args = types.SimpleNamespace(host='g4', run_name='sft-v10', work=str(work), launch_json=str(work / 'launch.json'),
                                 commit='c' * 40, state_sync_minutes=15.0, vram_gib=59.0, host_ram_gib=40.0)
    remote['truncate'] = True
    assert pop_rl_actor.run_remote_sft(args) == 1 and not (work / 'checkpoint.pt').exists()
    remote['truncate'] = False
    assert pop_rl_actor.run_remote_sft(args) == 0 and (work / 'checkpoint.pt').read_bytes() == b'trained' * 100
    remote_cmd = shlex.split(ran[-1][-1].split('&&', 1)[1])
    sft = remote_cmd[remote_cmd.index('--') + 1:]
    assert sft[1].endswith('scripts/neuralese_data/exp0_sft.py') and sft[2] == 'train'
    assert sft[sft.index('--epochs') + 1] == '1' and sft[sft.index('--val-every') + 1] == '20'
    assert sft[sft.index('--base') + 1] == '/content/natlang-work/base-lfm3'
    assert sft[sft.index('--out-dir') + 1] == '/content/natlang-work/runs/sft-sft-v10'
    assert sft[sft.index('--checkpoint') + 1].startswith('/content/natlang-work/checkpoints/')
    assert sft[sft.index('--val-chains') + 1].startswith('/content/natlang-work/runs/sft-sft-v10/val-0-')
    assert '--resume' in sft and sft[sft.index('--state-save-minutes') + 1] == '15.0'
    assert not any(str(tmp_path) in a for a in sft)       # no DGX path reaches the VM
    assert remote_cmd[remote_cmd.index('--vram-gib') + 1] == '59.0'


def test_cycle_sft_with_a_colab_profile_preflights_on_the_dgx_and_falls_back_without_adopting(tmp_path, monkeypatch):
    cycle = _cycle(tmp_path, '--sft-profile', 'colab_g4', '--colab-hosts', 'g4-1')
    cycle.mix.mkdir(parents=True)
    (cycle.mix / 'chains.sha256').write_text('abc\n')
    (cycle.mix / 'val-own.pkl').write_bytes(b'v')
    monkeypatch.setattr(ei.Cycle, 'choose_profile', lambda self, p, stage: (resolve_profile('colab_g4'), ['g4-1']))
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: False))
    monkeypatch.setattr(ei.Cycle, 'wait', lambda self, unit, poll=60: None)
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw.get('env') or {}))
        if 'run-sft' in argv:        # the remote run failed half-way and left a partial file under .pull
            (cycle.dir / 'sft').mkdir(exist_ok=True)
            (cycle.dir / 'sft' / 'checkpoint.pt.pull').write_bytes(b'half')
            return subprocess.CompletedProcess(argv, 1)
        return subprocess.CompletedProcess(argv, 0)
    monkeypatch.setattr(ei.subprocess, 'run', run)
    with pytest.raises(SystemExit, match='no SFT checkpoint'):      # the DGX run (stubbed) wrote none
        cycle.sft_stage()
    preflight, colab, dgx = calls[0], calls[1], calls[2]
    assert preflight[0][-2].endswith('launch-sft.sh') and preflight[1]['PREFLIGHT_ONLY'] == '1'
    assert preflight[1]['ACT'] == '44.0' and '--val-every 20' in preflight[1]['EXTRA']
    assert colab[0][colab[0].index('--launch-json') + 1] == str(cycle.dir / 'sft' / 'launch.json')
    assert dgx[0][-2].endswith('launch-sft.sh') and 'PREFLIGHT_ONLY' not in dgx[1]
    assert dgx[1]['ACT'] == str(cycle.a.sft_activation_gb) and dgx[1]['CHAINS_SHA256'] == 'abc'
    assert not cycle.done('sft')
    cost = json.loads((cycle.dir / 'cycle-cost.json').read_text())
    assert cost['sessions'][0]['host'] == 'g4-1' and cost['sessions'][0]['stopped'] is not None


# ---- collection ----------------------------------------------------------------------------------------------------------
def test_cycle_collect_on_colab_runs_the_same_command_with_variants_and_stops_its_actors(tmp_path, monkeypatch):
    cycle = _cycle(tmp_path, '--collect-profile', 'colab_g4', '--colab-hosts', 'g4-1,g4-2')
    monkeypatch.setattr(ei.Cycle, 'choose_profile', lambda self, p, stage: (resolve_profile('colab_g4'), ['g4-1', 'g4-2']))
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    calls, units = [], []

    def run(argv, **kw):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    def check_output(argv, **kw):
        host = argv[argv.index('--host') + 1]
        return json.dumps({'endpoint': f'http://{host}:18531'}) + '\n'
    monkeypatch.setattr(ei.subprocess, 'run', run)
    monkeypatch.setattr(ei.subprocess, 'check_output', check_output)
    active = iter([True, False])
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: next(active, False)))
    monkeypatch.setattr(ei.Cycle, 'ledger', lambda self, unit, budget, prio, command, log, detach=False, **k:
                        units.append((unit, budget, command, detach)))
    cycle.collect_dir.mkdir(parents=True)
    (cycle.collect_dir / 'collect.json').write_text(json.dumps({'stopped': False, 'groups_done_this_run': 3}))
    cycle.collect_stage()
    unit, budget, command, detach = units[0]
    assert detach and budget == round(256 * ei.COLLECTOR_SLOT_GB) + 4
    assert command[command.index('--neuralese-variants') + 1].endswith('cases.jsonl')          # one command builder
    assert [command[i + 1] for i, a in enumerate(command) if a == '--actor'] == ['http://g4-1:18531=128',
                                                                                 'http://g4-2:18531=128']
    stops = [c for c in calls if 'stop' in c and '--host-profile' in c]
    assert sorted(c[c.index('--host') + 1] for c in stops) == ['g4-1', 'g4-2']
    cost = json.loads((cycle.dir / 'cycle-cost.json').read_text())
    assert {s['host'] for s in cost['sessions']} == {'g4-1', 'g4-2'} and not cost['open_sessions']
    assert cycle.done('collect') and json.loads((cycle.dir / 'stages/collect.done').read_text())['profile'] == 'colab_g4'


def test_cycle_collect_on_colab_stops_for_a_recollection_and_still_stops_its_actor(tmp_path, monkeypatch):
    previous = tmp_path / 'sft-v9'
    previous.mkdir()
    (previous / 'verdict.json').write_text(json.dumps({'verdict': 'failure', 'adopt': False, 'next_start': '/v8.pt'}))
    cycle = _cycle(tmp_path, '--collect-profile', 'colab_g4', '--speculative-after', str(previous))
    monkeypatch.setattr(ei.Cycle, 'choose_profile', lambda self, p, stage: (resolve_profile('colab_g4'), ['g4-1']))
    monkeypatch.setattr(ei.time, 'sleep', lambda s: None)
    stopped, calls = [], []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: calls.append(argv) or subprocess.CompletedProcess(argv, 0))
    monkeypatch.setattr(ei.subprocess, 'check_output', lambda argv, **kw: '{"endpoint": "http://g4-1:18531"}\n')
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: True))
    monkeypatch.setattr(ei.Cycle, 'ledger', lambda *a, **k: None)
    monkeypatch.setattr(ei.Cycle, 'stop_unit', lambda self, unit: stopped.append(unit))
    cycle.collect_stage()
    assert stopped[0] == 'sft-v10-collect' and not cycle.done('collect')
    assert any('stop' in c and 'g4-1' in c for c in calls)


class _ShardBase:
    """The parts of grpo.Collector the sharded slot uses (``_abandon`` is grpo's own)."""
    _abandon = grpo.Collector._abandon

    def __init__(self, tasks, work):
        self.by_id = {t['id']: t for t in tasks}
        self.own, self.pending, self.lock, self.outstanding = {}, collections.deque(), threading.Lock(), 0
        self.paused, self.stop = threading.Event(), threading.Event()
        self.args = types.SimpleNamespace(max_outstanding=8, actor=None, slots=2)
        self.work = self.runtime = work
        self.env, self.groups, self.version, self.running, self.active_slots = {}, {}, 0, 0, 2
        self.policy_sha, self.scored, self.made = 'p' * 64, [], 0

    def _learnable(self):
        pass

    def _new_group(self):
        row = self._pick()
        group, self.made = self.made, self.made + 1
        directory = self.work / f'g{group}'
        directory.mkdir()
        self.groups[group] = {'dir': directory, 'size': 1, 'done': 0, 'versions': [], 'task_id': row['id']}
        self.outstanding += 1
        self.pending.append((group, {'key': f'k{group}'}))

    def _score(self, group):
        self.scored.append(self.groups[group]['task_id'])

    def _after_wave(self, group):
        return True


def test_sharded_slots_abandon_a_dead_endpoints_episodes_and_requeue_the_task(tmp_path, monkeypatch):
    """A dead endpoint's episodes exited nonzero and were counted done: all-fail groups burned the task order."""
    collector = ei.frontier_collector(_ShardBase)([{'id': 't1'}, {'id': 't2'}], tmp_path)
    collector.setup(['t1', 't2'], time.time() + 3600, endpoints=[('http://a', 1), ('http://b', 1)])
    seen = []

    class Proc:
        def __init__(self, argv, **kw):
            self.server = argv[argv.index('--server') + 1]
            seen.append(self.server)
            self.pid, self.returncode = 1, 0 if self.server == 'http://a' else 1

        def poll(self):
            return self.returncode
    monkeypatch.setattr(ei.subprocess, 'Popen', Proc)
    monkeypatch.setattr(ei, 'endpoint_alive', lambda url, sha, timeout=10: url == 'http://a')
    real_sleep = time.sleep
    monkeypatch.setattr(ei.time, 'sleep', lambda s: real_sleep(0.01))
    b = threading.Thread(target=collector.slot, args=(1,))
    b.start()
    for _ in range(200):
        if 'http://b' in collector.dead:
            break
        real_sleep(0.01)
    assert collector.dead == {'http://b'} and list(collector.order) == ['t1', 't2'] and not collector.scored
    assert ei.endpoint_for_slot(collector.live_endpoints(), 1) is None          # slot 1 idles; a is not overloaded
    a = threading.Thread(target=collector.slot, args=(0,))
    a.start()
    for _ in range(200):
        if len(collector.scored) == 2:
            break
        real_sleep(0.01)
    collector.stop.set()
    a.join(5)
    b.join(5)
    assert collector.scored == ['t1', 't2'] and seen.count('http://b') == 1
    assert collector.groups.get(0) is None          # the dead endpoint's group was dropped, never scored


def test_held_priority_claims_shrink_a_collection_and_expire(tmp_path):
    assert ei.priority_slots(128, []) == 128 and ei.priority_slots(128, [{}]) == 96 and ei.priority_slots(2, [{}], 0.1) == 1
    claim = tmp_path / 'sft-v10-held-candidate.json'
    claim.write_text(json.dumps({'pid': os.getpid()}))
    assert len(ei.held_priority_claims(tmp_path)) == 1
    assert ei.held_priority_claims(tmp_path, alive=lambda pid: False) == []                 # its process is gone
    assert ei.held_priority_claims(tmp_path, now=time.time() + ei.HELD_PRIORITY_STALE + 1) == []    # no heartbeat
    collector = ei.frontier_collector(_ShardBase)([{'id': 't1'}], tmp_path)
    collector.setup(['t1'], time.time() + 3600, endpoints=[('http://a', 64), ('http://b', 64)], priority_dir=tmp_path)
    collector._priority()
    assert collector.active_slots == 96
    claim.unlink()
    collector.priority_read = 0
    collector._priority()
    assert collector.active_slots == 128


def test_pop_rl_actor_profiles_keep_pop_settings():
    pop_settings, pop_fmt = pop_rl_actor.pop_sglang('fp8_e4m3', 20, profile='pop')
    assert pop_settings['max_running'] == 20 and pop_fmt == 'sglang-pop-rtx4060-bf16-vendor-projection-kv-fp8_e4m3'
    eps = ei.parse_actor_endpoints(['http://g4-1:18531=64', 'http://g4-2:18531=128'], default_slots=32)
    assert eps == [('http://g4-1:18531', 64), ('http://g4-2:18531', 128)]
    assigned = [ei.endpoint_for_slot(eps, i) for i in range(200)]
    assert assigned.count('http://g4-1:18531') == 64 and assigned.count('http://g4-2:18531') == 128
    assert assigned.count(None) == 8


def test_stop_colab_g4_kills_the_recorded_process_group_not_a_pattern(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path)
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', lambda host, command, **kw: sent.append(command) or '')
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    pop_rl_actor.colab_state_path('g4-1').write_text(json.dumps({'pid': 4242, 'policy_id': 'p'}))
    pop_rl_actor.stop_colab_g4('g4-1')
    assert sent and 'kill -TERM -- -4242' in sent[0] and 'pkill' not in sent[0]
    assert not pop_rl_actor.colab_state_path('g4-1').exists()
