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
    G4_HARDWARE, colab_layout,
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
    assert g4.name == 'colab' and resolve_profile('colab', hosts='g4-a') .hosts == ('g4-a',)   # the old name resolves
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


def test_a_normal_claim_evicts_tenants_newest_first_and_never_a_normal_one():
    """Program tuning is a tenant of the session (DECISIONS 2026-11-24 "program tuning on the A100"): a round's learner or an actor that
    does not fit evicts tenants (newest first, only as many as it needs); an actor (normal) is never a victim."""
    probes = dict(total_vram=78.0, total_host=160.0, free_vram=10.0, free_host=100.0)
    claims = {'actor': {'vram_gib': 34.0, 'host_ram_gib': 24.0, 'started': 1.0},
              'fit': {'vram_gib': 12.0, 'host_ram_gib': 16.0, 'preemptible': True, 'started': 2.0},
              'measure': {'vram_gib': 18.0, 'host_ram_gib': 16.0, 'preemptible': True, 'started': 3.0}}
    assert colab_admission.evictions(10.0, 4.0, claims, **probes) == []
    assert colab_admission.evictions(30.0, 4.0, claims, **probes) == ['measure']
    assert colab_admission.evictions(40.0, 4.0, claims, **probes) == ['measure', 'fit']
    assert colab_admission.evictions(50.0, 4.0, claims, **probes) is None       # the actor stays
    learner = colab_admission.evictions(77.0, 40.0, {k: c for k, c in claims.items() if k != 'actor'},
                                        **{**probes, 'free_vram': 48.0})
    assert learner == ['measure', 'fit']        # their memory counts as free once they are gone


def test_a_waiting_normal_claim_evicts_a_live_tenant_and_keeps_new_tenants_out(tmp_path):
    state = tmp_path / 'adm'
    state.mkdir()
    total = colab_admission.total_host_ram_gib()
    colab_admission.write_claims(state, {'actor': {'pid': os.getpid(), 'vram_gib': 0., 'host_ram_gib': 0.6 * total,
                                                   'started': 1.0}})
    tenant = subprocess.Popen([sys.executable, str(REPO / 'scripts/colab/admission.py'), '--state-dir', str(state), 'run',
                               '--vram-gib', '0', '--host-ram-gib', str(0.3 * total), '--preemptible', '--label', 'fit',
                               '--wait', '10', '--poll', '0.1', '--', 'sleep', '60'], start_new_session=True)
    for _ in range(100):
        if len(colab_admission.read_claims(state)) == 2:
            break
        time.sleep(0.1)
    assert any(c.get('preemptible') for c in colab_admission.read_claims(state).values())
    rc = colab_admission.wait_and_run(['/bin/true'], vram_gib=0., host_ram_gib=0.3 * total, label='sft', wait_seconds=20,
                                      poll_seconds=0.1, state_dir=state)
    assert rc == 0 and tenant.wait(timeout=20) is not None
    assert list(colab_admission.read_claims(state)) == ['actor'] and colab_admission.read_waiters(state) == {}
    colab_admission.write_waiters(state, {'sft-x': {'pid': os.getpid(), 'vram_gib': 0., 'host_ram_gib': 1.}})
    rc = colab_admission.wait_and_run(['/bin/true'], vram_gib=0., host_ram_gib=1., label='fit', wait_seconds=0.5,
                                      poll_seconds=0.1, state_dir=state, preemptible=True)
    assert rc == colab_admission.EXIT_TIMEOUT


def _fake_proc(root, pid, argv):
    (root / str(pid)).mkdir(parents=True, exist_ok=True)
    (root / str(pid) / 'cmdline').write_bytes(b'\0'.join(w.encode() for w in argv) + b'\0')


def test_pending_normal_requests_are_read_from_the_process_table(tmp_path):
    """A round's learner admitted by an older admission.py writes no waiting record: tenants see it in /proc."""
    proc = tmp_path / 'proc'
    _fake_proc(proc, 101, ['/py', '/c/natlang/scripts/colab/admission.py', 'run', '--vram-gib', '77', '--', 'python', 'x'])
    _fake_proc(proc, 102, ['/py', '/c/admission.py', 'run', '--preemptible', '--', 'python', 'tune'])
    _fake_proc(proc, 103, ['/py', '/c/admission.py', 'run', '--vram-gib', '34', '--', 'serve'])      # admitted (claim)
    _fake_proc(proc, 104, ['/py', '/c/admission.py', 'check', '--vram-gib', '1'])
    _fake_proc(proc, 105, ['bash', '-c', '. env && /py /c/admission.py run --vram-gib 77 -- x'])
    _fake_proc(proc, 106, ['/py', '/c/admission.py', 'run', '--', '/x/tool', '--preemptible'])  # the command's flag
    claims = {'actor': {'pid': 103}}
    assert colab_admission.pending_normal_requests(claims, proc) == [101, 106]


def test_a_tenant_yields_to_a_waiting_request_and_waits_beside_an_actor(tmp_path):
    state, proc = tmp_path / 'adm', tmp_path / 'proc'
    proc.mkdir()
    # no live actor claim: a tenant with --beside-label actor- is not admitted
    rc = colab_admission.wait_and_run(['/bin/true'], vram_gib=0., host_ram_gib=1., label='fit', wait_seconds=0.5,
                                      poll_seconds=0.1, state_dir=state, preemptible=True, beside_label='actor-',
                                      proc_root=proc)
    assert rc == colab_admission.EXIT_TIMEOUT
    state.mkdir(exist_ok=True)
    colab_admission.write_claims(state, {'actor-x': {'pid': os.getpid(), 'vram_gib': 0., 'host_ram_gib': 1.,
                                                     'label': 'actor-e15b20a86f1c', 'started': 1.0}})
    result = {}
    tenant = threading.Thread(target=lambda: result.update(rc=colab_admission.wait_and_run(
        ['sleep', '60'], vram_gib=0., host_ram_gib=1., label='fit', wait_seconds=5, poll_seconds=0.1, state_dir=state,
        preemptible=True, beside_label='actor-', proc_root=proc, watch_seconds=0.1)))
    started = time.monotonic()
    tenant.start()
    for _ in range(100):
        if len(colab_admission.read_claims(state)) == 2:
            break
        time.sleep(0.05)
    assert len(colab_admission.read_claims(state)) == 2
    _fake_proc(proc, 4242, ['/py', '/c/admission.py', 'run', '--vram-gib', '77', '--', 'exp0_sft.py'])
    tenant.join(timeout=20)
    assert not tenant.is_alive() and result['rc'] != 0 and time.monotonic() - started < 15
    assert list(colab_admission.read_claims(state)) == ['actor-x']


def test_admission_without_claims_needs_the_probed_free_memory():
    ok, info = colab_admission.decide(80.0, 32.0, {}, total_vram=94.0, total_host=160.0, free_vram=40.0, free_host=120.0)
    assert not ok and 'unadmitted' in info['reasons'][0]
    ok, _ = colab_admission.decide(80.0, 32.0, {}, total_vram=94.0, total_host=160.0, free_vram=90.0, free_host=120.0)
    assert ok
    ok, info = colab_admission.decide(1.0, 157.0, {}, total_vram=94.0, total_host=160.0, free_vram=90.0, free_host=160.0)
    assert not ok and info['reasons'][0].startswith('host_ram')


A100 = {'gpu_name': 'NVIDIA A100-SXM4-80GB', 'vram_total_gib': 80.0, 'host_ram_total_gib': 167.05}   # live session


@pytest.mark.parametrize('hardware', [A100, {**A100, 'compute_cap': '8.0'}, G4_HARDWARE],
                         ids=['a100-name', 'a100-cap', 'g4'])
def test_colab_layouts_fit_the_sessions_gpu_by_the_measured_memory_model(hardware):
    """Sizes come from the attested hardware: learner alone and learner + capped actor fit the GPU with the measured
    model (5.2 GiB + 1.2 x activation, 127 KB/token/layer), the admission accepts both, and FP8 weights only where the
    GPU has FP8 GEMMs (the sm_80 A100 serves BF16). The G4 keeps its measured 24 GB / 64 running / 34 + 59 GiB."""
    lay = colab_layout(hardware)
    total, usable = hardware['vram_total_gib'], hardware['vram_total_gib'] - 2.0
    for learner in (lay['learner_alone'], lay['learner_beside_actor']):
        assert learner['peak_gib'] + 2.0 <= learner['vram_gib'] and learner['activation_gib'] > 0
        assert learner['resident_tokens'] == int(learner['activation_gib'] * 2**30 // (127 * 1024 * 24))
    assert lay['learner_alone']['vram_gib'] + 1.0 <= usable
    assert lay['learner_beside_actor']['vram_gib'] + lay['actor_beside_learner']['vram_gib'] + 1.0 <= usable
    ok, _ = colab_admission.decide(lay['learner_beside_actor']['vram_gib'], lay['learner_beside_actor']['host_ram_gib'],
                                   {'actor': {'vram_gib': lay['actor_beside_learner']['vram_gib'], 'host_ram_gib': 24.0}},
                                   total_vram=usable, total_host=hardware['host_ram_total_gib'] - 7,
                                   free_vram=usable - lay['actor_beside_learner']['vram_gib'], free_host=100.0)
    assert ok
    ok, _ = colab_admission.decide(lay['learner_alone']['vram_gib'], 40.0, {}, total_vram=usable,
                                   total_host=hardware['host_ram_total_gib'] - 7, free_vram=usable, free_host=100.0)
    assert ok
    cfg = pop_rl_actor.host_config('colab', host='s')
    serve, claim, sglang, fmt = pop_rl_actor.colab_serve_command(cfg, checkpoint='/c.pt', sha='a' * 64, port=8570,
                                                                  work='/w', beside_learner=True, layout=lay,
                                                                  kv_cache_dtype='fp8_e4m3', weight_dtype='fp8')
    assert serve[serve.index('--sglang-kv-gb') + 1] == str(lay['actor_beside_learner']['kv_gb'])
    assert claim['vram_gib'] == lay['actor_beside_learner']['vram_gib'] and '--sglang-mem-fraction' not in serve
    assert ('--sglang-weight-dtype' in serve) == lay['fp8'] and fmt.endswith('-w-fp8') == lay['fp8']
    alone = pop_rl_actor.colab_serve_command(cfg, checkpoint='/c.pt', sha='a' * 64, port=8570, work='/w',
                                             beside_learner=False, kv_cache_dtype='fp8_e4m3', layout=lay)
    assert alone[0][alone[0].index('--sglang-max-running') + 1] == '128' and '--sglang-mem-fraction' not in alone[0]
    assert alone[0][alone[0].index('--sglang-kv-gb') + 1] == '24' and alone[1]['vram_gib'] == 34.0
    # tenants (program tuning) fit beside the capped actor alone: a fit and a measurement actor at once
    tenants = lay['tenants']
    assert tenants['fit']['vram_gib'] + tenants['measure_actor']['vram_gib'] <= tenants['vram_gib']
    assert 34.0 + tenants['vram_gib'] + 1.0 <= usable + 1e-9
    assert fmt.startswith('sglang-colab-bf16-vendor-projection-kv-fp8_e4m3')
    if total == 80.0:
        assert not lay['fp8'] and lay['compute_capability'] == '8.0'
        assert (lay['learner_alone']['activation_gib'], lay['learner_beside_actor']['activation_gib']) == (58, 33)
        assert lay['actor_beside_learner']['kv_gb'] == 20 and lay['actor_beside_learner']['vram_gib'] == 30.0
    else:
        assert lay['fp8'] and lay['actor_beside_learner'] == {'kv_gb': 24, 'max_running': 64, 'vram_gib': 34.0,
                                                               'host_ram_gib': 24.0}
        assert lay['learner_beside_actor']['vram_gib'] == 59.0 and lay['actor_alone']['vram_gib'] == 34.0


L4 = {'gpu_name': 'NVIDIA L4', 'vram_total_gib': 22.49, 'host_ram_total_gib': 52.96, 'compute_cap': '8.9'}  # live


def test_an_l4_serves_an_actor_and_takes_no_round_sft():
    """The L4 (24 GB, ~121 TFLOPS BF16, 300 GB/s: the GB10's class with an eighth of its memory) is slower than the
    DGX for SFT; the A100 (2.5x) and G4 (4x) are faster. An actor alone runs 128 there too (its 11.2 GB FP8 pool holds
    1.8M LFM3 tokens)."""
    l4, a100, g4 = colab_layout(L4), colab_layout(A100), colab_layout(G4_HARDWARE)
    assert not l4['sft_beats_dgx'] and 'NVIDIA L4' in l4['sft_reason'] and l4['fp8']
    assert a100['sft_beats_dgx'] and g4['sft_beats_dgx']
    assert colab_layout({**A100, 'gpu_name': 'NVIDIA A100-SXM4-40GB', 'vram_total_gib': 40.0})['sft_beats_dgx']
    assert not colab_layout({**A100, 'gpu_name': 'Some Future GPU'})['sft_beats_dgx']     # unknown: the DGX
    assert l4['actor_alone']['max_running'] == 128 and l4['actor_alone']['vram_gib'] <= l4['usable_gib']
    assert a100['actor_alone']['max_running'] == 128 and g4['actor_alone']['max_running'] == 128
    assert l4['actor_alone']['kv_gb'] is None and l4['actor_alone']['mem_fraction'] == 0.88   # below the cap


def test_the_actor_alone_cap_has_an_opt_out(monkeypatch):
    monkeypatch.setenv('NATLANG_COLAB_ACTOR_FULL', '1')
    lay = colab_layout(A100)
    assert lay['actor_alone']['kv_gb'] is None and lay['actor_alone']['vram_gib'] == 70.0


def test_run_sft_refuses_a_session_slower_than_the_dgx_before_anything_moves(tmp_path, monkeypatch):
    monkeypatch.setattr(pop_rl_actor, 'colab_hardware', lambda host: dict(L4))
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    moved = []
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', lambda *a, **k: moved.append(a))
    monkeypatch.setattr(pop_rl_actor, 'stop_colab', lambda host, **k: moved.append(('stop', host)))
    args = types.SimpleNamespace(host='g4', run_name='y', work=str(tmp_path), launch_json=str(tmp_path / 'l.json'),
                                 commit=None, state_sync_minutes=15.0, vram_gib=None, host_ram_gib=None)
    with pytest.raises(SystemExit, match='slower than on the DGX'):
        pop_rl_actor.run_remote_sft(args)
    assert moved == []
    monkeypatch.setenv('NATLANG_COLAB_SFT_ANY_GPU', '1')
    assert pop_rl_actor.colab_sft_fit('g4')[0]


def test_run_sft_sizes_the_learner_from_the_session_and_whether_an_actor_runs_beside(tmp_path, monkeypatch):
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path)
    monkeypatch.setattr(pop_rl_actor, 'colab_hardware', lambda host: dict(A100))
    alone = pop_rl_actor.colab_learner_layout('s')
    assert not alone['beside_actor'] and alone['vram_gib'] == 77.0 and alone['activation_gib'] == 58
    pop_rl_actor.colab_state_path('s').write_text(json.dumps({'pid': 7, 'beside_learner': True}))
    beside = pop_rl_actor.colab_learner_layout('s')
    assert beside['beside_actor'] and beside['vram_gib'] == 47.0 and beside['activation_gib'] == 33


def test_colab_hardware_reads_the_ready_file_and_asks_nvidia_smi_for_a_missing_compute_capability(monkeypatch):
    def ssh(host, command, **kw):
        if command.startswith('cat'):
            return json.dumps({'hardware': dict(A100)})
        assert 'compute_cap' in command
        return '8.0\n'
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh)
    assert pop_rl_actor.colab_hardware('s')['compute_cap'] == '8.0'


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
    monkeypatch.setattr(pop_rl_actor, 'colab_hardware', lambda host: dict(A100))
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path / 'dgx-root')
    monkeypatch.setattr(pop_rl_actor, 'sync_to_remote', sync_to)
    monkeypatch.setattr(pop_rl_actor, 'sync_from_remote', sync_from)
    from natlang_neuralese.common import data_registry
    state['consumed'] = []
    monkeypatch.setattr(data_registry, 'consume', lambda items, **kw: state['consumed'].append((list(items), kw)))
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
    ran, checkouts = [], []

    def run_detached(host, cfg, remote_run, command, **kw):
        ran.append(command)
        out_dir = command[command.index('--out-dir') + 1]
        dest = remote['to_local'](out_dir) / 'checkpoint.pt'
        dest.write_bytes(b'trained' * 100)
        return 0
    monkeypatch.setattr(pop_rl_actor, 'run_detached', run_detached)
    monkeypatch.setattr(pop_rl_actor, 'colab_checkout', lambda host, cfg, commit: checkouts.append(commit))
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    args = types.SimpleNamespace(host='g4', run_name='sft-v10', work=str(work), launch_json=str(work / 'launch.json'),
                                 commit='c' * 40, state_sync_minutes=15.0, vram_gib=59.0, host_ram_gib=40.0)
    remote['truncate'] = True
    assert pop_rl_actor.run_remote_sft(args) == 1 and not (work / 'checkpoint.pt').exists()
    remote['truncate'] = False
    assert pop_rl_actor.run_remote_sft(args) == 0 and (work / 'checkpoint.pt').read_bytes() == b'trained' * 100
    assert checkouts == ['c' * 40] * 2
    sft = ran[-1][ran[-1].index('--') + 1:]
    assert sft[:2] == ['/usr/bin/env', 'NATLANG_ALLOW_UNREGISTERED_DATA=1']   # the reads are recorded on the DGX
    assert sft[3].endswith('scripts/neuralese_data/exp0_sft.py') and sft[4] == 'train'
    (consumed,) = {tuple(c[0]) for c in remote['consumed']}
    assert consumed == (str(tmp_path / 'mix/chains.pkl'), str(tmp_path / 'mix/val-own.pkl'))
    assert sft[sft.index('--epochs') + 1] == '1' and sft[sft.index('--val-every') + 1] == '20'
    assert sft[sft.index('--base') + 1] == '/content/natlang-work/base-lfm3'
    assert sft[sft.index('--out-dir') + 1] == '/content/natlang-work/runs/sft-sft-v10'
    assert sft[sft.index('--checkpoint') + 1].startswith('/content/natlang-work/checkpoints/')
    assert sft[sft.index('--val-chains') + 1].startswith('/content/natlang-work/runs/sft-sft-v10/val-0-')
    assert '--resume' in sft and sft[sft.index('--state-save-minutes') + 1] == '15.0'
    assert not any(str(tmp_path) in a for a in sft)       # no DGX path reaches the VM
    assert ran[-1][ran[-1].index('--vram-gib') + 1] == '59.0'


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
    assert preflight[1]['ACT'] == str(resolve_profile('colab').activation_gib) and '--val-every 20' in preflight[1]['EXTRA']
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
    monkeypatch.setattr(ei, 'ledger_queued', lambda unit: False)   # the ledger owns no unit here (its import
    active = iter([True, False])                                      # under the patched subprocess.run failed)
    monkeypatch.setattr(ei.Cycle, 'active', staticmethod(lambda unit: next(active, False)))
    monkeypatch.setattr(ei.Cycle, 'ledger', lambda self, unit, budget, prio, command, log, detach=False, **k:
                        units.append((unit, budget, command, detach)))
    cycle.collect_dir.mkdir(parents=True)
    (cycle.collect_dir / 'collect.json').write_text(json.dumps({'stopped': False, 'groups_done_this_run': 3}))
    cycle.collect_stage()
    unit, budget, command, detach = units[0]
    assert detach and budget == round(256 * ei.REMOTE_COLLECTOR_SLOT_GB) + 4
    assert command[command.index('--neuralese-variants') + 1].endswith('cases.jsonl')          # one command builder
    assert [command[i + 1] for i, a in enumerate(command) if a == '--actor'] == ['http://g4-1:18531=128',
                                                                                 'http://g4-2:18531=128']
    stops = [c for c in calls if 'stop' in c and '--host-profile' in c]
    assert sorted(c[c.index('--host') + 1] for c in stops) == ['g4-1', 'g4-2']
    cost = json.loads((cycle.dir / 'cycle-cost.json').read_text())
    assert {s['host'] for s in cost['sessions']} == {'g4-1', 'g4-2'} and not cost['open_sessions']
    assert cycle.done('collect') and json.loads((cycle.dir / 'stages/collect.done').read_text())['profile'] == 'colab'


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
    """The parts of grpo.Collector around its ``slot`` (grpo's own slot, ``_abandon``, versions and lag)."""
    _abandon = grpo.Collector._abandon
    _episode_lost = grpo.Collector._episode_lost
    _mark_done = grpo.Collector._mark_done
    slot = grpo.Collector.slot
    episode_valid = grpo.Collector.episode_valid
    served_version = grpo.Collector.served_version
    actor_lagging = grpo.Collector.actor_lagging

    def __init__(self, tasks, work):
        self.by_id = {t['id']: t for t in tasks}
        self.own, self.pending, self.lock, self.outstanding = {}, collections.deque(), threading.Lock(), 0
        self.paused, self.stop = threading.Event(), threading.Event()
        self.args = types.SimpleNamespace(max_outstanding=8, actor=None, slots=2, max_staleness=2)
        self.work = self.runtime = work
        self.env, self.groups, self.version, self.running, self.active_slots = {}, {}, 0, 0, 2
        self.actor_versions = {}
        self.policy_sha, self.scored, self.made = 'p' * 64, [], 0
        self.dynamic = collections.Counter()

    def _learnable(self):
        pass

    def _new_group(self):
        row = self._pick()
        group, self.made = self.made, self.made + 1
        directory = self.work / f'g{group}'
        directory.mkdir()
        self.groups[group] = {'dir': directory, 'size': 1, 'done': 0, 'versions': [], 'task_id': row['id']}
        self.outstanding += 1
        self.pending.append((group, {'key': f'k{group}', 'output': str(directory / 'row-000/sample-00/results.jsonl')}))

    def _score(self, group):
        self.scored.append(self.groups[group]['task_id'])

    def _after_wave(self, group):
        return True


def test_sharded_slots_rerun_a_dead_endpoints_episodes_on_a_live_one(tmp_path, monkeypatch):
    """A dead endpoint's episodes exited nonzero and were counted done: all-fail groups burned the task order. Such an
    episode is no verdict: its files are set aside and it runs again (Collector._episode_lost) on a live endpoint."""
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
        if 'http://b' in collector.dead and collector.pending:
            break
        real_sleep(0.01)
    assert collector.dead == {'http://b'} and list(collector.order) == ['t2'] and not collector.scored
    assert [g for g, _ in collector.pending] == [0]                              # the lost episode waits for a rerun
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
    assert collector.groups[0]['done'] == 1 and (tmp_path / 'g0/workers/k0.done.json').exists()   # rerun on a
    with_reruns_off = ei.frontier_collector(_ShardBase)([{'id': 't1'}], tmp_path / 'off')
    (tmp_path / 'off').mkdir()
    with_reruns_off.args.episode_reruns = 0
    with_reruns_off.setup(['t1'], time.time() + 3600, endpoints=[('http://a', 1), ('http://b', 1)])
    with_reruns_off._new_group()
    with_reruns_off._episode_lost(0, with_reruns_off.pending[0][1], 1, 'endpoint')
    assert 0 not in with_reruns_off.groups and list(with_reruns_off.order) == ['t1']   # abandoned, its task first
    assert (tmp_path / 'off/g0/lost.json').exists()


def test_a_single_remote_actor_serving_another_policy_is_no_verdict_and_its_slots_wait(tmp_path, monkeypatch):
    """One remote actor (a Colab session) shares nothing with the collector: the keeper may replace the session and
    serve it again at another policy, or the driver rebind it. Each finished episode checks the actor still serves the
    collected policy; one that does not is no verdict (never scored under the wrong identity): its files are set
    aside, it waits to run again (``_episode_lost``) and the slots idle until the actor serves the policy again. A
    local actor is not checked."""
    collector = ei.frontier_collector(_ShardBase)([{'id': 't1'}, {'id': 't2'}], tmp_path)
    collector.setup(['t1', 't2'], time.time() + 3600, endpoints=[('http://a', 2)], verify=True)
    serves = {'policy': 'q' * 64}                     # the session came back serving another policy

    class Proc:
        def __init__(self, argv, **kw):
            self.pid, self.returncode = 1, 0

        def poll(self):
            return self.returncode
    monkeypatch.setattr(ei.subprocess, 'Popen', Proc)
    monkeypatch.setattr(ei, 'endpoint_alive', lambda url, sha, timeout=10: serves['policy'] == sha)
    real_sleep = time.sleep
    monkeypatch.setattr(ei.time, 'sleep', lambda s: real_sleep(0.01))
    a = threading.Thread(target=collector.slot, args=(0,))
    a.start()
    for _ in range(300):
        if 'http://a' in collector.dead and collector.pending:
            break
        real_sleep(0.01)
    assert collector.dead == {'http://a'} and not collector.scored and [g for g, _ in collector.pending] == [0]
    assert collector.endpoint_for_slot(0) is None and collector.endpoint_for_slot(1) is None   # slots idle
    serves['policy'] = 'p' * 64                       # served again at the collected policy: back after its retry
    collector.dead_checked['http://a'] = 0
    for _ in range(500):
        if len(collector.scored) == 2:
            break
        real_sleep(0.01)
    collector.stop.set()
    a.join(5)
    assert sorted(collector.scored) == ['t1', 't2'] and not collector.dead
    local = ei.frontier_collector(_ShardBase)([{'id': 't1'}], tmp_path)
    local.setup(['t1'], time.time() + 3600, endpoints=[('http://own', 2)], verify=False)
    assert local.episode_valid('http://own', {'task_id': 't1'})      # the collector's own actor: no check
    assert 'slot' not in vars(type(local))            # grpo's slot runs (memetic preparation, lag) for every actor


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



def test_round_preparation_claim_shrinks_the_dgx_collection_further_and_its_ledger_claim_follows(tmp_path, monkeypatch):
    """DECISIONS 2026-11-24 "the DGX collection yields to round preparation": a prep claim carries its own share (the
    smallest live share wins); entering and leaving a yield the collection lowers and restores its ledger claim
    (memory_ledger resize), once per transition; the claim is removed after the block, and share 1 writes none."""
    assert ei.priority_slots(128, [{}, {'share': 0.5}]) == 64 and ei.priority_slots(128, [{'share': 1.0}]) == 128
    with ei.yield_claim(tmp_path, 'aei-r0002-prep', 1.0) as none:
        assert none is None and not list(tmp_path.glob('*.json'))          # the opt-out
    resized = []
    monkeypatch.setattr(ei.threading, 'Thread', lambda target, args=(), daemon=None: types.SimpleNamespace(
        start=lambda: args and resized.append(args[0])))          # the claim's heartbeat thread is not started
    collector = ei.frontier_collector(_ShardBase)([{'id': 't1'}], tmp_path)
    collector.setup(['t1'], time.time() + 3600, endpoints=[('http://own', 128)], priority_dir=tmp_path,
                    ledger_unit='aei-collect-v0001')
    collector.active_slots = 128
    with ei.yield_claim(tmp_path, 'aei-r0002-prep', ei.PREP_YIELD_SHARE, extra={'reason': 'round preparation'}) as path:
        assert json.loads(path.read_text())['share'] == 0.5
        with ei.yield_claim(tmp_path, 'aei-r0002-prep', 0.5) as nested:
            assert nested is None                       # nested: the outer block owns the claim
        assert path.exists()
        collector._priority()
        assert collector.active_slots == 64 and resized == ['yield']
        held = tmp_path / 'aei-held-v0001.json'
        held.write_text(json.dumps({'pid': os.getpid()}))
        collector.priority_read = 0
        collector._priority()
        assert collector.active_slots == 64 and resized == ['yield']      # still yielding: no second resize
        held.unlink()
    assert not path.exists()
    collector.priority_read = 0
    collector._priority()
    assert collector.active_slots == 128 and resized == ['yield', 'restore']
    calls = []
    monkeypatch.setattr(ei.subprocess, 'run', lambda argv, **kw: calls.append(argv) or
                        types.SimpleNamespace(stdout='{"budget_gb": 54}'))
    collector._resize_claim('yield')
    assert calls == [[*ei.LEDGER, 'resize', '--unit', 'aei-collect-v0001', 'yield']]
    collector.ledger_full_gb = 70.0                           # launched shrunk during a round's preparation
    collector._resize_claim('restore')
    assert calls[-1] == [*ei.LEDGER, 'resize', '--unit', 'aei-collect-v0001', 'restore', '--budget-gb', '70.0']

def test_a_collection_launched_at_room_slots_runs_them_until_the_ledger_has_room(tmp_path, monkeypatch):
    """The collector started at --room-slots does not yield its (already fitted) claim; it takes its full slots and
    restores its claim once ledger_room_gb covers the growth to --ledger-full-gb."""
    resized = []
    monkeypatch.setattr(ei.threading, 'Thread', lambda target, args=(), daemon=None: types.SimpleNamespace(
        start=lambda: args and resized.append(args[0])))
    import scripts.memory_ledger as ml
    monkeypatch.setattr(ml, 'claimed_budget', lambda unit, state_path=None: 44 * ml.GIB)
    room = {'gb': 10.}
    monkeypatch.setattr(ei, 'ledger_room_gb', lambda u=None, priority='medium': room['gb'])
    collector = ei.frontier_collector(_ShardBase)([{'id': 't1'}], tmp_path)
    collector.active_slots = 192
    collector.setup(['t1'], time.time() + 3600, endpoints=[('http://own', 192)], priority_dir=tmp_path / 'prio',
                    ledger_unit='aei-collect-v0001', ledger_full_gb=70., room_slots=58)
    assert collector.active_slots == 58
    collector._priority()
    assert collector.active_slots == 58 and resized == []             # no yield: the launch budget fits the slots
    room['gb'] = 30.                                                   # 26 GB to grow: room
    collector.priority_read = 0
    collector._priority()
    assert collector.active_slots == 192 and collector.room_slots is None and resized == ['restore']
    plain = ei.frontier_collector(_ShardBase)([{'id': 't1'}], tmp_path)
    plain.active_slots = 64
    plain.setup(['t1'], time.time() + 3600, endpoints=[('http://own', 64)], room_slots=64)
    assert plain.room_slots is None and plain.active_slots == 64        # not fewer than its slots: no limit


def test_pop_rl_actor_profiles_keep_pop_settings():
    pop_settings, pop_fmt = pop_rl_actor.pop_sglang('fp8_e4m3', 20, profile='pop')
    assert pop_settings['max_running'] == 20 and pop_fmt == 'sglang-pop-rtx4060-bf16-vendor-projection-kv-fp8_e4m3'
    eps = ei.parse_actor_endpoints(['http://g4-1:18531=64', 'http://g4-2:18531=128'], default_slots=32)
    assert eps == [('http://g4-1:18531', 64), ('http://g4-2:18531', 128)]
    assigned = [ei.endpoint_for_slot(eps, i) for i in range(200)]
    assert assigned.count('http://g4-1:18531') == 64 and assigned.count('http://g4-2:18531') == 128
    assert assigned.count(None) == 8


def test_stop_colab_kills_the_recorded_process_group_not_a_pattern(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path)
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', lambda host, command, **kw: sent.append(command) or '')
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    pop_rl_actor.colab_state_path('g4-1').write_text(json.dumps({'pid': 4242, 'policy_id': 'p'}))
    pop_rl_actor.stop_colab('g4-1')
    assert sent and 'kill -TERM -- -4242' in sent[0] and 'pkill' not in sent[0]
    assert not pop_rl_actor.colab_state_path('g4-1').exists()


def test_run_remote_sft_of_a_round_maps_its_factor_dir_pulls_blob_first_and_starts_no_mps(tmp_path, remote, monkeypatch):
    """An async round's recipe (--round-id, --export-factors-dir, --factors-only in launch.json): the factor directory
    maps to the VM, the factors arrive whole at the recipe's DGX directory, and run-sft never starts MPS (that would
    turn a learner on a session whose actor started without MPS into a failing client)."""
    work = _launch_json(tmp_path)
    launch = json.loads((work / 'launch.json').read_text())
    launch['exp0_sft_args'] += f' --round-id r0007 --export-factors-dir {work}/factors --factors-only --resume'
    (work / 'launch.json').write_text(json.dumps(launch))
    commands = []
    real_ssh = pop_rl_actor.ssh_host
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', lambda h, c, **kw: commands.append(c) or real_ssh(h, c, **kw))
    monkeypatch.setattr(pop_rl_actor, 'ensure_colab_mps', lambda host: pytest.fail('run-sft started MPS'))

    def run_detached(host, cfg, remote_run, sft, **kw):
        factors = remote['to_local'](sft[sft.index('--export-factors-dir') + 1])
        factors.mkdir(parents=True, exist_ok=True)
        (factors / 'adapter_factors_fp32.safetensors').write_bytes(b'f' * 64)
        import hashlib
        (factors / 'policy_manifest.json').write_text(json.dumps({'adapter_factors_sha256':
                                                                  hashlib.sha256(b'f' * 64).hexdigest()}))
        learner.append(sft)
        return 0
    learner = []
    monkeypatch.setattr(pop_rl_actor, 'run_detached', run_detached)
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    args = types.SimpleNamespace(host='g4', run_name='r7', work=str(work), launch_json=str(work / 'launch.json'),
                                 commit=None, state_sync_minutes=15.0, vram_gib=59.0, host_ram_gib=40.0)
    assert pop_rl_actor.run_remote_sft(args) == 0
    sft = learner[-1]
    assert sft[sft.index('--export-factors-dir') + 1] == '/content/natlang-work/runs/sft-r7/factors'
    assert sft[sft.index('--round-id') + 1] == 'r0007' and sft.count('--resume') == 1
    assert (work / 'factors' / 'adapter_factors_fp32.safetensors').read_bytes() == b'f' * 64
    assert (work / 'factors' / 'policy_manifest.json').exists() and not (work / 'factors.pull').exists()
    assert not any('nvidia-cuda-mps' in c for c in commands)


def test_remote_rebind_merge_script_calls_the_exporters_with_their_real_signatures(tmp_path, monkeypatch):
    """The VM-side merge of shipped factors called export(out_ckpt, base_dir, out_hf) against
    export(checkpoint, out_dir, *, base=...): a TypeError on every factor rebind. The merge itself is ``remote_merge``
    (checked against the published sha), never a write to CHECKPOINTS/<sha>.pt that need not hash to its name."""
    import inspect
    from natlang_neuralese.export import native_hf
    factors = tmp_path / 'factors'
    factors.mkdir()
    (factors / 'policy_manifest.json').write_text(json.dumps({'base_checkpoint_sha256': 'b' * 64}))
    (factors / 'adapter_factors_fp32.safetensors').write_bytes(b'f')
    scripts, merges = [], []

    def ssh_host(host, command, **kw):
        words = shlex.split(command)
        if words[1:2] == ['-c']:
            scripts.append(words)
            return json.dumps({'export_dir': words[-1], 'weights_sha256': 'w' * 64}) + '\n'
        return ''
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh_host)
    monkeypatch.setattr(pop_rl_actor, 'sync_to_remote', lambda *a, **k: None)
    monkeypatch.setattr(pop_rl_actor, 'remote_merge', lambda host, cfg, **kw: merges.append(kw) or True)
    cfg = pop_rl_actor.host_config('colab_g4', host='g4')
    pop_rl_actor._prepare_remote_rebind_export(host='g4', cfg=cfg, policy_id='p' * 64, version=3, factors_dir=factors)
    assert merges == [{'base_sha': 'b' * 64, 'factors_dir': factors, 'expect_sha': 'p' * 64}]
    (words,) = scripts
    called = {}
    real_export = native_hf.export

    def fake_export(*a, **k):
        inspect.signature(real_export).bind(*a, **k)
        called['export'] = (a, k)
        return {'weights_sha256': 'w' * 64}
    monkeypatch.setattr(native_hf, 'export', fake_export)
    monkeypatch.setattr(sys, 'argv', ['-c', *words[3:]])
    exec(compile(words[2], '<remote-export>', 'exec'), {'__name__': '__main__'})
    (out_ckpt, out_hf), kw = called['export']
    assert kw == {'base': cfg['base']} and out_hf == words[-1] and out_ckpt == f'{cfg["checkpoints"]}/{"p" * 64}.pt'
    monkeypatch.setattr(pop_rl_actor, 'remote_merge', lambda host, cfg, **kw: False)
    with pytest.raises(RuntimeError, match='does not hash'):
        pop_rl_actor._prepare_remote_rebind_export(host='g4', cfg=cfg, policy_id='p' * 64, version=3, factors_dir=factors)


def test_ensure_remote_checkpoint_keeps_merges_or_ships_and_never_trusts_a_name(tmp_path, remote, monkeypatch):
    """A start checkpoint on the VM: kept when it hashes to its name, removed when not (an older VM merge wrote
    CHECKPOINTS/<sha>.pt unverified); merged on the VM from a published round's base and factors; shipped otherwise
    (also when the merge does not reproduce the DGX's sha)."""
    import hashlib
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    cfg = pop_rl_actor.host_config('colab', host='g4')
    local = tmp_path / 'policy' / 'checkpoint.pt'
    local.parent.mkdir()
    local.write_bytes(b'policy')
    sha = hashlib.sha256(b'policy').hexdigest()
    target = remote['to_local'](f'{cfg["checkpoints"]}/{sha}.pt')
    assert pop_rl_actor.ensure_remote_checkpoint('g4', cfg, local, sha) == 'shipped' and target.read_bytes() == b'policy'
    assert pop_rl_actor.ensure_remote_checkpoint('g4', cfg, local, sha) == 'present'
    target.write_bytes(b'not it')
    removed = []
    real_ssh = pop_rl_actor.ssh_host
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', lambda h, c, **kw: (
        removed.append(c) or target.unlink() or '') if c.startswith('rm -f') else real_ssh(h, c, **kw))
    assert pop_rl_actor.ensure_remote_checkpoint('g4', cfg, local, sha) == 'shipped' and target.read_bytes() == b'policy'
    assert removed
    target.unlink()
    (local.parent / 'published.json').write_text(json.dumps({
        'checkpoint_sha256': sha, 'factors_dir': str(tmp_path / 'f'), 'merge': {'base_checkpoint_sha256': 'b' * 64}}))
    merges = []

    def merge(host, cfg, **kw):
        merges.append(kw)
        target.write_bytes(b'policy')
        return True
    monkeypatch.setattr(pop_rl_actor, 'remote_merge', merge)
    monkeypatch.setattr(pop_rl_actor, 'sync_to_remote', lambda *a, **k: pytest.fail('shipped a mergeable policy'))
    assert pop_rl_actor.ensure_remote_checkpoint('g4', cfg, local, sha) == 'merged'
    assert merges == [{'base_sha': 'b' * 64, 'factors_dir': str(tmp_path / 'f'), 'expect_sha': sha, 'code': None}]
    target.unlink()
    shipped = []
    monkeypatch.setattr(pop_rl_actor, 'remote_merge', lambda host, cfg, **kw: False)
    monkeypatch.setattr(pop_rl_actor, 'sync_to_remote', lambda local, path, host, **kw: shipped.append(path) or
                        remote['to_local'](path).write_bytes(Path(local).read_bytes()))
    assert pop_rl_actor.ensure_remote_checkpoint('g4', cfg, local, sha) == 'shipped' and shipped


def test_run_detached_survives_the_link_and_never_starts_a_second_learner(monkeypatch):
    """The learner ran in the foreground of its SSH session: a dropped colab-ssh link killed it. It now runs under
    setsid with a pid and an exit-status file; a call that finds this run's learner alive waits for it; an unreachable
    host returns None after the away time; a learner gone without a status is a failure."""
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    monkeypatch.setattr(pop_rl_actor.time, 'sleep', lambda s: None)
    monkeypatch.setattr(pop_rl_actor, 'keeper_replaced_since', lambda host, since: None)
    cfg = pop_rl_actor.host_config('colab', host='g4')

    def fake(states):
        sent = []

        def ssh_host(host, command, **kw):
            sent.append(command)
            if 'setsid nohup' in command:
                return ''
            state = states.pop(0)
            if isinstance(state, Exception):
                raise state
            return state + '\n'
        return sent, ssh_host
    sent, ssh = fake(['none', 'running', 'running', 'rc 0'])
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh)
    assert pop_rl_actor.run_detached('g4', cfg, '/r', ['python', 'x.py']) == 0
    (launch,) = [c for c in sent if 'setsid nohup' in c]
    assert '/r/learner.rc' in launch and '/r/learner.pid' in launch and 'x.py' in launch
    sent, ssh = fake(['running', RuntimeError('link down'), 'rc 3'])
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh)
    assert pop_rl_actor.run_detached('g4', cfg, '/r', ['python', 'x.py']) == 3
    assert not any('setsid nohup' in c for c in sent)
    sent, ssh = fake(['running', RuntimeError('link down')])
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh)
    assert pop_rl_actor.run_detached('g4', cfg, '/r', ['python', 'x.py'], away_seconds=0) is None
    sent, ssh = fake(['running', 'none'])
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh)
    assert pop_rl_actor.run_detached('g4', cfg, '/r', ['python', 'x.py']) == 1
    sent, ssh = fake(['rc 0'])                       # finished before a crash of the caller: not trained again
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh)
    assert pop_rl_actor.run_detached('g4', cfg, '/r', ['python', 'x.py']) == 0 and len(sent) == 1
    sent, ssh = fake(['rc 1', 'rc 0'])               # a failed one is retried
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh)
    assert pop_rl_actor.run_detached('g4', cfg, '/r', ['python', 'x.py']) == 0
    (launch,) = [c for c in sent if 'setsid nohup' in c]
    # the launch is its own statement: `rm && setsid ... &` backgrounded a subshell that held the SSH channel open
    assert ' && setsid' not in launch and launch.startswith('rm -f /r/learner.rc; setsid nohup ')



def test_run_detached_ends_at_once_when_the_session_is_replaced(tmp_path, monkeypatch):
    """2026-11-24: the A100 session vanished at ~18:15 (the keeper requested a new one at 18:16, a CPU VM then
    answered under the alias with another host key); run-sft polled until the L4 answered at 18:27. The wait now ends
    with SESSION_LOST_RC as soon as the keeper requested a new session, the host key changed, or another VM (boot id)
    answers; run-sft then skips the post-run pull from the gone VM."""
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    monkeypatch.setattr(pop_rl_actor.time, 'sleep', lambda s: None)
    cfg = pop_rl_actor.host_config('colab', host='colab-g4')

    def fake(answers):
        def ssh_host(host, command, **kw):
            if 'setsid nohup' in command:
                return ''
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        return ssh_host
    lost = pop_rl_actor.SESSION_LOST_RC
    monkeypatch.setattr(pop_rl_actor, 'keeper_replaced_since', lambda host, since: None)
    # another VM answers as the host: its boot id differs
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', fake(['running\nboot-a\n', 'running\nboot-a\n', 'none\nboot-b\n']))
    assert pop_rl_actor.run_detached('colab-g4', cfg, '/r', ['python', 'x.py']) == lost
    # the host key changed (the replacement's sshd answers the alias)
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', fake(['running\nboot-a\n', RuntimeError(
        'Remote command on colab-g4 failed (255): @ WARNING: REMOTE HOST IDENTIFICATION HAS CHANGED! @')]))
    assert pop_rl_actor.run_detached('colab-g4', cfg, '/r', ['python', 'x.py']) == lost
    # the session keeper requested a new session after the wait began (its state file names the alias)
    root = tmp_path / 'colab'
    root.mkdir()
    monkeypatch.undo()
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    monkeypatch.setattr(pop_rl_actor.time, 'sleep', lambda s: None)
    (root / 'session-keeper.json').write_text(json.dumps({'last_request_try': 100.0}))          # default alias
    (root / 'session-keeper-g4b.json').write_text(json.dumps({'alias': 'colab-g4b', 'last_request_try': 5e9}))
    assert pop_rl_actor.keeper_replaced_since('colab-g4', 50.0, root)
    assert pop_rl_actor.keeper_replaced_since('colab-g4', 200.0, root) is None
    assert pop_rl_actor.keeper_replaced_since('colab-g4b', 200.0, root)
    assert pop_rl_actor.keeper_replaced_since('colab-g4c', 0.0, root) is None
    monkeypatch.setattr(pop_rl_actor, 'W', tmp_path)
    (tmp_path / 'ops' / 'ei').mkdir(parents=True)
    root.rename(tmp_path / 'ops' / 'ei' / 'colab')
    polls = []

    def keeper_requests_after_two_polls(host, command, **kw):
        if 'setsid nohup' in command:
            return ''
        polls.append(1)
        if len(polls) == 2:
            (tmp_path / 'ops/ei/colab/session-keeper.json').write_text(json.dumps({'last_request_try': 9e9}))
        return 'running\nboot-a\n'
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', keeper_requests_after_two_polls)
    assert pop_rl_actor.run_detached('colab-g4', cfg, '/r', ['python', 'x.py']) == lost and len(polls) == 2
    # the keeper records its alias in its state
    import types
    sys.path.insert(0, str(Path(pop_rl_actor.__file__).resolve().parents[1] / 'colab'))
    import session_keeper as sk
    k = sk.Keeper(types.SimpleNamespace(alias='colab-g4b', session='g4b', order='L4', state=str(tmp_path / 'k.json')))
    k.save({'last_request_try': 1.0})
    assert json.loads((tmp_path / 'k.json').read_text())['alias'] == 'colab-g4b'


def test_run_sft_yields_an_actor_served_alone_and_serves_it_again(tmp_path, remote, monkeypatch):
    """A recorded actor served alone (70 of 78 GiB) left the learner alone (77 GiB) waiting out its admission: run-sft
    stops it for the round and serves it again from its recorded arguments afterwards. An actor capped beside the
    learner keeps running."""
    work = _launch_json(tmp_path)
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    monkeypatch.setattr(pop_rl_actor, 'run_detached', lambda *a, **k: (
        remote['to_local']('/content/natlang-work/runs/sft-y/checkpoint.pt').write_bytes(b'ok') and 0))
    events = []
    monkeypatch.setattr(pop_rl_actor, 'stop_colab', lambda host, **k: events.append('stop') or pop_rl_actor.read_json(
        pop_rl_actor.colab_state_path(host)))
    monkeypatch.setattr(pop_rl_actor, 'serve_colab', lambda ns: events.append(('serve', ns.policy_id, ns.port)))
    args = types.SimpleNamespace(host='g4', run_name='y', work=str(work), launch_json=str(work / 'launch.json'),
                                 commit=None, state_sync_minutes=15.0, vram_gib=None, host_ram_gib=None)
    record = {'pid': 7, 'beside_learner': False, 'policy_id': 'p' * 64,
              'serve_args': {'policy_id': 'p' * 64, 'port': 8570, 'checkpoint': 'x', 'host': 'g4'}}
    pop_rl_actor.colab_state_path('g4').parent.mkdir(parents=True, exist_ok=True)
    pop_rl_actor.write_json(pop_rl_actor.colab_state_path('g4'), record)
    assert pop_rl_actor.run_remote_sft(args) == 0
    assert events == ['stop', ('serve', 'p' * 64, 8570)]
    events.clear()
    pop_rl_actor.write_json(pop_rl_actor.colab_state_path('g4'), {**record, 'beside_learner': True})
    assert pop_rl_actor.run_remote_sft(args) == 0 and events == []
    # a run-sft restarted mid-round: the first one yielded the actor (its record moved aside); this one restores it
    events.clear()
    pop_rl_actor.colab_state_path('g4').unlink()
    pop_rl_actor.write_json(pop_rl_actor.colab_yielded_path('g4'), record)
    assert pop_rl_actor.run_remote_sft(args) == 0 and events == [('serve', 'p' * 64, 8570)]


def test_learner_state_arrives_as_one_save_or_not_at_all(tmp_path, monkeypatch):
    """An rsync of the live learner-state directory across the learner's rotation (stage -> root) pulled files of
    two saves. The pull copies a hardlink snapshot taken while the directory kept its inode, into a staging directory
    that replaces the local state whole; a learner that has saved nothing leaves the local state as it was."""
    local = tmp_path / 'learner-state'
    local.mkdir()
    (local / 'learner_manifest.json').write_text('old')
    (local / 'stale.pt').write_text('old')
    replies, sent = ['ok'], []

    def ssh_host(host, command, **kw):
        sent.append(command)
        return replies.pop(0) + '\n'

    def sync_from(remote_path, local_path, host, **kw):
        dest = Path(local_path)
        dest.mkdir(parents=True)
        (dest / 'learner_manifest.json').write_text('new')
        (dest / 'adapter_factors_fp32.safetensors').write_text('new')
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh_host)
    monkeypatch.setattr(pop_rl_actor, 'sync_from_remote', sync_from)
    assert pop_rl_actor.pull_learner_state('g4', '/r', local) is True
    assert sorted(p.name for p in local.iterdir()) == ['adapter_factors_fp32.safetensors', 'learner_manifest.json']
    assert (local / 'learner_manifest.json').read_text() == 'new' and not list(tmp_path.glob('learner-state.*'))
    assert 'cp -al /r/learner-state /r/learner-state.snap' in sent[0] and 'stat -c %i /r/learner-state' in sent[0]
    replies[:] = ['empty']
    assert pop_rl_actor.pull_learner_state('g4', '/r', local) is False and (local / 'learner_manifest.json').exists()
    replies[:] = ['busy']
    with pytest.raises(RuntimeError, match='consistent snapshot'):
        pop_rl_actor.pull_learner_state('g4', '/r', local)
    assert (local / 'learner_manifest.json').read_text() == 'new'


def test_serve_colab_launches_the_actor_as_its_own_statement_and_records_its_serve(tmp_path, monkeypatch):
    """`. env && setsid nohup ... & echo $!` backgrounded the and-list in a subshell: the launch held the SSH channel
    until its 60 s timeout and `$!` named the subshell, whose group stop_colab could not signal. The record carries the
    serve's arguments (run-sft serves a yielded actor again from them)."""
    sent = []

    def ssh_host(host, command, **kw):
        sent.append(command)
        return '4242\n' if 'setsid nohup' in command else ''
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path)
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh_host)
    monkeypatch.setattr(pop_rl_actor, 'colab_hardware', lambda host: dict(A100))
    monkeypatch.setattr(pop_rl_actor, 'ensure_remote_checkpoint', lambda *a: 'present')
    monkeypatch.setattr(pop_rl_actor, 'colab_checkout', lambda *a: None)
    monkeypatch.setattr(pop_rl_actor, 'remote_endpoint', lambda host, port, bind=None: 'http://127.0.0.1:18731')
    monkeypatch.setattr(pop_rl_actor, 'attested', lambda endpoint, sha, adapters=None: True)
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    args = types.SimpleNamespace(host='g4', policy_id='p' * 64, commit='c' * 40, port=8570, bind_host=None,
                                 kv_cache_dtype='fp8_e4m3', weight_dtype='bfloat16', beside_learner=False,
                                 checkpoint='/dgx/ckpt.pt', ready_seconds=60)
    result = pop_rl_actor.serve_colab(args)
    (launch,) = [c for c in sent if 'setsid nohup' in c]
    assert '&& setsid' not in launch and launch.rstrip().endswith('& echo $!') and result['pid'] == 4242
    assert result['serve_args']['port'] == 8570 and result['serve_args']['checkpoint'] == '/dgx/ckpt.pt'


def _git(cwd, *args):
    return subprocess.run(['git', '-c', 'user.name=t', '-c', 'user.email=t@t', '-C', str(cwd), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def test_the_vm_merges_with_the_publishers_code_not_its_checkouts(tmp_path, monkeypatch):
    """2026-11-24: the r0002 rebind failed ('the merge ... does not hash to the published 7cfc4774'). The VM merged
    with its checkout, which the restored actor had put at bdf4c8721, before 0462075ea added the FP32 master rows to
    merged checkpoints; the publisher ran pin 0076be613. The merge now runs the publication's ``code`` commit,
    extracted with git archive beside the checkout (which stays where the serving actor needs it); a dirty publisher
    cannot be reproduced (ValueError: the caller ships); a publication without provenance uses the checkout."""
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    origin = tmp_path / 'origin'
    (origin / 'training' / 'neuralese').mkdir(parents=True)
    _git(tmp_path, 'init', '-q', str(origin))
    marker = origin / 'training' / 'neuralese' / 'merge_version.py'
    marker.write_text('VERSION = "publisher"\n')
    _git(origin, 'add', '-A')
    _git(origin, 'commit', '-qm', 'publisher')
    publisher = _git(origin, 'rev-parse', 'HEAD')
    marker.write_text('VERSION = "later"\n')
    _git(origin, 'commit', '-qam', 'later')
    vm = tmp_path / 'vm'
    vm.mkdir()
    _git(vm, 'clone', '-q', str(origin), 'natlang')
    _git(vm / 'natlang', 'checkout', '-q', publisher + '~0')          # the VM checkout somewhere else ...
    _git(origin, 'commit', '-q', '--allow-empty', '-m', 'unfetched')
    unfetched = _git(origin, 'rev-parse', 'HEAD')                       # ... and a commit it has not fetched yet
    cfg = {**pop_rl_actor.host_config('colab', host='g4'), 'root': str(vm), 'repo': str(vm / 'natlang')}
    sent = []

    def ssh_host(host, command, **kw):
        sent.append(command)
        return subprocess.run(['bash', '-c', command], check=True, capture_output=True, text=True).stdout
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', ssh_host)
    root = pop_rl_actor.remote_merge_code('g4', cfg, {'git_commit': publisher, 'dirty': False})
    assert root == f'{vm}/merge-code/{publisher}'
    assert (Path(root) / 'training/neuralese/merge_version.py').read_text() == 'VERSION = "publisher"\n'
    assert _git(vm / 'natlang', 'rev-parse', 'HEAD') == publisher            # the checkout was not moved
    assert pop_rl_actor.remote_merge_code('g4', cfg, {'git_commit': publisher, 'dirty': False}) == root   # kept
    later = pop_rl_actor.remote_merge_code('g4', cfg, {'git_commit': unfetched, 'dirty': False})
    assert (Path(later) / 'training/neuralese/merge_version.py').read_text() == 'VERSION = "later"\n'
    with pytest.raises(ValueError, match='dirty'):
        pop_rl_actor.remote_merge_code('g4', cfg, {'git_commit': publisher, 'dirty': True})
    with pytest.raises(ValueError, match='dirty'):
        pop_rl_actor.remote_merge_code('g4', cfg, {'git_commit': publisher, 'dirty': None})
    assert pop_rl_actor.remote_merge_code('g4', cfg, None) == cfg['repo']

    # remote_merge hands the merge script the publisher's tree
    factors = tmp_path / 'factors'
    factors.mkdir()
    (factors / 'policy_manifest.json').write_text(json.dumps({'base_checkpoint_sha256': 'b' * 64,
                                                              'adapter_factors_sha256': 'f' * 64}))
    runs = []

    def fake(host, command, **kw):
        words = shlex.split(command)
        if words[:2] == [cfg['python'], '-c']:
            runs.append(words[3])
            return json.dumps({'ok': True, 'checkpoint_sha256': 'p' * 64}) + '\n'
        if words[:1] == ['test']:
            return 'ok'
        return ssh_host(host, command) if command.startswith('bash -c') else ''
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', fake)
    assert pop_rl_actor.remote_merge('g4', cfg, base_sha='b' * 64, factors_dir=factors, expect_sha='p' * 64,
                                     code={'git_commit': publisher, 'dirty': False})
    assert runs == [root]


def test_ensure_remote_checkpoint_merges_with_the_published_code_and_ships_a_dirty_one(tmp_path, remote, monkeypatch):
    import hashlib
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    cfg = pop_rl_actor.host_config('colab', host='g4')
    local = tmp_path / 'policy' / 'checkpoint.pt'
    local.parent.mkdir()
    local.write_bytes(b'policy')
    sha = hashlib.sha256(b'policy').hexdigest()
    code = {'schema': 'natlang.code-provenance/1', 'git_commit': 'a' * 40, 'dirty': False}
    (local.parent / 'published.json').write_text(json.dumps({
        'checkpoint_sha256': sha, 'factors_dir': str(tmp_path / 'f'), 'merge': {'base_checkpoint_sha256': 'b' * 64},
        'code': code}))
    merges = []
    monkeypatch.setattr(pop_rl_actor, 'remote_merge', lambda host, cfg, **kw: merges.append(kw) or False)
    monkeypatch.setattr(pop_rl_actor, 'drive_enabled', lambda cfg: False)
    assert pop_rl_actor.ensure_remote_checkpoint('g4', cfg, local, sha) == 'shipped'
    assert merges[0]['code'] == code


def test_a_factor_rebind_with_its_publication_goes_through_ensure_remote_checkpoint(tmp_path, monkeypatch):
    """The rebind merged with the VM checkout and raised on a mismatch; given the DGX checkpoint it now merges with the
    publisher's code or ships the publication (``ensure_remote_checkpoint``)."""
    factors = tmp_path / 'factors'
    factors.mkdir()
    (factors / 'policy_manifest.json').write_text(json.dumps({'base_checkpoint_sha256': 'b' * 64}))
    ensured = []
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', lambda host, command, **kw: json.dumps(
        {'export_dir': 'x', 'weights_sha256': 'w' * 64}) if ' -c ' in command else '')
    monkeypatch.setattr(pop_rl_actor, 'ensure_remote_checkpoint', lambda *a: ensured.append(a) or 'shipped')
    monkeypatch.setattr(pop_rl_actor, 'remote_merge', lambda *a, **k: pytest.fail('merged with the VM checkout'))
    cfg = pop_rl_actor.host_config('colab', host='g4')
    pop_rl_actor._prepare_remote_rebind_export(host='g4', cfg=cfg, policy_id='p' * 64, version=3, factors_dir=factors,
                                               checkpoint='/dgx/policy/checkpoint.pt')
    assert ensured == [('g4', cfg, '/dgx/policy/checkpoint.pt', 'p' * 64)]


def test_a_yielded_actor_record_survives_a_run_sft_restart(tmp_path, monkeypatch):
    """stop_colab(yielded=True) keeps the record for the restore (a run-sft restarted mid-round adopts it); a successful
    restore removes it, a failed one leaves it for the keeper."""
    monkeypatch.setattr(pop_rl_actor, 'ROOT', tmp_path)
    monkeypatch.setattr(pop_rl_actor, 'ssh_host', lambda *a, **k: '')
    monkeypatch.setattr(pop_rl_actor, 'log', lambda **e: None)
    record = {'pid': 7, 'policy_id': 'a' * 64, 'host': 'colab-g4', 'serve_args': {'policy_id': 'a' * 64, 'host': 'colab-g4'}}
    pop_rl_actor.write_json(pop_rl_actor.colab_state_path('colab-g4'), record)
    pop_rl_actor.stop_colab('colab-g4', yielded=True)
    assert not pop_rl_actor.colab_state_path('colab-g4').exists()
    assert json.loads(pop_rl_actor.colab_yielded_path('colab-g4').read_text())['policy_id'] == 'a' * 64
    monkeypatch.setattr(pop_rl_actor, 'serve_colab', lambda ns: (_ for _ in ()).throw(RuntimeError('no GPU')))
    pop_rl_actor.restore_colab_actor(record)
    assert pop_rl_actor.colab_yielded_path('colab-g4').exists()          # failed: the keeper serves it
    monkeypatch.setattr(pop_rl_actor, 'serve_colab', lambda ns: {'ok': True})
    pop_rl_actor.restore_colab_actor(record)
    assert not pop_rl_actor.colab_yielded_path('colab-g4').exists()
    pop_rl_actor.stop_colab('colab-g4')                                   # a plain stop keeps nothing
    assert not pop_rl_actor.colab_yielded_path('colab-g4').exists()
