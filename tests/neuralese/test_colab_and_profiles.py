"""Unit tests for Strand 1: ExecutionProfile, Colab G4 admission/bootstrap, remote actor, held series, sharded collect."""
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'training/neuralese'))
sys.path.insert(0, str(REPO / 'scripts/neuralese_data'))
sys.path.insert(0, str(REPO / 'scripts/colab'))

from natlang_neuralese.common.execution_profile import (  # noqa: E402
    CostTelemetry,
    parse_colab_hosts,
    resolve_profile,
    select_stage_profile,
)
from natlang_neuralese.rl.held_eval import pairing_differences, series  # noqa: E402
import admission as colab_admission  # noqa: E402
import bootstrap as colab_bootstrap  # noqa: E402
import expert_iteration  # noqa: E402
import pop_rl_actor  # noqa: E402


def test_execution_profile_resolves_dgx_colab_g4_pop_and_rejects_l4():
    dgx = resolve_profile('dgx')
    g4 = resolve_profile('colab_g4', hosts='g4-a,g4-b')
    pop = resolve_profile('pop')

    assert dgx.is_discrete_gpu is False and dgx.activation_gib == 56.0
    assert g4.is_discrete_gpu is True and g4.activation_gib == 80.0 and g4.held_samples == 4
    assert g4.hosts == ('g4-a', 'g4-b') and g4.held_series_host == 'colab_g4'
    assert g4.fallback_profile == 'dgx' and g4.fallback_wait_minutes == 30.0
    assert pop.is_discrete_gpu is True and pop.sglang_max_running == 20

    for bad in ('l4', 'colab_l4', 'colab-l4', 'L4'):
        with pytest.raises(ValueError, match='excluded'):
            resolve_profile(bad)


def test_select_stage_profile_30min_unreachable_fallback_to_dgx():
    now = {'t': 0.0}
    slept = []

    def fake_clock():
        return now['t']

    def fake_sleep(dt):
        slept.append(dt)
        now['t'] += dt

    # Never reachable -> falls back to dgx after 30 minutes
    prof, hosts, reason = select_stage_profile(
        'colab_g4', hosts=['g4-1', 'g4-2'], reachable_fn=lambda h: False,
        wait_minutes=30.0, poll_seconds=300.0, sleep_fn=fake_sleep, clock_fn=fake_clock,
    )
    assert prof.name == 'dgx'
    assert hosts == ['local']
    assert reason is not None and 'unreachable' in reason
    assert sum(slept) == pytest.approx(1800.0)

    # Becomes reachable on second poll -> returns colab_g4 with reachable subset
    checks = {'n': 0}

    def becomes_up(h):
        checks['n'] += 1
        return checks['n'] > 2 and h == 'g4-2'

    prof2, hosts2, reason2 = select_stage_profile(
        'colab_g4', hosts=['g4-1', 'g4-2'], reachable_fn=becomes_up,
        wait_minutes=30.0, poll_seconds=60.0, sleep_fn=fake_sleep, clock_fn=fake_clock,
    )
    assert prof2.name == 'colab_g4'
    assert hosts2 == ['g4-2']
    assert reason2 is None


def test_cost_telemetry_computes_cu_and_tokens_per_cu(tmp_path):
    cost = CostTelemetry('sft-v10')
    cost.record_stage('collect', 'colab_g4', wall_seconds=3600.0, sessions=2)  # 26 CU
    cost.record_stage('sft', 'colab_g4', wall_seconds=1800.0, sessions=1, canonical_chain_tokens=50_000_000)  # 6.5 CU
    cost.record_stage('data', 'dgx', wall_seconds=600.0)  # 0 CU
    path = tmp_path / 'cycle-cost.json'
    summary = cost.save(path)
    assert summary['total_cu_spent'] == pytest.approx(32.5)
    assert summary['canonical_chain_tokens'] == 50_000_000
    assert summary['canonical_chain_tokens_per_cu'] == pytest.approx(50_000_000 / 32.5, rel=1e-3)
    assert summary['cu_per_million_units'] == pytest.approx(32.5 / 50.0, rel=1e-3)
    reloaded = CostTelemetry.load(path, 'sft-v10')
    assert len(reloaded.entries) == 3


def test_colab_admission_checks_vram_and_host_ram_separately():
    ok, info = colab_admission.check_admission(
        80.0, 32.0, vram_probe=lambda: 90.0, host_probe=lambda: 120.0,
    )
    assert ok is True and info['reasons'] == []

    # VRAM insufficient even though host RAM has plenty
    ok_vram, info_vram = colab_admission.check_admission(
        80.0, 32.0, vram_probe=lambda: 40.0, host_probe=lambda: 120.0,
    )
    assert ok_vram is False and any('vram' in r for r in info_vram['reasons'])

    # Host RAM insufficient even though VRAM has plenty
    ok_host, info_host = colab_admission.check_admission(
        80.0, 100.0, vram_probe=lambda: 90.0, host_probe=lambda: 50.0,
    )
    assert ok_host is False and any('host_ram' in r for r in info_host['reasons'])


def test_colab_bootstrap_dry_run_plan(tmp_path):
    ready = colab_bootstrap.bootstrap(
        tailscale_authkey='tskey-test', ssh_authorized_key='ssh-ed25519 AAAA',
        hostname='colab-g4-1', commit='abc1234', root=tmp_path / 'colab', dry_run=True,
    )
    assert ready['ready'] is True and ready['dry_run'] is True
    assert ready['plan']['hostname'] == 'colab-g4-1'
    assert 'install_tailscale_userspace' in ready['plan']['steps']
    assert 'fetch_base_model' in ready['plan']['steps']


def test_held_series_includes_engine_host_for_colab_g4(monkeypatch):
    monkeypatch.delenv('NATLANG_HELD_SERIES_HOST', raising=False)
    s_default = series()
    s_g4 = series(engine_host='colab_g4')
    assert 'engine_host' not in s_default
    assert s_g4['engine_host'] == 'colab_g4'
    assert s_g4['key'] != s_default['key']

    # Env var also sets it
    monkeypatch.setenv('NATLANG_HELD_SERIES_HOST', 'colab_g4')
    assert series()['key'] == s_g4['key']

    # pairing_differences reports engine_host mismatch
    sdiff, _ = pairing_differences({'series': s_default}, {'series': s_g4})
    assert sdiff['engine_host'] == [None, 'colab_g4']


def test_pop_rl_actor_colab_g4_profile_and_sglang_format():
    cfg = pop_rl_actor.host_config('colab_g4', host='g4-1')
    assert cfg['profile'] == 'colab_g4' and cfg['host'] == 'g4-1'
    assert cfg['max_running'] == 128 and cfg['admission'].endswith('scripts/colab/admission.py')
    settings, fmt = pop_rl_actor.pop_sglang('fp8_e4m3', 128, profile='colab_g4')
    assert settings['max_running'] == 128 and settings['mem_fraction'] == 0.88
    assert fmt == 'sglang-colab-g4-bf16-vendor-projection-kv-fp8_e4m3'
    # Pop settings untouched
    pop_settings, pop_fmt = pop_rl_actor.pop_sglang('fp8_e4m3', 20, profile='pop')
    assert pop_settings['max_running'] == 20 and pop_fmt == 'sglang-pop-rtx4060-bf16-vendor-projection-kv-fp8_e4m3'


def test_sharded_collection_endpoint_parsing_and_slot_assignment():
    eps = expert_iteration.parse_actor_endpoints(['http://g4-1:18531=64', 'http://g4-2:18531=128'], default_slots=32)
    assert eps == [('http://g4-1:18531', 64), ('http://g4-2:18531', 128)]
    assigned = [expert_iteration.endpoint_for_slot(eps, i) for i in range(192)]
    assert assigned.count('http://g4-1:18531') == 64
    assert assigned.count('http://g4-2:18531') == 128
