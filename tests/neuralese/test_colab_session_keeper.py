"""scripts/colab/session_keeper.py: decisions from fake `colab` output, and the request/stop/rename sequence."""
import json
import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('session_keeper', REPO / 'scripts/colab/session_keeper.py')
sk = importlib.util.module_from_spec(spec)
sys.modules['session_keeper'] = sk
spec.loader.exec_module(sk)

L4 = '[g4a] gpu-l4-s-ass1a1-aj7l81wd6hbq | Hardware: L4 | Shape: Standard | Variant: GPU'
A100 = '[g4a] gpu-a100-s-x | Hardware: A100 | Shape: High-RAM | Variant: GPU'
CPU = '[g4a] m-s-ass1c1-10rp9205mlf30 | Hardware: CPU | Shape: Standard | Variant: DEFAULT'
ORPHAN_CPU = '[?] m-s-ass1c1-10rp9205mlf30 | Hardware: CPU | Shape: Standard | Variant: DEFAULT'


def obs(*lines, ssh_ok=True, now=10_000.0, **kw):
    return sk.Observation(sessions=sk.parse_sessions('\n'.join(lines)), ssh_ok=ssh_ok, now=now, **kw)


def test_parse_sessions_reads_the_cli_lines_and_ranks_hardware():
    rows = sk.parse_sessions('\n'.join([L4, ORPHAN_CPU, 'Current balance: 234.31 compute units']))
    assert [(r['name'], r['hardware'], r['variant']) for r in rows] == [('g4a', 'L4', 'GPU'), ('?', 'CPU', 'DEFAULT')]
    assert [sk.tier(h) for h in ('A100', 'G4', 'L4', 'CPU', 'T4', None)] == [0, 1, 2, None, None, None]
    assert sk.parse_order('A100:hm,G4,L4') == sk.DEFAULT_ORDER


@pytest.mark.parametrize('lines,why', [((), 'missing'), ((CPU,), 'runs on CPU')])
def test_a_missing_or_cpu_session_is_replaced_in_preference_order(lines, why):
    d = sk.decide(obs(*lines), {}, session='g4a')
    assert d.action == 'replace' and why in d.reason
    assert [t[0] for t in d.tiers] == ['A100', 'G4', 'L4'] and d.tiers[0][2] is True     # A100 asks --high-mem


def test_replacement_requests_are_spaced_and_never_beside_another_gpu_session():
    assert sk.decide(obs(), {'last_request_try': 10_000 - 120}, session='g4a').action == 'wait'
    assert sk.decide(obs(), {'last_request_try': 10_000 - 400}, session='g4a').action == 'replace'
    other = '[g4b] gpu-g4-s-y | Hardware: G4 | Shape: Standard | Variant: GPU'
    d = sk.decide(obs(other), {}, session='g4a')
    assert d.action == 'wait' and 'g4b' in d.reason


def test_a_failed_tunnel_heals_first_and_is_a_lost_session_only_after_the_limit():
    state = sk.note_ssh({}, False, 10_000.0)
    assert sk.decide(obs(L4, ssh_ok=False, tunnel_429s=4), state, session='g4a').action == 'heal'
    assert sk.decide(obs(L4, ssh_ok=False), state, session='g4a').action == 'wait'
    late = obs(L4, ssh_ok=False, now=10_000.0 + 16 * 60)
    d = sk.decide(late, sk.note_ssh(state, False, late.now), session='g4a')
    assert d.action == 'replace' and 'unreachable' in d.reason
    assert sk.note_ssh(state, True, late.now)['ssh_fail_since'] is None


def test_a_lower_tier_retries_better_tiers_every_20_minutes_but_never_under_a_round():
    d = sk.decide(obs(L4), {'last_upgrade_try': 10_000 - 21 * 60}, session='g4a')
    assert d.action == 'upgrade' and [t[0] for t in d.tiers] == ['A100', 'G4']
    assert sk.decide(obs(L4), {'last_upgrade_try': 10_000 - 5 * 60}, session='g4a').action == 'ok'
    assert sk.decide(obs(L4, sft_active=True), {}, session='g4a').action == 'ok'
    g4 = '[g4a] gpu-g4-s | Hardware: G4 | Shape: Standard | Variant: GPU'
    assert [t[0] for t in sk.decide(obs(g4), {}, session='g4a').tiers] == ['A100']
    assert sk.decide(obs(A100), {}, session='g4a').action == 'ok'


def test_redact_masks_token_like_output():
    text = 'token=abc123 Authorization: Bearer xyz ' + 'Q' * 48 + ' ok'
    out = sk.redact(text)
    assert 'abc123' not in out and 'Q' * 48 not in out and out.endswith(' ok')


class FakeColab:
    """`colab` sessions/new/stop/usage over a dict of assignments; `refuse` lists GPUs without capacity."""

    def __init__(self, sessions, refuse=()):
        self.sessions, self.refuse, self.calls = dict(sessions), set(refuse), []

    def line(self, name, hw):
        return f'[{name}] ep-{name}-{hw.lower()} | Hardware: {hw} | Shape: Standard | Variant: GPU'

    def __call__(self, cmd, timeout=None, env=None, check=False):
        if cmd[0] != sk.COLAB:
            self.calls.append(('other', cmd[0]))
            return 0, ''
        args = cmd[2:]
        self.calls.append(tuple(args))
        if args[0] == 'sessions':
            return 0, '\n'.join(self.line(n, h) for n, h in self.sessions.items())
        if args[0] == 'usage':
            return 0, 'Usage rate: 1.94/hr'
        if args[0] == 'stop':
            self.sessions.pop(args[2], None)
            return 0, '[colab] Session terminated.'
        if args[0] == 'new':
            gpu = args[args.index('--gpu') + 1]
            if gpu in self.refuse:
                return 1, f'https://colab.../assign?nbh=........&variant=GPU&accelerator={gpu}: Service Unavailable'
            self.sessions[args[2]] = gpu
            return 0, ''
        return 0, ''


def keeper(tmp_path, fake, monkeypatch, *, ssh_ok=True, sft=False, state=None):
    a = types.SimpleNamespace(order='A100:hm,G4,L4', state=str(tmp_path / 'state.json'), session='g4a',
                              alias='colab-g4', tunnel_unit='colab-g4a-tunnel', forward=['19731:9570'],
                              tunnel_fail_minutes=15, upgrade_minutes=20, request_minutes=5, serve_minutes=10,
                              actor_root=str(tmp_path / 'actors'))
    k = sk.Keeper(a, runner=fake)
    if state is not None:
        k.save(state)
    events = []
    monkeypatch.setattr(k, 'ssh_ok', lambda: ssh_ok)
    monkeypatch.setattr(k, 'sft_active', lambda: sft)
    monkeypatch.setattr(k, 'reforward', lambda: None)
    monkeypatch.setattr(k, 'unassign_orphans', lambda s: None)
    monkeypatch.setattr(k, 'rename', lambda old, new: events.append(('rename', old, new)) or fake.sessions.__setitem__(
        new, fake.sessions.pop(old)))
    monkeypatch.setattr(k, 'after_new_session', lambda hw, rate: events.append(('bootstrap', hw)))
    return k, events


def test_a_cpu_session_is_stopped_and_replaced_by_the_best_tier_with_capacity(tmp_path, monkeypatch):
    fake = FakeColab({'g4a': 'CPU'}, refuse={'A100', 'G4'})
    k, events = keeper(tmp_path, fake, monkeypatch)
    assert k.step().action == 'replace'
    news = [c for c in fake.calls if c[0] == 'new']
    assert ('stop', '-s', 'g4a') in fake.calls and fake.calls.index(('stop', '-s', 'g4a')) < fake.calls.index(news[0])
    assert [c[c.index('--gpu') + 1] for c in news] == ['A100', 'G4', 'L4'] and '--high-mem' in news[0]
    assert fake.sessions == {'g4a': 'L4'} and events == [('bootstrap', 'L4')]


def test_an_upgrade_stops_the_old_session_once_the_new_one_is_assigned(tmp_path, monkeypatch):
    fake = FakeColab({'g4a': 'L4'}, refuse={'A100'})
    k, events = keeper(tmp_path, fake, monkeypatch, state={'last_upgrade_try': 0})
    assert k.step().action == 'upgrade'
    news = [c for c in fake.calls if c[0] == 'new']
    assert [c[2] for c in news] == ['g4a-next', 'g4a-next'] and [c[c.index('--gpu') + 1] for c in news] == ['A100', 'G4']
    assert fake.calls.index(('stop', '-s', 'g4a')) > fake.calls.index(news[-1])      # only after the G4 is assigned
    assert fake.sessions == {'g4a': 'G4'} and events == [('rename', 'g4a-next', 'g4a'), ('bootstrap', 'G4')]


def test_a_refused_upgrade_keeps_the_session_and_waits_20_minutes(tmp_path, monkeypatch):
    fake = FakeColab({'g4a': 'L4'}, refuse={'A100', 'G4'})
    k, events = keeper(tmp_path, fake, monkeypatch, state={'last_upgrade_try': 0})
    assert k.step().action == 'upgrade'
    assert fake.sessions == {'g4a': 'L4'} and events == [] and not any(c[0] == 'stop' for c in fake.calls)
    fake.calls.clear()
    assert k.step().action == 'ok' and not any(c[0] == 'new' for c in fake.calls)


def test_no_upgrade_while_a_round_trains_on_the_session(tmp_path, monkeypatch):
    fake = FakeColab({'g4a': 'L4'})
    k, events = keeper(tmp_path, fake, monkeypatch, sft=True, state={'last_upgrade_try': 0})
    assert k.step().action == 'ok' and not any(c[0] == 'new' for c in fake.calls)


def test_a_request_answered_with_other_hardware_is_stopped(tmp_path, monkeypatch):
    fake = FakeColab({})
    real_call = fake.__call__

    def cpu_instead(cmd, **kw):
        rc, out = real_call(cmd, **kw)
        if cmd[0] == sk.COLAB and cmd[2] == 'new':
            fake.sessions[cmd[4]] = 'CPU'
        return rc, out
    k, events = keeper(tmp_path, cpu_instead, monkeypatch)
    k.run = cpu_instead
    assert k.request('g4a', 'A100', True, retries=0) is None and 'g4a' not in fake.sessions


def test_a_down_or_oversized_actor_is_served_again_but_never_under_a_round():
    """2026-11-23 22:37: the A100's actor serve failed once and the keeper answered 'A100 serves' for hours (the lost
    VM's record already moved aside): a failed serve stays pending and is retried every 10 min; an actor restored
    by an older pin at the full fraction is re-served at the current layout (tenants need the room)."""
    up = [{'name': 'g4a', 'endpoint': 'e', 'hardware': 'A100', 'shape': 'High-RAM', 'variant': 'GPU'}]
    obs = lambda **kw: sk.Observation(sessions=up, ssh_ok=True, now=10_000.0, **kw)
    assert sk.decide(obs(actor_down=True), {}, session='g4a').action == 'serve'
    assert sk.decide(obs(actor_oversized=True), {}, session='g4a').action == 'serve'
    assert sk.decide(obs(actor_down=True, sft_active=True), {}, session='g4a').action == 'ok'
    assert sk.decide(obs(actor_down=True), {'last_serve_try': 9_700.0}, session='g4a').action == 'ok'
    assert sk.decide(obs(), {}, session='g4a').action == 'ok'


def test_a_pending_serve_is_retried_and_cleared_when_it_serves(tmp_path, monkeypatch):
    fake = FakeColab({'g4a': 'A100'})
    serve = {'checkpoint': '/c.pt', 'policy_id': 'p' * 64, 'commit': 'c' * 40, 'port': 8570}
    k, _ = keeper(tmp_path, fake, monkeypatch, state={'pending_serve': {'serve': serve, 'commit': 'c' * 40}})
    d = k.step()
    assert d.action == 'serve' and ('other', str(sk.REPO / '.venv-neuralese/bin/python')) in fake.calls
    assert 'pending_serve' not in k.load() and k.load()['last_serve_try'] > 0
    assert k.step().action == 'ok'                       # served: nothing pending, no retry


def test_a_recorded_actor_whose_process_is_gone_is_served_again(tmp_path, monkeypatch):
    """2026-11-24 03:47: a tenant's GPU fault killed the actor's engine; its record still named it and the keeper said
    'serves'. A recorded actor whose admission process is gone on the VM is down."""
    fake = FakeColab({'g4a': 'A100'})
    k, _ = keeper(tmp_path, fake, monkeypatch)
    (tmp_path / 'actors').mkdir()
    serve = {'checkpoint': '/c.pt', 'policy_id': 'p' * 64, 'commit': 'c' * 40, 'port': 8570}
    (tmp_path / 'actors' / 'colab-actor-colab-g4.json').write_text(json.dumps({'pid': 4242, 'serve_args': serve,
                                                                               'claim': {'vram_gib': 34.0}}))
    served = []
    monkeypatch.setattr(k, 'actor_dead', lambda: True)
    monkeypatch.setattr(k, 'actor_oversized', lambda: False)
    monkeypatch.setattr(k, 'serve', lambda s, c: served.append((s['policy_id'], c)) or True)
    assert k.step().action == 'serve' and served == [('p' * 64, 'c' * 40)]
    monkeypatch.setattr(k, 'actor_dead', lambda: False)
    assert k.step().action == 'ok'


def test_a_sibling_session_is_not_a_stray_and_an_unnamed_gpu_is_never_unassigned(tmp_path, monkeypatch):
    """Program tuning's L4 'g4b' beside aei1's A100 'g4a' (2026-11-24): each keeper names the other as a sibling, so a
    lost g4a is still replaced; and the L4 keeper (order L4 alone) never takes an unnamed A100 for a CPU orphan."""
    lost = [{'name': 'g4b', 'endpoint': 'e2', 'hardware': 'L4', 'shape': 'Standard', 'variant': 'GPU'}]
    obs = sk.Observation(sessions=lost, ssh_ok=False, now=10_000.0)
    assert sk.decide(obs, {}, session='g4a').action == 'wait'                       # an unknown GPU session blocks
    assert sk.decide(obs, {}, session='g4a', siblings=('g4b',)).action == 'replace'
    fake = FakeColab({})
    k, _ = keeper(tmp_path, fake, monkeypatch)
    k.a.order, k.order = 'L4', sk.parse_order('L4')
    monkeypatch.undo()          # the real unassign_orphans
    k.run = lambda cmd, timeout=None, env=None, check=False: fake.calls.append(('unassign', tuple(cmd[3:]))) or (0, '')
    k.unassign_orphans([{'name': '?', 'endpoint': 'a100-ep', 'hardware': 'A100'},
                        {'name': '?', 'endpoint': 'cpu-ep', 'hardware': 'CPU'}])
    assert [c for c in fake.calls if c[0] == 'unassign'] == [('unassign', ('cpu-ep',))]
