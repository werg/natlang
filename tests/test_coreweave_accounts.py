"""Multi-account CoreWeave tooling (scripts/coreweave/accounts.py and its cwctl/diloco/exports/bring-up wiring).

Everything here is offline: the SDK is a fake, no credential is real and nothing is created."""
import importlib.util
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts' / 'coreweave'))


def load(name):
    spec = importlib.util.spec_from_file_location(f'cwacct_{name}', ROOT / 'scripts' / 'coreweave' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cwctl = load('cwctl')
accounts = cwctl.cwaccounts


class Headers:
    def __init__(self, headers, strategy):
        self.headers, self.strategy = headers, strategy


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Isolated profiles, state and a fake cwsandbox module."""
    profiles = tmp_path / 'profiles'
    profiles.mkdir()
    default = tmp_path / 'coreweave.env'

    def profile(name, text):
        path = default if name == 'default' else profiles / f'{name}.env'
        path.write_text(text)
        path.chmod(0o600)
        return path
    monkeypatch.setattr(accounts, 'ACCOUNTS_DIR', profiles)
    monkeypatch.setattr(accounts, 'ENV_FILE', default)
    monkeypatch.setattr(accounts, 'REPO_SETTINGS', tmp_path / 'none.json')
    monkeypatch.setattr(cwctl, 'STATE', tmp_path / 'state')
    monkeypatch.setattr(cwctl, 'TOKENS', tmp_path / 'state' / 'tokens')
    monkeypatch.delenv('NATLANG_CW_ACCOUNT', raising=False)
    monkeypatch.setitem(sys.modules, 'cwsandbox', SimpleNamespace(AuthHeaders=Headers))
    profile('default', 'WANDB_API_KEY=default-key-000000\n')
    profile('acct-a', 'WANDB_API_KEY=a-key-0000000000\nWANDB_ENTITY=org-a\n')
    profile('acct-b', 'export CWSANDBOX_API_KEY="b-key-0000000000"\n')
    return SimpleNamespace(profile=profile, state=tmp_path / 'state', tmp=tmp_path)


def stamp(seconds):
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(seconds))


# Profiles -------------------------------------------------------------------------------------------------------------

def test_profiles_select_their_own_credentials_without_touching_environ(env, monkeypatch):
    monkeypatch.setenv('WANDB_API_KEY', 'process-default')
    assert accounts.profile_names() == ['default', 'acct-a', 'acct-b']
    a = accounts.auth_headers('acct-a')
    assert a.headers['x-wandb-api-key'] == 'a-key-0000000000' and a.headers['x-entity-id'] == 'org-a'
    assert accounts.auth_headers('acct-b').headers == {'Authorization': 'Bearer b-key-0000000000'}
    assert os.environ['WANDB_API_KEY'] == 'process-default' and 'CWSANDBOX_API_KEY' not in os.environ
    child = accounts.cli_env('acct-b', {'WANDB_API_KEY': 'process-default', 'PATH': '/bin'})
    assert child == {'PATH': '/bin', 'CWSANDBOX_AUTH': 'coreweave_api_key', 'CWSANDBOX_API_KEY': 'b-key-0000000000'}


def test_profile_must_be_private_and_named(env):
    env.profile('acct-c', 'WANDB_API_KEY=c-key-0000000000\n').chmod(0o644)
    with pytest.raises(accounts.AccountError, match='readable by others'):
        accounts.read_profile('acct-c')
    with pytest.raises(accounts.AccountError, match='lower-case'):
        accounts.check_name('../etc')
    with pytest.raises(accounts.AccountError, match='no credential profile'):
        accounts.read_profile('missing')


def test_account_keys_never_enter_a_sandbox(env):
    accounts.refuse_credentials({'diloco-token': b'ours', 'parent-read': b'{"url": "https://storage.googleapis.com/x"}'})
    with pytest.raises(accounts.AccountError, match='never enter sandboxes'):
        accounts.refuse_credentials({'peers': b'[{"share": "a-key-0000000000"}]'})


# Binding: every later call uses the sandbox's account ----------------------------------------------------------------

class FakeSandbox:
    created = []
    opened = []

    def __init__(self, ident):
        self.sandbox_id = ident

    @classmethod
    def run(cls, *args, auth=None, **kwargs):
        cls.created.append((args, auth, kwargs))
        return cls(f'sb-{len(cls.created)}')

    @classmethod
    def from_id(cls, ident, auth=None, **kwargs):
        cls.opened.append((ident, auth))
        return SimpleNamespace(result=lambda timeout=None: cls(ident))


def base_job(**changes):
    value = {'schema': 'natlang.coreweave-job/1', 'id': 'eval-one', 'kind': 'eval', 'placement': {'reason': 'test'},
             'image': 'python:3.11', 'resources': {'cpu': '8', 'memory': '32Gi', 'gpu': 1}, 'lifetime_hours': 2,
             'command': ['true'], 'outputs': ['/work/out']}
    value.update(changes)
    return cwctl.jobs.validate(value)


def test_launch_binds_the_account_and_later_reads_use_it(env, monkeypatch):
    FakeSandbox.created, FakeSandbox.opened = [], []
    monkeypatch.setitem(sys.modules, 'cwsandbox', SimpleNamespace(AuthHeaders=Headers, Sandbox=FakeSandbox))
    monkeypatch.setattr(cwctl, 'CREATE_INTERVAL_SECONDS', 0.0)
    monkeypatch.setattr(cwctl, 'bootstrap_text', lambda: '#!/bin/bash')
    monkeypatch.setattr(cwctl, 'sdk_auth', lambda: 'default-auth')
    entry = cwctl.launch(base_job(account='acct-a'), False, False, None)
    assert entry['account'] == 'acct-a' and FakeSandbox.created[0][1].headers['x-wandb-api-key'] == 'a-key-0000000000'
    assert entry['exports'] == [{'kind': 'tree', 'remote': '/work/out', 'role': 'out'}]
    assert accounts.bindings(env.state)['sandboxes'] == {'sb-1': 'acct-a'}
    # A later read selects the bound account even when another account (or none) is selected for the process.
    monkeypatch.setenv('NATLANG_CW_ACCOUNT', 'acct-b')
    cwctl.read_sandbox('sb-1')
    assert FakeSandbox.opened[-1][1].headers['x-wandb-api-key'] == 'a-key-0000000000'
    with pytest.raises(cwctl.ControllerError, match='belongs to account'):
        cwctl.sandbox_account('sb-1', 'acct-b')
    with pytest.raises(cwctl.ControllerError, match='belong to'):
        cwctl.passthrough_account(['stop', 'sb-1'])
    monkeypatch.delenv('NATLANG_CW_ACCOUNT')
    assert cwctl.passthrough_account(['stop', 'sb-1']) == 'acct-a'
    # Legacy sandboxes (launch record without an account) stay with the default profile.
    cwctl.record({'event': 'launched', 'sandbox_id': 'old', 'lifetime_hours': 1, 'resources': {'gpu': 1}})
    cwctl.read_sandbox('old')
    assert FakeSandbox.opened[-1][1] == 'default-auth'
    with pytest.raises(cwctl.ControllerError, match='declares account'):
        cwctl.launch(base_job(account='acct-a'), False, False, None, account='acct-b')


def test_launch_refuses_account_scoped_inputs_from_another_account(env):
    accounts.bind(env.state, 'snapshots', 'snap-a', 'acct-a')
    job = base_job(scratch={'path': '/work', 'size': '20Gi', 'restore_from_snapshot_id': 'snap-a'})
    cwctl.check_launch_account(job, 'acct-a')
    with pytest.raises(cwctl.ControllerError, match='account-scoped'):
        cwctl.check_launch_account(job, 'acct-b')
    with pytest.raises(cwctl.ControllerError, match='org-scoped'):
        cwctl.check_launch_account(base_job(bucket='natlang-default'), 'acct-b')
    accounts.set_lifecycle(env.state, 'acct-b', 'retired', 'test')
    with pytest.raises(cwctl.ControllerError, match='retired'):
        cwctl.check_launch_account(base_job(), 'acct-b')
    with pytest.raises(accounts.AccountError, match='bound to account'):
        accounts.bind(env.state, 'snapshots', 'snap-a', 'acct-b')


# Ledger --------------------------------------------------------------------------------------------------------------

def records_for(now):
    return [
        {'at': stamp(now - 10 * 3600), 'event': 'launched', 'sandbox_id': 'a1', 'account': 'acct-a', 'kind': 'train',
         'job': 'learner-0', 'lifetime_hours': 720, 'resources': {'gpu': 1, 'cpu': '8', 'memory': '64Gi'}},
        {'at': stamp(now - 4 * 3600), 'event': 'export_verified', 'sandbox_id': 'a1', 'source': '/work/x',
         'sha256': 'f' * 64, 'bytes': 1},
        {'at': stamp(now - 5 * 3600), 'event': 'launched', 'sandbox_id': 'b1', 'account': 'acct-b', 'kind': 'eval',
         'job': 'eval', 'lifetime_hours': 720, 'resources': {'cpu': '4', 'memory': '16Gi'}},
        {'at': stamp(now - 3 * 3600), 'event': 'stop', 'account': 'acct-b', 'args': ['b1', '--graceful-shutdown-seconds', '600']},
        {'at': now - 2 * 3600, 'event': 'launched', 'sandbox_id': 'd1', 'job': 'legacy', 'lifetime_hours': 1,
         'resources': {'gpu': 1}},
    ]


def test_ledger_counts_hours_spend_reservations_headroom_and_unexported_work(env):
    now = 2_000_000_000
    config = {**accounts.settings(), 'gpu_hour_price_usd': 2.0,
              'accounts': {'acct-a': {'cap_usd': 500}, 'acct-b': {'cap_gpu_hours': 100}}}
    value = accounts.ledger(records_for(now), config, profiles=['default', 'acct-a', 'acct-b'], now=now)
    a, b, d = (value['accounts'][n] for n in ('acct-a', 'acct-b', 'default'))
    rate = 8 * accounts.PRICE_CPU_CORE_HOUR + 64 * accounts.PRICE_MEMORY_GIB_HOUR + 2.0
    assert a['gpu_hours'] == 10 and a['running'] == 1 and a['spend_usd'] == round(10 * rate, 4)
    assert a['reserved_usd'] == round(24 * rate, 4) and a['headroom_usd'] == round(500 - 34 * rate, 4)
    box = a['sandboxes'][0]
    assert box['unexported_hours'] == 4 and box['last_verified_export']['sha256'] == 'f' * 64
    assert b['running'] == 0 and b['cpu_hours'] == 8 and b['gpu_hours'] == 0 and b['headroom_gpu_hours'] == 100
    # Epoch-stamped legacy record, no account: default; undeclared CPU/memory are assumed and flagged.
    assert d['gpu_hours'] == 1 and d['sandboxes'][0]['assumed'] == ['cpu', 'memory'] and not d['sandboxes'][0]['running']
    assert 'no usage or billing API' in value['estimate']
    unknown = accounts.ledger(records_for(now), {**config, 'gpu_hour_price_usd': None}, now=now)['accounts']['acct-a']
    assert unknown['spend_usd'] is None and unknown['headroom_usd'] is None and 'GPU price is unknown' in unknown['note']


# Placement ------------------------------------------------------------------------------------------------------------

def fresh_ledger(env, caps=None, records=(), now=2_000_000_000, lifecycle=None):
    config = {**accounts.settings(), 'gpu_hour_price_usd': 2.0,
              'accounts': {name: {'cap_usd': cap} for name, cap in (caps or {}).items()}}
    names = ['acct-a', 'acct-b', 'acct-c', 'acct-d']
    for name in names[2:]:
        env.profile(name, f'WANDB_API_KEY={name}-key-00000000\n')
    return accounts.ledger(list(records), config, profiles=['default', *names], now=now,
                           state_lifecycle=lifecycle), config


def test_placement_balances_spend_and_uses_the_full_cap(env):
    now = 2_000_000_000
    used = [{'at': stamp(now - 50 * 3600), 'event': 'launched', 'sandbox_id': 'x', 'account': 'acct-a',
             'lifetime_hours': 40, 'resources': {'gpu': 1, 'cpu': '8', 'memory': '64Gi'}}]
    ledger, config = fresh_ledger(env, {'acct-a': 500, 'acct-b': 500, 'acct-c': 500, 'acct-d': 500}, used, now)
    learner = {'gpu': 1, 'cpu': '8', 'memory': '64Gi'}
    first = accounts.choose_account(ledger, config, learner, 720)
    assert first['account'] != 'acct-a' and 'of its cap left' in first['reason']
    assert any(c.get('skip') == 'placement disabled' for c in first['considered'] if c['account'] == 'default')
    # No early retirement: a nearly spent account still takes work while headroom remains.
    nearly = [{'at': stamp(now - 200 * 3600), 'event': 'launched', 'sandbox_id': 'y', 'account': 'acct-b',
               'lifetime_hours': 100, 'resources': {'gpu': 1, 'cpu': '8', 'memory': '64Gi'}}]
    ledger, config = fresh_ledger(env, {'acct-b': 500}, nearly, now)
    for name in ('acct-a', 'acct-c', 'acct-d'):
        ledger['accounts'][name]['eligible'] = False
    assert accounts.choose_account(ledger, config, learner, 1)['account'] == 'acct-b'
    exhausted = [{**nearly[0], 'lifetime_hours': 120}]
    ledger, config = fresh_ledger(env, {'acct-b': 500}, exhausted, now)
    assert ledger['accounts']['acct-b']['exhausted'] and not ledger['accounts']['acct-b']['eligible']


def test_unknown_caps_balance_on_use(env):
    ledger, config = fresh_ledger(env)
    picks = []
    for _ in range(8):
        chosen = accounts.choose_account(ledger, config, {'gpu': 1, 'cpu': '8', 'memory': '64Gi'}, 720)
        accounts.reserve(ledger, chosen['account'], chosen['need'])
        picks.append(chosen['account'])
    assert Counter(picks) == Counter({'acct-a': 2, 'acct-b': 2, 'acct-c': 2, 'acct-d': 2})


def test_diloco_plan_spreads_learners_and_separates_the_replica(env):
    ledger, config = fresh_ledger(env)
    learner, coordinator = {'gpu': 1, 'cpu': '8', 'memory': '64Gi'}, {'cpu': '8', 'memory': '48Gi'}
    plan = accounts.plan_diloco(ledger, config, workers=4, learner=learner, coordinator=coordinator,
                                cohosted=False, hours=720, replica=coordinator)
    assert sorted(plan['learners']) == ['acct-a', 'acct-b', 'acct-c', 'acct-d']
    assert plan['replica'] != plan['coordinator']
    cohosted = accounts.plan_diloco(ledger, config, workers=4, learner=learner, coordinator=coordinator,
                                    cohosted=True, hours=720)
    assert cohosted['learners'][0] == cohosted['coordinator'] and len(set(cohosted['learners'])) == 4
    ledger['accounts']['acct-d']['eligible'] = False
    ledger['accounts']['acct-c']['eligible'] = False
    ledger['accounts']['acct-b']['eligible'] = False
    with pytest.raises(accounts.NoAccount):
        accounts.plan_diloco(ledger, config, workers=1, learner=learner, coordinator=coordinator, cohosted=False,
                             hours=720, replica=coordinator)


# Freeze: only an explicit quota error ---------------------------------------------------------------------------------

QUOTA = 'SandboxResourceExhaustedError: Create sandbox resource exhausted: quota exceeded, check plan and billing details'


def test_only_explicit_quota_errors_freeze_an_account(env):
    now = 2_000_000_000
    refusals = [
        {'at': stamp(now), 'event': 'create_refused', 'account': 'acct-a', 'reason': 'CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED',
         'error': 'SandboxResourceExhaustedError: capacity exhausted'},
        {'at': stamp(now), 'event': 'create_refused', 'account': 'acct-a', 'reason': 'CWSANDBOX_OCCUPANCY_QUOTA_EXCEEDED',
         'error': 'occupancy quota exceeded'},
        {'at': stamp(now), 'event': 'launched', 'sandbox_id': 'c1', 'account': 'acct-c', 'kind': 'train',
         'lifetime_hours': 720, 'resources': {'gpu': 1}},
        {'at': stamp(now + 1), 'event': 'status', 'sandbox_id': 'c1', 'status': 'completed', 'status_reason': 'user_requested'},
        {'at': stamp(now + 2), 'event': 'api_error', 'account': 'acct-d', 'error': 'quota exceeded'},
    ]
    assert accounts.freeze_signals(refusals) == {}
    frozen = refusals + [{'at': stamp(now + 3), 'event': 'create_refused', 'account': 'acct-b', 'reason': None,
                          'error': QUOTA}]
    signals = accounts.freeze_signals(frozen)
    assert list(signals) == ['acct-b'] and signals['acct-b']['signals'][0]['error'] == QUOTA
    killed = refusals + [{'at': stamp(now + 3), 'event': 'status', 'sandbox_id': 'c1', 'status': 'terminated',
                          'status_reason': 'Sandbox terminated: quota exceeded'}]
    assert list(accounts.freeze_signals(killed)) == ['acct-c']
    ledger, config = fresh_ledger(env, records=frozen, now=now + 10)
    assert ledger['accounts']['acct-b']['freeze']['frozen'] and not ledger['accounts']['acct-b']['eligible']
    # An operator's later activation (a reviewed false alarm) clears it.
    lifecycle = {'acct-b': {'state': 'active', 'at': stamp(now + 5)}}
    ledger, _ = fresh_ledger(env, records=frozen, now=now + 10, lifecycle=lifecycle)
    assert not ledger['accounts']['acct-b']['freeze']['frozen']


def test_quota_refusal_is_not_retried_but_capacity_is(env):
    quota = type('SandboxResourceExhaustedError', (Exception,), {})(QUOTA.split(': ', 1)[1])
    capacity = type('SandboxResourceExhaustedError', (Exception,), {'reason': 'CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED'})('x')
    assert cwctl.error_facts(quota)['quota_error'] and not cwctl.error_facts(quota)['retryable']
    assert cwctl.error_facts(capacity)['retryable'] and not cwctl.error_facts(capacity)['quota_error']


# Evacuation ------------------------------------------------------------------------------------------------------------

def test_first_frozen_account_evacuates_every_account_by_unexported_work(env, monkeypatch):
    exports = load('account_exports')
    monkeypatch.setattr(exports.cwctl, 'STATE', env.state)
    now = time.time()
    tree = [{'kind': 'tree', 'remote': '/work/out', 'role': 'out'}]
    records = [
        {'at': stamp(now - 9 * 3600), 'event': 'launched', 'sandbox_id': 'a-old', 'account': 'acct-a', 'job': 'l0',
         'kind': 'train', 'lifetime_hours': 720, 'resources': {'gpu': 1}, 'exports': tree},
        {'at': stamp(now - 2 * 3600), 'event': 'launched', 'sandbox_id': 'a-new', 'account': 'acct-a', 'job': 'l1',
         'kind': 'train', 'lifetime_hours': 720, 'resources': {'gpu': 1}, 'exports': tree},
        {'at': stamp(now - 5 * 3600), 'event': 'launched', 'sandbox_id': 'b1', 'account': 'acct-b', 'job': 'l2',
         'kind': 'train', 'lifetime_hours': 720, 'resources': {'gpu': 1}, 'exports': tree},
        {'at': stamp(now - 1 * 3600), 'event': 'launched', 'sandbox_id': 'b2', 'account': 'acct-b', 'job': 'e',
         'kind': 'eval', 'lifetime_hours': 720, 'resources': {'cpu': '1', 'memory': '2Gi'}, 'exports': []},
    ]
    for entry in records:
        exports.cwctl.record(entry)
    plan = accounts.evacuation_plan(exports.cwctl.account_ledger(write=False), 'acct-b', reason='quota')
    assert [b['sandbox_id'] for b in plan['accounts']['acct-a']] == ['a-old', 'a-new']
    assert all(b['best_effort'] for b in plan['accounts']['acct-b'])
    pulled = []

    def pull(sandbox_id, target, dest, owner):
        pulled.append((sandbox_id, str(dest)))
        if sandbox_id == 'b1':
            raise RuntimeError('frozen account rejects file reads')
        return {'verified': True, 'sha256': sandbox_id * 4, 'bytes': 7, 'local': str(dest / 'out.tar')}
    # Watch: the first start takes quota errors already recorded as the baseline; a new one evacuates everything.
    env.profile('acct-c', 'WANDB_API_KEY=c-key-0000000000\n')
    exports.cwctl.record({'event': 'create_refused', 'account': 'acct-c', 'reason': None, 'error': QUOTA})
    first = exports.watch_once(refresh=False, evacuate_fn=lambda *a, **k: pytest.fail('baseline must not evacuate'))
    assert first['baseline'] == {'acct-c': first['baseline']['acct-c']} and first['frozen'] == []
    exports.cwctl.record({'event': 'create_refused', 'account': 'acct-b', 'reason': None, 'error': QUOTA})
    calls = []
    report = exports.watch_once(refresh=False, evacuate_fn=lambda trigger, reason, ledger: calls.append(
        exports.evacuate(trigger, reason, ledger=ledger, pull=pull, root=env.tmp / 'evac')) or calls[-1])
    assert report['frozen'] == ['acct-b'] and len(calls) == 1
    result = calls[0]
    assert {s for s, _ in pulled} == {'a-old', 'a-new', 'b1'}
    assert '/acct-a/a-old/out' in dict(pulled)['a-old']
    assert result['results']['acct-b'][1]['error'].startswith('no declared exports')
    newest = json.loads((env.tmp / 'evac' / 'newest-verified.json').read_text())
    assert set(newest) == {'l0/out', 'l1/out'} and newest['l0/out']['account'] == 'acct-a'
    assert accounts.lifecycle(env.state)['acct-b']['state'] == 'frozen'
    assert exports.watch_once(refresh=False, evacuate_fn=lambda *a, **k: pytest.fail('handled once'))['frozen'] == []


# DiLoCo -----------------------------------------------------------------------------------------------------------------

def test_diloco_launch_places_each_role_in_its_planned_account(env, monkeypatch, tmp_path):
    diloco = load('diloco')
    diloco.cwctl = cwctl
    monkeypatch.setattr(diloco, 'RUNS', tmp_path / 'records')
    trainer = ['python', '-m', 'natlang_neuralese.train.native_state_sft']
    learner, coord = tmp_path / 'learner.json', tmp_path / 'coord.json'
    learner.write_text(json.dumps(base_job(id='lfm', kind='train', command=trainer)))
    coord.write_text(json.dumps(base_job(id='coord', resources={'cpu': '8', 'memory': '32Gi'}, transport='api')))
    run = {'schema': diloco.SCHEMA, 'id': 'run', 'resident': 'off', 'placement': 'separate', 'accounts': 'auto',
           'settings': {'run_id': 'run', 'workers': 4},
           'coordinator': {'job': str(coord), 'bundle_file': 'c', 'parent': '/work/in/parent.pt',
                           'replica': {'resources': {'cpu': '8', 'memory': '32Gi'}}},
           'learners': {'job': str(learner), 'bundle_file': 'l'}}
    ledger, _ = fresh_ledger(env)
    monkeypatch.setattr(cwctl, 'account_ledger', lambda write=False: ledger)
    placed = {}

    def fake_launch(job, *args, account=None, **kwargs):
        placed[job['id']] = account
        return {'sandbox_id': job['id'], 'push': {}, **({'service_url': 'https://c'} if job.get('service') else {})}
    monkeypatch.setattr(cwctl, 'launch', fake_launch)
    monkeypatch.setattr(cwctl, 'share_token', lambda sandbox_id: b'gateway')
    monkeypatch.setattr(cwctl, 'record', lambda event: None)
    monkeypatch.setattr(diloco, 'wait_coordinator', lambda *a, **k: None)
    record = diloco.launch(run)
    learners = [placed[f'run-learner-{i}'] for i in range(4)]
    assert sorted(learners) == ['acct-a', 'acct-b', 'acct-c', 'acct-d']
    assert record['accounts']['replica'] != record['accounts']['coordinator'] == placed['run-coordinator']
    with pytest.raises(cwctl.ControllerError, match='another account'):
        diloco.validate({**run, 'accounts': {'coordinator': 'acct-a', 'replica': 'acct-a', 'learners': ['acct-a'] * 4}})
    with pytest.raises(cwctl.ControllerError, match='learner 0 shares'):
        diloco.validate({**run, 'placement': 'cohosted',
                         'accounts': {'coordinator': 'acct-a', 'learners': ['acct-b'] * 4}})


# Bring-up --------------------------------------------------------------------------------------------------------------

def test_bringup_builds_a_cross_account_matrix_and_stops_everything(env, monkeypatch):
    bringup = load('account_bringup')
    bringup.cwctl = cwctl
    created, stopped, files = [], [], {}

    class Box:
        def __init__(self, ident, account):
            self.sandbox_id, self.account = ident, account
            self.endpoint_share_token = SimpleNamespace(get_secret_value=lambda: f'share-{ident}')
            self.service_urls = [(None, None, f'https://{ident}.example')]

        def exec(self, argv, timeout_seconds=None):
            if argv[0] == 'mkdir':
                return SimpleNamespace(result=lambda timeout=None: SimpleNamespace(returncode=0))
            peers = json.loads(files[(self.sandbox_id, argv[-1])])
            out = {p['account']: {'full': {'status': 200, 'body': {'account': p['account']}},
                                  'no_share_token': {'status': 401}, 'no_bearer': {'status': 401}} for p in peers}
            return SimpleNamespace(result=lambda timeout=None: SimpleNamespace(returncode=0, stdout=json.dumps(out)))

        def write_file(self, path, data):
            files[(self.sandbox_id, path)] = data
            return SimpleNamespace(result=lambda timeout=None: None)
    boxes = {}

    def create(args, keywords, account, entry):
        assert keywords['resources'] == {'cpu': '1', 'memory': '2Gi'} and keywords['services'][0]['endpoint']['auth'] == 'share_token'
        compile(args[3], '<server>', 'exec')
        boxes[account] = Box(f'id-{account}', account)
        created.append(account)
        return boxes[account], entry

    class Response:
        def __init__(self, account):
            self.status, self.account = 200, account

        def read(self):
            return json.dumps({'account': self.account}).encode()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False
    opener = lambda request, timeout=None: Response(request.full_url.split('//id-')[1].split('.')[0])
    report = bringup.run(['acct-a', 'acct-b', 'acct-c'], create=create, read=lambda ident: next(
        b for b in boxes.values() if b.sandbox_id == ident), stop=lambda box: stopped.append(box.sandbox_id),
        opener=opener, wait_ready=lambda box: None)
    assert created == ['acct-a', 'acct-b', 'acct-c'] and sorted(stopped) == ['id-acct-a', 'id-acct-b', 'id-acct-c']
    assert report['cross_account_ok'], report['problems']
    assert set(report['matrix']) == {'dgx', 'acct-a', 'acct-b', 'acct-c'} and set(report['matrix']['acct-a']) == {'acct-b', 'acct-c'}
    for data in files.values():
        accounts.refuse_credentials({'x': data})
    assert bringup.verdict({'acct-a': {'acct-b': {'full': {'status': 403}}}})['cross_account_ok'] is False
