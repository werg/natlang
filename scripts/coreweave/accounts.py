"""CoreWeave accounts: credential profiles, sandbox/snapshot bindings, our own usage ledger and account placement.

Owner decision 2026-10-30: several CoreWeave sandbox accounts (each a separate W&B org with its own API key and a
lifetime trial cap). All keys live only on the DGX. Work is spread across accounts; the DGX is the data hub.

Profiles. ``default`` is ~/.config/natlang/coreweave.env (NATLANG_CW_ENV_FILE). Every other account NAME is
~/.config/natlang/coreweave-accounts/NAME.env (NATLANG_CW_ACCOUNTS_DIR), mode 600, in the same KEY=VALUE format
(WANDB_API_KEY with optional WANDB_ENTITY, or CWSANDBOX_API_KEY / COREWEAVE_API_TOKEN; optional CWSANDBOX_AUTH and
NATLANG_CW_BUCKET). A non-default profile is never loaded into os.environ: its key only becomes the explicit
per-call SDK auth headers (``auth_headers``) or the environment of a cwsandbox CLI child (``cli_env``). The default
profile keeps its existing role as the W&B storage account for private exports (``storage_account``).

Non-secret settings: repository defaults in training/coreweave-placement.json ("accounts"), overridden by
~/.config/natlang/coreweave-accounts/accounts.json:

    {"accounts": {"acct1": {"cap_usd": null, "cap_gpu_hours": null, "placement": true}},
     "gpu_hour_price_usd": null, "reservation_horizon_hours": 24,
     "storage_account": "default", "evacuation_root": "~/data/dgxops/coreweave-accounts"}

A cap may be unknown (null; the owner expects about 500 USD each). The ledger (STATE/accounts.json) is OUR estimate
from our records: cwsandbox exposes no usage or billing call, so hours and spend are derived from launch, stop,
status and export records and declared resources, never read from the provider. Running sandboxes reserve their
next ``reservation_horizon_hours``.

Owner decisions 2026-10-30: no early retirement. Every account is used to its full cap; placement balances spend so
the accounts reach their caps at about the same time. An account is frozen only on an explicit quota error from the
API (``quota_error``); the FIRST frozen account triggers an immediate evacuation of every sandbox in every account to
the DGX (account_exports.py), as insurance. Exports keep their normal cadence; the ledger shows per sandbox the
earned work not yet independently verified on the DGX (a frozen account's sandboxes become unreadable).
"""
from __future__ import annotations

import calendar
import copy
import fcntl
import json
import os
import re
import stat
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT = 'default'
NAME = re.compile(r'^[a-z0-9][a-z0-9-]{0,31}$')
ENV_FILE = Path(os.environ.get('NATLANG_CW_ENV_FILE', Path.home() / '.config/natlang/coreweave.env'))
ACCOUNTS_DIR = Path(os.environ.get('NATLANG_CW_ACCOUNTS_DIR', Path.home() / '.config/natlang/coreweave-accounts'))
REPO_SETTINGS = REPO / 'training' / 'coreweave-placement.json'
KEY_NAMES = ('CWSANDBOX_API_KEY', 'COREWEAVE_API_TOKEN', 'WANDB_API_KEY')
TERMINAL = ('completed', 'failed', 'terminated', 'absent')
# Sandbox price card (job.py): CPU and memory billed on the request. An undeclared CPU/memory request gets the platform
# default; AGENTS.md (2026-10-29) observed 1 CPU/16Gi and elsewhere 2 CPUs, so the estimate assumes the larger.
PRICE_CPU_CORE_HOUR = 0.128
PRICE_MEMORY_GIB_HOUR = 0.0212
ASSUMED_CPU = 2.0
ASSUMED_MEMORY_GIB = 16.0
DEFAULTS = {'reservation_horizon_hours': 24.0, 'storage_account': DEFAULT,
            'gpu_hour_price_usd': None, 'evacuation_root': str(Path.home() / 'data/dgxops/coreweave-accounts'),
            'accounts': {}}


class AccountError(RuntimeError):
    pass


def check_name(name: str) -> str:
    if not isinstance(name, str) or not NAME.match(name):
        raise AccountError(f'account name {name!r}: lower-case letters, digits and dashes, at most 32 characters')
    return name


def profile_path(name: str) -> Path:
    return ENV_FILE if check_name(name) == DEFAULT else ACCOUNTS_DIR / f'{name}.env'


def profile_names() -> list[str]:
    names = [DEFAULT] if ENV_FILE.is_file() else []
    if ACCOUNTS_DIR.is_dir():
        names += sorted(p.stem for p in ACCOUNTS_DIR.glob('*.env') if NAME.match(p.stem) and p.stem != DEFAULT)
    return names


def read_profile(name: str) -> dict[str, str]:
    """KEY=VALUE lines of one profile, without touching os.environ. Refuses a group/world-readable file."""
    path = profile_path(name)
    if not path.is_file():
        raise AccountError(f'account {name!r} has no credential profile ({path}); see plans/COREWEAVE_SANDBOXES.md '
                           '"Multiple accounts"')
    if path.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise AccountError(f'{path} is readable by others; chmod 600 {path}')
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        key, value = line.removeprefix('export ').split('=', 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def credential(values: dict[str, str]) -> tuple[str, str, str] | None:
    """(auth strategy, variable name, value) of a profile, with cwctl.auth_strategy's precedence."""
    chosen = values.get('CWSANDBOX_AUTH')
    if chosen not in (None, '', 'coreweave_api_key', 'wandb'):
        raise AccountError(f'CWSANDBOX_AUTH is coreweave_api_key or wandb, got {chosen!r}')
    if not chosen:
        chosen = ('coreweave_api_key' if values.get('CWSANDBOX_API_KEY') or values.get('COREWEAVE_API_TOKEN')
                  else 'wandb' if values.get('WANDB_API_KEY') else 'coreweave_api_key')
    names = ('WANDB_API_KEY',) if chosen == 'wandb' else ('CWSANDBOX_API_KEY', 'COREWEAVE_API_TOKEN')
    for key in names:
        if values.get(key):
            return chosen, key, values[key]
    return None


def has_credential(name: str) -> bool:
    try:
        return credential(read_profile(name)) is not None
    except AccountError:
        return False


def auth_headers(name: str):
    """Explicit per-call cwsandbox auth for a non-default account (the SDK prefers it over any environment)."""
    from cwsandbox import AuthHeaders
    values = read_profile(name)
    found = credential(values)
    if found is None:
        raise AccountError(f'account {name!r}: its profile has no WANDB_API_KEY or CWSANDBOX_API_KEY')
    strategy, _, key = found
    if strategy == 'coreweave_api_key':
        return AuthHeaders(headers={'Authorization': f'Bearer {key}'}, strategy='api_key')
    try:
        from importlib.metadata import version
        sdk = version('wandb')
    except Exception:  # noqa: BLE001 - the header is informational
        sdk = 'unknown'
    headers = {'x-wandb-api-key': key, 'x-wandb-sdk-version': sdk}
    if values.get('WANDB_ENTITY'):
        headers['x-entity-id'] = values['WANDB_ENTITY']
    if values.get('WANDB_PROJECT'):
        headers['x-project-name'] = values['WANDB_PROJECT']
    return AuthHeaders(headers=headers, strategy='wandb_api_key')


def cli_env(name: str, base: dict[str, str]) -> dict[str, str]:
    """Environment of a cwsandbox CLI child for one account: no other account's credential survives in it."""
    if name == DEFAULT:
        return dict(base)
    values = read_profile(name)
    found = credential(values)
    if found is None:
        raise AccountError(f'account {name!r}: its profile has no credential')
    env = {k: v for k, v in base.items() if k not in KEY_NAMES and k not in ('CWSANDBOX_AUTH', 'WANDB_ENTITY',
                                                                                'WANDB_PROJECT')}
    strategy, key_name, value = found
    env['CWSANDBOX_AUTH'] = strategy
    env['CWSANDBOX_API_KEY' if key_name == 'COREWEAVE_API_TOKEN' else key_name] = value
    for extra in ('WANDB_ENTITY', 'WANDB_PROJECT'):
        if values.get(extra):
            env[extra] = values[extra]
    return env


def credential_values() -> list[str]:
    """Every credential value of every profile (used only to refuse delivering one into a sandbox)."""
    found = []
    for name in profile_names():
        try:
            values = read_profile(name)
        except AccountError:
            continue
        found += [values[k] for k in KEY_NAMES if values.get(k)]
    return [value for value in found if len(value) >= 8]


def refuse_credentials(secrets: dict[str, bytes]) -> None:
    """No account key may be delivered into a sandbox, under any secret name."""
    for value in credential_values():
        needle = value.encode()
        for name, payload in secrets.items():
            if needle in payload:
                raise AccountError(f'secret {name!r} contains an account API key; account keys never enter sandboxes')


def bucket_for(name: str) -> str | None:
    """The object-storage bucket an account's sandboxes may use (buckets are org-scoped)."""
    if name == DEFAULT:
        return os.environ.get('NATLANG_CW_BUCKET')
    return read_profile(name).get('NATLANG_CW_BUCKET') or None


# Settings -------------------------------------------------------------------------------------------------------------

def settings(local: Path | None = None, repo: Path | None = None) -> dict:
    merged = copy.deepcopy(DEFAULTS)
    repo_path = repo or REPO_SETTINGS
    if repo_path.is_file():
        top = json.loads(repo_path.read_text())
        merged['gpu_hour_price_usd'] = top.get('gpu_hour_price_usd')
        merged.update({k: v for k, v in (top.get('accounts') or {}).items() if k != 'description'})
    local = local or ACCOUNTS_DIR / 'accounts.json'
    if local.is_file():
        override = json.loads(local.read_text())
        accounts = {**merged.get('accounts', {}), **(override.get('accounts') or {})}
        merged.update({k: v for k, v in override.items() if k != 'accounts'})
        merged['accounts'] = accounts
    for name in merged['accounts']:
        check_name(name)
    return merged


def storage_account(config: dict | None = None) -> str:
    return check_name((config or settings()).get('storage_account', DEFAULT))


def storage_wandb_key(config: dict | None = None) -> str:
    """The W&B key of the storage account that holds the private exports (werg/natlang). Default profile: the
    environment loaded by cwctl.load_env_file wins, as before; another storage account reads its own profile."""
    name = storage_account(config)
    if name == DEFAULT and os.environ.get('WANDB_API_KEY'):
        return os.environ['WANDB_API_KEY']
    key = read_profile(name).get('WANDB_API_KEY')
    if not key:
        raise AccountError(f'storage account {name!r} has no WANDB_API_KEY')
    return key


# Bindings: which account owns a sandbox or snapshot ------------------------------------------------------------------

def _bindings_path(state: Path) -> Path:
    return Path(state) / 'account-bindings.json'


def bindings(state: Path) -> dict:
    path = _bindings_path(state)
    return json.loads(path.read_text()) if path.is_file() else {'sandboxes': {}, 'snapshots': {}}


def bind(state: Path, kind: str, ident: str, account: str) -> None:
    """Record that ``ident`` (a sandbox or snapshot) belongs to ``account``. A binding never changes."""
    if kind not in ('sandboxes', 'snapshots'):
        raise ValueError(kind)
    check_name(account)
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'account-bindings.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        value = bindings(state)
        value.setdefault(kind, {})
        if value[kind].get(ident, account) != account:
            raise AccountError(f'{kind[:-1]} {ident} is bound to account {value[kind][ident]!r}, not {account!r}')
        value[kind][ident] = account
        tmp = _bindings_path(state).with_suffix('.tmp')
        tmp.write_text(json.dumps(value, indent=1, sort_keys=True) + '\n')
        os.replace(tmp, _bindings_path(state))


def bound(state: Path, kind: str, ident: str, records: list[dict] | None = None) -> str | None:
    found = bindings(state).get(kind, {}).get(ident)
    if found or kind != 'sandboxes':
        return found
    for entry in records or ():
        if entry.get('event') == 'launched' and entry.get('sandbox_id') == ident:
            return entry.get('account', DEFAULT)
    return None


# Lifecycle: active / frozen / retired (our observation or decision, recorded) ----------------------------------------

LIFECYCLE = ('active', 'frozen', 'retired')


def lifecycle(state: Path) -> dict:
    path = Path(state) / 'account-lifecycle.json'
    return json.loads(path.read_text()) if path.is_file() else {}


def set_lifecycle(state: Path, account: str, value: str, reason: str) -> dict:
    if value not in LIFECYCLE:
        raise ValueError(value)
    check_name(account)
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    with (state / 'account-lifecycle.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        current = lifecycle(state)
        previous = current.get(account)
        history = [] if previous is None else [*previous.get('history', []),
                                               {k: v for k, v in previous.items() if k != 'history'}][-20:]
        current[account] = {'state': value, 'reason': reason, 'at': _stamp(time.time()), 'history': history}
        tmp = state / 'account-lifecycle.tmp'
        tmp.write_text(json.dumps(current, indent=1, sort_keys=True) + '\n')
        os.replace(tmp, state / 'account-lifecycle.json')
    return current[account]


# Ledger --------------------------------------------------------------------------------------------------------------

def _stamp(seconds: float) -> str:
    return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(seconds))


def _seconds(stamp) -> float:
    """A record's time: our UTC stamp, or the epoch seconds a few other writers recorded."""
    if isinstance(stamp, (int, float)):
        return float(stamp)
    return calendar.timegm(time.strptime(stamp, '%Y-%m-%dT%H:%M:%SZ'))


def _cores(value) -> float:
    text = str(value)
    return float(text[:-1]) / 1000 if text.endswith('m') else float(text)


def _gib(value) -> float:
    text = str(value)
    for unit, factor in (('Ki', 2 ** -20), ('Mi', 2 ** -10), ('Gi', 1), ('Ti', 1024)):
        if text.endswith(unit):
            return float(text[:-2]) * factor
    raise ValueError(f'memory needs a binary unit, got {value!r}')


def resource_rate(resources: dict, gpu_price: float | None) -> dict:
    """Hourly usage and USD rate of one sandbox request (None when a GPU is used and its price is unknown)."""
    assumed = []
    cpu = _cores(resources['cpu']) if 'cpu' in resources else (assumed.append('cpu') or ASSUMED_CPU)
    memory = _gib(resources['memory']) if 'memory' in resources else (assumed.append('memory') or ASSUMED_MEMORY_GIB)
    gpu = resources.get('gpu', 0)
    gpu = int(gpu.get('count', 1)) if isinstance(gpu, dict) else int(gpu or 0)
    cpu_memory = cpu * PRICE_CPU_CORE_HOUR + memory * PRICE_MEMORY_GIB_HOUR
    usd = None if gpu and gpu_price is None else cpu_memory + gpu * (gpu_price or 0.0)
    return {'gpu': gpu, 'cpu': cpu, 'memory_gib': memory, 'usd_per_hour': usd,
            'cpu_memory_usd_per_hour': cpu_memory, 'assumed': assumed}


def _stopped_by_us(records: list[dict]) -> dict[str, float]:
    stops: dict[str, float] = {}
    for entry in records:
        if 'at' not in entry:
            continue
        at = _seconds(entry['at'])
        if entry.get('event') == 'stop':
            for arg in entry.get('args', []):
                if not str(arg).startswith('-') and not str(arg).isdigit():
                    stops[arg] = min(stops.get(arg, at), at)
        elif entry.get('event') == 'stopped' and entry.get('sandbox_id'):
            stops[entry['sandbox_id']] = min(stops.get(entry['sandbox_id'], at), at)
    return stops


def _ends(records: list[dict]) -> dict[str, float]:
    """Earliest observed end of each sandbox: our stop, or a terminal status observation (an upper bound)."""
    ends = dict(_stopped_by_us(records))
    for entry in records:
        if entry.get('event') == 'status' and str(entry.get('status', '')).lower() in TERMINAL and 'at' in entry:
            at = _seconds(entry['at'])
            ends[entry['sandbox_id']] = min(ends.get(entry['sandbox_id'], at), at)
    return ends


def _exports(records: list[dict]) -> dict[str, dict]:
    """Newest DGX-verified export per sandbox (recovery_pull.py records ``export_verified`` after the hash check)."""
    newest: dict[str, dict] = {}
    for entry in records:
        if entry.get('event') == 'export_verified' and entry.get('sandbox_id') and 'at' in entry:
            if entry['sandbox_id'] not in newest or _seconds(entry['at']) >= _seconds(newest[entry['sandbox_id']]['at']):
                newest[entry['sandbox_id']] = entry
    return newest


def ledger(records: list[dict], config: dict, *, state_lifecycle: dict | None = None, profiles: list[str] = (),
           credentials: dict[str, bool] | None = None, now: float | None = None) -> dict:
    """Per-account GPU/CPU hours, estimated spend, reservations, caps, headroom and, per sandbox, the earned work not
    yet independently on the DGX, from our own records."""
    now = time.time() if now is None else now
    price = config.get('gpu_hour_price_usd')
    horizon = float(config.get('reservation_horizon_hours', DEFAULTS['reservation_horizon_hours']))
    ends, exports = _ends(records), _exports(records)
    names = set(profiles) | set(config.get('accounts', {}))
    launched = [r for r in records if r.get('event') == 'launched' and r.get('sandbox_id')]
    names |= {r.get('account', DEFAULT) for r in launched}
    frozen = freeze_signals(records, now=now)
    accounts = {}
    for name in sorted(names):
        settings_of = config.get('accounts', {}).get(name, {})
        accounts[name] = {
            'account': name, 'sandboxes': [], 'gpu_hours': 0.0, 'cpu_hours': 0.0, 'memory_gib_hours': 0.0,
            'spend_usd': 0.0, 'cpu_memory_spend_usd': 0.0, 'running': 0, 'running_gpu': 0,
            'reserved_usd': 0.0, 'reserved_gpu_hours': 0.0, 'unexported_gpu_hours': 0.0,
            'cap_usd': settings_of.get('cap_usd'), 'cap_gpu_hours': settings_of.get('cap_gpu_hours'),
            'placement': settings_of.get('placement', name != DEFAULT or set(profiles) <= {DEFAULT}),
            'credential': (credentials or {}).get(name, name in profiles),
            'lifecycle': (state_lifecycle or {}).get(name, {}).get('state', 'active'),
            'freeze': _cleared(frozen.get(name, {'frozen': False, 'signals': []}),
                               (state_lifecycle or {}).get(name, {}))}
    for entry in launched:
        account = accounts[entry.get('account', DEFAULT)]
        ident = entry['sandbox_id']
        start = _seconds(entry['at'])
        limit = start + float(entry.get('lifetime_hours', 0)) * 3600
        running = ident not in ends and limit > now
        end = min(ends.get(ident, limit), limit, now)
        hours = max(0.0, end - start) / 3600
        rate = resource_rate(entry.get('resources') or {}, price)
        account['gpu_hours'] += rate['gpu'] * hours
        account['cpu_hours'] += rate['cpu'] * hours
        account['memory_gib_hours'] += rate['memory_gib'] * hours
        account['cpu_memory_spend_usd'] += rate['cpu_memory_usd_per_hour'] * hours
        if account['spend_usd'] is not None:
            account['spend_usd'] = None if rate['usd_per_hour'] is None else account['spend_usd'] + rate['usd_per_hour'] * hours
        export = exports.get(ident)
        exported_at = _seconds(export['at']) if export else None
        # Earned work since the newest DGX-verified export (since start when none): lost if the account freezes now.
        unexported = max(0.0, end - max(start, exported_at or start)) / 3600
        if running:
            ahead = min(horizon, (limit - now) / 3600)
            account['running'] += 1
            account['running_gpu'] += rate['gpu']
            account['reserved_gpu_hours'] += rate['gpu'] * ahead
            if account['reserved_usd'] is not None:
                account['reserved_usd'] = None if rate['usd_per_hour'] is None else account['reserved_usd'] + rate['usd_per_hour'] * ahead
            account['unexported_gpu_hours'] += max(rate['gpu'], 1) * unexported
        account['sandboxes'].append({
            'sandbox_id': ident, 'job': entry.get('job'), 'start': _stamp(start), 'end': None if running else _stamp(end),
            'running': running, 'hours': round(hours, 3), 'gpu': rate['gpu'], 'assumed': rate['assumed'],
            'exports': entry.get('exports', []),
            'last_verified_export': None if export is None else {**{k: export.get(k) for k in (
                'source', 'sha256', 'bytes', 'step', 'local')}, 'at': _stamp(exported_at)},
            'unexported_hours': round(unexported, 3)})
    for account in accounts.values():
        _headroom(account)
    return {'schema': 'natlang.coreweave-accounts-ledger/2', 'at': _stamp(now),
            'estimate': ('ours: derived from this machine\'s launch/stop/status/export records and declared resources; '
                         'the cwsandbox client exposes no usage or billing API'),
            'gpu_hour_price_usd': price, 'price_source': 'configured' if price is not None else 'unknown',
            'reservation_horizon_hours': horizon,
            'storage_account': config.get('storage_account', DEFAULT), 'accounts': accounts}


def _cleared(freeze: dict, life: dict) -> dict:
    """An operator's later `accounts activate` (a reviewed false alarm) clears the freeze signals before it."""
    if not freeze['frozen'] or life.get('state') != 'active' or not life.get('at'):
        return freeze
    if all((signal.get('at') or '') <= life['at'] for signal in freeze['signals']):
        return {**freeze, 'frozen': False, 'cleared_at': life['at']}
    return freeze


def _round(value):
    return None if value is None else round(value, 4)


def _headroom(account: dict) -> None:
    """headroom = cap - estimated spend - committed reservation, per known unit. No early retirement (owner
    2026-10-30): an account is used to its full cap; only a freeze or an explicit retirement removes it."""
    def left(cap, used, reserved):
        return None if cap is None or used is None or reserved is None else cap - used - reserved
    account['headroom_usd'] = left(account['cap_usd'], account['spend_usd'], account['reserved_usd'])
    account['headroom_gpu_hours'] = left(account['cap_gpu_hours'], account['gpu_hours'], account['reserved_gpu_hours'])
    if account['cap_usd'] is not None and account['spend_usd'] is None:
        account['note'] = 'cap in USD but the GPU price is unknown: set gpu_hour_price_usd or cap_gpu_hours'
    exhausted = [h for h in (account['headroom_usd'], account['headroom_gpu_hours']) if h is not None and h <= 0]
    account['exhausted'] = bool(exhausted)
    account['eligible'] = bool(account['credential'] and account['placement'] and account['lifecycle'] == 'active'
                               and not account['freeze']['frozen'] and not exhausted)
    for key in ('gpu_hours', 'cpu_hours', 'memory_gib_hours', 'spend_usd', 'cpu_memory_spend_usd', 'reserved_usd',
                'reserved_gpu_hours', 'headroom_usd', 'headroom_gpu_hours', 'unexported_gpu_hours'):
        account[key] = _round(account[key])


def write_ledger(state: Path, value: dict) -> Path:
    path = Path(state) / 'accounts.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=1) + '\n')
    os.replace(tmp, path)
    return path


# Freeze detection ----------------------------------------------------------------------------------------------------

# Owner rule 2026-10-30: an account is frozen ONLY on an explicit quota error from the API: a create refused with it, or
# a sandbox whose terminal status reason carries it. Every other refusal (capacity, placement, throttling, occupancy,
# transient) takes the normal retry path and never triggers an evacuation. Strings actually seen:
#   2026-10-23T20:00Z and 2026-10-29T18:09Z, create_refused, reason None:
#   "SandboxResourceExhaustedError: Create sandbox resource exhausted: quota exceeded, check plan and billing details"
# (On 2026-10-29 the running sandboxes then COMPLETED with reason user_requested: no quota text, so not a signal.)
QUOTA_ERROR = re.compile(r'quota exceeded|quota exhausted|check plan and billing', re.I)
NOT_FREEZE_REASONS = {'CWSANDBOX_OCCUPANCY_QUOTA_EXCEEDED', 'CWSANDBOX_SERVERLESS_CREATE_THROTTLED',
                      'CWSANDBOX_RUNNER_CAPACITY_EXHAUSTED', 'CWSANDBOX_RUNNER_OVERLOADED',
                      'CWSANDBOX_RUNNER_UNAVAILABLE', 'CWSANDBOX_BACKEND_UNAVAILABLE', 'CWSANDBOX_GPU_COUNT_EXCEEDED'}


def quota_error(entry: dict) -> str | None:
    """The explicit quota-error text of one recorded provider answer (create refusal or terminal status), or None."""
    if entry.get('reason') in NOT_FREEZE_REASONS:
        return None
    if entry.get('event') == 'create_refused':
        text = str(entry.get('error') or '')
    elif entry.get('event') == 'status':
        text = str(entry.get('status_reason') or '')
    else:
        return None
    match = QUOTA_ERROR.search(text)
    return text[:300] if match else None


def freeze_signals(records: list[dict], *, now: float | None = None) -> dict[str, dict]:
    """Per account: frozen when we recorded an explicit quota error for it (quota_error); nothing else counts."""
    owners = {r['sandbox_id']: r.get('account', DEFAULT) for r in records
              if r.get('event') == 'launched' and r.get('sandbox_id')}
    found: dict[str, dict] = {}
    for entry in records:
        account = entry.get('account') or owners.get(entry.get('sandbox_id')) or (
            DEFAULT if entry.get('event') == 'create_refused' else None)
        text = quota_error(entry)
        if account and text:
            found.setdefault(account, {'frozen': True, 'signals': []})['signals'].append(
                {'at': _stamp(_seconds(entry['at'])) if 'at' in entry else None, 'event': entry['event'],
                 'sandbox_id': entry.get('sandbox_id'), 'error': text})
    return found


# Placement: balance spend so every account reaches its cap at about the same time -------------------------------------

class NoAccount(AccountError):
    pass


def _need(resources: dict, hours: float, config: dict) -> dict:
    rate = resource_rate(resources, config.get('gpu_hour_price_usd'))
    ahead = min(float(hours), float(config.get('reservation_horizon_hours', 24)))
    return {'usd': None if rate['usd_per_hour'] is None else rate['usd_per_hour'] * ahead,
            'gpu_hours': rate['gpu'] * ahead, 'gpu': rate['gpu']}


def _remaining_fraction(account: dict) -> float | None:
    parts = [h / c for h, c in ((account['headroom_usd'], account['cap_usd']),
                                (account['headroom_gpu_hours'], account['cap_gpu_hours'])) if h is not None and c]
    return min(parts) if parts else None


def choose_account(ledger_value: dict, config: dict, resources: dict, hours: float, *, exclude=(),
                   load: Counter | None = None) -> dict:
    """The account for one new sandbox. Eligible accounts (credential, placement on, active, not frozen, headroom
    left) are ranked by ``load`` first when given (spreading DiLoCo learners), then by the largest remaining fraction
    of the cap (known caps), else by the least used so far; spend therefore evens out across accounts."""
    need = _need(resources, hours, config)
    considered, fits = [], []
    for name, account in sorted(ledger_value['accounts'].items()):
        if name in exclude:
            considered.append({'account': name, 'skip': 'excluded'})
            continue
        if not account['eligible']:
            why = ('no credential' if not account['credential'] else 'placement disabled' if not account['placement']
                   else 'frozen' if account['freeze']['frozen'] else account['lifecycle']
                   if account['lifecycle'] != 'active' else 'cap reached')
            considered.append({'account': name, 'skip': why})
            continue
        fraction = _remaining_fraction(account)
        used = account['spend_usd'] if account['spend_usd'] is not None else account['gpu_hours']
        loaded = (load or {}).get(name, 0)
        considered.append({'account': name, 'load': loaded, 'remaining_fraction': fraction, 'used': used})
        fits.append((loaded, fraction is None, -(fraction or 0.0), used or 0.0, name))
    if not fits:
        raise NoAccount(f'no account can take this sandbox: {json.dumps(considered)}')
    fits.sort()
    loaded, unknown, fraction, used, name = fits[0]
    reason = (f'{name}: ' + (f'load {loaded}, ' if load is not None else '')
              + (f'least used so far ({used:.2f}; cap unknown)' if unknown else f'{-fraction:.0%} of its cap left'))
    return {'account': name, 'reason': reason, 'considered': considered, 'need': need}


def reserve(ledger_value: dict, account: str, need: dict) -> None:
    """Count a planned sandbox against an account (for successive placement decisions in one plan)."""
    entry = ledger_value['accounts'][account]
    entry['running'] += 1
    entry['reserved_gpu_hours'] = (entry['reserved_gpu_hours'] or 0.0) + need['gpu_hours']
    if entry['reserved_usd'] is not None:
        entry['reserved_usd'] = None if need['usd'] is None else entry['reserved_usd'] + need['usd']
    _headroom(entry)
    if entry['headroom_usd'] is None and entry['cap_usd'] is None and entry['cap_gpu_hours'] is None:
        # Unknown caps: balance on use, so count the planned reservation as use for the next decision.
        if entry['spend_usd'] is not None and need['usd'] is not None:
            entry['spend_usd'] = round(entry['spend_usd'] + need['usd'], 4)
        else:
            entry['gpu_hours'] = round((entry['gpu_hours'] or 0.0) + need['gpu_hours'], 4)


def plan_diloco(ledger_value: dict, config: dict, *, workers: int, learner: dict, coordinator: dict,
                cohosted: bool, hours: float, replica: dict | None = None) -> dict:
    """Accounts for a DiLoCo stage: learners spread across accounts; the coordinator replica never shares the
    primary's account. ``learner``/``coordinator``/``replica`` are resource requests (job.py form)."""
    plan = copy.deepcopy(ledger_value)
    learners: list[str | None] = [None] * workers
    spread: Counter = Counter()
    reasons = {}

    def take(role, resources, *, exclude=(), load=None):
        chosen = choose_account(plan, config, resources, hours, exclude=exclude, load=load)
        reserve(plan, chosen['account'], chosen['need'])
        reasons[role] = chosen['reason']
        return chosen['account']

    if cohosted:
        primary = take('coordinator+learner-0', learner, load=Counter(spread))
        learners[0] = primary
        spread[primary] += 1
    else:
        primary = take('coordinator', coordinator)
    replica_account = take('replica', replica, exclude={primary}) if replica is not None else None
    for worker in range(workers):
        if learners[worker] is None:
            learners[worker] = take(f'learner-{worker}', learner, load=Counter(spread))
            spread[learners[worker]] += 1
    return {'coordinator': primary, 'replica': replica_account, 'learners': learners, 'reasons': reasons}


# Evacuation: the first frozen account moves every account's newest state to the DGX ----------------------------------

def evacuation_plan(ledger_value: dict, trigger: str, *, reason: str) -> dict:
    """Every running sandbox of every account, highest unexported work first, grouped per account (pulls run in
    parallel across accounts, each with that account's own key). The frozen account's sandboxes are attempted too
    (best effort: a frozen account may already reject file reads)."""
    groups = {}
    for name, account in sorted(ledger_value['accounts'].items()):
        boxes = [s for s in account['sandboxes'] if s['running']]
        boxes.sort(key=lambda s: (-(max(s['gpu'], 1) * s['unexported_hours']), s['sandbox_id']))
        if boxes:
            groups[name] = [{'sandbox_id': s['sandbox_id'], 'job': s['job'], 'unexported_hours': s['unexported_hours'],
                             'exports': s['exports'], 'best_effort': name == trigger} for s in boxes]
    return {'schema': 'natlang.coreweave-evacuation/1', 'trigger': trigger, 'reason': reason, 'at': ledger_value['at'],
            'order': 'per account: highest unexported GPU-hours first; accounts in parallel', 'accounts': groups}
