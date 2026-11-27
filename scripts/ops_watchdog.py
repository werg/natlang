#!/usr/bin/env python3
"""Ops watchdog: operational failures reach the supervising agent within minutes (DECISIONS 2026-11-23 "ops watchdog").

    ops_watchdog.py check [--only CHECK ...]     # the natlang-ops-watchdog timer, every 5 minutes, outside the ledger
    ops_watchdog.py wait [--state F] [--timeout S]   # the supervisor's background command: blocks until unseen alerts

``check`` runs crisp checks (plumbing only; every decision about what to do stays with the agent) and appends one JSON
line per alert to ``OPS/alerts.jsonl`` (``time``, ``ts``, ``check``, ``key``, ``severity``, ``message``, ``detail``,
``repeat``). An alert key fires once, again every ``realert_minutes`` while its condition holds, and at once when the
condition clears and returns. Checks:

  disk        free space under a mount's floor; a fall faster than ``disk_fall_gb_per_hour``
  colab       ``colab --auth=adc sessions``: a configured session missing or not a GPU, a stray session, the tunnel
              unit's restart counter rising or the unit down; ``colab usage``: credits burning while the session's
              GPU is idle or unreachable, a balance under ``colab_min_balance_hours``
  gpu         DGX ``nvidia-smi``, Pop over ssh, Colab over the session's existing ControlMaster only (never a second
              Colab connection): idle (< ``gpu_busy_percent``) or unreachable for ``gpu_idle_minutes`` while an
              experiment that expects the GPU has an active unit
  units       unit failures from the user journal: more than ``storm.failures`` under one unit-name prefix within
              ``storm.window_minutes``; any failure of an experiment's unit
  ledger      ``memory_ledger.py status``: headroom under ``ledger.headroom_gb``; the queue head waiting longer than
              ``ledger.queue_wait_minutes``
  driver      an experiment's ``drive_log`` with no new stage line in ``stall_hours`` while its units are active
  health      new flags in an experiment's ``health_glob`` files (``collection_health.py``); new lines matching a
              ``logs`` pattern (``COLLECTION HEALTH`` in ``storage-alerts.log``)

The in-flight experiments (``experiments``) and every threshold come from ``OPS/watchdog.json``, which the heartbeat
edits; keys it leaves out take the defaults below. ``wait`` keeps a byte cursor in ``--state`` and resumes exactly where
the last call stopped (no alert falls between two waits); it also reports a watchdog whose timer stopped running.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import fnmatch
import glob
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))
from natlang_neuralese.common import paths  # noqa: E402  (stdlib only)

OPS = Path(os.environ.get('NATLANG_OPS_DIR') or paths.resolve('data_nvme', 'lfm3-rl-dgx-20261029', 'ops'))
STATE = Path(os.environ.get('NATLANG_OPS_WATCHDOG_STATE')
             or Path.home() / '.local/state/natlang/ops-watchdog.json')
WAIT_STATE = Path.home() / '.cache/natlang-watch/ops-watchdog-wait.json'
UNIT_FAILED_MESSAGE_ID = 'd9b373ed55a64feb8242e02dbe79a49c'   # systemd "Failed with result" (unit_log_failure)
GPU_SAMPLE = ('for i in 1 2 3; do nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits; '
              'sleep 1; done')
CHECKS = ('disk', 'gpu', 'colab', 'units', 'ledger', 'driver', 'health')

DEFAULTS = {
    'realert_minutes': 60,
    'disks': [{'mount': '/', 'floor_gb': 40}, {'mount': str(paths.root('data_hdd')), 'floor_gb': 50}],
    'disk_fall_gb_per_hour': 8,
    'disk_fall_window_minutes': 60,
    'gpu_idle_minutes': 20,
    'gpu_busy_percent': 10,
    # name -> {"kind": "local" | "ssh" | "colab", "host": ssh alias}
    'gpus': {'dgx': {'kind': 'local'}, 'pop': {'kind': 'ssh', 'host': 'pop'},
             'colab-g4a': {'kind': 'colab', 'host': 'colab-g4'}},
    'colab': {'sessions': [{'session': 'g4a', 'gpu': 'colab-g4a', 'tunnel_unit': 'colab-g4a-tunnel'}],
              'min_balance_hours': 24, 'tunnel_restarts': 3},
    'storm': {'failures': 3, 'window_minutes': 30, 'ignore': [r'^snap\.']},
    'ledger': {'headroom_gb': 15, 'queue_wait_minutes': 60},
    'logs': [{'path': 'storage-alerts.log', 'pattern': 'COLLECTION HEALTH'}],
    # [{"name", "units": [glob], "gpus": [gpu name], "drive_log", "stall_hours", "health_glob"}]; paths relative to OPS
    'experiments': [],
}


# ---------------------------------------------------------------------------------------------------------- plumbing

def run(command, timeout=60):
    """(returncode, stdout, stderr) of ``command``; a timeout or a missing binary is returncode 124 / 127."""
    try:
        r = subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, '', f'timed out after {timeout} s'
    except FileNotFoundError as e:
        return 127, '', str(e)


def disk_free_gb(mount):
    st = os.statvfs(mount)
    return st.f_bavail * st.f_frsize / 2**30


def is_mount(mount):
    return mount == '/' or os.path.ismount(mount)


def load_config(path):
    config = copy.deepcopy(DEFAULTS)
    if path.exists():
        for key, value in json.loads(path.read_text()).items():
            if isinstance(value, dict) and isinstance(config.get(key), dict):
                config[key] = {**config[key], **value}
            else:
                config[key] = value
    return config


def ops_path(p):
    p = Path(p)
    return p if p.is_absolute() else OPS / p


def iso(ts):
    return datetime.fromtimestamp(ts).astimezone().isoformat(timespec='seconds')


def unit_prefix(unit):
    """``colab-g4a-fill-semantic-1795...-nores2.service`` -> ``colab-g4a-fill-semantic``: trailing dash-separated parts
    that carry a digit (ids, counters, versions) are dropped."""
    name = re.sub(r'\.(service|scope|timer|socket)$', '', unit)
    parts = name.split('-')
    while len(parts) > 1 and re.search(r'\d', parts[-1]):
        parts.pop()
    return '-'.join(parts)


def unit_matches(unit, globs):
    return any(fnmatch.fnmatch(unit, g) or fnmatch.fnmatch(unit, g + '.service') for g in globs)


def unit_tail(unit, lines=4):
    rc, out, _ = run(['journalctl', '--user', '-u', unit, '-n', str(lines), '-o', 'cat', '--no-pager'], timeout=20)
    return [line[:300] for line in out.splitlines()] if rc == 0 else []


class Watchdog:
    def __init__(self, config, state, now, alerts_path):
        self.config, self.state, self.now, self.alerts_path = config, state, now, alerts_path
        self.emitted, self.seen = [], set()
        self.state.setdefault('alerts', {})
        self._active_units = None

    def alert(self, check, key, severity, message, detail=None, once=False):
        """Record an alert for ``key`` unless it fired within ``realert_minutes`` (``once``: never again)."""
        key = f'{check}:{key}'
        self.seen.add(key)
        prev = self.state['alerts'].get(key)
        if prev and (once or self.now - prev['last'] < self.config['realert_minutes'] * 60):
            return
        repeat = prev['repeat'] + 1 if prev else 0
        self.state['alerts'][key] = {'last': self.now, 'repeat': repeat, 'once': once}
        self.emitted.append({'time': iso(self.now), 'ts': round(self.now, 1), 'check': check, 'key': key,
                             'severity': severity, 'message': message, 'detail': detail or {}, 'repeat': repeat})

    def finish(self, ran):
        """Forget keys whose condition cleared (a recurrence alerts at once); one-shot keys are kept a week. Keys of
        checks that did not run this time are left alone."""
        for key in list(self.state['alerts']):
            entry = self.state['alerts'][key]
            if key.split(':', 1)[0] not in ran or key in self.seen:
                continue
            if not entry.get('once') or self.now - entry['last'] > 7 * 86400:
                del self.state['alerts'][key]
        self.state['last_run'] = self.now
        if self.emitted:
            self.alerts_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.alerts_path, 'a') as handle:
                fcntl.flock(handle, fcntl.LOCK_EX)
                handle.write(''.join(json.dumps(a, sort_keys=True) + '\n' for a in self.emitted))
        return self.emitted

    def active_units(self):
        if self._active_units is None:
            rc, out, _ = run(['systemctl', '--user', 'list-units', '--type=service', '--plain', '--no-legend',
                              '--no-pager', '--state=active,activating,reloading'], timeout=30)
            self._active_units = [line.split()[0] for line in out.splitlines() if line.strip()] if rc == 0 else []
        return self._active_units

    def experiment_units(self, exp):
        return [u for u in self.active_units() if unit_matches(u, exp.get('units', []))]


# ------------------------------------------------------------------------------------------------------------ checks

def check_disk(w):
    cfg, hist = w.config, w.state.setdefault('disk', {})
    window = cfg['disk_fall_window_minutes'] * 60
    for d in cfg['disks']:
        mount, floor = d['mount'], d['floor_gb']
        if not is_mount(mount):
            continue
        free = disk_free_gb(mount)
        samples = [s for s in hist.get(mount, []) if w.now - s[0] <= 3 * window] + [[w.now, round(free, 2)]]
        hist[mount] = samples
        if free < floor:
            w.alert('disk', f'floor:{mount}', 'critical', f'{mount} has {free:.0f} GiB free < floor {floor} GiB',
                    {'mount': mount, 'free_gb': round(free, 1), 'floor_gb': floor})
        past = [s for s in samples if window / 3 <= w.now - s[0] <= window * 1.5]
        if past:
            t0, f0 = min(past, key=lambda s: abs(w.now - s[0] - window))
            rate = (f0 - free) / ((w.now - t0) / 3600)
            if rate >= cfg['disk_fall_gb_per_hour']:
                eta = max(free - floor, 0) / rate
                w.alert('disk', f'fall:{mount}', 'critical',
                        f'{mount} free space falling {rate:.0f} GiB/h ({f0:.0f} -> {free:.0f} GiB in '
                        f'{(w.now - t0) / 60:.0f} min); floor {floor} GiB in {eta:.1f} h',
                        {'mount': mount, 'rate_gb_per_hour': round(rate, 1), 'free_gb': round(free, 1),
                         'hours_to_floor': round(eta, 2)})


def gpu_sample(spec):
    """(max utilization %, error) of a GPU host from three one-second samples; Colab only over its live master."""
    kind, host = spec.get('kind', 'local'), spec.get('host')
    if kind == 'local':
        command = ['bash', '-c', GPU_SAMPLE]
    elif kind == 'ssh':
        command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, GPU_SAMPLE]
    elif kind == 'colab':
        # The session admits one SSH connection, held by its tunnel unit's ControlMaster: use it or nothing.
        # ProxyCommand=/bin/false makes a missing master fail instead of dialling the session a second time.
        if run(['ssh', '-O', 'check', host], timeout=15)[0] != 0:
            return None, 'no ControlMaster (tunnel down)'
        command = ['ssh', '-o', 'ControlMaster=no', '-o', 'ProxyCommand=/bin/false', '-o', 'BatchMode=yes', host,
                   GPU_SAMPLE]
    else:
        return None, f'unknown kind {kind}'
    rc, out, err = run(command, timeout=45)
    values = [int(v) for v in re.findall(r'^\s*(\d+)\s*$', out, re.M)]
    if rc != 0 or not values:
        return None, (err.strip() or out.strip() or f'exit {rc}')[-200:]
    return max(values), None


def check_gpu(w):
    cfg, gst = w.config, w.state.setdefault('gpu', {})
    expected = {}
    for exp in w.config['experiments']:
        if w.experiment_units(exp):
            for g in exp.get('gpus', []):
                expected.setdefault(g, []).append(exp['name'])
    gap = w.now - w.state.get('last_run', 0) > 20 * 60   # no history across a stopped timer
    for name, spec in cfg['gpus'].items():
        util, error = gpu_sample(spec)
        s = gst.setdefault(name, {'since': w.now, 'last_busy': None})
        if gap:
            s.update(since=w.now, last_busy=None)
        s.update(util=util, error=error, checked=w.now)
        if util is not None and util >= cfg['gpu_busy_percent']:
            s['last_busy'] = w.now
        idle_s = w.now - (s['last_busy'] or s['since'])
        s['idle_s'] = idle_s
        if name in expected and idle_s >= cfg['gpu_idle_minutes'] * 60:
            what = f'unreachable ({error})' if util is None else f'idle ({util}% now)'
            w.alert('gpu', name, 'critical' if util is None else 'warning',
                    f'GPU {name} {what} for {idle_s / 60:.0f} min while {", ".join(expected[name])} expects work',
                    {'gpu': name, 'util': util, 'error': error, 'idle_minutes': round(idle_s / 60),
                     'experiments': expected[name]})


SESSION_RE = re.compile(r'^\[(?P<name>[^\]]*)\]\s+(?P<id>\S+)\s*\|\s*Hardware:\s*(?P<hardware>[^|]+?)\s*'
                        r'\|\s*Shape:\s*(?P<shape>[^|]+?)\s*\|\s*Variant:\s*(?P<variant>\S+)')


def colab_sessions(text):
    return [m.groupdict() for m in map(SESSION_RE.match, text.splitlines()) if m]


def check_colab(w):
    cfg = w.config['colab']
    rc, out, err = run(['colab', '--auth=adc', 'sessions'], timeout=90)
    if rc != 0:
        w.alert('colab', 'cli', 'warning', f'colab sessions failed (exit {rc}): {(err or out).strip()[-200:]}')
        sessions = None
    else:
        sessions = colab_sessions(out)
    wanted = {s['session'] for s in cfg['sessions']}
    tunnels = w.state.setdefault('tunnels', {})
    for s in cfg['sessions']:
        name = s['session']
        if sessions is not None:
            found = [x for x in sessions if x['name'] == name]
            if not found:
                w.alert('colab', f'missing:{name}', 'critical', f'Colab session {name} is gone',
                        {'session': name, 'sessions': sessions})
            for x in found:
                if x['hardware'].upper() == 'CPU' or x['variant'].upper() != 'GPU':
                    w.alert('colab', f'not_gpu:{name}', 'critical',
                            f'Colab session {name} is {x["hardware"]} ({x["variant"]}), not a GPU: {x["id"]}', x)
        unit = s.get('tunnel_unit')
        if unit:
            rc, out, _ = run(['systemctl', '--user', 'show', unit, '-p', 'NRestarts', '-p', 'ActiveState',
                              '-p', 'SubState'], timeout=20)
            props = dict(line.split('=', 1) for line in out.splitlines() if '=' in line)
            restarts = int(props.get('NRestarts') or 0)
            t = tunnels.setdefault(unit, {'restarts': restarts, 'down': 0})
            rose = restarts if restarts < t['restarts'] else restarts - t['restarts']
            t['down'] = 0 if props.get('ActiveState') == 'active' else t['down'] + 1
            t['restarts'] = restarts
            if rose >= cfg['tunnel_restarts'] or t['down'] >= 2:
                w.alert('colab', f'tunnel:{unit}', 'critical',
                        f'{unit} is {props.get("ActiveState")}/{props.get("SubState")}, restart counter {restarts} '
                        f'(+{max(rose, 0)} since the last check)',
                        {'unit': unit, 'restarts': restarts, 'rose': rose, 'tail': unit_tail(unit)})
    for x in sessions or []:
        if x['name'] not in wanted:
            w.alert('colab', f'stray:{x["id"]}', 'warning',
                    f'Colab assignment {x["id"]} ([{x["name"]}], {x["hardware"]}) is in no configured session and '
                    f'may burn credit', x)
    rc, out, _ = run(['colab', '--auth=adc', 'usage'], timeout=90)
    balance = re.search(r'balance:\s*([\d.]+)', out)
    rate = re.search(r'rate:\s*([\d.]+)', out)
    if rc != 0 or not balance or not rate:
        return
    balance, rate = float(balance.group(1)), float(rate.group(1))
    if rate > 0 and balance / rate < cfg['min_balance_hours']:
        w.alert('colab', 'balance', 'warning', f'Colab balance {balance:.0f} units lasts {balance / rate:.0f} h '
                f'at {rate}/h', {'balance': balance, 'rate': rate})
    gst = w.state.get('gpu', {})
    idle = [s['gpu'] for s in cfg['sessions'] if s.get('gpu') in gst
            and gst[s['gpu']].get('idle_s', 0) >= w.config['gpu_idle_minutes'] * 60]
    if rate > 0 and idle:
        w.alert('colab', 'burn_idle', 'warning',
                f'Colab burns {rate}/h (balance {balance:.0f}) while {", ".join(idle)} is idle or unreachable for '
                f'{max(gst[g]["idle_s"] for g in idle) / 60:.0f} min',
                {'rate': rate, 'balance': balance, 'idle': {g: gst[g] for g in idle}})


def journal_failures(cursor, since):
    command = ['journalctl', '--user', f'MESSAGE_ID={UNIT_FAILED_MESSAGE_ID}', '-o', 'json', '--no-pager']
    command += ['--after-cursor', cursor] if cursor else ['--since', f'@{int(since)}']
    rc, out, _ = run(command, timeout=60)
    events, last = [], cursor
    for raw in out.splitlines():
        try:
            r = json.loads(raw)
        except ValueError:
            continue
        last = r.get('__CURSOR', last)
        unit = r.get('USER_UNIT') or r.get('UNIT')
        if unit:
            events.append([int(r.get('__REALTIME_TIMESTAMP', 0)) / 1e6, unit, r.get('UNIT_RESULT', '')])
    return events, last


def check_units(w):
    cfg = w.config['storm']
    window = cfg['window_minutes'] * 60
    new, w.state['journal_cursor'] = journal_failures(w.state.get('journal_cursor'), w.now - window)
    failures = [f for f in w.state.get('failures', []) + new if w.now - f[0] <= window]
    w.state['failures'] = failures
    ignore = [re.compile(p) for p in cfg['ignore']]
    globs = {exp['name']: exp.get('units', []) for exp in w.config['experiments']}
    for ts, unit, result in new:
        for name, gs in globs.items():
            if unit_matches(unit, gs):
                w.alert('units', f'failed:{unit}', 'critical', f'{name} unit {unit} failed ({result}) at {iso(ts)}',
                        {'experiment': name, 'unit': unit, 'result': result, 'tail': unit_tail(unit)})
    groups = {}
    for ts, unit, result in failures:
        if not any(p.search(unit) for p in ignore):
            groups.setdefault(unit_prefix(unit), []).append((ts, unit, result))
    for prefix, rows in groups.items():
        if len(rows) > cfg['failures']:
            last = max(rows)
            w.alert('units', f'storm:{prefix}', 'critical',
                    f'{len(rows)} failures under {prefix}* in {cfg["window_minutes"]} min (last {last[1]}: {last[2]})',
                    {'prefix': prefix, 'failures': len(rows), 'units': sorted({r[1] for r in rows})[:10],
                     'tail': unit_tail(last[1])})


def check_ledger(w):
    cfg = w.config['ledger']
    rc, out, err = run([sys.executable, str(REPO / 'scripts' / 'memory_ledger.py'), 'status'], timeout=90)
    if rc != 0:
        w.alert('ledger', 'status', 'warning', f'memory_ledger status failed (exit {rc}): {err.strip()[-200:]}')
        return
    status = json.loads(out)
    headroom = status.get('headroom_gb')
    if headroom is not None and headroom < cfg['headroom_gb']:
        top = sorted(status.get('claims', {}).items(), key=lambda kv: -kv[1].get('budget_gb', 0))[:5]
        w.alert('ledger', 'headroom', 'warning', f'ledger headroom {headroom} GB < {cfg["headroom_gb"]} GB',
                {'headroom_gb': headroom, 'largest_claims': {u: c.get('budget_gb') for u, c in top}})
    waiting = status.get('waiting') or []
    if waiting and (waiting[0].get('waiting_s') or 0) > cfg['queue_wait_minutes'] * 60:
        head = waiting[0]
        why = head.get('cannot_fit', {}).get('remedy') or ', '.join(b['unit'] for b in head.get('blocked_by', []))
        w.alert('ledger', f'queue:{head["unit"]}', 'warning',
                f'ledger queue head {head["unit"]} ({head.get("budget_gb")} GB) waiting '
                f'{head["waiting_s"] / 60:.0f} min; {len(waiting)} queued; {why}'[:400],
                {'head': head, 'queued': [x['unit'] for x in waiting]})


def last_stage_time(path):
    """Time of the last line in ``path`` that is a JSON record with a ``time`` field (local ISO), else None."""
    with open(path, 'rb') as handle:
        handle.seek(0, 2)
        handle.seek(max(0, handle.tell() - 65536))
        lines = handle.read().decode(errors='replace').splitlines()
    for line in reversed(lines):
        try:
            record = json.loads(line)
            return datetime.fromisoformat(record['time']).timestamp()
        except (ValueError, KeyError, TypeError):
            continue
    return None


def check_driver(w):
    for exp in w.config['experiments']:
        if not exp.get('drive_log'):
            continue
        units = w.experiment_units(exp)
        if not units:
            continue
        path, hours = ops_path(exp['drive_log']), exp.get('stall_hours', 3)
        last = last_stage_time(path) if path.exists() else None
        if last is None:
            last = path.stat().st_mtime if path.exists() else None
        if last is None or w.now - last > hours * 3600:
            age = f'{(w.now - last) / 3600:.1f} h' if last else 'ever (no log)'
            w.alert('driver', exp['name'], 'critical',
                    f'{exp["name"]} drive log has no new stage line for {age} (> {hours} h) while {len(units)} '
                    f'units are active', {'drive_log': str(path), 'last': iso(last) if last else None,
                                         'units': units[:10]})


def check_health(w):
    seen = w.state.setdefault('health_files', {})
    for exp in w.config['experiments']:
        if not exp.get('health_glob'):
            continue
        for f in sorted(glob.glob(str(ops_path(exp['health_glob'])))):
            try:
                health = json.loads(Path(f).read_text())
            except (OSError, ValueError):
                continue
            stamp = health.get('time')
            if seen.get(f) == stamp:
                continue
            seen[f] = stamp
            for flag in health.get('flags', []):
                subject = flag.get('family') or flag.get('construct') or flag.get('reason') or ''
                detail = {k: v for k, v in flag.items() if k != 'flag'}
                w.alert('health', f'{f}:{stamp}:{flag["flag"]}:{subject}', 'warning',
                        f'{exp["name"]} {Path(f).parent.name}: {flag["flag"]}{" " + subject if subject else ""}'
                        f'{" [sfp]" if flag.get("sfp") else ""} {json.dumps(detail, sort_keys=True)[:250]}',
                        {'file': f, **flag}, once=True)
    cursors = w.state.setdefault('log_cursors', {})
    for spec in w.config['logs']:
        path = ops_path(spec['path'])
        if not path.exists():
            continue
        key, size = str(path), path.stat().st_size
        if key not in cursors or size < cursors[key]:   # new source starts at its end; a truncated one at 0
            cursors[key] = size if key not in cursors else 0
        with open(path, 'rb') as handle:
            handle.seek(cursors[key])
            chunk = handle.read()
        complete = chunk[:chunk.rfind(b'\n') + 1]
        offset = cursors[key]
        cursors[key] += len(complete)
        rx = re.compile(spec['pattern'])
        for n, line in enumerate(complete.decode(errors='replace').splitlines()):
            if rx.search(line):
                w.alert('health', f'log:{path.name}:{offset}:{n}', 'warning', line[:500], {'log': key}, once=True)


CHECK_FUNCS = {'disk': check_disk, 'gpu': check_gpu, 'colab': check_colab, 'units': check_units,
               'ledger': check_ledger, 'driver': check_driver, 'health': check_health}


def run_checks(config, state, now, alerts_path, only=None):
    w = Watchdog(config, state, now, alerts_path)
    ran = set()
    if config.get('_missing'):
        w.alert('watchdog', 'config', 'warning', f'no roster at {config["_missing"]}: no experiment is watched')
    for name in CHECKS:   # gpu before colab: the burn check reads this run's idle times
        if only and name not in only:
            continue
        try:
            CHECK_FUNCS[name](w)
            ran.add(name)
        except Exception as e:  # a broken check is itself an alert; preserve its active alerts (do not add to ran)
            w.alert('watchdog', f'error:{name}', 'warning', f'check {name} raised {type(e).__name__}: {e}'[:400])
    ran.add('watchdog')
    return w.finish(ran)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def cmd_check(a):
    config_path = Path(a.config)
    STATE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE.with_suffix('.lock'), 'w') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('another check is running')
            return
        state = json.loads(STATE.read_text()) if STATE.exists() else {}
        config = load_config(config_path)
        if not config_path.exists():
            config['_missing'] = str(config_path)
        emitted = run_checks(config, state, time.time(), Path(a.alerts), set(a.only or ()) or None)
        save_json(STATE, state)
    for alert in emitted:
        print(f'{alert["severity"].upper()} {alert["key"]}: {alert["message"]}', flush=True)


# -------------------------------------------------------------------------------------------------------------- wait

def read_new(alerts_path, cursor):
    """(alerts, new cursor) after byte ``cursor``; never a half-written line."""
    if not alerts_path.exists():
        return [], cursor
    if alerts_path.stat().st_size < cursor:
        cursor = 0
    with open(alerts_path, 'rb') as handle:
        handle.seek(cursor)
        chunk = handle.read()
    complete = chunk[:chunk.rfind(b'\n') + 1]
    alerts = []
    for line in complete.decode(errors='replace').splitlines():
        try:
            alerts.append(json.loads(line))
        except ValueError:
            continue
    return alerts, cursor + len(complete)


def start_cursor(alerts_path, since):
    """Byte offset of the first alert at or after ``since`` (a new waiter sees the recent backlog)."""
    if not alerts_path.exists():
        return 0
    offset = 0
    with open(alerts_path, 'rb') as handle:
        for line in handle:
            try:
                if json.loads(line).get('ts', 0) >= since:
                    return offset
            except ValueError:
                pass
            offset += len(line)
    return offset


def format_alert(a):
    return f'{str(a.get("time", ""))[:16]} {str(a.get("severity", "?")).upper()} {a.get("key")}: {a.get("message")}'


def wait_once(alerts_path, wait_state, watchdog_state, now, stale_minutes):
    """Unseen alert lines (advancing the cursor in ``wait_state``), plus a stale-watchdog line once per stale run."""
    out = []
    alerts, wait_state['cursor'] = read_new(alerts_path, wait_state['cursor'])
    out += [format_alert(a) for a in alerts]
    last_run = watchdog_state.get('last_run')
    if last_run is None or now - last_run > stale_minutes * 60:
        if wait_state.get('stale_reported') != last_run:
            wait_state['stale_reported'] = last_run
            when = f'{(now - last_run) / 60:.0f} min ago' if last_run else 'never'
            out.append(f'{iso(now)[:16]} CRITICAL watchdog:stale: the ops watchdog last ran {when} '
                       f'(systemctl --user status natlang-ops-watchdog.timer)')
    return out


def cmd_wait(a):
    state_path = Path(a.state).expanduser()
    alerts_path = Path(a.alerts)
    wait_state = json.loads(state_path.read_text()) if state_path.exists() else {}
    if 'cursor' not in wait_state:
        wait_state['cursor'] = start_cursor(alerts_path, time.time() - a.backlog_minutes * 60)
    started = time.time()
    last_saved = json.dumps(wait_state, sort_keys=True)
    save_json(state_path, wait_state)
    while True:
        watchdog_state = json.loads(STATE.read_text()) if STATE.exists() else {}
        lines = wait_once(alerts_path, wait_state, watchdog_state, time.time(), a.stale_minutes)
        serialized = json.dumps(wait_state, sort_keys=True)
        if serialized != last_saved:
            save_json(state_path, wait_state)
            last_saved = serialized
        if lines:
            print('\n'.join(lines), flush=True)
            return
        if a.timeout and time.time() - started > a.timeout:
            print('quiet', flush=True)
            return
        time.sleep(a.interval)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--alerts', default=str(OPS / 'alerts.jsonl'))
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('check', help='run the checks once (the timer)')
    c.add_argument('--config', default=str(OPS / 'watchdog.json'))
    c.add_argument('--only', action='append', choices=CHECKS)
    c.set_defaults(func=cmd_check)
    wt = sub.add_parser('wait', help='block until unseen alerts exist, print them, exit')
    wt.add_argument('--state', default=str(WAIT_STATE), help='JSON file holding this waiter\'s cursor')
    wt.add_argument('--interval', type=float, default=15)
    wt.add_argument('--timeout', type=float, default=0, help='seconds; 0 waits indefinitely')
    wt.add_argument('--backlog-minutes', type=float, default=60, help='a new cursor starts this far back')
    wt.add_argument('--stale-minutes', type=float, default=15, help='report a watchdog that has not run this long')
    wt.set_defaults(func=cmd_wait)
    a = p.parse_args(argv)
    a.func(a)


if __name__ == '__main__':
    main()
