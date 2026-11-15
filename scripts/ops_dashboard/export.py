#!/usr/bin/env python3
"""Live ops dashboard exporter: every 30 s append machine, run, process and event records as JSONL.

Contract: <data_dir>/ops/dashboard/CONTRACT.md (data_dir, floors and the Pop ssh host are in config.json). Stdlib only, Python 3.10+.
The same file runs on Pop (`ssh POP python3 - --host-json CFG < export.py`) to collect that host's metrics,
memory ledger and systemd units in one ssh call.
"""
import argparse
import fnmatch
import glob
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

try:     # absent when the file runs from stdin on Pop (collect_host only)
    CONFIG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')))
except (OSError, NameError):
    CONFIG = {}
DATA = os.path.expanduser(CONFIG.get('data_dir', ''))
OPS = DATA + '/ops'
DASH = OPS + '/dashboard'
FALLOW = os.path.expanduser(CONFIG.get('repo', '')) + '/plans/neuralese/FALLOW_REGISTER.md'
LEDGER = os.path.expanduser(CONFIG.get('repo', '')) + '/scripts/memory_ledger.py'
POP_LEDGER = '~/natlang/scripts/memory_ledger.py'     # Pop's checkout, expanded on Pop
LIMIT = 28000          # bytes per line; the contract's hard limit is 30 KiB
KEEP_S = 6 * 3600      # finished/failed units stay listed this long
SLOW_S = 600           # queue/investigations/alerts/fallow are re-emitted at most this often unless changed
FLOORS = CONFIG.get('floors_gb', {})
DEFAULT_PATTERNS = ['hunt-*', 'grpo-*', 'popsoak-*', 'front-*', 'checkmain-*', 'test-*', 'trlref-*', 'perf-*', 'exp*',
                    'held-*', 'opsdash*', 'ops-dashboard-*', 'probe-*', 'bench-*', 'sft-*', 'natlang-gb10-compactor*']
SKIP_DIRS = {'actor', 'groups', 'held', 'rollouts', 'decisions', 'code', 'workers', '__pycache__', 'dashboard'}

ALWAYS = ['reward_mean', 'kl_mean', 'grad_norm', 'grad_norm_vs_median', 'lr', 'lr_data_scale', 'spread_fraction', 'groups',
          'groups_with_spread', 'episodes', 'rho_mean', 'abs_log_rho_mean', 'ref_abs_dlogp_mean', 'ref_dlogp_max', 'mean_logp',
          'reasoning_tokens_per_call', 'reply_tokens_per_call', 'reasoning_vs_first_step', 'learner_tokens_per_s',
          'actor_tokens_per_s_collect_phase', 'actor_tokens_per_s_learn_phase', 'cuda_peak_gb', 'bf16_changed_step_fraction',
          'bf16_changed_vs_start_fraction', 'grad_moment_cosine']
ALWAYS_PREFIX = ('behaviour.', 'grad_mass_share.', 'phase_s.', 'gain.')   # nested hunt gain object: always kept
SKIP_PREFIX = ('meminfo', 'swap', 'learner_swap', 'staleness.', 'decision_diagnostics.worst', 'lowering.')
FLAGS = ('grad_spike', 'reasoning_shift', 'behaviour_shift')
FALLOW_COLUMNS = {'#': 'id', 'id': 'id', 'piece': 'piece', 'capability': 'piece', 'code now': 'code_now',
                  'built where': 'code_now', 'blocker': 'blocker', 'fix before it returns': 'blocker',
                  'next action': 'next_action'}


# ---------- small helpers ----------

def clip(s, n):
    s = '' if s is None else str(s)
    return s if len(s) <= n else s[:n - 1] + '…'


def ms(t):
    return int(t * 1000)


def sh(cmd, timeout=20):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ''


def line_size(rec):
    return len(json.dumps(rec, ensure_ascii=False, separators=(',', ':')).encode())


def read_json(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def run_id(ops, path):
    rel = os.path.relpath(path, ops)
    return None if rel.startswith('..') else rel.replace('/', '.')


def last_arg(cmd, flag):
    found = re.findall(r'(?:^|\s)' + re.escape(flag) + r'[ =](\S+)', cmd)
    return found[-1] if found else None


def summarize(cmd):
    """Command without `env`, VAR=x words and the interpreter path, clipped to 160 chars."""
    words = [w for w in cmd.split() if w != '/usr/bin/env' and not re.match(r'^[A-Za-z_][A-Za-z0-9_]*=', w)
             and not re.search(r'(^|/)python[0-9.]*$', w)]
    return clip(' '.join(os.path.basename(w) if i == 0 and '/' in w else w for i, w in enumerate(words)), 160)


# ---------- host collection (runs locally and on Pop) ----------

def parse_systemd_time(s):
    m = re.search(r'(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)', s or '')
    return time.mktime(time.strptime(m.group(1), '%Y-%m-%d %H:%M:%S')) if m else None


def cpu_jiffies():
    v = [int(x) for x in open('/proc/stat').readline().split()[1:]]
    return sum(v), v[3] + (v[4] if len(v) > 4 else 0)


def host_metrics(mounts):
    t0, i0 = cpu_jiffies()
    time.sleep(0.4)
    t1, i1 = cpu_jiffies()
    mem = {k: int(v.split()[0]) * 1024 for k, v in (l.split(':', 1) for l in open('/proc/meminfo'))}
    gib = 2 ** 30
    out = {'cpu_percent': round(100 * (1 - (i1 - i0) / max(t1 - t0, 1)), 1), 'load1': float(open('/proc/loadavg').read().split()[0]),
           'mem_total_gb': round(mem['MemTotal'] / gib, 1), 'mem_available_gb': round(mem['MemAvailable'] / gib, 1),
           'mem_used_gb': round((mem['MemTotal'] - mem['MemAvailable']) / gib, 1),
           'swap_used_gb': round((mem['SwapTotal'] - mem['SwapFree']) / gib, 1), 'gpu_util_percent': None, 'gpu_mem_used_gb': None}
    smi = sh(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'], 10).strip().splitlines()
    if smi:
        util, used = [x.strip() for x in smi[0].split(',')][:2]
        out['gpu_util_percent'] = float(util) if util.replace('.', '').isdigit() else None
        out['gpu_mem_used_gb'] = round(float(used) / 1024, 1) if used.replace('.', '').isdigit() else None
    disks = []
    for m in mounts:
        try:
            u = shutil.disk_usage(m)
            disks.append({'mount': m, 'size_gb': round(u.total / gib), 'free_gb': round(u.free / gib)})
        except OSError:
            pass
    out['disks'] = disks
    return out


def show_units(names):
    props = ['Id', 'LoadState', 'ActiveState', 'Result', 'MainPID', 'ActiveEnterTimestamp', 'InactiveEnterTimestamp',
             'ExecMainStartTimestamp', 'CPUUsageNSec', 'ExecStart']
    text = sh(['systemctl', '--user', 'show', *names] + [a for p in props for a in ('-p', p)], 30)
    units = {}
    for block in text.split('\n\n'):
        kv = dict(l.split('=', 1) for l in block.splitlines() if '=' in l)
        if 'Id' not in kv:
            continue
        m = re.search(r'argv\[\]=(.*?) ;', kv.get('ExecStart', ''))
        nsec = kv.get('CPUUsageNSec', '')
        units[kv['Id']] = {
            'loaded': kv.get('LoadState') == 'loaded', 'active': kv.get('ActiveState', ''), 'result': kv.get('Result', ''),
            'pid': int(kv.get('MainPID') or 0), 'cmd': m.group(1) if m else '',
            'start': parse_systemd_time(kv.get('ActiveEnterTimestamp')) or parse_systemd_time(kv.get('ExecMainStartTimestamp')),
            'end': parse_systemd_time(kv.get('InactiveEnterTimestamp')), 'cpu_ns': int(nsec) if nsec.isdigit() else None}
    return units


def collect_host(cfg):
    """Metrics, ledger status, matching systemd units and short journal tails of this machine."""
    host = host_metrics(cfg['mounts'])
    ledger = None
    try:
        ledger = json.loads(sh(['/usr/bin/python3', os.path.expanduser(cfg['ledger']), 'status'], 30))
    except Exception:
        pass
    listed = []
    for l in sh(['systemctl', '--user', 'list-units', '--all', '--plain', '--no-legend', '--type=service']).splitlines():
        w = [x for x in l.split() if x != '●']
        if w:
            listed.append(w[0])
    names = {n for n in listed if any(fnmatch.fnmatch(n, p) or fnmatch.fnmatch(n[:-8], p) for p in cfg['patterns'])}
    names |= set((ledger or {}).get('claims', {})) | set(cfg.get('known', []))
    units = show_units(sorted(names)) if names else {}
    journal = {}
    for n, u in units.items():
        if u['active'] in ('active', 'activating', 'reloading', 'deactivating') or (u['active'] == 'failed' and n not in cfg.get('have_log', [])):
            lines = [clip(x, 400) for x in sh(['journalctl', '--user', '-u', n, '-n', '12', '-o', 'cat', '--no-pager'], 10).splitlines() if x.strip()]
            journal[n] = lines[-12:]
    return {'host': host, 'ledger': ledger, 'units': units, 'journal': journal}


def collect_pop(cfg, timeout=45):
    cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=8', CONFIG['pop_ssh_host'], '/usr/bin/python3', '-', '--host-json',
           shlex.quote(json.dumps(cfg))]
    r = subprocess.run(cmd, input=open(__file__).read(), capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(clip(r.stderr.strip().splitlines()[-1] if r.stderr.strip() else f'ssh exit {r.returncode}', 160))
    return json.loads(r.stdout.strip().splitlines()[-1])


# ---------- metrics ----------

def flatten(obj, prefix='', out=None):
    out = {} if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f'{prefix}{k}.' if isinstance(v, dict) else f'{prefix}{k}', out)
    elif isinstance(obj, bool):
        out[prefix] = int(obj)
    elif isinstance(obj, (int, float)) and math.isfinite(obj):
        out[prefix] = round(obj, 6) if isinstance(obj, float) else obj
    return out


def select_metrics(rec, cap=80):
    flat = flatten({k: v for k, v in rec.items() if k != 'step'})
    chosen = {k: v for k, v in flat.items() if k in ALWAYS or k.startswith(ALWAYS_PREFIX) or 'gain' in k}
    for k, v in flat.items():
        if len(chosen) >= cap:
            break
        if k not in chosen and not k.startswith(SKIP_PREFIX):
            chosen[k] = v
    for f in FLAGS:
        chosen[f] = int(bool(rec.get(f)))
    return chosen


def is_step(rec):
    return isinstance(rec, dict) and 'step' in rec and ('cuda_peak_gb' in rec or 'behaviour' in rec)


def discover_runs(ops):
    found = []

    def walk(d, parts):
        if parts and os.path.isfile(d + '/train.jsonl'):
            found.append((parts, d))
        if len(parts) >= 3:
            return
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            if e.is_dir(follow_symlinks=False) and e.name not in SKIP_DIRS and not e.name.startswith(('pin-', '.')):
                walk(e.path, parts + [e.name])
    walk(ops, [])
    return [('.'.join(p), d) for p, d in found]


def run_kind(run, rec):
    if rec.get('replay_groups') or any(x in run.split('.') for x in ('hunt', 'bench', 'replay-controls')):
        return 'bench'
    return 'grpo' if run.startswith('grpo') else 'other'


def tail_run(st, path):
    """New complete lines of train.jsonl since st['offset']; returns the parsed step records."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return [], 0
    off = st.get('offset', 0) if size >= st.get('offset', 0) else 0
    with open(path, 'rb') as f:
        f.seek(off)
        data = f.read()
    end = data.rfind(b'\n') + 1
    st['offset'] = off + end
    steps = []
    for raw in data[:end].splitlines():
        try:
            rec = json.loads(raw.decode('utf-8', 'replace'))
        except ValueError:
            continue
        if is_step(rec):
            steps.append(rec)
    return steps, os.path.getmtime(path)


def step_records(run, st, steps, mtime, now):
    """Metric records and sentinel events; a step's ts is estimated from the file mtime and elapsed_s."""
    out, events = [], []
    last = max([s.get('elapsed_s') or 0 for s in steps] or [0])
    group = run.rsplit('.', 1)[0] if '.' in run else run
    for s in steps:
        el = s.get('elapsed_s')
        ts = (mtime - (last - el)) if isinstance(el, (int, float)) and last else mtime
        st['max_step'] = max(st.get('max_step', 0), int(s['step']))
        st.setdefault('recent', []).append([int(s['step']), el])
        st['recent'] = st['recent'][-6:]
        data = {'run': run, 'group': group, 'kind': run_kind(run, s), 'step': int(s['step']),
                'wall_s': el if isinstance(el, (int, float)) else None, 'metrics': select_metrics(s)}
        out.append({'ts': ms(ts), 'type': 'train', 'data': data})
        flags = [f for f in FLAGS if s.get(f)]
        if flags and now - ts < KEEP_S:
            events.append({'ts': ms(ts), 'type': 'event', 'data': {
                'kind': 'flag', 'level': 'warn', 'run': run, 'step': data['step'], 'flags': flags,
                'text': f'{run} step {data["step"]}: {", ".join(flags)}'}})
    return out, events


def held_record(run, path):
    d = read_json(path, None)
    if not d or 'primary' not in d:
        return None
    name = os.path.basename(os.path.dirname(path))
    m = re.match(r'step-(\d+)$', name)
    first, second, prim, paired = d.get('first', {}), d.get('second', {}), d['primary'], d.get('paired', {})
    return {'run': run, 'step': int(m.group(1)) if m else None, 'held_dir': name, 'policy': second.get('policy'),
            'delta_pts': round(100 * prim['delta'], 2), 'p': prim.get('permutation_p'),
            'first_rate': prim.get('first_rate'), 'second_rate': prim.get('second_rate'), 'tasks': prim.get('tasks'),
            'ran': second.get('ran'), 'of': first.get('ran'), 'complete': bool(second.get('complete')),
            'loop_rate_delta': d.get('loop_rate_delta'), 'ci_lo': (prim.get('ci95_task_bootstrap') or [None, None])[0],
            'ci_hi': (prim.get('ci95_task_bootstrap') or [None, None])[1], 'mcnemar_p': paired.get('mcnemar_exact_p'),
            'only_first': paired.get('only_first'), 'only_second': paired.get('only_second'),
            'tasks_up': paired.get('tasks_up'), 'tasks_down': paired.get('tasks_down')}


def scan_runs(ops, state, now):
    metrics, events = [], []
    for run, d in discover_runs(ops):
        st = state.setdefault('runs', {}).setdefault(run, {})
        steps, mtime = tail_run(st, d + '/train.jsonl')
        if steps:
            m, e = step_records(run, st, steps, mtime, now)
            metrics += m
            events += e
        if mtime:
            st['mtime'] = mtime
        for p in sorted(glob.glob(d + '/held/*/compare-vs-step0.json')):
            sig = f'{os.path.getmtime(p):.0f}:{os.path.getsize(p)}'
            key = f'{run}/{os.path.basename(os.path.dirname(p))}'
            fresh = key not in state.setdefault('held', {})
            if state['held'].get(key) == sig:
                continue
            state['held'][key] = sig
            h = held_record(run, p)
            if not h:
                continue
            ts = ms(os.path.getmtime(p))
            metrics.append({'ts': ts, 'type': 'held', 'data': h})
            if not fresh or now - ts / 1000 < 24 * 3600:
                events.append({'ts': ts, 'type': 'event', 'data': {
                    'kind': 'held', 'level': 'info', 'run': run, 'step': h['step'],
                    'text': f'Held {h["held_dir"]} on {run}: {h["delta_pts"]:+.1f} pts (p={h["p"]}), {h["ran"]}/{h["of"]} ran'}})
    return metrics, events


# ---------- fallow register ----------

def split_row(line):
    cells = re.findall(r'((?:`[^`]*`|\\\||[^|`]|`)*)(?:\||$)', line.strip()[1:].rstrip('|') + '|')
    return [c.strip().replace('\\|', '|') for c in cells[:-1]]


def parse_fallow(text):
    rows, section, header, sep_next = [], '', None, False
    for line in text.splitlines():
        h = re.match(r'#{2,3}\s+(.*)', line)
        if h:
            section, header = h.group(1).strip(), None
        elif not line.startswith('|'):
            header = None
        elif header is None:
            header, sep_next = split_row(line), True
        elif sep_next and all(re.fullmatch(r':?-{2,}:?', c) for c in split_row(line) if c):
            sep_next = False
        else:
            cells = split_row(line)
            cells = cells[:len(header) - 1] + [' | '.join(cells[len(header) - 1:])] if len(cells) > len(header) else cells
            row = {'section': section, 'id': '', 'piece': '', 'code_now': '', 'blocker': '', 'next_action': '', 'fields': {}}
            for name, cell in zip(header, cells):
                key = FALLOW_COLUMNS.get(name.lower())
                if key:
                    row[key] = clip(cell, 300)
                else:
                    row['fields'][name] = clip(cell, 200)
            rows.append(row)
    return rows


# ---------- processes, queue, investigations, alerts ----------

def annotation_for(ann, name):
    short = name[:-8] if name.endswith('.service') else name
    return next((u for u in ann.get('units', []) if fnmatch.fnmatch(name, u['match']) or fnmatch.fnmatch(short, u['match'])), {})


def kind_of(name, cmd):
    if 'grpo.py' in cmd and ' train' in cmd:
        return 'bench' if '--replay' in cmd or name.startswith('hunt-') else 'grpo'
    if 'held_eval' in cmd and 'measure' in cmd:
        return 'held-eval'
    if 'pytest' in cmd or 'check-main' in cmd:
        return 'test'
    if 'probe' in name or 'behaviour_probe' in cmd:
        return 'probe'
    return 'service' if re.search(r'serve|compactor|ops-dashboard|collector', name + cmd) else 'other'


def held_progress(lines):
    for l in reversed(lines):
        if '"held_eval_progress"' in l:
            try:
                d = json.loads(l[l.index('{'):])
                return d['held_eval_progress'], d['of'], d.get('minutes')
            except (ValueError, KeyError):
                pass
    return None


def read_tail(path, n=8192):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - n))
            return f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return []


def progress_of(rec, state, lines, now):
    """(progress, eta_s) for a unit record: grpo --work step count or a held-eval measure line."""
    cmd = rec['cmd']
    work = last_arg(cmd, '--work')
    if work:
        r = state.get('runs', {}).get(run_id(OPS, work) or '')
        if not r or 'max_step' not in r:
            return None, None
        total = last_arg(cmd, '--max-steps')
        total = int(total) if total else None
        per = [b - a for (s0, a), (s1, b) in zip(r.get('recent', []), r.get('recent', [])[1:])
               if s1 == s0 + 1 and a is not None and b is not None and b > a]
        per_step = sum(per) / len(per) if per else None
        eta = (total - r['max_step']) * per_step if total and per_step and rec['state'] == 'running' else None
        return {'done': r['max_step'], 'total': total, 'unit': 'steps'}, round(eta) if eta is not None else None
    hp = None
    if kind_of(rec['unit'], cmd) == 'held-eval':      # progress line in the unit journal, else WORK/held-pop.log (out = WORK/held/step-N)
        out = last_arg(cmd, '--out')
        hp = held_progress(lines) or (held_progress(read_tail(os.path.join(out, '..', '..', 'held-pop.log'))) if out else None)
    if hp:
        done, of, minutes = hp
        eta = (of - done) * minutes * 60 / done if done and minutes and rec['state'] == 'running' else None
        return {'done': done, 'total': of, 'unit': 'episodes'}, round(eta) if eta is not None else None
    return None, None


def update_units(state, machine, data, now, ann):
    """Merge this cycle's systemd/ledger view into state; returns (process items, events, failed-recently alerts)."""
    store = state.setdefault('units', {})
    claims = (data.get('ledger') or {}).get('claims', {})
    events, items, alerts = [], [], []
    names = {n for n, u in data['units'].items() if u['loaded']} | set(claims)
    for key in [k for k in store if k.startswith(machine + ':') and k.split(':', 1)[1] not in names and store[k]['state'] == 'running']:
        store[key].update(state='finished', ended=now)        # gone from systemd and the ledger: it finished
        events.append(unit_event(store[key], now, ann))
    for name in sorted(names):
        u, key = data['units'].get(name), f'{machine}:{name}'
        status = 'running' if (not u or u['active'] in ('active', 'activating', 'reloading', 'deactivating')) else 'failed' if u['active'] == 'failed' else 'finished'
        prior = store.get(key)
        started = (u or {}).get('start') or (prior or {}).get('started')
        ended = (u or {}).get('end') if status != 'running' else None
        if status != 'running' and not (ended or (prior or {}).get('ended')):
            continue
        rec = prior or {'machine': machine, 'unit': name, 'state': None, 'emitted': None}
        rec.update(started=started, cmd=(u or {}).get('cmd') or rec.get('cmd') or claims.get(name, {}).get('command', ''),
                   result=(u or {}).get('result', ''), claim=claims.get(name), pid=(u or {}).get('pid'))
        if status != 'running':
            rec['ended'] = ended or rec.get('ended')
        if rec['state'] != status:
            if status == 'running' and started and now - started < 3600 or status != 'running' and now - rec['ended'] < KEEP_S:
                events.append(unit_event({**rec, 'state': status}, now, ann))
            rec['state'] = status
        lines = data['journal'].get(name)
        if lines:
            rec['last_log'] = clip(lines[-1], 200)
        if u and u['cpu_ns'] is not None:
            p = state.setdefault('cpu', {}).get(key)
            rec['cpu_percent'] = round(100 * (u['cpu_ns'] - p[0]) / 1e9 / max(now - p[1], 1), 1) if p and status == 'running' and u['cpu_ns'] >= p[0] else rec.get('cpu_percent')
            state['cpu'][key] = [u['cpu_ns'], now]
        store[key] = rec
        if status == 'failed' and rec.get('ended') and now - rec['ended'] < 3600:
            alerts.append({'level': 'warn', 'machine': machine,
                           'text': f'{name} failed {int((now - rec["ended"]) / 60)} min ago ({rec["result"] or "failed"})'})
    for key in [k for k, r in store.items() if r['machine'] == machine and r['state'] != 'running' and now - (r.get('ended') or 0) > KEEP_S]:
        del store[key]
    for key, rec in store.items():
        if rec['machine'] == machine:
            items.append(process_item(rec, state, data['journal'].get(rec['unit'], []), now, ann))
    return items, events, alerts


def unit_event(rec, now, ann):
    t = {'running': 'started', 'finished': 'finished', 'failed': 'failed'}[rec['state']]
    return {'ts': ms(rec.get('ended') if t != 'started' and rec.get('ended') else rec.get('started') or now), 'type': 'event', 'data': {
        'kind': f'unit_{t}', 'level': 'warn' if t == 'failed' else 'info', 'machine': rec['machine'], 'unit': rec['unit'],
        'title': clip(annotation_for(ann, rec['unit']).get('title') or rec['unit'], 100),
        'text': f'{rec["unit"]} {t} on {rec["machine"]}'}}


def process_item(rec, state, lines, now, ann):
    a, cmd = annotation_for(ann, rec['unit']), rec.get('cmd', '')
    end = now if rec['state'] == 'running' else rec.get('ended') or now
    progress, eta = progress_of({**rec, 'cmd': cmd}, state, lines, now)
    work = last_arg(cmd, '--work')
    claim = rec.get('claim') or {}
    run = run_id(OPS, work) if work else None
    return {'machine': rec['machine'], 'unit': rec['unit'], 'kind': kind_of(rec['unit'], cmd),
            'title': clip(a.get('title') or rec['unit'], 100), 'description': clip(a.get('description') or summarize(cmd), 400),
            'rationale': clip(a.get('rationale', ''), 400), 'investigation': a.get('investigation'),
            'started_ts': ms(rec['started']) if rec.get('started') else None,
            'elapsed_s': round(end - rec['started']) if rec.get('started') else None, 'state': rec['state'],
            'progress': progress, 'eta_s': eta,
            'resources': {'budget_gb': claim.get('budget_gb'), 'used_gb': claim.get('used_gb'), 'peak_gb': claim.get('peak_gb'),
                          'cpu_percent': rec.get('cpu_percent'), 'gpu': bool(claim.get('gpu'))},
            'run': run, 'last_log': rec.get('last_log', ''),
            'last_update_ts': ms(state.get('runs', {}).get(run or '', {}).get('mtime') or now)}


def bench_queue(hunt, active_units):
    """`run ARM ...` / `replay.sh ARM ...` lines of a live queue*.sh whose arm has not started."""
    ps = sh(['ps', '-eo', 'args', '--no-headers'])
    items = []
    for script in sorted(glob.glob(hunt + '/queue*.sh')):
        if script not in ps:
            continue        # script no longer running: its remaining lines will never start
        for line in open(script):
            m = re.match(r'\s*(?:run|\S*replay\.sh)\s+(\w+)\s*(.*)', line)
            if not m or m.group(1) in ('arm',):
                continue
            arm = m.group(1)
            tj = f'{hunt}/{arm}/train.jsonl'
            if (os.path.exists(tj) and os.path.getsize(tj) > 0) or f'hunt-{arm}.service' in active_units:
                continue
            items.append({'title': f'Bench arm {arm}', 'description': clip(f'{os.path.basename(script)}: {m.group(2)}', 400),
                          'rationale': '', 'machine': 'dgx', 'blocked_on': 'previous arm in the queue', 'eta_s': None,
                          'source': 'bench-queue'})
    return items


def build_queue(ann, ledgers, hunt, active_units):
    items = []
    for machine, ledger in ledgers.items():
        for w in (ledger or {}).get('waiting', []):
            w = w if isinstance(w, dict) else {'unit': str(w)}
            name = w.get('unit') or w.get('name') or '?'
            a = annotation_for(ann, name)
            need = w.get('budget_gb')
            items.append({'title': clip(a.get('title') or name, 100), 'description': clip(a.get('description') or summarize(w.get('command', '')), 400),
                          'rationale': clip(a.get('rationale', ''), 400), 'machine': machine,
                          'blocked_on': f'memory ledger: needs {need} GB, headroom {ledger.get("headroom_gb")} GB' if need else 'memory ledger',
                          'eta_s': None, 'source': 'ledger-waiting'})
    for q in ann.get('queue', []):
        if q.get('done_when_exists') and os.path.exists(q['done_when_exists']):
            continue
        items.append({'title': clip(q.get('title'), 100), 'description': clip(q.get('description'), 400),
                      'rationale': clip(q.get('rationale'), 400), 'machine': q.get('machine'),
                      'blocked_on': clip(q.get('blocked_on'), 200), 'eta_s': None, 'source': 'annotations'})
    return items + bench_queue(hunt, active_units)


def build_alerts(machines, ledgers, unit_alerts, errors):
    alerts = list(unit_alerts) + [{'level': 'warn', 'machine': None, 'text': e} for e in errors]
    for machine, host in machines.items():
        for d in host['disks']:
            floor = (FLOORS.get(machine) or {}).get(d['mount'])
            if floor is not None and d['free_gb'] < floor + 20:
                alerts.append({'level': 'crit' if d['free_gb'] < floor else 'warn', 'machine': machine,
                               'text': f'{d["mount"]} has {d["free_gb"]} GB free (floor {floor} GB)'})
        head = (ledgers.get(machine) or {}).get('headroom_gb')
        if head is not None and head < 15:
            alerts.append({'level': 'warn', 'machine': machine, 'text': f'memory ledger headroom {head} GB (< 15 GB)'})
    return alerts


# ---------- output ----------

def shrink(item):
    return {k: clip(v, 200) if isinstance(v, str) else v for k, v in item.items()}


def snapshot_records(type_, items, ts, cap=None):
    """Lines for one snapshot type: items split over parts so every line stays under LIMIT."""
    items = items[:cap] if cap else items
    chunks, cur = [], []
    for it in items:
        if line_size({'ts': ts, 'type': type_, 'data': {'items': cur + [it], 'part': 99, 'parts': 99}}) > LIMIT and cur:
            chunks.append(cur)
            cur = []
        cur.append(it if line_size({'ts': ts, 'type': type_, 'data': {'items': [it]}}) < LIMIT else shrink(it))
    chunks.append(cur)
    return [{'ts': ts, 'type': type_, 'data': {'items': c, 'part': i + 1, 'parts': len(chunks)}} for i, c in enumerate(chunks)]


def append_lines(out, name, recs):
    if recs:
        with open(os.path.join(out, name), 'a') as f:
            f.write(''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n' for r in recs))
            f.flush()


def slow_due(state, key, payload, now, force=False):
    h = hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    last = state.setdefault('slow', {}).get(key)
    if force or not last or last[0] != h or now - last[1] >= SLOW_S:
        state['slow'][key] = [h, now]
        return True
    return False


def machine_record(machine, d, now):
    ledger = d.get('ledger') or {}
    disks = [{**k, 'floor_gb': (FLOORS.get(machine) or {}).get(k['mount'])} for k in d['host']['disks']]
    host = {k: v for k, v in d['host'].items() if k != 'disks'}
    return {'ts': ms(now), 'type': 'machine', 'data': {
        'machine': machine, **host,
        'ledger': {'available_gb': ledger.get('available_gb'), 'headroom_gb': ledger.get('headroom_gb'),
                   'claims': len(ledger.get('claims', {})), 'waiting': len(ledger.get('waiting', []))}, 'disks': disks}}


def cycle(out, ops, ann_path, state, once=False, use_pop=True):
    now, errors = time.time(), []
    ann = read_json(ann_path, {})
    pats = DEFAULT_PATTERNS + [u['match'] for u in ann.get('units', []) if u['match'] != '*']

    def cfg(machine, mounts, ledger):
        store = state.get('units', {})
        return {'mounts': mounts, 'ledger': ledger, 'patterns': pats,
                'known': [k.split(':', 1)[1] for k, r in store.items() if r['machine'] == machine and r['state'] == 'running'],
                'have_log': [k.split(':', 1)[1] for k, r in store.items() if r['machine'] == machine and r.get('last_log')]}
    with ThreadPoolExecutor(2) as ex:
        fl = ex.submit(collect_host, cfg('dgx', list(FLOORS['dgx']), LEDGER))
        fp = ex.submit(collect_pop, cfg('pop', list(FLOORS['pop']), POP_LEDGER)) if use_pop else None
        hosts = {'dgx': fl.result()}
        if fp:
            try:
                hosts['pop'] = fp.result()
            except Exception as e:
                errors.append(f'Pop unreachable: {clip(e, 120)}')
    metrics, events = [], []
    try:
        metrics, events = scan_runs(ops, state, now)
    except Exception as e:
        errors.append(f'exporter: run scan failed: {clip(e, 120)}')
    items, unit_alerts = [], []
    for machine, d in hosts.items():
        i, e, a = update_units(state, machine, d, now, ann)
        items += i
        events += e
        unit_alerts += a
    order = {'running': 0, 'failed': 1, 'finished': 2}
    items.sort(key=lambda x: (order[x['state']], -(x['started_ts'] or 0)))
    ledgers = {m: d.get('ledger') for m, d in hosts.items()}
    active = {i['unit'] for i in items if i['state'] == 'running'}
    ts = ms(now)
    snap = snapshot_records('processes', items, ts, cap=60)
    queue = build_queue(ann, ledgers, ops + '/grpo-v2/hunt', active)
    invs = [{**i, 'updated_ts': ms(os.path.getmtime(ann_path))} for i in ann.get('investigations', [])] if os.path.exists(ann_path) else []
    alerts = build_alerts({m: d['host'] for m, d in hosts.items()}, ledgers, unit_alerts, errors)
    for type_, payload, cap in (('queue', queue, None), ('investigations', invs, None), ('alerts', alerts, None)):
        if slow_due(state, type_, payload, now, once):
            snap += snapshot_records(type_, payload, ts, cap)
    try:
        mt = os.path.getmtime(FALLOW)
        if slow_due(state, 'fallow', mt, now, once):
            snap += snapshot_records('fallow', parse_fallow(open(FALLOW).read()), ts, cap=80)
    except OSError:
        pass
    mach = []
    for m, d in hosts.items():
        if once or now - state.setdefault('machine_ts', {}).get(m, 0) >= 55:
            state.setdefault('machine_ts', {})[m] = now
            mach.append(machine_record(m, d, now))
    append_lines(out, 'metrics.jsonl', metrics)
    append_lines(out, 'machines.jsonl', mach)
    append_lines(out, 'snapshot.jsonl', snap)
    append_lines(out, 'events.jsonl', sorted(events, key=lambda r: r['ts']))
    return {'metrics': len(metrics), 'machines': len(mach), 'snapshot': len(snap), 'events': len(events), 'errors': errors}


def load_state(path):
    return read_json(path, {})


def save_state(path, state):
    with open(path + '.tmp', 'w') as f:
        json.dump(state, f)
    os.replace(path + '.tmp', path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--once', action='store_true')
    ap.add_argument('--interval', type=float, default=30)
    ap.add_argument('--ops', default=OPS)
    ap.add_argument('--out', default=DASH + '/out')
    ap.add_argument('--annotations', default=DASH + '/annotations.json')
    ap.add_argument('--no-pop', action='store_true')
    ap.add_argument('--host-json', help='internal: collect this host and print JSON (used over ssh)')
    args = ap.parse_args(argv)
    if args.host_json:
        print(json.dumps(collect_host(json.loads(args.host_json))))
        return 0
    os.makedirs(args.out, exist_ok=True)
    state_path = args.out + '/state.json'
    while True:
        t0 = time.time()
        state = load_state(state_path)
        try:
            res = cycle(args.out, args.ops, args.annotations, state, args.once, not args.no_pop)
            save_state(state_path, state)
            print(time.strftime('%H:%M:%S'), json.dumps(res), f'{time.time() - t0:.1f}s', flush=True)
        except Exception as e:     # keep looping: one bad cycle must not stop the feed
            print('cycle failed:', repr(e), file=sys.stderr, flush=True)
            if args.once:
                return 1
        if args.once:
            return 0
        time.sleep(max(1.0, args.interval - (time.time() - t0)))


if __name__ == '__main__':
    sys.exit(main())
