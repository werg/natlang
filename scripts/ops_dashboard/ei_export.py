"""Expert-iteration section of the ops snapshot (`snap['ei']`): the async round driver (aei1), its rounds, SFT, collection,
held-out gate, teacher campaign, Colab credits, watchdog alerts and machine history.

Every reader is incremental or cached: logs are read from a byte offset kept in the exporter state, JSON files are parsed again only
when their (mtime, size) changed, finished rounds and collections are not re-read. Stdlib only; imported lazily by export.py (never
shipped to Pop).
"""
import glob
import json
import os
import re
import subprocess
import threading
import time

CONFIG = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json')))
DATA = os.path.expanduser(CONFIG['data_dir'])
REPO = os.path.expanduser(CONFIG.get('repo', '~/natlang'))
OPS = DATA + '/ops'
EI_ROOT = OPS + '/ei'
WATCHDOG_STATE = os.path.expanduser(CONFIG.get('watchdog_state', '~/.local/state/natlang/ops-watchdog.json'))
TUNE = os.path.expanduser(CONFIG.get('program_tune_dir', '~/data/program-tune-20261117'))
PROBE_EVERY_S = 600
LOG_EVENTS_KEEP = 300
STATE_VERSION = 3


# ---------- small helpers ----------

def ms(t):
    return int(t * 1000)


def clip(s, n):
    s = '' if s is None else str(s)
    return s if len(s) <= n else s[:n - 1] + '…'


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def sig(path):
    try:
        s = os.stat(path)
        return [round(s.st_mtime, 1), s.st_size]
    except OSError:
        return None


def read_tail(path, n=65536):
    try:
        with open(path, 'rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - n))
            return f.read().decode('utf-8', 'replace').splitlines()
    except OSError:
        return []


def read_new(path, st):
    """Complete new lines of an append-only file since st['off'] (resets when the file shrank)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        return []
    off = st.get('off', 0)
    if size < off:
        off, st.clear()
    with open(path, 'rb') as f:
        f.seek(off)
        data = f.read()
    end = data.rfind(b'\n') + 1
    st['off'] = off + end
    return [l for l in data[:end].decode('utf-8', 'replace').splitlines() if l.strip()]


def jline(raw):
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else None
    except ValueError:
        return None


def local_ts(s):
    try:
        return time.mktime(time.strptime(str(s)[:19], '%Y-%m-%dT%H:%M:%S'))
    except (ValueError, TypeError):
        return None


def bucket(points, n=70):
    """Mean-bucket a [[x, y], ...] list down to at most n points (keeps the last x)."""
    if len(points) <= n:
        return [[a, round(b, 5)] for a, b in points]
    size = len(points) / n
    out = []
    for i in range(n):
        chunk = points[int(i * size):int((i + 1) * size)] or [points[-1]]
        out.append([chunk[-1][0], round(sum(p[1] for p in chunk) / len(chunk), 5)])
    return out


def pick_run():
    """The driver directory written most recently (ops/ei/*/drive.log): discovered, not configured."""
    cands = [(mtime(p), os.path.dirname(p)) for p in glob.glob(EI_ROOT + '/*/drive.log') if mtime(p)]
    return max(cands)[1] if cands else None


# ---------- unit roles ----------

ROLE_RULES = [   # (regex, role, where, title template using named groups)
    (r'^aei1-drive', 'driver', 'dgx', 'Round driver (aei1)'),
    (r'^aei1-collect-v(?P<v>\d+)-fill', 'fill', 'dgx', 'Student fill of v{v}'),
    (r'^aei1-collect-v(?P<v>\d+)-remote\d*', 'collect', 'colab', 'Collection of v{v} on the Colab A100'),
    (r'^aei1-collect-v(?P<v>\d+)-pop', 'collect', 'pop', 'Collection of v{v} on Pop'),
    (r'^aei1-collect-v(?P<v>\d+)', 'collect', 'dgx', 'Collection of v{v} on the DGX'),
    (r'^aei1-teacher', 'teacher', 'api', 'Teacher campaign (API models)'),
    (r'^aei1-r(?P<r>\d+)-data', 'prepare', 'dgx', 'Round r{r}: data and build'),
    (r'^aei1-r(?P<r>\d+)-sft', 'sft', 'dgx', 'Round r{r}: SFT'),
    (r'^popheld-', 'held', 'pop', 'Held-v3 evaluation (Pop engine)'),
    (r'^(progtune|progdev)-', 'tuning', 'colab', 'Program tuning'),
    (r'^tqual-', 'taskqual', 'dgx', 'Task qualification'),
    (r'^(vibe-student-fill|vsingest|colab-g4[ab]-fill)', 'datafill', 'dgx', 'Vibestudio student data'),
    (r'^colab-g4[ab]-(keeper|tunnel)', 'colab', 'colab', 'Colab session keeper or tunnel'),
    (r'^natlang-(storage-sweep|drive-tier)', 'housekeeping', 'dgx', 'Storage housekeeping'),
    (r'^(ops-peer|ops-dashboard|opsdash|natlang-gb10)', 'plumbing', 'dgx', 'Ops plumbing'),
    (r'^(check-main|test-|pytest)', 'test', 'dgx', 'Tests'),
]
ROLE_LABELS = {'driver': 'Round driver', 'collect': 'Collection', 'fill': 'Student fill', 'teacher': 'Teacher campaign', 'prepare': 'Round prepare',
               'sft': 'SFT', 'held': 'Held evaluation', 'tuning': 'Program tuning', 'taskqual': 'Task qualification', 'datafill': 'Student data fill',
               'colab': 'Colab keeper', 'housekeeping': 'Housekeeping', 'plumbing': 'Plumbing', 'test': 'Tests', 'other': 'Other'}


def unit_role(name):
    short = name[:-8] if name.endswith('.service') else name
    for pat, role, where, title in ROLE_RULES:
        m = re.match(pat, short)
        if m:
            g = {k: str(int(v)) if v and v.isdigit() else v for k, v in m.groupdict().items()}
            return {'role': role, 'where': where, 'title': title.format(**g)}
    return {'role': 'other', 'where': 'dgx', 'title': short}


# ---------- lineage ----------

class Lineage:
    def __init__(self, phase):
        self.by_sha, self.rounds = {}, {}
        self.reference = (phase.get('reference_policy') or phase.get('start_policy') or {})
        for e in phase.get('lineage', []):
            self.by_sha[e['sha256']] = e['version']
            if e.get('round_id'):
                self.rounds[e['round_id']] = e['version']
        ref = self.reference.get('sha256')
        m = re.search(r'/(sft-v\d+)/', self.reference.get('checkpoint', '') or '')
        self.ref_name = m.group(1) if m else 'reference'
        if ref:
            self.by_sha.setdefault(ref, 0)
        self.latest = max(self.by_sha.values()) if self.by_sha else 0

    def ver(self, sha):
        return self.by_sha.get(sha)

    def label(self, sha):
        v = self.ver(sha)
        if v is None:
            return (sha or '')[:8]
        return f'v{v}' + (f' ({self.ref_name})' if v == 0 else '')


# ---------- drive.log: events, held gate failures ----------

def short_error(e):
    e = re.sub(r'\S*/ops/', '', str(e))
    if 'held_eval' in e and 'compare' in e:
        return 'held compare failed (the pair is not comparable yet)'
    if 'incomplete' in e:
        return 'held measurement incomplete'
    if 'rebind' in e.lower() or 'merge of' in e:
        return clip(e, 110)
    return clip(e, 110)


def process_drive_log(st, path, lin, now):
    """Turns new drive.log lines into curated events; returns nothing, updates st (events, gate failures, stamps)."""
    ev = st.setdefault('events', [])
    last = st.setdefault('last', {})
    pub = st.setdefault('published', {})
    size = st.setdefault('sizes', {})

    def emit(ts, kind, level, text, **extra):
        ev.append({'ts': ms(ts), 'kind': kind, 'level': level, 'text': text, **extra})

    for raw in read_new(path, st):
        r = jline(raw)
        if not r:
            continue
        ts = local_ts(r.get('time')) or now
        st['last_ts'] = ts
        stage = r.get('stage')
        ver = lambda p: lin.ver(p)
        vtxt = lambda p: f'v{ver(p)}' if ver(p) is not None else (p or '')[:8]
        if stage == 'pin' and r.get('done'):
            emit(ts, 'driver', 'info', f'Driver (re)started on code pin {os.path.basename(str(r.get("pin")))[4:16]}')
            st['restart_ts'] = ts
        elif stage == 'held reuse':
            emit(ts, 'held', 'info', f'Held result reused for {os.path.basename(str(r.get("out")))[7:19]}')
        elif 'published' in r:
            pub[r['round']] = ts
            emit(ts, 'publish', 'ok', f'Published v{r.get("version")} from {r["round"]}', round=r['round'])
        elif 'phase' in r and 'round' in r:
            key = [r['round'], r['phase'], r.get('version')]
            if last.get('phase') != key:
                last['phase'] = key
                emit(ts, 'round', 'info', f'{r["round"]}: {r["phase"]} phase on v{r.get("version")}', round=r['round'])
        elif 'round_size' in r:
            s = r['round_size']
            if last.get('size') != [r['round'], s.get('decided_at')]:
                last['size'] = [r['round'], s.get('decided_at')]
                size[r['round']] = s
                emit(ts, 'round', 'info', f'{r["round"]} sized: {s.get("hours")} h on {s.get("gpu") or s.get("placement")}, {(s.get("max_target_tokens") or 0) / 1e6:.1f} M target tokens', round=r['round'])
        elif 'collection_started' in r:
            m = re.search(r'-v(\d+)', r['collection_started'])
            emit(ts, 'collect', 'info', f'Collection of v{int(m.group(1)) if m else "?"} started on the DGX')
        elif 'collection_stopped' in r:
            m = re.search(r'-v(\d+)(.*)', r['collection_stopped'])
            if m and not m.group(2).startswith('-fill'):
                emit(ts, 'collect', 'info', f'Collection of v{int(m.group(1))}{m.group(2)} stopped')
        elif 'remote_collection' in r:
            m = re.search(r'-v(\d+)', r['remote_collection'])
            emit(ts, 'collect', 'info', f'{"Pop" if "pop" in str(r.get("host")) else "Colab A100"} joined the collection of v{int(m.group(1)) if m else "?"} ({r.get("slots", "?")} slots)')
        elif 'fill_started' in r:
            m = re.search(r'-v(\d+)', r['fill_started'])
            emit(ts, 'collect', 'info', f'Student fill of v{int(m.group(1)) if m else "?"} started')
        elif 'teacher' in r and 'frontier' in r:
            m = re.search(r'/(r\d+)/', r['frontier'])
            emit(ts, 'teacher', 'info', f'Teacher campaign for {m.group(1) if m else "next round"}: {r.get("frontier_tasks")} frontier tasks{" (resumed)" if r.get("resume") else ""}')
        elif 'takeover' in r:
            emit(ts, 'sft', 'warn', f'{r["round"]}: the DGX took over the SFT (Colab session unfit: {", ".join(r.get("unfit_sessions", []))})', round=r['round'])
        elif 'handback_check' in r:
            h = r['handback_check']
            emit(ts, 'sft', 'ok' if h.get('move') else 'info', f'{r["round"]}: hand-back to {r.get("host")}: {"moved" if h.get("move") else "kept on DGX"}, speedup x{h.get("speedup")}, saves {((h.get("gain_s") or 0) / 3600):.1f} h', round=r['round'])
        elif 'held_failed' in r:
            err = short_error(r.get('error'))
            g = st.setdefault('gate_fail', {})
            if g.get('error') == err:
                g['count'] += 1
                g['last_ts'] = ts
            else:
                st['gate_fail'] = {'error': err, 'count': 1, 'first_ts': ts, 'last_ts': ts, 'candidate': r.get('held_failed')}
                emit(ts, 'held', 'warn', f'Held verdict for {vtxt(r.get("held_failed"))} failed: {err}')
        elif 'held_pair_refused' in r:
            emit(ts, 'held', 'warn', f'Held pair refused: {vtxt(r.get("candidate"))} and {vtxt(r.get("reference"))} were measured with different engine settings; the reference is re-measured')
        elif 'held_v4_queued' in r:
            emit(ts, 'held', 'info', f'held-v4 queued for {vtxt(r["held_v4_queued"])} ({", ".join(r.get("reasons", []))})')
        elif 'rebind_failed' in r:
            key = ['rebind', r.get('policy')]
            if last.get('rebind') != key:
                last['rebind'] = key
                emit(ts, 'sft', 'warn', f'Colab rebind failed for {vtxt(r.get("policy"))}: {clip(short_error(r["rebind_failed"]), 90)}')
        elif 'arguments_reconciled' in r:
            emit(ts, 'driver', 'info', 'Driver arguments reconciled with the recorded state')
        elif 'health' in r:
            text = re.sub(r'^health \S+ \S+: ', '', r['health'])
            emit(ts, 'health', 'warn' if r.get('health_flags') else 'info', f'{r["round"]} collection health: {text}', round=r['round'])
        elif 'round_speed' in r:
            for gpu, v in r['round_speed'].items():
                emit(ts, 'sft', 'info', f'{r["round"]} SFT speed measured: {v.get("chain_tokens_per_s")} tokens/s on {gpu}', round=r['round'])
        elif 'pop_served' in r:
            emit(ts, 'held', 'info', f'Pop now serves {vtxt(r["pop_served"])} (copy {r.get("seconds")} s)')
        elif 'prepare_ahead_failed' in r:
            emit(ts, 'driver', 'warn', f'Prepare-ahead failed: {short_error(r["prepare_ahead_failed"])}')
        elif 'prepare_continues_with' in r:
            emit(ts, 'round', 'info', f'{r["round"]} prepare continues with earlier policies')
    del ev[:-LOG_EVENTS_KEEP]


# ---------- SFT logs ----------

def tail_sft(st, path):
    """Incremental read of a round's SFT train.jsonl into compact series."""
    lines = read_new(path, st)
    st.setdefault('loss', [])
    st.setdefault('val', [])
    st.setdefault('tm', [])
    for raw in lines:
        r = jline(raw)
        if not r:
            continue
        vals = [(k, v) for k, v in r.items() if k.startswith('val_loss') and isinstance(v, (int, float))]
        if vals:
            st['val'].append([r.get('step', 0), vals[0][1]])
        if 'of' not in r or 'step' not in r:
            continue
        st['of'] = r['of']
        st['step'] = r['step']
        st['el'] = r.get('elapsed_s') or st.get('el')
        st['lr'] = r.get('lr')
        st['chain_tokens'] = st.get('chain_tokens', 0) + (r.get('chain_tokens') or 0)
        st.setdefault('recent', []).append([r['step'], r.get('elapsed_s')])
        st['recent'] = st['recent'][-60:]
        if isinstance(r.get('loss'), (int, float)):
            (st['tm'] if r.get('loss_kind') == 'tm' else st['loss']).append([r['step'], r['loss']])
    for k in ('loss', 'val', 'tm'):
        if len(st[k]) > 1500:
            st[k] = [[a, round(b, 5)] for a, b in bucket(st[k], 700)]


def sft_summary(st, launch_at, running, now):
    if not st.get('step'):
        return None
    rec = st.get('recent', [])
    per = None
    if len(rec) > 5 and rec[-1][1] and rec[0][1] and rec[-1][0] > rec[0][0]:
        per = (rec[-1][1] - rec[0][1]) / (rec[-1][0] - rec[0][0])
    step_ts = (launch_at + (st.get('el') or 0)) if launch_at else None
    eta = None
    if running and per and step_ts and st['of'] > st['step']:
        eta = step_ts + (st['of'] - st['step']) * per
    loss = st.get('loss', [])
    out = {'step': st['step'], 'of': st['of'], 'elapsed_s': st.get('el'), 's_per_step': round(per, 2) if per else None,
           'chain_tok_s': round(st['chain_tokens'] / st['el']) if st.get('el') else None, 'lr': st.get('lr'),
           'loss': bucket(loss, 70), 'val': bucket(st.get('val', []), 40), 'tm': bucket(st.get('tm', []), 40),
           'loss_first': round(sum(p[1] for p in loss[:10]) / len(loss[:10]), 4) if loss else None,
           'loss_last': round(sum(p[1] for p in loss[-10:]) / len(loss[-10:]), 4) if loss else None,
           'val_first': st['val'][0][1] if st.get('val') else None, 'val_last': st['val'][-1][1] if st.get('val') else None,
           'step_ts': ms(step_ts) if step_ts else None, 'eta_ts': ms(eta) if eta else None}
    return out


# ---------- rounds ----------

def cached(st, key, path, fn):
    """Parse a JSON file again only when (mtime, size) changed."""
    s = sig(path)
    if s is None:
        return None
    hit = st.get(key)
    if not hit or hit[0] != s:
        hit = st[key] = [s, fn(path)]
    return hit[1]


def build_summary(path):
    b = read_json(path, {})
    tc = b.get('target_cap') or {}
    parts = {}
    mixp = (b.get('mix') or {}).get('parts', {})
    for name, p in (tc.get('parts') or {}).items():
        m = mixp.get(name, {})
        parts[name] = {'kept': p.get('kept_episodes'), 'unseen': p.get('unseen_episodes'), 'repeated': p.get('kept_trained_before'),
                       'available': p.get('episodes'), 'loss_share': m.get('loss_share'), 'forward_share': m.get('forward_share'), 'chains': m.get('chains')}
    for name, m in mixp.items():
        parts.setdefault(name, {'loss_share': m.get('loss_share'), 'forward_share': m.get('forward_share'), 'chains': m.get('chains')})
    return {'chains': b.get('chains'), 'forward_tokens': b.get('forward_tokens'), 'target_tokens': (b.get('mix') or {}).get('weighted_target_tokens'),
            'binding': tc.get('binding'), 'keep_fraction': tc.get('fraction'), 'tasks': (b.get('mix') or {}).get('tasks'), 'episodes': b.get('episodes'),
            'epoch_hours_dgx': b.get('epoch_hours_at_5500_tokens_per_second'), 'core_base_rate': (b.get('core_base_rate') or {}).get('mean'),
            'parts': parts, 'wrapper_share': (b.get('policy') or {}).get('wrapper_share')}


def health_summary(path):
    h = read_json(path, {})
    cur = (h.get('current') or {}).get('overall') or {}
    prev = (h.get('previous') or {}).get('overall') or {}
    return {'pass_rate': cur.get('pass_rate'), 'episodes': cur.get('episodes'), 'tasks': cur.get('tasks'), 'prev_pass_rate': prev.get('pass_rate'),
            'flags': [f.get('flag') + ':' + str(f.get('reason', '')) for f in h.get('flags', [])][:4], 'categories': cur.get('categories')}


def round_json_summary(path):
    r = read_json(path, {})
    return {'committed_at': r.get('committed_at'), 'episodes': len(r.get('episode_ids') or []), 'trained_from_version': r.get('trained_from_version'),
            'collections': [os.path.basename(c) for c in r.get('collections', [])], 'expected_hours': r.get('expected_hours')}


def rounds_info(st, run, lin, log_st, running_units, now):
    out = []
    rids = sorted(os.path.basename(p) for p in glob.glob(run + '/r[0-9][0-9][0-9][0-9]'))[-6:]
    cache = st.setdefault('rcache', {})
    sfts = st.setdefault('sft', {})
    speed = read_json(run + '/round-speed.json', {})
    for rid in rids:
        d = f'{run}/{rid}'
        c = cache.setdefault(rid, {})
        size = read_json(d + '/size.json', None) or (log_st.get('sizes', {}).get(rid))
        stages = {k: mtime(f'{d}/stages/{k}.done') for k in ('frontier', 'data', 'build')}
        launch = read_json(d + '/sft/launch.json', None)
        launch_at = (launch or {}).get('at')
        version = lin.rounds.get(rid)
        published = version is not None
        sst = sfts.setdefault(rid, {})
        if launch and not sst.get('final'):
            tail_sft(sst, d + '/sft/train.jsonl')
            if published:
                sst['final'] = True
        placement = read_json(d + '/sft/colab-placement.json', None)
        phase_active = (read_json(run + '/phase-state.json', {}).get('active_round') or {})
        sft_running = bool(launch) and not published and phase_active.get('round_id') == rid and phase_active.get('phase') == 'sft'
        sft = sft_summary(sst, launch_at, sft_running, now)
        rs = [v for v in speed.values() if isinstance(v, dict) and v.get('round') == rid]
        sft_end = rs[0].get('at') if rs else None
        if sft_end is None and published and mtime(d + '/sft/train.jsonl'):
            sft_end = mtime(d + '/sft/train.jsonl')
        pub_ts = log_st.get('published', {}).get(rid) or (mtime(d + '/policy/published.json') if published else None)
        if published:
            state = 'published'
        elif launch:
            state = 'sft'
        elif stages['build']:
            state = 'ready'
        else:
            state = 'preparing'
        rj = cached(c, 'round_json', d + '/round.json', round_json_summary)
        bs = cached(c, 'build', d + '/mix/build.json', build_summary)
        hs = cached(c, 'health', d + '/health.json', health_summary)
        fr = read_json(d + '/frontier.json', None) if state in ('preparing', 'ready') and 'frontier' not in c else c.get('frontier')
        if fr and 'frontier' not in c:
            c['frontier'] = {'tiers': fr.get('tiers'), 'tasks': len(fr.get('tasks', {}))}
            fr = c['frontier']
        data_prog = None
        if state == 'preparing' and not stages['data']:
            tail = read_tail(d + '/data.log', 2048)
            for raw in reversed(tail):
                j = jline(raw)
                if j and 'task_success' in j:
                    data_prog = j.get('task_success')
                    break
        start_ts = (size or {}).get('decided_at') or mtime(d + '/size.json') or mtime(d + '/stages/frontier.exit') or stages['frontier']
        out.append({
            'id': rid, 'state': state, 'version': version, 'trained_from': (rj or {}).get('trained_from_version'), 'size': size,
            'host': (placement or {}).get('host') or (size or {}).get('host'), 'gpu': (placement or {}).get('gpu') or (size or {}).get('gpu'),
            'stamps': {'start': ms(start_ts) if start_ts else None, 'frontier': ms(stages['frontier']) if stages['frontier'] else None,
                       'data': ms(stages['data']) if stages['data'] else None, 'build': ms(stages['build']) if stages['build'] else None,
                       'sft_start': ms(launch_at) if launch_at else None, 'sft_end': ms(sft_end) if sft_end else None,
                       'published': ms(pub_ts) if pub_ts else None},
            'sft': sft, 'sft_running': sft_running, 'mix': bs, 'frontier': fr, 'health': hs, 'episodes_in': (rj or {}).get('episodes'),
            'collections_in': (rj or {}).get('collections'), 'expected_hours': (rj or {}).get('expected_hours'), 'data_progress': data_prog})
    return out


# ---------- teacher campaigns ----------

def campaign_rows(path):
    rows = []
    try:
        for line in open(path, errors='replace'):
            j = jline(line)
            if j:
                rows.append(j)
    except OSError:
        pass
    return rows


def summarize_campaign(pool, state_cache, now):
    """Per-teacher outcomes of one teacher-rNNNN directory. Finished campaigns are cached by manifest signature."""
    man, dfr = pool + '/manifest.jsonl', pool + '/deferred.jsonl'
    key = pool
    s = [sig(man), sig(dfr)]
    hit = state_cache.get(key)
    if hit and hit[0] == s:
        return hit[1]
    rows, deferred = campaign_rows(man), campaign_rows(dfr)
    per, fam = {}, {}
    mk = lambda t: per.setdefault(t['model_id'], {'id': t['model_id'], 'provider': t.get('provider'), 'reasoning': t.get('reasoning'), 'effort': t.get('reasoning_effort'),
                                                'tasks': 0, 'episodes': 0, 'success': 0, 'outcomes': {}, 'seconds': 0.0, 'tokens': 0, 'deferred': {}, 'provider_failures': 0})
    tasks_done, tasks_won = set(), set()
    for r in rows:
        t = mk(r['teacher'])
        t['tasks'] += 1
        t['seconds'] += r.get('seconds') or 0
        t['provider_failures'] += 1 if r.get('provider_failure') else 0
        tasks_done.add(r['task_id'])
        won = False
        for e in r.get('episodes', []):
            t['episodes'] += 1
            t['outcomes'][e['outcome']] = t['outcomes'].get(e['outcome'], 0) + 1
            t['tokens'] += e.get('completion_tokens') or 0
            ok = e['outcome'] == 'task_success'
            won |= ok
            if ok:
                tasks_won.add(r['task_id'])
            t['success'] += ok
        f = fam.setdefault(r.get('family') or '?', {'family': r.get('family') or '?', 'tasks': 0, 'success': 0})
        f['tasks'] += 1
        f['success'] += won
    for r in deferred:
        t = mk(r['teacher'])
        o = (r.get('episodes') or [{}])[0].get('outcome', 'unknown')
        t['deferred'][o] = t['deferred'].get(o, 0) + 1
    for t in per.values():
        t['success_rate'] = round(t['success'] / t['episodes'], 3) if t['episodes'] else None
        t['seconds_mean'] = round(t['seconds'] / t['tasks'], 1) if t['tasks'] else None
        t['tokens_mean'] = round(t['tokens'] / t['episodes']) if t['episodes'] else None
        del t['seconds'], t['tokens']
    launch = read_json(pool + '/driver-launch.json', {})
    out = {'pool': os.path.basename(pool), 'tasks_done': len(tasks_done), 'tasks_won': len(tasks_won), 'success': sum(t['success'] for t in per.values()), 'deferred': len(deferred),
           'eligible': launch.get('tasks'), 'launched_at': ms(launch['at']) if launch.get('at') else None,
           'last_activity_ts': ms(max([mtime(f) or 0 for f in (man, dfr)] or [0])),
           'teachers': sorted(per.values(), key=lambda t: -t['success']),
           'families': sorted(fam.values(), key=lambda f: (-f['success'], -f['tasks']))[:12], 'families_total': len(fam),
           'families_with_success': sum(1 for f in fam.values() if f['success'])}
    state_cache[key] = [s, out]
    return out


def teacher_section(st, run, now):
    pools = sorted(glob.glob(run + '/teacher-r[0-9][0-9][0-9][0-9]'))
    if not pools:
        return None
    cache = st.setdefault('tcache', {})
    campaigns = []
    for p in pools[-6:]:
        s = summarize_campaign(p, cache, now)
        campaigns.append({k: s[k] for k in ('pool', 'tasks_done', 'tasks_won', 'success', 'eligible', 'deferred', 'last_activity_ts')})
    cur = summarize_campaign(pools[-1], cache, now)
    cur = dict(cur)
    hist = st.setdefault('thist', {})
    h = hist.setdefault(cur['pool'], [])
    if not h or h[-1][1:] != [cur['tasks_done'], cur['success']]:
        h.append([ms(now), cur['tasks_done'], cur['success']])
        del h[:-300]
    hour = [x for x in h if x[0] >= ms(now) - 3600_000]
    cur['success_last_hour'] = (h[-1][2] - hour[0][2]) if len(hour) > 1 else 0
    cur['tasks_last_hour'] = (h[-1][1] - hour[0][1]) if len(hour) > 1 else 0
    cur['history'] = h[-60:]
    cur['pool_rel'] = None
    cur['campaigns'] = campaigns
    cur['claimed'] = len(glob.glob(pools[-1] + '/claims/*'))
    return cur


# ---------- collection ----------

def collection_dirs(run, lin):
    """vNNNN-<sha12>[-pop|-remoteN] directories of the newest policy versions."""
    found = []
    for p in glob.glob(run + '/collect/v[0-9][0-9][0-9][0-9]-*'):
        name = os.path.basename(p)
        m = re.match(r'v(\d+)-([0-9a-f]{12})(?:-(pop|remote\d*))?$', name)
        if m:
            found.append((int(m.group(1)), m.group(3) or '', name, p))
    versions = sorted({f[0] for f in found})[-3:]
    return [f for f in found if f[0] in versions]


HOST_OF = {'': 'dgx', 'pop': 'pop'}


def scan_groups(st, path):
    """Per-group [episodes, successes, finished_ts] from groups/*/score.json, each read once."""
    gdir = path + '/groups'
    gsig = mtime(gdir)
    if gsig is None:
        return st
    if st.get('gsig') == gsig and st.get('done'):
        return st
    seen = st.setdefault('g', {})
    try:
        names = os.listdir(gdir)
    except OSError:
        return st
    for n in names:
        if n in seen:
            continue
        sp = f'{gdir}/{n}/score.json'
        t = mtime(sp)
        if t is None:
            continue
        s = read_json(sp, {})
        seen[n] = [int(s.get('rollouts') or 0), int(s.get('scored') or 0), int(s.get('successes') or 0), int(t)]
    st['gsig'] = gsig
    return st


def progress_tail(path):
    last = None
    for raw in reversed(read_tail(path, 16384)):
        j = jline(raw)
        if j and j.get('event') == 'progress':
            last = j
            break
    return last


def actor_tok_s(path):
    for l in reversed(read_tail(path, 49152)):
        m = re.search(r'#running-req: (\d+).*gen throughput \(token/s\): ([\d.]+)', l)
        if m:
            return {'running': int(m.group(1)), 'tok_s': float(m.group(2))}
    return None


def collection_section(st, run, lin, running_units, now):
    cs = st.setdefault('coll', {})
    hourly = {}
    policies = {}
    actors = []
    cutoff = now - 24 * 3600
    for version, suffix, name, path in sorted(collection_dirs(run, lin)):
        host = 'pop' if suffix == 'pop' else 'colab' if suffix.startswith('remote') else 'dgx'
        c = scan_groups(cs.setdefault(name, {}), path)
        unit = f'aei1-collect-v{version:04d}' + (f'-{suffix}' if suffix else '') + '.service'
        running = unit in running_units
        c['done'] = (not running) and bool(c.get('g'))
        groups = c.get('g', {})
        eps = sum(g[0] for g in groups.values())
        scored = sum(g[1] for g in groups.values())
        succ = sum(g[2] for g in groups.values())
        recent = [g for g in groups.values() if g[3] >= now - 3600]
        hour_eps = sum(g[0] for g in recent)
        hour_scored = sum(g[1] for g in recent)
        hour_succ = sum(g[2] for g in recent)
        for g in groups.values():
            if g[3] >= cutoff:
                b = hourly.setdefault(int(g[3] // 3600) * 3600, {'dgx': 0, 'colab': 0, 'pop': 0})
                b[host] += g[0]
        pol = policies.setdefault(version, {'version': version, 'sha12': name.split('-')[1], 'episodes': 0, 'scored': 0, 'successes': 0, 'groups': 0, 'hosts': {}, 'last_ts': None})
        pol['episodes'] += eps
        pol['scored'] += scored
        pol['successes'] += succ
        pol['groups'] += len(groups)
        last = max([g[3] for g in groups.values()] or [0])
        pol['last_ts'] = max(pol['last_ts'] or 0, last) or None
        h = pol['hosts'].setdefault(host, {'episodes': 0, 'scored': 0, 'successes': 0, 'groups': 0})
        h['episodes'] += eps
        h['scored'] += scored
        h['successes'] += succ
        h['groups'] += len(groups)
        if running:
            prog = progress_tail(path + '/collect.log')
            tok = actor_tok_s(path + '/actor.log') if host == 'dgx' else None
            actors.append({'host': host, 'unit': unit, 'version': version, 'running': True, 'groups': len(groups), 'episodes': eps, 'rate_h': hour_eps,
                           'pass_rate_h': round(hour_succ / hour_scored, 3) if hour_scored else None, 'pass_rate': round(succ / scored, 3) if scored else None, 'scored': scored,
                           'in_flight': (prog or {}).get('running'), 'tasks_left': (prog or {}).get('tasks_left'), 'progress_ts': ms(prog['time']) if prog and prog.get('time') else None,
                           'last_group_ts': ms(last) if last else None, 'actor': tok})
    pols = []
    for v in sorted(policies):
        p = policies[v]
        p['pass_rate'] = round(p['successes'] / p['scored'], 3) if p['scored'] else None
        for h in p['hosts'].values():
            h['pass_rate'] = round(h['successes'] / h['scored'], 3) if h['scored'] else None
        pols.append(p)
    fill = None
    for version, suffix, name, path in collection_dirs(run, lin):
        if suffix == '' and f'aei1-collect-v{version:04d}-fill.service' in running_units:
            fj = None
            for raw in read_tail(path + '/fill.log', 2048):
                fj = jline(raw) or fj
            fill = {'version': version, 'info': fj}
    hrs = [{'t': ms(t), **hourly[t]} for t in sorted(hourly)]
    return {'policies': pols[-3:], 'hourly': hrs[-25:], 'actors': actors, 'fill': fill}


# ---------- held-out ----------

def held_record(path):
    d = read_json(path, None)
    if not d or 'success_rate' not in d:
        return None
    # held_eval.compare's rule: an episode lost to infrastructure (``infra_lost``; a result written before the field:
    # an infrastructure_error) is left out, and of both sides of a pair (``paired``); any other unscored one fails
    # (outcomes and lost are keyed '<task>#<sample>': the record is cached in the JSON exporter state)
    rates, outcomes, lost = {}, {}, []
    for s in d.get('samples', []):
        if not s.get('ran', True):
            continue
        key = f"{s['task_id']}#{s.get('sample')}"
        if (bool(s['infra_lost']) if 'infra_lost' in s else s.get('outcome') == 'infrastructure_error'):
            lost.append(key)
            continue
        outcomes[key] = s.get('outcome') == 'task_success'
        t = rates.setdefault(s['task_id'], [0, 0])
        t[1] += 1
        t[0] += 1 if s.get('outcome') == 'task_success' else 0
    per = [a / b for a, b in rates.values() if b]
    n = len(per)
    mean = sum(per) / n if n else None
    se = (sum((x - mean) ** 2 for x in per) / max(n - 1, 1)) ** 0.5 / n ** 0.5 if n > 1 else None
    oc = d.get('outcome_counts') or {}
    return {'sha': d.get('policy_id'), 'series': (d.get('series') or {}).get('key'), 'rate': d.get('success_rate'), 'successes': d.get('successes'),
            'scored': d.get('scored'), 'ran': d.get('ran'), 'tasks': d.get('tasks'), 'spt': d.get('samples_per_task'), 'finished': ms(d['finished_at']) if d.get('finished_at') else None,
            'ran_out_rate': d.get('ran_out_rate'), 'loop_rate': d.get('loop_rate'), 'neuralese_use': d.get('neuralese_use_rate'),
            'graded': d.get('mean_graded_reward'), 'infra': len(lost), 'counted': len(outcomes), 'timeouts': d.get('timeouts'),
            'engine': ((d.get('engine') or {}).get('dtype'), (d.get('engine') or {}).get('kv_cache_dtype')),
            'task_mean': round(mean, 4) if mean is not None else None, 'task_se': round(se, 4) if se is not None else None, 'task_n': n,
            'rates': {k: v for k, v in rates.items()}, 'outcomes': outcomes, 'lost': lost}


def _pair_rates(rec, drop):
    rates = {}
    for key, ok in rec['outcomes'].items():
        if key in drop:
            continue
        t = rates.setdefault(key.rpartition('#')[0], [0, 0])
        t[1] += 1
        t[0] += ok
    return rates


def paired(a, b):
    """Task-level paired difference b - a over the tasks both measured, an episode lost to infrastructure on either side
    left out of both (normal approximation; the driver's verdict uses held_eval compare's permutation test)."""
    if 'outcomes' in a and 'outcomes' in b:
        drop = set(a['lost']) | set(b['lost'])
        ra, rb = _pair_rates(a, drop), _pair_rates(b, drop)
    else:
        ra, rb = a['rates'], b['rates']
    common = [k for k in ra if k in rb and ra[k][1] and rb[k][1]]
    if len(common) < 8:
        return None
    d = [rb[k][0] / rb[k][1] - ra[k][0] / ra[k][1] for k in common]
    m = sum(d) / len(d)
    se = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5 / len(d) ** 0.5
    return {'delta_pts': round(100 * m, 2), 'ci_pts': [round(100 * (m - 1.96 * se), 2), round(100 * (m + 1.96 * se), 2)], 'tasks': len(d),
            'up': sum(1 for x in d if x > 0), 'down': sum(1 for x in d if x < 0)}


def run_kind(path):
    parts = path.split('/')
    try:
        i = parts.index('held')
        return parts[i - 1], parts[i + 1]
    except (ValueError, IndexError):
        return os.path.basename(os.path.dirname(path)), ''



def held_section(st, run, lin, log_st, phase, now):
    hc = st.setdefault('held', {})
    paths = set(glob.glob(EI_ROOT + '/*/held/*/result.json')) | set(glob.glob(run + '/held/*/result.json'))
    for p in paths:
        s = sig(p)
        if hc.get(p, [None])[0] != s:
            hc[p] = [s, held_record(p)]
    cand_name = {}
    for p in paths:
        rec = (hc.get(p) or [None, None])[1]
        run_name, kind = run_kind(p)
        if rec and kind == 'candidate' and not run_name.startswith('policy-'):
            cand_name[rec['sha']] = run_name
    seen, results = set(), []
    for p, (s, rec) in sorted(hc.items(), key=lambda kv: (kv[1][1] or {}).get('finished') or 0):
        if not rec or p not in paths:
            continue
        k = (rec['sha'], rec['series'], rec['finished'])
        if k in seen:
            continue
        seen.add(k)
        run_name, kind = run_kind(p)
        if lin.ver(rec['sha']) is not None:
            label = lin.label(rec['sha'])
        elif rec['sha'] in cand_name:
            label = cand_name[rec['sha']]
        else:
            label = 'base model' if kind == 'base' else f'{run_name} {kind}'.strip()
        results.append({**rec, 'label': label, 'version': lin.ver(rec['sha']), 'unpaired': 'unpaired' in p})
    # drop results of a policy superseded in the same series (keep every distinct (sha, series))
    pairs = []
    by_series = {}
    for r in results:
        by_series.setdefault(r['series'], []).append(r)
    for series, rs in by_series.items():
        rs = sorted(rs, key=lambda r: r['finished'] or 0)
        for a, b in zip(rs, rs[1:]):
            pr = paired(a, b)
            if pr and a['sha'] != b['sha']:
                pairs.append({'series': series, 'a': a['label'], 'b': b['label'], **pr})
    # in-progress measurements (Pop's held log)
    inprog = []
    for d in glob.glob(run + '/held/policy-*'):
        if os.path.exists(d + '/result.json') or 'unpaired' in d:
            continue
        log = d + '/pop-held.log'
        t = mtime(log)
        if t is None or now - t > 6 * 3600:
            continue
        prog = None
        for raw in reversed(read_tail(log, 4096)):
            j = jline(raw)
            if j and 'held_eval_progress' in j:
                prog = j
                break
        sha12 = os.path.basename(d)[7:19]
        sha = next((s for s in lin.by_sha if s.startswith(sha12)), sha12)
        if prog:
            done, of, minutes = prog['held_eval_progress'], prog['of'], prog.get('minutes')
            inprog.append({'label': lin.label(sha), 'done': done, 'of': of, 'minutes': minutes, 'updated': ms(t),
                           'eta_s': round((of - done) * minutes * 60 / done) if done and minutes else None})
    meas = phase.get('measuring') or {}
    cand, ref = (meas.get('candidate') or {}), (meas.get('reference') or {})
    def latest(sha, series=None):
        rs = [r for r in results if r['sha'] == sha and not r['unpaired'] and (series is None or r['series'] == series)]
        return rs[-1] if rs else None
    gate = None
    if cand:
        c = latest(cand.get('sha256'))
        r_same = latest(ref.get('sha256'), c['series'] if c else None)
        gate = {'candidate': lin.label(cand.get('sha256')), 'candidate_version': cand.get('version'), 'reference': lin.label(ref.get('sha256')),
                'candidate_result': {k: c.get(k) for k in ('rate', 'scored', 'counted', 'ran', 'series', 'infra', 'finished')} if c else None,
                'reference_same_series': bool(r_same), 'pair': paired(r_same, c) if c and r_same else None,
                'failing': log_st.get('gate_fail'), 'every': phase.get('held_every'), 'since_held': phase.get('since_held'),
                'neutral': phase.get('consecutive_neutral')}
    q = read_json(run + '/held-v4/queue.json', {}) or {}
    queue = [{'label': lin.label(e.get('sha256')), 'state': e.get('state'), 'reasons': e.get('reasons'), 'queued_at': ms(e['queued_at']) if e.get('queued_at') else None}
             for e in q.get('entries', [])]
    out = [{k: v for k, v in r.items() if k not in ('rates', 'outcomes', 'lost')} for r in results]
    return {'results': out[-16:], 'pairs': pairs[-8:], 'in_progress': inprog, 'gate': gate, 'queue': queue}


# ---------- Colab credits ----------

_PROBE = {'busy': False, 'result': None, 'at': 0}


def _probe():
    res = {'at': time.time()}
    try:
        r = subprocess.run(['colab', '--auth=adc', 'usage'], capture_output=True, text=True, timeout=60)
        bal = re.search(r'balance:\s*([\d.]+)', r.stdout)
        rate = re.search(r'rate:\s*([\d.]+)', r.stdout)
        act = re.search(r'assignments:\s*(\d+)', r.stdout)
        if bal and rate:
            res.update(balance=float(bal.group(1)), rate=float(rate.group(1)), assignments=int(act.group(1)) if act else None)
    except Exception as e:
        res['error'] = clip(e, 100)
    try:
        r = subprocess.run(['colab', '--auth=adc', 'sessions'], capture_output=True, text=True, timeout=60)
        sess = []
        for l in r.stdout.splitlines():
            m = re.match(r'^\[(?P<name>[^\]]*)\]\s+(?P<id>\S+)\s*\|\s*Hardware:\s*(?P<hw>[^|]+?)\s*\|', l)
            if m:
                sess.append({'name': m.group('name'), 'hardware': m.group('hw')})
        res['sessions'] = sess
    except Exception:
        pass
    _PROBE['result'], _PROBE['busy'] = res, False


def colab_section(st, run, now):
    c = st.setdefault('colab', {'hist': []})
    if not _PROBE['busy'] and now - max(_PROBE['at'], c.get('at', 0)) > PROBE_EVERY_S:
        _PROBE['busy'], _PROBE['at'] = True, now
        threading.Thread(target=_probe, daemon=True).start()
    r = _PROBE.get('result')
    if r and r.get('at', 0) > c.get('at', 0):
        if r.get('balance') is not None:
            c.update(balance=r['balance'], rate=r['rate'], assignments=r.get('assignments'), at=r['at'])
            c['hist'].append([ms(r['at']), r['balance'], r['rate']])
            del c['hist'][:-400]
        if r.get('sessions') is not None:
            c['sessions'] = r['sessions']
        c['probe_error'] = r.get('error')
    cost = read_json(run + '/cycle-cost.json', {}) or {}
    sessions = [{'host': s.get('host'), 'cu': s.get('cu_spent'), 'hours': round((s.get('seconds') or 0) / 3600, 2), 'cu_per_hour': s.get('cu_per_hour'),
                 'started': ms(s['started']) if s.get('started') else None, 'stopped': ms(s['stopped']) if s.get('stopped') else None}
                for s in cost.get('sessions', [])]
    out = {k: c.get(k) for k in ('balance', 'rate', 'assignments', 'sessions', 'probe_error')}
    out['at'] = ms(c['at']) if c.get('at') else None
    out['hist'] = c.get('hist', [])[-200:]
    if out['balance'] is not None and out['rate']:
        out['runway_h'] = round(out['balance'] / out['rate'], 1)
        out['empty_ts'] = ms(now + out['balance'] / out['rate'] * 3600)
    out['spent_cu'] = cost.get('session_cu_spent')
    out['spent_sessions'] = sessions[-6:]
    return out


# ---------- watchdog, machine history, disks ----------

def watchdog_section(now):
    ws = read_json(WATCHDOG_STATE, {}) or {}
    rows = {}
    for raw in read_tail(OPS + '/alerts.jsonl', 200000):
        j = jline(raw)
        if j and j.get('key'):
            rows[j['key']] = j
    active = []
    for key, meta in (ws.get('alerts') or {}).items():
        j = rows.get(key)
        if not j or key.startswith('health:log:'):
            continue
        age = now - meta.get('last', 0)
        if meta.get('once') and age > 6 * 3600:
            continue
        active.append({'key': key, 'check': j.get('check'), 'severity': j.get('severity'), 'message': clip(j.get('message'), 220), 'ts': ms(j.get('ts') or meta['last']),
                       'last_ts': ms(meta['last']), 'repeat': meta.get('repeat', 0), 'once': bool(meta.get('once'))})
    active.sort(key=lambda a: (a['severity'] != 'critical', -a['last_ts']))
    gpu = {}
    for name, g in (ws.get('gpu') or {}).items():
        gpu[name] = {'util': g.get('util'), 'idle_s': g.get('idle_s'), 'error': clip(g.get('error'), 80), 'checked': ms(g['checked']) if g.get('checked') else None}
    return {'alerts': active, 'gpu': gpu, 'tunnels': ws.get('tunnels'), 'checked': ms(ws['last_run']) if ws.get('last_run') else None}


def history_section(st, machines, gpu, now):
    h = st.setdefault('mhist', {})
    due = now - h.get('_ts', 0) >= 590
    if due:
        h['_ts'] = now
        for name, m in machines.items():
            row = [ms(now), m.get('mem_used_gb'), m.get('gpu_util_percent'), m.get('cpu_percent'), (m.get('ledger') or {}).get('headroom_gb')]
            h.setdefault(name, []).append(row)
            del h[name][:-144]
            for d in m.get('disks', []):
                h.setdefault(f'disk:{name}:{d["mount"]}', []).append([ms(now), d['free_gb']])
                del h[f'disk:{name}:{d["mount"]}'][:-144]
        for name in ('colab-g4a', 'colab-g4b'):
            g = (gpu or {}).get(name)
            if g and g.get('util') is not None:
                h.setdefault(name, []).append([ms(now), None, g['util'], None, None])
                del h[name][:-144]
    out = {k: v for k, v in h.items() if not k.startswith(('_', 'disk:'))}
    disks = {}
    for k, v in h.items():
        if k.startswith('disk:'):
            recent = [p for p in v if p[0] >= ms(now) - 3 * 3600_000]
            rate = None
            if len(recent) >= 3 and recent[-1][0] > recent[0][0]:
                rate = round((recent[0][1] - recent[-1][1]) / ((recent[-1][0] - recent[0][0]) / 3600_000), 1)
            disks[k[5:]] = {'rate_gb_h': rate}
    return out, disks


# ---------- program tuning, registry, open items, decisions ----------

def tuning_section(now, units):
    mine = [u for u in units if u['role'] == 'tuning']
    lines = [l for l in read_tail(TUNE + '/logs/colab-seq.log', 4096) if l.strip()]
    return {'running': [{'unit': u['unit'], 'state': u['state'], 'title': u['title'], 'machine': u['machine'], 'elapsed_s': u['elapsed_s']} for u in mine if u['state'] == 'running'],
            'failed_recent': sum(1 for u in mine if u['state'] == 'failed'), 'last_log': clip(lines[-1], 220) if lines else None,
            'last_log_ts': ms(mtime(TUNE + '/logs/colab-seq.log')) if mtime(TUNE + '/logs/colab-seq.log') else None}


def registry_section(st, now):
    if now - st.get('registry_ts', 0) < 300 and 'registry' in st:
        return st['registry']
    items = []
    for p in glob.glob(REPO + '/training/corpora/*.json'):
        t = mtime(p)
        if t:
            items.append((t, p))
    items.sort(reverse=True)
    rows = []
    for t, p in items[:8]:
        d = read_json(p, {})
        rows.append({'id': clip(d.get('id') or os.path.basename(p), 90), 'kind': d.get('kind'), 'rows': d.get('rows'), 'ts': ms(t), 'owner': d.get('owner')})
    st['registry'], st['registry_ts'] = rows, now
    return rows


def open_items_section(st):
    p = REPO + '/plans/semantic-fp/OPEN_ITEMS.md'
    s = [sig(p), sig(OPS + '/teacher-free-models/candidates.jsonl')]
    if st.get('open_items_sig') == s and 'open_items' in st:
        return st['open_items']
    rows, in_flight = [], False
    try:
        for line in open(p):
            h = re.match(r'#{2,3}\s+(.*)', line)
            if h:
                in_flight = h.group(1).strip().lower().startswith('in flight')
            elif in_flight and line.startswith('|') and not re.match(r'\|\s*(Item|-)', line):
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                if len(cells) >= 2:
                    rows.append({'item': clip(re.sub(r'`', '', cells[0]), 260), 'owner': clip(' | '.join(cells[1:]), 120)})
    except OSError:
        pass
    rows += free_teacher_candidates()
    st['open_items'], st['open_items_sig'] = rows, s
    return rows


def free_teacher_candidates():
    """Free models first seen by the daily check (scripts/teacher_free_models.py) and not yet qualified: open items."""
    rows = []
    try:
        for line in open(OPS + '/teacher-free-models/candidates.jsonl'):
            c = json.loads(line) if line.strip() else {}
            if c.get('status') == 'new':
                rows.append({'item': clip(f"Free teacher candidate {c['model']} on {c['provider']} (first seen "
                                          f"{str(c.get('first_seen', ''))[:10]}): qualify on the teacher set", 260),
                             'owner': 'teacher-providers (scripts/teacher_free_models.py mark)'})
    except (OSError, ValueError, KeyError):
        pass
    return rows


def decisions_section(st):
    p = REPO + '/plans/neuralese/DECISIONS.md'
    s = sig(p)
    if st.get('decisions_sig') == s and 'decisions' in st:
        return st['decisions']
    heads = []
    for l in read_tail(p, 400000):
        m = re.match(r'##\s+(2026-\d\d-\d\d \d\d:\d\d)\s+(.*)', l)
        if m:
            heads.append({'when': m.group(1), 'title': clip(m.group(2), 200)})
    st['decisions'], st['decisions_sig'] = heads[-6:][::-1], s
    return st['decisions']


# ---------- the section ----------

def build(state, machines, units, now, run=None):
    """The `ei` dict of the snapshot. `machines` maps name -> host_metrics dict (with ledger), `units` the exporter's process items."""
    run = run or pick_run()
    if not run:
        return None
    st = state.setdefault('ei', {})
    if st.get('v') != STATE_VERSION:        # a reader's cached shape changed: rebuild the caches once
        st.clear()
        st['v'] = STATE_VERSION
    phase = read_json(run + '/phase-state.json', {}) or {}
    lin = Lineage(phase)
    log_st = st.setdefault('log', {})
    errors = []
    sections = {}

    def part(name, fn):
        try:
            sections[name] = fn()
        except Exception as e:      # one broken reader never blanks the section
            errors.append(f'{name}: {clip(e, 100)}')
            sections[name] = None
    part('log', lambda: process_drive_log(log_st, run + '/drive.log', lin, now))
    running_units = {u['unit'] for u in units if u['state'] == 'running'}
    part('rounds', lambda: rounds_info(st, run, lin, log_st, running_units, now))
    part('collection', lambda: collection_section(st, run, lin, running_units, now))
    part('held', lambda: held_section(st, run, lin, log_st, phase, now))
    part('teacher', lambda: teacher_section(st, run, now))
    part('colab', lambda: colab_section(st, run, now))
    part('watchdog', lambda: watchdog_section(now))
    wd = sections.get('watchdog') or {}
    hist, disks = {}, {}
    try:
        hist, disks = history_section(st, machines, wd.get('gpu'), now)
    except Exception as e:
        errors.append(f'history: {clip(e, 100)}')
    part('tuning', lambda: tuning_section(now, units))
    part('registry', lambda: registry_section(st, now))
    part('open_items', lambda: open_items_section(st))
    part('decisions', lambda: decisions_section(st))
    active = phase.get('active_round') or {}
    drive = next((u for u in units if u['role'] == 'driver' and u['state'] == 'running'), None)
    snap = {
        'schema': 1, 'run': os.path.basename(run), 'ts': ms(now),
        'driver': {'running': bool(drive), 'unit': drive['unit'] if drive else None, 'started_ts': drive['started_ts'] if drive else None,
                   'last_log_ts': ms(log_st['last_ts']) if log_st.get('last_ts') else None, 'restart_ts': ms(log_st['restart_ts']) if log_st.get('restart_ts') else None},
        'phase': {'round': active.get('round_id'), 'phase': active.get('phase') or phase.get('phase'), 'version_counter': phase.get('version_counter'),
                  'latest_version': lin.latest, 'current_policy': lin.label(((phase.get('lineage') or [{}])[-1]).get('sha256')),
                  'held_every': phase.get('held_every'), 'since_held': phase.get('since_held')},
        'lineage': [{'version': e['version'], 'round': e.get('round_id'), 'sha12': e['sha256'][:12]} for e in phase.get('lineage', [])],
        'reference': {'label': lin.label(lin.reference.get('sha256')), 'name': lin.ref_name},
        'rounds': sections.get('rounds') or [], 'collection': sections.get('collection'), 'held': sections.get('held'), 'teacher': sections.get('teacher'),
        'colab': sections.get('colab'), 'watchdog': sections.get('watchdog'), 'hist': hist, 'disks': disks, 'tuning': sections.get('tuning'),
        'registry': sections.get('registry'), 'open_items': sections.get('open_items'), 'decisions': sections.get('decisions'),
        'events': log_st.get('events', [])[-60:], 'errors': errors}
    return snap


def derived_alerts(ei, floors, machines, now):
    """Alerts the page lists beside the exporter's own (unit failures, disk floors): driver, SFT sync, credits, disk fall, held gate, watchdog."""
    out = []
    add = lambda i, level, text, machine=None: out.append({'id': i, 'level': level, 'machine': machine, 'text': text})
    d = ei.get('driver') or {}
    if not d.get('running'):
        add('ei:driver', 'crit', 'The round driver (aei1-drive-*) is not running: no rounds will advance.', 'dgx')
    for r in ei.get('rounds', []):
        s = r.get('sft')
        if r.get('sft_running') and s and s.get('step_ts') and now * 1000 - s['step_ts'] > 70 * 60_000:
            add(f'ei:syncstale:{r["id"]}', 'warn', f'{r["id"]} SFT log is {int((now * 1000 - s["step_ts"]) / 60000)} min behind (the Colab learner state syncs about every 20 min).')
    c = ei.get('colab') or {}
    if c.get('runway_h') is not None and c['runway_h'] < 12:
        add('ei:credits', 'crit' if c['runway_h'] < 4 else 'warn', f'Colab credits: {c["balance"]:.0f} units last {c["runway_h"]:.0f} h at {c["rate"]}/h. Refill soon.')
    for k, v in (ei.get('disks') or {}).items():
        machine, mount = k.split(':', 1)
        floor = ((floors or {}).get(machine) or {}).get(mount)
        free = next((x['free_gb'] for x in (machines.get(machine) or {}).get('disks', []) if x['mount'] == mount), None)
        if floor is not None and free is not None and v.get('rate_gb_h') and v['rate_gb_h'] > 1:
            hrs = (free - floor) / v['rate_gb_h']
            if 0 < hrs < 8:
                add(f'ei:diskfall:{k}', 'crit' if hrs < 3 else 'warn', f'{mount} on {machine} is filling {v["rate_gb_h"]} GB/h: the {floor} GB floor in {hrs:.1f} h.', machine)
    for a in (ei.get('collection') or {}).get('actors', []):
        if a.get('last_group_ts') and now * 1000 - a['last_group_ts'] > 40 * 60_000:
            add(f'ei:collect-stall:{a["unit"]}', 'warn', f'The {a["host"]} collector of v{a["version"]} has finished no group for {int((now * 1000 - a["last_group_ts"]) / 60000)} min.', a['host'] if a['host'] in ('dgx', 'pop') else None)
    t = ei.get('teacher') or {}
    if t.get('eligible') and t['tasks_done'] < t['eligible'] and t.get('last_activity_ts') and now * 1000 - t['last_activity_ts'] > 2 * 3600_000:
        add('ei:teacher-stall', 'warn', f'The teacher campaign {t["pool"]} has made no progress for {int((now * 1000 - t["last_activity_ts"]) / 3600_000)} h ({t["tasks_done"]}/{t["eligible"]} tasks).')
    g = (ei.get('held') or {}).get('gate') or {}
    f = g.get('failing')
    if f and f.get('count', 0) >= 6:
        add('ei:gate', 'warn', f'Held verdict for {g.get("candidate")} has failed {f["count"]} times since {time.strftime("%H:%M", time.localtime(f["first_ts"]))}: {f["error"]}.')
    wd = (ei.get('watchdog') or {}).get('alerts', [])
    tunnels = {a['key'].split(':', 2)[2] for a in wd if a['key'].startswith('colab:tunnel:')}
    for a in wd:
        if a['check'] in ('disk', 'ledger', 'health') or a['once'] or a['key'] == 'colab:balance':
            continue
        if a['key'].startswith('units:storm:') and a['key'].split(':', 2)[2] in tunnels:
            continue        # the tunnel alert of the same unit says it better
        add('wd:' + a['key'], 'crit' if a['severity'] == 'critical' else 'warn', a['message'])
    return out
