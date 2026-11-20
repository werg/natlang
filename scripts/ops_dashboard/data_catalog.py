"""Training-data catalog (data.catalog / data.dataset queries): every dataset we have, wherever it lives.

Datasets come from providers, each reading a source of truth rather than a hand list where one exists:
  code       the constants the trainers actually use: grpo.py POOLS / MEMETIC_TASKS / HELD / ERRATA / SD_TEACHER_SOURCES
             (parsed with ast, never imported), so a new supply round appears as soon as it lands in POOLS
  runs       GRPO runs (ops/grpo-*/run*/train.jsonl), their held evals, SFT runs (ops/sft-v*/run*), teacher-gen campaigns
  registry   training/neuralese_corpora.json (3k+ entries) as one browsable dataset, plus registry facts for every path
  sources    catalog_sources.json: the explicit hand layer for what no registry knows (to be replaced by a producer-written
             registry; see plans/neuralese/TRAINING_DATA_DISCOVERABILITY_20261120.md)
Each dataset: {id, title, group, kind, reader, path, parts, protected, size_bytes, rows, provenance, consumers, relations}.
Expensive facts (line counts, sizes) are cached under the dashboard cache and refreshed when files change.
"""
import ast
import glob
import json
import os
import re
import time

from data_service import handler, QueryError, DATA, REPO, DASH
from data_items import OPS, CACHE, read_json, jsonl, sig, line_index, task_summary, manifest_summary, is_held

HERE = os.path.dirname(os.path.abspath(__file__))
from natlang_neuralese.common.paths import root as _root  # noqa: E402  (machine roots; the catalog never hard-codes them)
ROOTS = {name: str(_root(name)) for name in ('data_nvme', 'data_hdd', 'repo', 'archive')}
HDD = ROOTS['data_hdd']
GRPO = REPO + '/scripts/neuralese_data/grpo.py'
REGISTRY = REPO + '/training/neuralese_corpora.json'


# ---------- cheap facts ----------

_FACTS = {}


def file_facts(path):
    """{bytes, lines} of a file; lines counted once per (size, mtime) for files under 1.5 GB (zst: not counted)."""
    try:
        s = sig(path)
    except OSError:
        return {'bytes': None, 'lines': None, 'missing': True}
    hit = _FACTS.get(path)
    if hit and hit[0] == s:
        return hit[1]
    cp = f'{CACHE}/facts.json'
    disk = _FACTS.setdefault('__disk__', (None, read_json(cp, {})))[1]
    if disk.get(path, {}).get('sig') == s:
        _FACTS[path] = (s, disk[path]['v'])
        return disk[path]['v']
    size = os.path.getsize(path)
    lines = None
    estimated = False
    if path.endswith('.jsonl'):
        slow = path.startswith(HDD) and size > 32e6 or size > 1.5e9
        n = 0
        with open(path, 'rb') as f:      # exact count on NVMe; on the HDD a 4 MB sample scaled to the file size
            for chunk in iter(lambda: f.read(1 << 22), b''):
                n += chunk.count(b'\n')
                if slow:
                    n = round(n * size / max(f.tell(), 1))
                    estimated = True
                    break
        lines = n
    v = {'bytes': size, 'lines': lines, 'estimated': estimated, 'mtime': int(os.path.getmtime(path))}
    _FACTS[path] = (s, v)
    disk[path] = {'sig': s, 'v': v}
    os.makedirs(CACHE, exist_ok=True)
    tmp = cp + '.tmp'
    json.dump(disk, open(tmp, 'w'))
    os.replace(tmp, cp)
    return v


def dir_bytes(d, max_files=4000):
    """Bytes of a directory's files (top two levels, bounded): an estimate for the catalog, not an accounting."""
    total, n = 0, 0
    for root, dirs, files in os.walk(d):
        if root.count('/') - d.count('/') >= 2:
            dirs[:] = []
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
            n += 1
            if n > max_files:
                return total
    return total


# ---------- code constants (grpo.py) ----------

def _resolve_call(node):
    if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'resolve_str':
        args = [a.value for a in node.args if isinstance(a, ast.Constant)]
        if len(args) == len(node.args) and args[0] in ROOTS:
            return os.path.join(ROOTS[args[0]], *args[1:])
    return None


_CODE = {}


def code_constants():
    """{name: path | [paths]} of the data constants in grpo.py, with line numbers."""
    s = sig(GRPO)
    if _CODE.get('sig') == s:
        return _CODE['v']
    tree = ast.parse(open(GRPO).read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name not in ('POOLS', 'MEMETIC_TASKS', 'HELD', 'ERRATA', 'SD_TEACHER_SOURCES'):
                continue
            if isinstance(node.value, ast.List):
                out[name] = {'line': node.lineno, 'value': [p for p in map(_resolve_call, node.value.elts) if p]}
            else:
                out[name] = {'line': node.lineno, 'value': _resolve_call(node.value)}
    _CODE.update(sig=s, v=out)
    return out


def _slug(path):
    for r in (HDD, ROOTS['data_nvme'], REPO):
        path = path.replace(r + '/', '')
    return re.sub(r'[^A-Za-z0-9._-]+', '-', path).strip('-')


def task_pools():
    c = code_constants()
    pools = []
    for p in c.get('POOLS', {}).get('value', []):
        d = os.path.dirname(p)
        name = os.path.basename(d) if os.path.basename(d) != os.path.basename(os.path.dirname(d)) else os.path.basename(d)
        pools.append({'id': f'pool:{name}', 'path': p, 'title': name, 'line': c['POOLS']['line']})
    return pools


# ---------- runs ----------

def grpo_runs():
    out = []
    for tj in sorted(glob.glob(OPS + '/grpo-*/run*/train.jsonl') + glob.glob(OPS + '/grpo-*/*/run*/train.jsonl')):
        d = os.path.dirname(tj)
        run = os.path.relpath(d, OPS)
        if '/pin-' in run or '/hunt/' in run:
            continue
        out.append({'id': f'grpo:{run}', 'run': run, 'path': d})
    return out


def teacher_campaigns():
    out = []
    for m in sorted(glob.glob(OPS + '/teacher-gen-*/*/manifest.jsonl')):
        d = os.path.dirname(m)
        name = os.path.relpath(d, OPS)
        out.append({'id': f'teacher-gen:{name}', 'path': d, 'name': name,
                    'parts_map': {'tasks': m, 'deferred': d + '/deferred.jsonl'}})
    return out


def sft_mixes():
    out = []
    for d in sorted(glob.glob(REPO + '/runs/sft-*-2026111*-v*') + glob.glob(REPO + '/runs/sft-mix-*')):
        rows = d + '/rows.jsonl.zst'
        out.append({'id': f'sft:{os.path.basename(d)}', 'path': d,
                    'parts_map': {'rows': rows if os.path.exists(rows) else None, 'cases': d + '/cases.jsonl' if os.path.exists(d + '/cases.jsonl') else None}})
    return out


def sft_runs():
    out = []
    for lj in sorted(glob.glob(OPS + '/sft-v*/run*/launch.json')):
        d = os.path.dirname(lj)
        out.append({'run': os.path.relpath(d, OPS), 'path': d, 'launch': read_json(lj, {})})
    return out


# ---------- registry ----------

_REG = {}


def registry():
    s = sig(REGISTRY)
    if _REG.get('sig') != s:
        reg = read_json(REGISTRY, {})
        by_path = {}
        links = {}

        def top_target(rel):      # readlink of runs/X or data/X only: the repo side (NVMe), never a hop onto the HDD
            top = '/'.join(rel.split('/')[:2])
            if top not in links:
                try:
                    links[top] = os.path.normpath(os.path.join(REPO, os.path.dirname(top), os.readlink(os.path.join(REPO, top))))
                except OSError:
                    links[top] = None
            t = links[top]
            return t and os.path.normpath(t + rel[len(top):])
        for e in reg.get('corpora', []):
            rel = e.get('path', '').strip('/')
            by_path[os.path.join(REPO, rel)] = e
            t = top_target(rel)
            if t:
                by_path[t] = e
        _REG.update(sig=s, v=reg, by_path=by_path)
    return _REG['v'], _REG['by_path']


def registry_entry_for(path):
    _, by = registry()
    for p in (path, os.path.dirname(path)):
        if p in by:
            return by[p]
    return None


# ---------- consumers (who read a dataset, and how much) ----------

_CONS = {}


def consumers_index():
    """path -> [{run, how, amount}] from what runs recorded: GRPO pool.json pools, teacher-gen pool.json, sft-mix build.json
    inputs, SFT launch.json chains."""
    key = tuple(sig(p) for p in glob.glob(OPS + '/grpo-*/run*/pool.json') + glob.glob(OPS + '/teacher-gen-*/*/pool.json')
                + glob.glob(REPO + '/runs/sft-*/build.json') + glob.glob(OPS + '/sft-v*/run*/launch.json'))
    if _CONS.get('key') == key:
        return _CONS['v']
    idx = {}

    def add(path, rec):
        idx.setdefault(os.path.normpath(path), []).append(rec)
    for pj in glob.glob(OPS + '/grpo-*/run*/pool.json'):
        run = os.path.relpath(os.path.dirname(pj), OPS)
        pj_ = read_json(pj, {})
        for p in pj_.get('pools') or []:
            add(p.get('path'), {'run': f'grpo:{run}', 'how': 'GRPO task pool', 'amount': f"{pj_.get('eligible')} eligible of {pj_.get('union_rows')} union rows (all pools)", 'sha256': p.get('sha256')})
        for e in pj_.get('excluded_errata') or []:
            add(e.get('path'), {'run': f'grpo:{run}', 'how': 'errata (excluded ids)', 'amount': f"{pj_.get('excluded_errata_ids')} ids"})
        if pj_.get('held_manifest'):
            add(pj_['held_manifest'], {'run': f'grpo:{run}', 'how': 'held exclusion', 'amount': f"{pj_.get('excluded_ids')} ids excluded"})
    for pj in glob.glob(OPS + '/teacher-gen-*/*/pool.json'):
        name = os.path.relpath(os.path.dirname(pj), OPS)
        pj_ = read_json(pj, {})
        for p in pj_.get('pools') or []:
            add(p.get('path'), {'run': f'teacher-gen:{name}', 'how': 'teacher task pool', 'amount': f"{pj_.get('eligible')} eligible"})
    for bj in glob.glob(REPO + '/runs/sft-*/build.json'):
        mix = os.path.basename(os.path.dirname(bj))
        b = read_json(bj, {})
        inputs = b.get('inputs') or {}
        for k, v in inputs.items():
            vals = v if isinstance(v, list) else [v]
            for x in vals:
                p = x.get('path') if isinstance(x, dict) else x
                if isinstance(p, str) and p.startswith('/'):
                    add(p, {'run': f'sft:{mix}', 'how': f'SFT build input ({k})', 'amount': None})
                    add(os.path.dirname(p), {'run': f'sft:{mix}', 'how': f'SFT build input ({k})', 'amount': None})
        for name, aux in ((b.get('mix') or {}).get('aux') or {}).items():
            if isinstance(aux, dict) and aux.get('path'):
                add(aux['path'], {'run': f'sft:{mix}', 'how': f'SFT aux {name}', 'amount': f"share {aux.get('share')}"})
                add(os.path.dirname(aux['path']), {'run': f'sft:{mix}', 'how': f'SFT aux {name}', 'amount': f"share {aux.get('share')}"})
    for r in sft_runs():
        ch = (r['launch'].get('args') or r['launch']).get('chains') if isinstance(r['launch'], dict) else None
        if isinstance(ch, str):
            add(ch, {'run': f"sft-run:{r['run']}", 'how': 'SFT training chains', 'amount': None})
            add(os.path.dirname(ch), {'run': f"sft-run:{r['run']}", 'how': 'SFT training chains', 'amount': None})
    _CONS.update(key=key, v=idx)
    return idx


def consumers_of(*paths):
    idx = consumers_index()
    out, seen = [], set()
    for p in paths:
        for rec in idx.get(os.path.normpath(p), []) if p else []:
            k = (rec['run'], rec['how'])
            if k not in seen:
                seen.add(k)
                out.append(rec)
    return out


# ---------- dataset assembly ----------

def _prov_registry(path):
    e = registry_entry_for(path)
    if not e:
        return None
    return {k: e.get(k) for k in ('id', 'admission', 'training_admission', 'license_provenance', 'license', 'derived_from',
                                  'supersedes', 'superseded_by', 'note', 'owner') if e.get(k) is not None}


def _task_pool_ds(p, title, group, *, code_ref=None, protected=False, extra=None):
    d = os.path.dirname(p)
    f = file_facts(p)
    sides = {n: os.path.exists(f'{d}/{n}') for n in ('receipt.json', 'qualification.json', 'verification.json', 'selection.json', 'custody.json', 'repetition.json', 'build.json', 'protection-union.json')}
    reg = _prov_registry(d)
    ds = {'id': None, 'title': title, 'group': group, 'kind': 'held' if protected else 'rl-tasks', 'reader': 'held-set' if protected else 'jsonl',
          'item_kind': 'task', 'path': p, 'dir': d, 'format': 'JSONL, one natlang.program/2 task per line',
          'parts': [{'id': 'tasks', 'title': 'Tasks'}], 'parts_map': {'tasks': p}, 'part_kind': {'tasks': 'task'},
          'protected': protected, 'size_bytes': f['bytes'], 'rows': f['lines'], 'rows_estimated': f.get('estimated'), 'mtime': f.get('mtime'), 'missing': f.get('missing'),
          'provenance': {'registry': reg, 'code_ref': code_ref, 'records': [k for k, v in sides.items() if v],
                         'receipt': read_json(f'{d}/receipt.json', None) and {k: v for k, v in read_json(f'{d}/receipt.json', {}).items() if isinstance(v, (str, int, float))}},
          'consumers': consumers_of(p, d)}
    if extra:
        ds.update(extra)
    return ds


_CAT = {}


_BG = {}


def build_catalog(wait=100):
    """All datasets. Served from the last build (memory, else <cache>/catalog.json) while a background thread rebuilds it
    when it is older than 10 minutes: the first build stats and samples files on a contended HDD and can take minutes."""
    import threading
    if not _CAT.get('v'):
        disk = read_json(CACHE + '/catalog.json')
        if disk:
            _CAT.update(ts=disk['ts'], v=disk['v'])
    stale = _CAT.get('ts', 0) < time.time() - 600
    t = _BG.get('thread')
    if stale and not (t and t.is_alive()):
        t = _BG['thread'] = threading.Thread(target=_rebuild, daemon=True)
        t.start()
    if not _CAT.get('v'):
        t.join(wait)
        if not _CAT.get('v'):
            raise QueryError('The catalog is still being built (first build reads the HDD); try again in a minute.', 'busy')
    return _CAT['v']


def _rebuild():
    try:
        v = _build_catalog()
        _CAT.update(ts=time.time(), v=v)
        os.makedirs(CACHE, exist_ok=True)
        tmp = CACHE + '/catalog.json.tmp'
        json.dump({'ts': _CAT['ts'], 'v': v}, open(tmp, 'w'), default=str)
        os.replace(tmp, CACHE + '/catalog.json')
    except Exception:
        import traceback
        traceback.print_exc()


def _build_catalog():
    out = []
    c = code_constants()
    # RL task pools in POOLS
    for pool in task_pools():
        out.append({**_task_pool_ds(pool['path'], pool['title'], 'RL task pools', code_ref=f"grpo.py:{pool['line']} POOLS"), 'id': pool['id']})
    mt = (c.get('MEMETIC_TASKS') or {}).get('value')
    if mt:
        out.append({**_task_pool_ds(mt, os.path.basename(os.path.dirname(mt)), 'RL task pools', code_ref=f"grpo.py:{c['MEMETIC_TASKS']['line']} MEMETIC_TASKS"),
                    'id': 'memetic-tasks-' + os.path.basename(os.path.dirname(mt)).split('-')[-1]})
    held = (c.get('HELD') or {}).get('value')
    if held:
        d = os.path.dirname(held)
        out.append({**_task_pool_ds(d + '/cases.jsonl', 'held-v3 (the held-out set)', 'Held-out (protected)', code_ref=f"grpo.py:{c['HELD']['line']} HELD", protected=True),
                    'id': 'held-v3', 'manifest': held,
                    'note': 'View only. 256 tasks paired against the base policy; training excludes these ids and their split groups (held_eval.exclusion_ids).'})
    er = (c.get('ERRATA') or {}).get('value')
    if er:
        files = sorted(glob.glob(er + '/*.jsonl'))
        out.append({'id': 'pool-errata', 'title': 'Pool errata (excluded task ids)', 'group': 'Edit layers', 'kind': 'index', 'reader': 'jsonl', 'item_kind': 'raw',
                    'path': er, 'parts': [{'id': os.path.basename(f), 'title': os.path.basename(f)} for f in files],
                    'parts_map': {os.path.basename(f): f for f in files}, 'part_kind': {os.path.basename(f): 'errata' for f in files},
                    'rows': sum(file_facts(f)['lines'] or 0 for f in files), 'size_bytes': sum(file_facts(f)['bytes'] or 0 for f in files),
                    'provenance': {'code_ref': f"grpo.py:{c['ERRATA']['line']} ERRATA"}, 'consumers': consumers_of(*files),
                    'note': 'Every *.jsonl here is excluded from GRPO and teacher-gen task pools at start/resume. ops-dashboard.jsonl is written by page edits.'})
    for sd in (c.get('SD_TEACHER_SOURCES') or {}).get('value') or []:
        pass       # covered by the catalog_sources globs (sft-teacher-demos-*, sft-sources-*), tagged as consumed below
    sd_set = {os.path.normpath(p) for p in (c.get('SD_TEACHER_SOURCES') or {}).get('value') or []}
    # GRPO runs and their held evals
    for r in grpo_runs():
        d = r['path']
        f = file_facts(d + '/train.jsonl')
        pj = read_json(d + '/pool.json', {})
        out.append({'id': r['id'], 'run': r['run'], 'title': r['run'], 'group': 'GRPO runs', 'kind': 'run', 'reader': 'grpo-run', 'path': d,
                    'format': 'run directory: train.jsonl, groups/, step-chains/, rollouts archive, held/',
                    'parts': [{'id': 'steps', 'title': 'Steps (selection)'}, {'id': 'groups', 'title': 'Groups and episodes'}, {'id': 'held', 'title': 'Held evals'}],
                    'rows': None, 'size_bytes': f['bytes'], 'mtime': f.get('mtime'),
                    'provenance': {'pool': {k: pj.get(k) for k in ('union_rows', 'excluded_ids', 'eligible', 'excluded_errata_ids', 'excluded_family_regex', 'held_manifest')},
                                   'pools': [p.get('path') for p in pj.get('pools') or []], 'rollouts': None},
                    'consumers': consumers_of(d)})
        for hd in sorted(glob.glob(d + '/held/step-*/result.json')):
            h = os.path.dirname(hd)
            out.append({'id': f"held-eval:{r['run']}:{os.path.basename(h)}", 'title': f"{r['run']} {os.path.basename(h)}", 'group': 'Held-out evaluations (protected)',
                        'kind': 'held', 'reader': 'held-eval', 'path': h, 'protected': True, 'parts': [{'id': 'samples', 'title': 'Episodes'}],
                        'format': 'held eval directory (result.json, row-RRR/sample-KK/results.jsonl native traces)', 'run': r['run'], 'hidden': True})
    # teacher-gen campaigns
    for tc in teacher_campaigns():
        f = file_facts(tc['parts_map']['tasks'])
        pj = read_json(tc['path'] + '/pool.json', {})
        out.append({**tc, 'title': tc['name'], 'group': 'Teacher-generation campaigns', 'kind': 'teacher-demos', 'reader': 'teacher-gen', 'item_kind': 'manifest',
                    'format': 'manifest.jsonl (one task, its teacher episodes and outcomes) + episodes/tNNNNNN/rollouts.jsonl',
                    'parts': [{'id': 'tasks', 'title': 'Tasks and teacher episodes'}], 'part_kind': {'tasks': 'manifest'},
                    'rows': f['lines'], 'size_bytes': f['bytes'], 'mtime': f.get('mtime'),
                    'provenance': {'pools': [p.get('path') for p in pj.get('pools') or []], 'eligible': pj.get('eligible'),
                                   'launches': len(jsonl(tc['path'] + '/launches.jsonl', 200)), 'producer': 'scripts/neuralese_data/teacher_gen.py'},
                    'consumers': consumers_of(tc['path'], tc['parts_map']['tasks'])})
    # registry as one dataset
    reg, _ = registry()
    out.append({'id': 'corpus-registry', 'title': 'Corpus registry (training/neuralese_corpora.json)', 'group': 'Registries', 'kind': 'index', 'reader': 'registry',
                'path': REGISTRY, 'parts': [{'id': 'entries', 'title': 'Entries'}], 'rows': len(reg.get('corpora', [])),
                'size_bytes': os.path.getsize(REGISTRY), 'format': 'natlang.neuralese-corpus-registry/1 + corpus-manifests/<id>.json (natlang.corpus-snapshot/1)',
                'provenance': {'writer': 'scripts/neuralese_data/register_corpora.py'}})
    # hand layer
    src = read_json(HERE + '/catalog_sources.json', {})
    seen = {d.get('path') for d in out} | {d.get('dir') for d in out}
    for s in src.get('sources', []):
        pats = [s['glob']]
        m = re.search(r'\{([^}]*)\}', s['glob'])
        if m:
            pats = [s['glob'].replace(m.group(0), x) for x in m.group(1).split(',')]
        for d in sorted(x for pat in pats for x in glob.glob(pat)):
            if d in seen or not os.path.isdir(d):
                continue
            name = os.path.basename(d.rstrip('/')) if not d.endswith('/rollouts') else os.path.relpath(d, HDD).replace('/rollouts', '')
            ds = {'id': f"{s['id']}:{name}" if '*' in s['glob'] or '{' in s['glob'] else s['id'], 'title': f"{s['title']} · {name}" if '*' in s['glob'] or '{' in s['glob'] else s['title'],
                  'group': s['group'], 'kind': s['kind'], 'reader': s['reader'], 'item_kind': s.get('item_kind'), 'path': d, 'dir': d,
                  'protected': bool(s.get('protected')), 'provenance': {'producer': s.get('producer'), 'registry': _prov_registry(d),
                  'records': [x for x in s.get('side', []) if os.path.exists(f'{d}/{x}')], 'source': 'catalog_sources.json'},
                  'relations': {k: s[k] for k in ('superseded_by',) if s.get(k)}, 'consumers_note': s.get('consumers_note')}
            if os.path.normpath(d) in sd_set:
                ds['code_consumer'] = f"grpo.py:{c['SD_TEACHER_SOURCES']['line']} SD_TEACHER_SOURCES (TeacherDemos, self-distillation)"
            files = s.get('files') or ({'tasks': s['file']} if s.get('file') else {})
            if s['reader'] == 'sft-mix':
                files = {'rows': 'rows.jsonl.zst', 'cases': 'cases.jsonl'}
                b = read_json(d + '/build.json', None) or read_json(d + '/merge.json', None)
                if b:
                    mix = b.get('mix') or {}
                    ds['mix'] = {'parts': mix.get('parts'), 'aux': {k: {x: y for x, y in v.items() if x in ('path', 'share', 'chains', 'loss_share')} for k, v in (mix.get('aux') or {}).items()} if isinstance(mix.get('aux'), dict) else None,
                                 'funnel': b.get('funnel'), 'stats': b.get('stats'), 'code': b.get('code'), 'policy': b.get('policy'),
                                 'inputs': {k: (v if not isinstance(v, dict) else {x: y for x, y in v.items() if not isinstance(y, (list, dict))}) for k, v in (b.get('inputs') or {}).items()} if isinstance(b.get('inputs'), dict) else None}
            pm = {k: f'{d}/{v}' for k, v in files.items() if os.path.exists(f'{d}/{v}')}
            kinds = {'sft-mix': {'rows': 'sft_row', 'cases': 'task'}, 'teacher-demos': {k: 'demo' for k in pm}}.get(s['reader'], {k: s.get('item_kind') or 'raw' for k in pm})
            ds.update(parts=[{'id': k, 'title': k} for k in pm], parts_map=pm, part_kind=kinds)
            sizes = [file_facts(p) for p in pm.values()]
            ds['size_bytes'] = sum(x['bytes'] or 0 for x in sizes) if sizes else None
            ds['rows'] = sum(x['lines'] or 0 for x in sizes) if sizes and all(x['lines'] is not None for x in sizes) else None
            ds['rows_estimated'] = any(x.get('estimated') for x in sizes)
            ds['consumers'] = consumers_of(d, *pm.values())
            if s['reader'] in ('jsonl', 'held-set') and 'tasks' in pm:
                ds['path'] = pm['tasks']
            out.append(ds)
    for ds in out:
        ds.setdefault('protected', False)
        ds.setdefault('relations', {})
        if ds.get('code_consumer'):
            ds['consumers'] = (ds.get('consumers') or []) + [{'run': 'grpo (default)', 'how': ds['code_consumer'], 'amount': None}]
    _relations(out)
    return out


def _relations(out):
    """Derived-from / superseded-by / duplicates from registry fields and identical file signatures."""
    by_sig = {}
    for ds in out:
        p = ds.get('path')
        if p and os.path.isfile(p):
            try:
                by_sig.setdefault((os.path.getsize(p), os.path.basename(p)), []).append(ds['id'])
            except OSError:
                pass
        reg = (ds.get('provenance') or {}).get('registry') or {}
        for k in ('derived_from', 'supersedes', 'superseded_by'):
            if reg.get(k):
                ds['relations'][k] = reg[k]
    for ids in by_sig.values():
        if len(ids) > 1:
            for ds in out:
                if ds['id'] in ids:
                    ds['relations']['same_size_as'] = [i for i in ids if i != ds['id']]


def dataset(ds_id):
    for ds in build_catalog():
        if ds['id'] == ds_id:
            return ds
    raise QueryError(f'unknown dataset {ds_id}', 'not_found')


def _card(ds):
    keep = ('id', 'title', 'group', 'kind', 'reader', 'path', 'protected', 'size_bytes', 'rows', 'rows_estimated', 'mtime', 'missing', 'consumers',
            'relations', 'consumers_note', 'note', 'run', 'hidden')
    out = {k: ds.get(k) for k in keep if ds.get(k) is not None}
    prov = ds.get('provenance') or {}
    out['provenance_brief'] = {'registered': bool(prov.get('registry')), 'code_ref': prov.get('code_ref'), 'producer': prov.get('producer'),
                               'records': prov.get('records'), 'license': (prov.get('registry') or {}).get('license_provenance') or (prov.get('registry') or {}).get('license')}
    return out


@handler('data.catalog')
def data_catalog(_):
    cat = build_catalog()
    return {'datasets': [_card(d) for d in cat if not d.get('hidden')], 'generated': int(_CAT.get('ts', time.time()) * 1000),
            'sources': {'code': GRPO, 'registry': REGISTRY, 'hand_layer': HERE + '/catalog_sources.json'}}


@handler('data.dataset')
def data_dataset(inp):
    ds = dict(dataset(inp.get('ds')))
    ds.pop('parts_map', None)
    if ds['reader'] == 'grpo-run':
        from data_items import rollout_archive, run_args
        args, rec = run_args(ds['path'])
        ds['args'] = args
        ds['code'] = (rec or {}).get('code')
        ds['rollouts'] = rollout_archive(ds['path'])
    if ds.get('manifest'):
        m = read_json(ds['manifest'], {})
        ds['held_manifest'] = {k: m.get(k) for k in ('seed', 'included', 'by_level', 'families', 'also_excludes', 'cases', 'pool') if k in m}
    return ds


@handler('data.registry')
def data_registry(inp):
    """Corpus registry entries: {q?, filter{admission, owner}, offset, limit} or {id} for one entry with its manifest."""
    reg, _ = registry()
    if inp.get('id'):
        e = next((x for x in reg.get('corpora', []) if x.get('id') == inp['id']), None)
        if not e:
            raise QueryError('no such registry entry', 'not_found')
        man = read_json(f"{REPO}/training/corpus-manifests/{e['id']}.json", None)
        full = os.path.join(REPO, e.get('path', ''))
        return {'kind': 'registry_entry', 'entry': e, 'manifest': man and {**{k: v for k, v in man.items() if k != 'files'}, 'files': (man.get('files') or [])[:300]},
                'exists': os.path.exists(full), 'realpath': os.path.realpath(full)}
    q = (inp.get('q') or '').lower()
    f = inp.get('filter') or {}
    rows = []
    for e in reg.get('corpora', []):
        r = {'id': e.get('id'), 'path': e.get('path'), 'owner': e.get('owner'), 'admission': e.get('admission') or e.get('source_class'),
             'training_admission': e.get('training_admission'), 'superseded_by': e.get('superseded_by')}
        if q and q not in json.dumps(e).lower():
            continue
        if any(v and str(r.get(k)) != str(v) for k, v in f.items()):
            continue
        rows.append(r)
    fac = {}
    for k in ('admission', 'owner'):
        cnt = {}
        for r in rows:
            cnt[r.get(k)] = cnt.get(r.get(k), 0) + 1
        fac[k] = sorted(([str(a), b] for a, b in cnt.items()), key=lambda x: -x[1])[:40]
    off, lim = int(inp.get('offset') or 0), min(int(inp.get('limit') or 100), 500)
    return {'total': len(rows), 'offset': off, 'rows': rows[off:off + lim], 'facets': fac}


@handler('data.search')
def data_search(inp):
    """Global search: dataset titles/ids, and task ids / families across the task pools (from their cached indexes)."""
    q = (inp.get('q') or '').strip().lower()
    if len(q) < 2:
        return {'datasets': [], 'tasks': []}
    cat = build_catalog()
    dss = [_card(d) for d in cat if q in d['id'].lower() or q in (d.get('title') or '').lower()][:40]
    tasks = []
    for ds in cat:
        if ds.get('item_kind') == 'task' and ds.get('path') and os.path.isfile(ds['path']) and (ds.get('rows') or 0) < 60000:
            for loc, s in line_index(ds['path'], task_summary):
                if q in (s.get('id') or '').lower() or q in (s.get('family') or '').lower():
                    tasks.append({'ds': ds['id'], 'protected': ds.get('protected'), **{k: s.get(k) for k in ('id', 'family', 'level', 'source')}})
                    if len(tasks) >= 200:
                        break
        if len(tasks) >= 200:
            break
    return {'datasets': dss, 'tasks': tasks}
