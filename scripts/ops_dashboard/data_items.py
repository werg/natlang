"""Readers and normalizers for the training-data viewer (data.* queries of data_service.py).

Every dataset in the catalog (data_catalog.py) names a `reader`; the readers here list its items cheaply (cached line
indexes, substring scans over raw bytes) and return one item normalized for the page:

  task        natlang.program/2 row: program files, inputs, oracle (expected value / files), world (folder files),
              curriculum, qualification checks (wrong/alias answers, wrong code), provenance
  trajectory  a multi-call episode (natlang.rollout/1 from GRPO archives and teacher campaigns, held-eval native traces):
              calls with new context messages, reasoning, actions (tool calls), tool results, decisions, Neuralese
              imports, sampling (token ids and logps) and the training role / weights that can be reconstructed
  sft_row     one SFT call row (rows.jsonl.zst): context messages, target with its weighted / zero-weight parts
  chain       one learner chain (GRPO step-chains): token ids with trained segments, roles, advantages
  raw         anything else, as JSON

What the files cannot tell is returned as `unknown` notes beside the values (see the module docstrings of each reader).
"""
import glob
import hashlib
import io
import json
import os
import pickle
import re
import statistics
import time

from data_service import handler, QueryError, DATA, REPO, DASH

OPS = DATA + '/ops'
CACHE = DASH + '/cache'
TEXT_CLIP = 60000          # characters of one message or file shown before "show more"


# ---------- low-level: line access to JSONL / JSONL.zst ----------

def is_zst(path):
    return path.endswith('.zst')


def zst_lines(path):
    import zstandard
    with open(path, 'rb') as f:
        reader = io.BufferedReader(zstandard.ZstdDecompressor().stream_reader(f), 1 << 20)
        for line in reader:
            yield line


def sig(path):
    st = os.stat(path)
    return f'{st.st_size}:{int(st.st_mtime)}'


def _cache_path(kind, path):
    os.makedirs(CACHE, exist_ok=True)
    return f'{CACHE}/{kind}-{hashlib.sha1(path.encode()).hexdigest()[:16]}.json'


_IDX = {}


def line_index(path, summarize, version=1):
    """[(loc, summary)] of a JSONL file: loc is the byte offset (plain) or the line number (zst). Cached on disk by
    (size, mtime) so a restart or a later query does not rescan; `summarize(record) -> dict` keeps a few small fields."""
    key = (path, summarize.__name__, version)
    s = sig(path)
    hit = _IDX.get(key)
    if hit and hit[0] == s:
        return hit[1]
    cp = _cache_path(f'idx-{summarize.__name__}-v{version}', path)
    try:
        cached = json.load(open(cp))
        if cached['sig'] == s:
            _IDX[key] = (s, cached['rows'])
            return cached['rows']
    except (OSError, ValueError, KeyError):
        pass
    rows = []
    if is_zst(path):
        for n, raw in enumerate(zst_lines(path)):
            try:
                rows.append([n, summarize(json.loads(raw))])
            except ValueError:
                continue
    else:
        with open(path, 'rb') as f:
            off = 0
            for raw in f:
                if raw.strip():
                    try:
                        rows.append([off, summarize(json.loads(raw))])
                    except ValueError:
                        pass
                off += len(raw)
    tmp = cp + '.tmp'
    with open(tmp, 'w') as f:
        json.dump({'sig': s, 'path': path, 'rows': rows}, f, separators=(',', ':'))
    os.replace(tmp, cp)
    _IDX[key] = (s, rows)
    return rows


_LINE_CACHE = {}


def read_at(path, loc):
    """One record at a loc from line_index."""
    k = (path, loc)
    if k in _LINE_CACHE:
        return _LINE_CACHE[k]
    if is_zst(path):
        rec = None
        for n, raw in enumerate(zst_lines(path)):
            if n == loc:
                rec = json.loads(raw)
                break
    else:
        with open(path, 'rb') as f:
            f.seek(loc)
            rec = json.loads(f.readline())
    if len(_LINE_CACHE) > 24:
        _LINE_CACHE.pop(next(iter(_LINE_CACHE)))
    _LINE_CACHE[k] = rec
    return rec


def grep_locs(path, needle, limit=2000):
    """Locs of lines containing `needle` (case-insensitive bytes search; one pass, no JSON parsing)."""
    nb = needle.lower().encode()
    out = []
    if is_zst(path):
        for n, raw in enumerate(zst_lines(path)):
            if nb in raw.lower():
                out.append(n)
                if len(out) >= limit:
                    break
        return out
    with open(path, 'rb') as f:
        off = 0
        for raw in f:
            if nb in raw.lower():
                out.append(off)
                if len(out) >= limit:
                    break
            off += len(raw)
    return out


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def jsonl(path, limit=None):
    out = []
    try:
        it = zst_lines(path) if is_zst(path) else open(path, 'rb')
        for raw in it:
            if raw.strip():
                try:
                    out.append(json.loads(raw))
                except ValueError:
                    pass
                if limit and len(out) >= limit:
                    break
    except OSError:
        pass
    return out


def clip(s, n=TEXT_CLIP):
    if not isinstance(s, str):
        return s
    return s if len(s) <= n else s[:n] + f'\n… [{len(s) - n:,} more characters]'


# ---------- held-out protection ----------

_HELD = {}


def held_ids():
    """Task ids and split groups excluded from training by held-v3 (and held-v2 via also_excludes)."""
    path = OPS + '/held-v3/manifest.json'
    s = sig(path)
    if _HELD.get('sig') == s:
        return _HELD['ids'], _HELD['groups']
    ids, groups = set(), set()
    for p in [path] + [OPS + '/held-v2/manifest.json']:
        m = read_json(p, {})
        ids |= {t['id'] for t in m.get('tasks', [])}
        groups |= set(m.get('excluded_split_groups', []))
    ex = read_json(OPS + '/held-v3/exclusion-ids.json', [])
    ids |= set(ex if isinstance(ex, list) else [])
    _HELD.update(sig=s, ids=ids, groups=groups)
    return ids, groups


def is_held(task_id, row=None):
    ids, groups = held_ids()
    if task_id in ids:
        return True
    if row:
        sg = (row.get('curriculum') or {}).get('split_group') or row.get('id')
        return sg in groups
    return False


# ---------- normalizers ----------

def task_summary(r):
    cur = r.get('curriculum') or {}
    ext = cur.get('external') or {}
    sem = r.get('semantics') or {}
    return {'id': r.get('id'), 'family': r.get('family') or cur.get('family'), 'split': r.get('split'),
            'source': r.get('source'), 'level': (cur.get('rl') or {}).get('level'), 'shape': cur.get('shape'),
            'constructs': (ext.get('constructs') or [])[:8], 'oracle': sem.get('oracle'),
            'files': len(sem.get('files') or {}), 'world': len(sem.get('folder_files') or {}),
            'world_kind': (sem.get('world') or {}).get('kind'), 'sg': cur.get('split_group')}


def normalize_task(r, where=None):
    """A natlang task program as the task editor shows it."""
    cur = r.get('curriculum') or {}
    ext = cur.get('external') or {}
    sem = r.get('semantics') or {}
    checks = ext.get('checks') or {}
    return {
        'kind': 'task', 'id': r.get('id'), 'family': r.get('family'), 'split': r.get('split'),
        'held': is_held(r.get('id'), r),
        'program': {'root': sem.get('root'), 'files': {k: clip(v) for k, v in (sem.get('files') or {}).items()},
                    'inputs': sem.get('inputs')},
        'oracle': {'kind': sem.get('oracle'), 'expected': sem.get('expected'),
                   'expected_files': {k: clip(v, 20000) for k, v in (sem.get('expected_files') or {}).items()},
                   'files_oracle': sem.get('files_oracle'), 'depth': ext.get('oracle_depth')},
        'world': {'kind': (sem.get('world') or {}).get('kind') or ('folder' if sem.get('folder_files') else None),
                  'folder_files': {k: clip(v, 20000) for k, v in (sem.get('folder_files') or {}).items()},
                  'spec': {k: v for k, v in (sem.get('world') or {}).items() if k != 'kind'} or None,
                  'evidence': cur.get('evidence')},
        'qualification': {'wrong_answers': checks.get('wrong_answers'), 'alias_answers': checks.get('alias_answers'),
                          'wrong_code': checks.get('wrong_code'), 'basis': (cur.get('rl') or {}).get('basis'),
                          'gold_sources': r.get('gold_sources'), 'reference': cur.get('reference'),
                          'smoke': (ext.get('structure') or {}).get('smoke')},
        'curriculum': {k: cur.get(k) for k in ('shape', 'variant', 'slice', 'domain', 'mode', 'inline', 'named',
                                               'split_group', 'pair_group', 'minimum_sequence', 'decisive', 'assumptions')},
        'constructs': ext.get('constructs'), 'level': (cur.get('rl') or {}).get('level'),
        'provenance': {'source': r.get('source'), 'license': r.get('license'), 'generation': r.get('generation'),
                       'source_ids': r.get('source_ids'), 'source_groups': r.get('source_groups'),
                       'source_revisions': r.get('source_revisions'), 'upstream': {k: ext.get(k) for k in ('source', 'upstream', 'upstream_id', 'upstream_split', 'unit', 'framing', 'subject_domain')}},
        'where': where,
    }


def _text(c):
    if isinstance(c, list):      # content parts
        return '\n'.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in c)
    return c if isinstance(c, str) else ('' if c is None else json.dumps(c))


def norm_message(m):
    out = {'role': m.get('role'), 'content': clip(_text(m.get('content')))}
    if m.get('reasoning_content'):
        out['reasoning'] = clip(m['reasoning_content'])
    if m.get('tool_calls'):
        out['tool_calls'] = [norm_tool_call(t) for t in m['tool_calls']]
    if m.get('tool_call_id'):
        out['tool_call_id'] = m['tool_call_id']
    if m.get('natlang_host_generated'):
        out['host_generated'] = True
    if m.get('name'):
        out['name'] = m['name']
    return out


def norm_tool_call(t):
    fn = t.get('function') or {}
    args = fn.get('arguments')
    try:
        parsed = json.loads(args) if isinstance(args, str) else args
    except ValueError:
        parsed = None
    return {'id': t.get('id'), 'name': fn.get('name') or t.get('name'), 'arguments': clip(args if isinstance(args, str) else json.dumps(args)),
            'parsed': parsed if isinstance(parsed, dict) else None}


def new_messages(prev, cur):
    """The messages of `cur` that extend the conversation `prev` (same invocation): the context added since that call."""
    n = 0
    while n < min(len(prev), len(cur)) and prev[n].get('role') == cur[n].get('role') and _text(prev[n].get('content')) == _text(cur[n].get('content')):
        n += 1
    return n, cur[n:]


NEURALESE_RE = re.compile(r'<\|neuralese[^|]*\|>|⟦neuralese[^⟧]*⟧', re.I)


def norm_turns(turns, *, sampled=True):
    """natlang.rollout/1 turns -> calls. Each call keeps only the context added since the previous call of its invocation."""
    last = {}
    calls = []
    for i, t in enumerate(turns):
        msgs = t.get('messages') or []
        inv = t.get('invocation_id')
        prev = last.get(inv)
        shared, fresh = new_messages(prev, msgs) if prev is not None else (0, msgs)
        last[inv] = msgs + [t.get('target') or {}]
        tgt = t.get('target') or {}
        s = t.get('sampling') or {}
        samples = s.get('samples') or []
        lp = [x.get('logp') for x in samples if isinstance(x.get('logp'), (int, float))]
        decision = s.get('decision_sample')
        call = {
            'i': i, 'invocation': inv, 'function': t.get('function'), 'definition': t.get('definition_id'),
            'role': t.get('invocation_role'), 'parent': t.get('parent_invocation'),
            'context_shared': shared, 'context_total': len(msgs), 'context': [norm_message(m) for m in fresh],
            'target': norm_message(tgt),
            'tools': [((x.get('function') or {}).get('name')) for x in t.get('tools') or []],
            'usage': t.get('usage'),
            'neuralese': bool(NEURALESE_RE.search(json.dumps(fresh)[:200000])) or any((p or {}).get('kind') == 'neuralese' for p in s.get('prompt') or []),
        }
        if s:
            call['sampling'] = {'finish': s.get('finish_reason'), 'law': s.get('sampling_law'), 'tokens': len(samples),
                                'prompt_tokens': sum(len(p.get('token_ids') or []) for p in s.get('prompt') or []),
                                'prompt_parts': [p.get('kind') for p in s.get('prompt') or []],
                                'logp_mean': round(statistics.fmean(lp), 4) if lp else None,
                                'tail_tokens': sum(1 for x in lp if x < -7),
                                'actor': (s.get('actor_export') or {}).get('sha256')}
        if decision:
            call['decision'] = {k: decision.get(k) for k in ('options', 'chosen', 'log_probs', 'labels') if k in decision} or decision
        calls.append(call)
    return calls


def graph_summary(graph):
    """Invocation tree from the execution graph: [{call_id, definition, signature, parent, phase counts}]."""
    nodes = {}
    for ev in graph or []:
        cid = ev.get('call_id')
        if not cid:
            continue
        n = nodes.setdefault(cid, {'call_id': cid, 'definition': (ev.get('definition') or {}).get('id'),
                                   'signature': ev.get('signature'), 'context': ev.get('context'), 'events': 0, 'phases': []})
        n['events'] += 1
        if ev.get('phase') and ev['phase'] not in n['phases']:
            n['phases'].append(ev['phase'])
        po = (ev.get('native_occurrence') or {}).get('parent_occurrence_sha256')
        if po:
            n['parent_occurrence'] = po
    return list(nodes.values())[:400]


def normalize_rollout(e, *, advantage=None, role=None, unknown=()):
    calls = norm_turns(e.get('turns') or [])
    graph = graph_summary(e.get('execution_graph'))
    defs = {n['call_id']: n['definition'] for n in graph if n.get('definition')}
    for c in calls:          # name each call by the natlang definition its invocation runs (the turn says "unknown")
        if (not c.get('function') or c['function'] == 'unknown') and defs.get(c['invocation']):
            c['function'] = defs[c['invocation']]
    rw = e.get('reward') or {}
    return {'kind': 'trajectory', 'format': e.get('schema'), 'id': f"g{e.get('group')}:{e.get('sample')}" if e.get('group') is not None else e.get('task_id'),
            'task_id': e.get('task_id'), 'family': e.get('family'), 'group': e.get('group'), 'sample': e.get('sample'),
            'seed': e.get('seed'), 'round': e.get('round'), 'outcome': e.get('outcome'), 'unscored': e.get('unscored'),
            'reward': rw.get('quality'), 'graded': rw.get('graded'), 'efficiency': rw.get('execution_efficiency'),
            'policy': e.get('policy'), 'code': e.get('code'), 'model_facing': e.get('model_facing'),
            'advantage': advantage, 'role': role, 'calls': calls, 'graph': graph,
            'held': is_held(e.get('task_id')), 'unknown': list(unknown)}


def normalize_native_trace(r):
    """held-eval / native teacher trajectories (natlang.teacher_trajectory.native/1): phases with context and the reply."""
    calls, last = [], {}
    for i, ph in enumerate(r.get('trajectory') or []):
        ctx = ph.get('context') or []
        inv = ph.get('invocation_id')
        prev = last.get(inv)
        shared, fresh = new_messages(prev, ctx) if prev is not None else (0, ctx)
        last[inv] = ctx
        a = ph.get('assistant') or {}
        mr = ph.get('model_response') or {}
        calls.append({'i': i, 'invocation': inv, 'function': ph.get('phase'), 'role': None, 'context_shared': shared,
                      'context_total': len(ctx), 'context': [norm_message(m) for m in fresh],
                      'target': {'role': 'assistant', 'content': clip(a.get('content') or mr.get('text') or ''),
                                 'reasoning': clip(a.get('reasoning') or ''),
                                 'tool_calls': [norm_tool_call(c if 'function' in c else {'function': c}) for c in a.get('calls') or mr.get('calls') or []]},
                      'tools': [((x.get('function') or {}).get('name') or x.get('name')) for x in ph.get('tools_offered') or []],
                      'usage': {'completion_tokens': mr.get('completion_tokens')}})
    o = r.get('outcome') or {}
    prog = (r.get('task') or {}).get('program_ir') or {}
    return {'kind': 'trajectory', 'format': r.get('version'), 'id': r.get('id'), 'task_id': prog.get('id'), 'family': prog.get('family'),
            'outcome': o.get('status'), 'accepted': o.get('accepted'), 'checks': o.get('checks'), 'value': o.get('value'),
            'rejection_reasons': o.get('rejection_reasons'), 'oracle': o.get('oracle'), 'provenance': r.get('provenance'),
            'calls': calls, 'graph': graph_summary(o.get('execution_graph')), 'held': True,
            'unknown': ['Held-eval traces keep no token ids or log-probabilities.']}


def normalize_sft_row(r):
    tgt = r.get('target') or {}
    src = r.get('source')
    reasoning = tgt.get('reasoning_content') or ''
    # Build-time weights (sft_sources.think_context): teacher replies train neither their reasoning nor the lead-in text
    # before the first tool call; own replies zero only an empty think block. loss_from = leading zero-weight tokens.
    parts = []
    if reasoning:
        parts.append({'part': 'reasoning', 'text': clip(reasoning), 'weight': 0 if src == 'teacher' else 1})
    if tgt.get('content'):
        lead_zero = src == 'teacher' and bool(tgt.get('tool_calls'))
        parts.append({'part': 'content', 'text': clip(_text(tgt.get('content'))), 'weight': 0 if lead_zero else 1})
    for t in tgt.get('tool_calls') or []:
        parts.append({'part': 'tool_call', 'call': norm_tool_call(t), 'weight': 1})
    return {'kind': 'sft_row', 'id': r.get('id'), 'episode': r.get('episode'), 'source': src, 'run': r.get('run'),
            'task_id': r.get('task_id'), 'family': r.get('family'), 'domain': r.get('domain'), 'teacher': r.get('teacher'),
            'call': r.get('call'), 'invocation': r.get('invocation'), 'function': r.get('function'),
            'role': r.get('invocation_role'), 'parent': r.get('parent_invocation'),
            'context': [norm_message(m) for m in r.get('messages') or []],
            'tools': [((x.get('function') or {}).get('name')) for x in r.get('tools') or []],
            'target': norm_message(tgt), 'parts': parts, 'prompt_tokens': r.get('prompt_tokens'),
            'target_tokens': r.get('target_tokens'), 'loss_from': r.get('loss_from'), 'too_long': r.get('too_long'),
            'live': {k: r.get(k) for k in ('verification_mode', 'live_source', 'world') if r.get(k) is not None} or None,
            'held': is_held(r.get('task_id')),
            'unknown': ['Load-time masks (empty think, after first <|im_end|>, second </think>) are applied by exp0_sft at training '
                        'time and are not stored; the part weights here are the build-time rule (sft_sources.think_context).',
                        'Which optimizer step trained this row is not recorded (exp0_sft.step_groups re-derives it from the seed).']}


# ---------- tokens ----------

_TOK = {}


def tokenizer():
    """The actor's tokenizer (as served by the GRPO actor export)."""
    if 'tok' in _TOK:
        return _TOK['tok']
    from tokenizers import Tokenizer
    for p in sorted(glob.glob(OPS + '/grpo-v*/run*/actor/sglang-hf/tokenizer.json'), key=os.path.getmtime, reverse=True) + \
            glob.glob(DATA + '/*/actor-tokenizer/tokenizer.json'):
        try:
            _TOK['tok'] = Tokenizer.from_file(p)
            _TOK['path'] = p
            return _TOK['tok']
        except Exception:
            continue
    raise QueryError('no LFM3 tokenizer.json found for token views', 'unavailable')


def token_pieces(ids):
    tok = tokenizer()
    return tok.decode_batch([[int(i)] for i in ids], skip_special_tokens=False)


def special_id(text):
    return tokenizer().token_to_id(text)


def sampled_tokens(turn, advantage=None, tail_floor=-7.0):
    """Token view of one sampled call: pieces, actor logp, and the per-token masks the GRPO learner applies that are
    reconstructible from the record: tail floor (logp < floor with A > 0) and the second-</think> mask (A > 0)."""
    s = turn.get('sampling') or {}
    samples = s.get('samples') or []
    ids = [x['token_id'] for x in samples]
    lps = [x.get('logp') for x in samples]
    pieces = token_pieces(ids) if ids else []
    tail = [bool(advantage is not None and advantage > 0 and lp is not None and lp < tail_floor) for lp in lps]
    second = [False] * len(ids)
    close = special_id('</think>')
    if close is not None and advantage is not None and advantage > 0:
        import numpy as np
        from natlang_neuralese.train.reply_format import second_close_mask, SECOND_CLOSE_STOPS
        stops = {special_id(t) for t in SECOND_CLOSE_STOPS} - {None}
        m = second_close_mask(np.array(ids), close, stops)
        if m is not None:
            second = [bool(v == 0) for v in m]
    return {'ids': ids, 'pieces': pieces, 'logp': lps, 'tail_masked': tail, 'second_close_masked': second,
            'roles': [x.get('input_role') for x in samples], 'advantage': advantage,
            'weight_rule': 'policy weight = min(rho, 2) x A; rho = exp(learner logp - actor logp) is computed by the learner '
                           'and not stored, so trained weights show A and the masks only (tail floor -7 and second </think> apply when A > 0).'}


# ---------- GRPO runs ----------

def run_dir(run):
    d = os.path.join(OPS, run)
    if not os.path.isfile(d + '/train.jsonl'):
        raise QueryError(f'no GRPO run {run}', 'not_found')
    return d


def run_args(d):
    rec = None
    for rec in jsonl(d + '/runs.jsonl'):
        pass
    argv = (rec or {}).get('argv') or []
    out = {}
    for i, a in enumerate(argv):
        if a.startswith('--'):
            out[a[2:]] = argv[i + 1] if i + 1 < len(argv) and not argv[i + 1].startswith('--') else True
    return out, rec


def rollout_archive(d):
    args, _ = run_args(d)
    ck = args.get('checkpoint-dir') or args.get('external') or ''
    cands = [os.path.join(ck, 'rollouts')] if ck else []
    from natlang_neuralese.common.paths import root
    name = os.path.relpath(d, OPS)                     # grpo-v4/run2 -> <data_hdd>/natlang-grpo-v4/run2/rollouts
    parts = name.split('/')
    cands.append(f'{root("data_hdd")}/natlang-{parts[0]}/{"/".join(parts[1:])}/rollouts')
    cands.append(d + '/rollouts')
    for c in cands:
        if os.path.isdir(c):
            return c
    return None


def train_steps(d):
    return [r for r in jsonl(d + '/train.jsonl') if 'step' in r and 'event' not in r]


SEL_KEYS = ('groups', 'groups_with_spread', 'spread_fraction', 'dropped_stale_groups', 'staleness', 'episodes', 'reward_mean',
            'episodes_trained', 'sequences', 'loss_tokens', 'decisions', 'lowering', 'dynamic_sampling', 'memetic_groups',
            'replay_corpora', 'sd', 'teacher', 'reasoning_rl', 'tail_logp_floor', 'tail_masked_fraction', 'second_close_masked',
            'actor_version', 'lr', 'kl_mean', 'grad_norm', 'grad_spike', 'behaviour_shift', 'reasoning_shift', 'elapsed_s')


def group_rows(d):
    """One row per group: task, family, rewards, spread, actor versions (cached by groups/ mtime)."""
    cp = _cache_path('groups', d)
    gd = d + '/groups'
    s = str(int(os.path.getmtime(gd))) if os.path.isdir(gd) else '0'
    cached = read_json(cp)
    if cached and cached.get('sig') == s:
        return cached['rows']
    rows = []
    for g in sorted(os.listdir(gd)) if os.path.isdir(gd) else []:
        rw = jsonl(f'{gd}/{g}/rewards.jsonl')
        vers = read_json(f'{gd}/{g}/versions.json')
        if not rw:
            continue
        rewards = [r.get('reward') for r in rw]
        outs = {}
        for r in rw:
            outs[r.get('outcome')] = outs.get(r.get('outcome'), 0) + 1
        vs = sorted(set(vers.values())) if isinstance(vers, dict) else (sorted(set(vers)) if isinstance(vers, list) else [])
        rows.append({'group': g, 'task_id': rw[0].get('task_id'), 'family': rw[0].get('family'), 'n': len(rw),
                     'mean': round(statistics.fmean([x for x in rewards if isinstance(x, (int, float))] or [0]), 3),
                     'spread': len({x for x in rewards}) > 1, 'outcomes': outs, 'versions': vs,
                     'retired': os.path.exists(f'{gd}/{g}/retired'), 'memetic': any(r.get('jolt_role') for r in rw) or None})
    tmp = cp + '.tmp'
    json.dump({'sig': s, 'rows': rows}, open(tmp, 'w'), separators=(',', ':'))
    os.replace(tmp, cp)
    return rows


def group_advantages(rewards):
    """(r - mean) / (std + 1e-6) of one group (grpo.group_advantages); None for a group without spread."""
    vals = [float(x) for x in rewards]
    if len(vals) < 2:
        return [None] * len(vals)
    m = statistics.fmean(vals)
    sd = statistics.pstdev(vals)
    if sd < 1e-6:
        return [0.0] * len(vals)
    return [round((v - m) / (sd + 1e-6), 4) for v in vals]


def group_episodes(d, group):
    arch = rollout_archive(d)
    g = group if str(group).startswith('g') else f'g{int(group):06d}'
    p = f'{arch}/{g}.rollouts.jsonl.zst' if arch else None
    if not p or not os.path.exists(p):
        raise QueryError(f'rollouts of {g} are not archived (lowered groups keep rewards only)', 'not_found')
    eps = jsonl(p)
    args, _ = run_args(d)
    graded = args.get('reward') == 'graded'
    rewards = [((e.get('reward') or {}).get('graded' if graded else 'quality') or 0.0) for e in eps]
    return eps, group_advantages(rewards), ('graded' if graded else 'binary')


def load_step_chains(d, step):
    p = f'{d}/step-chains/step-{int(step):05d}.pkl.zst'
    if not os.path.exists(p):
        return None
    import zstandard
    with open(p, 'rb') as f:
        return pickle.loads(zstandard.ZstdDecompressor().stream_reader(f).read())


def _num(v):
    """A scalar from a float, a numpy scalar or a per-token array (its mean)."""
    if v is None:
        return None
    try:
        if hasattr(v, 'shape') and getattr(v, 'shape', ()) != ():
            return round(float(v.mean()), 4) if v.size else None
        return round(float(v), 4)
    except (TypeError, ValueError):
        return None


def chain_summary(c, i):
    segs = c.get('segs') or []
    ids = c.get('ids')
    rnd = c.get('round')
    return {'i': i, 'role': c.get('role'), 'task': c.get('task'),
            'round': int(rnd) if isinstance(rnd, (int,)) or hasattr(rnd, 'dtype') else (rnd if isinstance(rnd, str) else None),
            'episode': _num(c.get('episode')), 'advantage': _num(c.get('advantage')),
            'per_token_advantage': hasattr(c.get('advantage'), 'shape') and getattr(c.get('advantage'), 'shape', ()) != (),
            'tokens': int(len(ids)) if ids is not None else 0, 'trained_tokens': int(sum(int(s[1]) for s in segs)), 'segments': len(segs),
            'replay_corpus': c.get('replay_corpus'), 'token_scale': _num(c.get('token_scale'))}


def chain_view(c, i):
    """Token-level view of one learner chain: every token, whether it is trained (inside a seg), its seg's call, logp."""
    ids = [int(x) for x in c.get('ids')]
    trained = [None] * len(ids)
    segs = []
    for k, s in enumerate(c.get('segs') or []):
        start, length, row = int(s[0]), int(s[1]), s[2] if len(s) > 2 else {}
        lp = row.get('logp') if isinstance(row, dict) else None
        w = row.get('weight') if isinstance(row, dict) else None
        lp = None if lp is None or not hasattr(lp, '__len__') else lp
        w = None if w is None or not hasattr(w, '__len__') else w
        for j in range(length):
            if start + j < len(ids):
                trained[start + j] = [k, None if lp is None else round(float(lp[j]), 3), None if w is None else round(float(w[j]), 3)]
        segs.append({'k': k, 'start': start, 'length': length, 'invocation': row.get('invocation') if isinstance(row, dict) else None,
                     'index': row.get('index') if isinstance(row, dict) else None})
    if len(ids) > 40000:
        raise QueryError(f'chain has {len(ids):,} tokens; too long to show', 'too_large')
    return {'kind': 'chain', **chain_summary(c, i), 'pieces': token_pieces(ids), 'trained': trained, 'segs': segs,
            'imports': [[int(a), str(b)] for a, b in (c.get('imports') or [])][:200],
            'unknown': ['rho (learner/actor ratio) and the final per-token policy weight are computed in the learner and not stored.']}


# ---------- handlers ----------

def _page(rows, inp):
    off, lim = int(inp.get('offset') or 0), min(int(inp.get('limit') or 100), 500)
    return {'total': len(rows), 'offset': off, 'rows': rows[off:off + lim]}


def _facets(rows, keys=('family',)):
    out = {}
    for k in keys:
        c = {}
        for r in rows:
            v = r.get(k)
            if isinstance(v, list):
                for x in v:
                    c[x] = c.get(x, 0) + 1
            elif v is not None:
                c[v] = c.get(v, 0) + 1
        out[k] = sorted(c.items(), key=lambda kv: -kv[1])[:60]
    return out


def _filter(rows, inp):
    for k, v in (inp.get('filter') or {}).items():
        if v in (None, ''):
            continue
        rows = [r for r in rows if ((v in r.get(k)) if isinstance(r.get(k), list) else str(r.get(k)).lower() == str(v).lower())]
    sort = inp.get('sort')
    if sort:
        key = sort.lstrip('-')
        rows = sorted(rows, key=lambda r: (r.get(key) is None, r.get(key) if isinstance(r.get(key), (int, float, str)) else str(r.get(key))),
                      reverse=sort.startswith('-'))
    return rows


def jsonl_items(path, summarize, inp, facets=('family',)):
    idx = line_index(path, summarize)
    rows = [{**s, '_loc': loc} for loc, s in idx]
    q = (inp.get('q') or '').strip()
    if q:
        locs = set(grep_locs(path, q))
        rows = [r for r in rows if r['_loc'] in locs]
    fac = _facets(rows, facets)
    rows = _filter(rows, inp)
    return {**_page(rows, inp), 'facets': fac, 'searched': bool(q)}


def jsonl_item(path, summarize, item_id, key='id'):
    for loc, s in line_index(path, summarize):
        if str(s.get(key)) == str(item_id) or str(loc) == str(item_id):
            return read_at(path, loc), loc
    raise QueryError(f'{item_id} not found in {os.path.basename(path)}', 'not_found')


def sft_summary(r):
    t = r.get('target') or {}
    return {'id': r.get('id'), 'source': r.get('source'), 'task_id': r.get('task_id'), 'family': r.get('family'),
            'teacher': r.get('teacher'), 'call': r.get('call'), 'role': r.get('invocation_role'), 'episode': r.get('episode'),
            'target_tokens': r.get('target_tokens'), 'prompt_tokens': r.get('prompt_tokens'), 'loss_from': r.get('loss_from'),
            'tool': ((t.get('tool_calls') or [{}])[0].get('function') or {}).get('name') if t.get('tool_calls') else None}


def demo_summary(r):
    return {'id': r.get('key'), 'source': r.get('source'), 'run': r.get('run'), 'task_id': r.get('task_id'),
            'family': r.get('family'), 'teacher': (r.get('teacher') or {}).get('model_id'), 'calls': len(r.get('turns') or []),
            'base_rank': r.get('base_rank'), 'root_mode': r.get('root_mode')}


def rollout_summary(r):
    rw = r.get('reward') or {}
    return {'id': f"g{r.get('group')}:{r.get('sample')}", 'task_id': r.get('task_id'), 'family': r.get('family'),
            'outcome': r.get('outcome'), 'reward': rw.get('quality'), 'calls': len(r.get('turns') or [])}


def manifest_summary(r):
    eps = r.get('episodes') or []
    return {'id': r.get('task_id'), 'index': r.get('index'), 'family': r.get('family'), 'base_rank': r.get('base_rank'),
            'teacher': (r.get('teacher') or {}).get('model_id'), 'episodes': len(eps),
            'successes': sum(1 for e in eps if e.get('outcome') == 'task_success'),
            'outcome': (eps[0] if eps else {}).get('outcome'), 'seconds': r.get('seconds')}


def raw_summary(r):
    keys = [k for k in r][:8] if isinstance(r, dict) else []
    rid = next((r.get(k) for k in ('id', 'key', 'task_id', 'name') if isinstance(r, dict) and r.get(k) is not None), None)
    return {'id': rid, 'family': r.get('family') if isinstance(r, dict) else None, 'source': r.get('source') if isinstance(r, dict) else None, 'keys': keys}


def errata_summary(r):
    return {'id': r.get('id'), 'family': r.get('family'), 'reason': (r.get('reason') or '')[:200], 'source': r.get('source'), 'type': r.get('type')}


SUMMARIZERS = {'raw': raw_summary, 'errata': errata_summary, 'task': task_summary, 'sft_row': sft_summary, 'demo': demo_summary, 'rollout': rollout_summary, 'manifest': manifest_summary}


def _ds(inp):
    from data_catalog import dataset
    return dataset(inp.get('ds'))


@handler('data.items')
def data_items(inp):
    """Items of one dataset part: {ds, part?, q?, filter?, sort?, offset?, limit?}."""
    ds = _ds(inp)
    r = ds['reader']
    part = inp.get('part') or (ds.get('parts') or [{}])[0].get('id')
    if r == 'registry':
        from data_catalog import data_registry
        return data_registry({k: v for k, v in inp.items() if k != 'ds'})
    if r == 'info':
        raise QueryError(f'{ds["id"]} has no row reader yet (format: {ds.get("format") or ds.get("kind")}); see its provenance and files', 'unsupported')
    if r == 'grpo-run':
        d = run_dir(ds['run'])
        if part == 'steps':
            rows = []
            for s in train_steps(d):
                low = s.get('lowering') or {}
                rows.append({'id': s['step'], 'step': s['step'], 'groups': s.get('groups'), 'groups_with_spread': s.get('groups_with_spread'),
                             'episodes': s.get('episodes'), 'reward_mean': s.get('reward_mean'), 'dropped_stale': s.get('dropped_stale_groups'),
                             'unscored': low.get('episodes_unscored'), 'truncated': low.get('episodes_truncated'),
                             'budget_stopped': low.get('episodes_budget_stopped'), 'tail_masked': s.get('tail_masked_fraction'),
                             'second_close': s.get('second_close_masked'), 'loss_tokens': s.get('loss_tokens'),
                             'flags': [f for f in ('grad_spike', 'behaviour_shift', 'reasoning_shift') if s.get(f)],
                             'chains_kept': os.path.exists(f'{d}/step-chains/step-{int(s["step"]):05d}.pkl.zst')})
            rows.reverse()
            return {**_page(_filter(rows, inp), inp), 'facets': {}}
        if part == 'groups':
            rows = group_rows(d)
            q = (inp.get('q') or '').lower()
            if q:
                rows = [g for g in rows if q in (g['task_id'] or '').lower() or q in (g['family'] or '').lower()]
            fac = _facets(rows, ('family',))
            return {**_page(_filter(rows, inp), inp), 'facets': fac}
        if part == 'held':
            out = []
            for p in sorted(glob.glob(d + '/held/step-*/result.json')):
                res = read_json(p, {})
                cmp_ = read_json(os.path.dirname(p) + '/compare-vs-step0.json', {})
                prim = cmp_.get('primary') or {}
                out.append({'id': os.path.basename(os.path.dirname(p)), 'success_rate': res.get('success_rate'), 'ran': res.get('ran'),
                            'complete': res.get('complete'), 'delta_pts': round(100 * prim['delta'], 2) if 'delta' in prim else None,
                            'p': prim.get('permutation_p'), 'samples': len(res.get('samples') or [])})
            return {**_page(out[::-1], inp), 'facets': {}}
    if r in ('jsonl', 'sft-mix', 'teacher-gen', 'teacher-demos', 'held-set', 'held-eval'):
        p = (ds.get('parts_map') or {}).get(part) or ds.get('path')
        kind = (ds.get('part_kind') or {}).get(part) or ds.get('item_kind') or 'task'
        if r == 'held-eval' and part == 'samples':
            res = read_json(ds['path'] + '/result.json', {})
            rows = [{'id': f"{s.get('task_id')}#{s.get('sample')}", **{k: s.get(k) for k in ('task_id', 'family', 'level', 'sample', 'outcome', 'reward', 'completion_tokens', 'model_calls', 'finish_status')}} for s in res.get('samples') or []]
            q = (inp.get('q') or '').lower()
            if q:
                rows = [x for x in rows if q in json.dumps(x).lower()]
            return {**_page(_filter(rows, inp), inp), 'facets': _facets(rows, ('family', 'outcome'))}
        summ = SUMMARIZERS.get(kind)
        if not summ or not p or not os.path.exists(p):
            raise QueryError(f'{ds["id"]}: part {part} has no readable file', 'not_found')
        return jsonl_items(p, summ, inp, facets=('family', 'source', 'outcome', 'teacher') if kind != 'task' else ('family', 'source', 'constructs'))
    raise QueryError(f'{ds["id"]}: reader {r} lists no items', 'unsupported')


def _task_where(task_id):
    """Where a task id appears: the pools that hold it, held status, errata, base rate."""
    from data_catalog import task_pools
    where = {'pools': [], 'held': is_held(task_id), 'errata': [], 'base_rate': None}
    for pool in task_pools():
        if not os.path.exists(pool['path']):
            continue
        for loc, s in line_index(pool['path'], task_summary):
            if s.get('id') == task_id:
                where['pools'].append({'ds': pool['id'], 'path': pool['path']})
                break
    for p in glob.glob(OPS + '/pool-errata/*.jsonl'):
        for r in jsonl(p):
            if r.get('id') == task_id:
                where['errata'].append({'file': os.path.basename(p), **{k: r.get(k) for k in ('reason', 'source', 'type')}})
    br = _base_rates().get(task_id)
    if br:
        where['base_rate'] = br
    return where


_BR = {}


def _base_rates():
    p = OPS + '/exp1-rft/base-pass-rates.json'
    try:
        s = sig(p)
    except OSError:
        return {}
    if _BR.get('sig') != s:
        _BR.update(sig=s, v=read_json(p, {}))
    return _BR['v']


@handler('data.item')
def data_item(inp):
    """One item: {ds, id, part?}."""
    ds = _ds(inp)
    r, iid = ds['reader'], inp.get('id')
    part = inp.get('part') or (ds.get('parts') or [{}])[0].get('id')
    if r == 'registry':
        from data_catalog import data_registry
        return data_registry({'id': iid})
    if r == 'grpo-run':
        d = run_dir(ds['run'])
        if part == 'groups' or str(iid).startswith('g'):
            g, _, sample = str(iid).partition(':')
            eps, adv, mode = group_episodes(d, g)
            if sample == '':
                return {'kind': 'group', 'id': g, 'reward_mode': mode, 'task_id': eps[0].get('task_id') if eps else None,
                        'episodes': [{**rollout_summary(e), 'advantage': a, 'sample': e.get('sample')} for e, a in zip(eps, adv)],
                        'task_ref': {'task_id': eps[0].get('task_id')} if eps else None}
            e = next((x for x in eps if str(x.get('sample')) == sample), None)
            if not e:
                raise QueryError(f'{iid}: no such sample', 'not_found')
            a = adv[eps.index(e)]
            return {**normalize_rollout(e, advantage=a, role='student', unknown=[
                f'Advantage reconstructed from the group\'s {mode} rewards ((r - mean) / (std + 1e-6)); the learner may have '
                'dropped this group (staleness > 2, no spread) or episodes (unscored, truncated): see the step\'s lowering counts.',
                'Which step trained this group is recorded only while its step-chains file is kept (last 4 steps).']),
                'id': f'{g}:{sample}', 'group_label': g, 'spread': len({x for x in adv if x is not None}) > 1}
        if part == 'steps':
            return step_detail(d, int(iid))
    if r == 'held-eval':
        tid, _, sample = str(iid).partition('#')
        return held_trace(ds['path'], tid, sample)
    if r == 'teacher-gen' and part == 'tasks':
        rec, loc = jsonl_item(ds['parts_map']['tasks'], manifest_summary, iid)
        tdir = f"{ds['path']}/episodes/t{int(rec.get('index') or 0):06d}"
        eps = jsonl(tdir + '/rollouts.jsonl')
        return {'kind': 'teacher_task', 'id': iid, 'manifest': rec, 'episodes': [normalize_rollout(e) for e in eps[:4]],
                'task_ref': {'task_id': iid}, 'dir': tdir}
    p = (ds.get('parts_map') or {}).get(part) or ds.get('path')
    kind = (ds.get('part_kind') or {}).get(part) or ds.get('item_kind') or 'task'
    summ = SUMMARIZERS.get(kind)
    if not summ:
        raise QueryError(f'{ds["id"]}: no item reader', 'unsupported')
    rec, loc = jsonl_item(p, summ, iid)
    digest = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()[:16]
    if kind == 'task':
        out = normalize_task(rec, _task_where(rec.get('id')))
    elif kind == 'sft_row':
        out = normalize_sft_row(rec)
    elif kind in ('demo',):
        out = normalize_rollout({**rec, 'group': None}, role='teacher')
        out.update(id=rec.get('key'), teacher=rec.get('teacher'), task_id=rec.get('task_id'), family=rec.get('family'),
                   task=normalize_task(rec['task']['row']) if (rec.get('task') or {}).get('row') else None)
    elif kind == 'rollout':
        out = normalize_rollout(rec)
    else:
        out = {'kind': 'raw', 'value': rec}
    out['ref'] = f'{p}@{loc}'
    out['digest'] = digest
    return out


def held_trace(path, task_id, sample):
    res = read_json(path + '/result.json', {})
    q = read_json(path + '/sealed-queue.jsonl') or jsonl(path + '/sealed-queue.jsonl')
    row = None
    for i, t in enumerate(sorted({s['task_id'] for s in res.get('samples') or []}, key=lambda x: x)):
        pass
    # rows are numbered by the eval's task order (packet.jsonl)
    order = [r.get('id') or (r.get('row') or {}).get('id') for r in jsonl(path + '/packet.jsonl')]
    if task_id in order:
        row = order.index(task_id)
    if row is None:
        raise QueryError(f'{task_id} not in this held eval', 'not_found')
    p = f'{path}/row-{row:03d}/sample-{int(sample or 0):02d}/results.jsonl'
    recs = jsonl(p, 1)
    if not recs:
        raise QueryError(f'no transcript at {p}', 'not_found')
    out = normalize_native_trace(recs[0])
    out['ref'] = p
    return out


def step_detail(d, step):
    rec = next((s for s in train_steps(d) if s.get('step') == step), None)
    if not rec:
        raise QueryError(f'no step {step}', 'not_found')
    sel = {k: rec.get(k) for k in SEL_KEYS if k in rec}
    sc = load_step_chains(d, step)
    chains = [chain_summary(c, i) for i, c in enumerate(sc['chains'])] if sc else None
    by_role = {}
    for c in chains or []:
        b = by_role.setdefault(c['role'] or '?', {'chains': 0, 'trained_tokens': 0})
        b['chains'] += 1
        b['trained_tokens'] += c['trained_tokens']
    groups = sorted({c['round'] for c in chains or [] if isinstance(c['round'], int) and c['role'] == 'student'})
    decisions = jsonl(f'{d}/decisions/step-{step:05d}.jsonl', 500)
    return {'kind': 'step', 'id': step, 'selection': sel, 'chains': chains, 'by_role': by_role,
            'groups': [f'g{g:06d}' for g in groups], 'decisions': [{k: x.get(k) for k in ('group', 'task_id', 'stale', 'advantage', 'chosen', 'options', 'log_rho')} for x in decisions[:200]],
            'unknown': ([] if sc else ['This step\'s learner batch (step-chains) is no longer kept: only the last 4 steps are saved. '
                                       'Groups trained here can be approximated from groups/*/versions.json.']) +
                       ['Whether a group came from the pass band or exploration is decided in memory and not recorded.']}


@handler('data.chain')
def data_chain(inp):
    from data_catalog import dataset
    ds = dataset(inp.get('ds'))
    d = run_dir(ds['run'])
    sc = load_step_chains(d, int(inp['step']))
    if not sc:
        raise QueryError('step-chains not kept for this step', 'not_found')
    i = int(inp['chain'])
    return chain_view(sc['chains'][i], i)


@handler('data.tokens')
def data_tokens(inp):
    """Token view of one sampled call of a GRPO episode: {ds, id: 'g000123:3', call}."""
    from data_catalog import dataset
    ds = dataset(inp.get('ds'))
    d = run_dir(ds['run'])
    g, _, sample = str(inp['id']).partition(':')
    eps, adv, _ = group_episodes(d, g)
    e = next((x for x in eps if str(x.get('sample')) == sample), None)
    if not e:
        raise QueryError('no such episode', 'not_found')
    turn = (e.get('turns') or [])[int(inp['call'])]
    args, _ = run_args(d)
    return sampled_tokens(turn, adv[eps.index(e)], float(args.get('tail-logp-floor', -7)))


@handler('data.task_refs')
def data_task_refs(inp):
    """Everywhere a task appears: pools, held, errata, base rate, GRPO groups, teacher campaigns, SFT mixes."""
    tid = inp.get('task_id')
    where = _task_where(tid)
    from data_catalog import grpo_runs, teacher_campaigns, sft_mixes
    where['groups'] = []
    for run in grpo_runs():
        for g in group_rows(os.path.join(OPS, run['run'])):
            if g['task_id'] == tid:
                where['groups'].append({'ds': run['id'], 'group': g['group'], 'mean': g['mean'], 'n': g['n'], 'spread': g['spread']})
    where['teacher'] = []
    for c in teacher_campaigns():
        p = c['parts_map'].get('tasks')
        if p and os.path.exists(p):
            for loc, s in line_index(p, manifest_summary):
                if s.get('id') == tid:
                    where['teacher'].append({'ds': c['id'], **s})
    where['sft'] = []
    for m in sft_mixes():
        p = m['parts_map'].get('rows')
        if p and os.path.exists(p):
            n = sum(1 for loc, s in line_index(p, sft_summary) if s.get('task_id') == tid)
            if n:
                where['sft'].append({'ds': m['id'], 'rows': n})
    return where
