#!/usr/bin/env python3
"""Collection health: per task family and per construct tag, what the episodes a collection already scored say about
the system (DECISIONS 2026-11-23 "collection health"). Collection scores every episode, so pass rates and failure
reasons cost no extra compute; most defects found on 2026-11-23 (std imports, memetic bindings, prompts) showed in them
and not in held eval.

    collection_health.py report --collect DIR [--collect DIR ...] [--previous DIR ...] [--earlier DIR ...]
                                --out health.json [--cache-dir D] [--label L]

A collection DIR is an ``expert_iteration collect`` WORK (``groups/gNNNNNN``, ``collect.log``). Per episode it reads
the group's ``rewards.jsonl`` (outcome), the sample's ``results.jsonl`` (action diagnostics, scope failures, host
events) and ``jobs/*.error.json`` (episode-level stops); per group the collection log's ``group abandoned`` and
``group scoring failed`` events. Guided attempts (``jolt_role``) are counted apart and never enter own pass rates.
Scored groups are immutable, so their summaries are cached (``--cache-dir``).

Flags (never a halt; the driver writes them to its drive log and the ops alerts log the heartbeat reads):
  - ``family_collapse`` / ``construct_collapse``: 0% own pass with failure reasons, where the previous or earlier
    collections had passes;
  - ``family_never_passes`` / ``construct_never_passes``: absent from every compared collection and never passes;
  - ``error_share_jump``: the share of own episodes carrying a failure reason rose sharply against the baseline;
  - ``infrastructure``: unscored, crashed or abandoned episodes above ``--infra-rate``;
  - ``no_episodes``: tasks whose group yielded no scored episode (import, loader, binding and wall-clock failures).
A flag that touches a semantic-FP construct (``SEMANTIC_FP_CONSTRUCTS``) carries ``sfp: true``.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = 'natlang.ei-collection-health/1'
CACHE_SCHEMA = 'natlang.ei-collection-health-group/1'
SUCCESS = 'task_success'
UNSCORED = {'unresolved', 'infrastructure_error'}
BUDGET = {'budget_exhaustion'}
BUDGET_CODES = {'NATLANG_MODEL_REQUEST_BUDGET'}
# thresholds (report CLI flags override)
MIN_EPISODES = 8            # own scored episodes before a family/construct may be flagged
MIN_JUMP_EPISODES = 10      # episodes carrying a reason before its share may be flagged
JUMP_RATIO = 2.0            # share now >= ratio x baseline share ...
JUMP_ABS = 0.05             # ... and >= baseline + this
INFRA_RATE = 0.05           # unscored/crashed/abandoned share of started episodes


def semantic_fp_constructs():
    """The semantic-FP construct tags (the authored-task source's definition)."""
    from scripts.rl_tasks.sources.authored import SEMANTIC_FP_CONSTRUCTS
    return SEMANTIC_FP_CONSTRUCTS


def _jsonl(path):
    try:
        with open(path) as handle:
            return [json.loads(line) for line in handle if line.strip()]
    except (OSError, ValueError):
        return []


def _last_jsonl(path):
    """The last complete JSON line of ``path`` (a retried sample appends)."""
    last = None
    try:
        with open(path) as handle:
            for line in handle:
                if line.strip():
                    last = line
    except OSError:
        return None
    try:
        return json.loads(last) if last else None
    except ValueError:
        return None


def _message_reason(message):
    """A short stable reason for a runtime message: its error class, else its first words with numbers folded."""
    text = str(message or '').strip()
    if text.startswith('Traceback'):            # a Python traceback: its last line names the error
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return 'python:' + _message_reason(lines[-1]) if len(lines) > 1 else 'python:traceback'
    match = re.match(r'(?:Uncaught\s+)?([A-Z][A-Za-z]*Error)\b', text)
    if match:
        return match.group(1)
    match = re.match(r'\d+:\d+\s+([a-z][a-z0-9-]+):', text)
    if match:
        return match.group(1)
    text = re.sub(r'\d+', 'N', text.splitlines()[0] if text else '')
    return ' '.join(text.split()[:6])[:60] or 'unknown'


def episode_reasons(result):
    """Failure reasons one sample's ``results.jsonl`` record shows: action diagnostics of failed actions
    (``diag:<code>``), runtime scope failures (``runtime:<class or head>``), host events (``host:<operation>``), and a
    non-``done`` status (``status:<status>``)."""
    out = set()
    outcome = (result or {}).get('outcome') or {}
    for action in outcome.get('action_ledger') or []:
        if isinstance(action, dict) and action.get('outcome') in ('error', 'rejected'):
            codes = [d.get('code') if isinstance(d, dict) else d for d in action.get('diagnostics') or []]
            codes = [c for c in codes if c]
            out |= {f'diag:{c}' for c in codes}
            if not codes and action.get('outcome') == 'error':
                out.add(f'runtime:{_message_reason(action.get("result_text"))}')
    for failure in outcome.get('scope_failures') or []:
        if isinstance(failure, dict) and failure.get('failure_kind') in ('runtime', 'boundary'):
            out.add(f'runtime:{_message_reason(failure.get("message"))}')
    for event in outcome.get('host_events') or []:
        op = ((event or {}).get('event') or {}).get('operation') if isinstance(event, dict) else None
        if op:
            out.add(f'host:{op}')
    status = outcome.get('status')
    if status and status != 'done':
        out.add(f'status:{status}')
    return out


def _category(outcome, reasons, job_codes):
    """One category per episode: success, unscored, crash, budget_stop, runtime_error, wrong_answer."""
    if outcome == SUCCESS:
        return 'success'
    if outcome in UNSCORED or outcome is None:
        return 'unscored'
    if job_codes - BUDGET_CODES:
        return 'crash'
    if outcome in BUDGET or job_codes & BUDGET_CODES:
        return 'budget_stop'
    if any(r.startswith(('runtime:', 'host:')) for r in reasons):
        return 'runtime_error'
    return 'wrong_answer'


def group_summary(gdir):
    """Summary of one group directory: task, family, construct tags, state (``scored``, ``pending``) and its
    episodes [{sample, outcome, category, guided, reasons}]."""
    gdir = Path(gdir)
    case = None
    try:
        with open(gdir / 'cases.jsonl') as handle:
            case = json.loads(handle.readline())
    except (OSError, ValueError):
        pass
    case = case or {}
    external = ((case.get('curriculum') or {}).get('external') or {})
    summary = {'group': gdir.name, 'task_id': case.get('id'), 'family': case.get('family') or 'unknown',
               'constructs': sorted(set(external.get('constructs') or [])), 'episodes': []}
    rewards = _jsonl(gdir / 'rewards.jsonl')
    summary['state'] = 'scored' if rewards else 'pending'
    for row in rewards:
        summary['task_id'] = summary['task_id'] or row.get('task_id')
        if row.get('family'):
            summary['family'] = row['family']
        sample = gdir / 'row-000' / f'sample-{int(row.get("sample", 0)):02d}'
        reasons = episode_reasons(_last_jsonl(sample / 'results.jsonl'))
        job_codes = set()
        for path in sorted((sample / 'jobs').glob('*.error.json')) if (sample / 'jobs').is_dir() else ():
            try:
                code = json.loads(path.read_text()).get('code')
            except (OSError, ValueError):
                code = 'unreadable-error-record'
            job_codes.add(code or 'uncoded-error')
        reasons |= {f'job:{c}' for c in job_codes}
        outcome, finish = row.get('outcome'), row.get('finish_status')
        if outcome != SUCCESS:
            reasons.add(f'outcome:{outcome}')
            if finish and finish not in ('complete',):
                reasons.add(f'finish:{finish}')
        summary['episodes'].append({
            'sample': row.get('sample'), 'outcome': outcome, 'guided': bool(row.get('jolt_role')),
            'category': _category(outcome, reasons, job_codes),
            'reasons': sorted(reasons) if outcome != SUCCESS else []})
    return summary


def log_events(work):
    """Group-level collector events of ``collect.log``: abandoned groups, failed scorings, dropped endpoints."""
    out = {'abandoned': collections.Counter(), 'scoring_failed': collections.Counter(), 'endpoint_dropped': 0}
    try:
        handle = open(Path(work) / 'collect.log', errors='replace')
    except OSError:
        return out
    with handle:
        for line in handle:
            if '"group abandoned"' not in line and '"group scoring failed"' not in line and '"endpoint dropped"' not in line:
                continue
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if event.get('event') == 'group abandoned':
                out['abandoned'][event.get('group')] += 1
            elif event.get('event') == 'group scoring failed':
                out['scoring_failed'][event.get('group')] += 1
            elif event.get('event') == 'endpoint dropped':
                out['endpoint_dropped'] += 1
    return out


def collection_summary(work, cache_dir=None):
    """Every group of one collection (scored groups from ``cache_dir`` when cached) and its log events."""
    work = Path(work)
    cache, cache_path = {}, None
    if cache_dir:
        digest = hashlib.sha256(str(work.resolve()).encode()).hexdigest()[:12]
        cache_path = Path(cache_dir) / f'{work.name}-{digest}.json'
        try:
            stored = json.loads(cache_path.read_text())
            if stored.get('schema') == CACHE_SCHEMA:
                cache = stored['groups']
        except (OSError, ValueError, KeyError):
            cache = {}
    groups, fresh = [], False
    for gdir in sorted(work.glob('groups/g*')):
        cached = cache.get(gdir.name)
        if cached is None:
            cached = group_summary(gdir)
            if cached['state'] == 'scored':
                cache[gdir.name], fresh = cached, True
        groups.append(cached)
    if cache_path and fresh:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix('.tmp')
        tmp.write_text(json.dumps({'schema': CACHE_SCHEMA, 'work': str(work), 'groups': cache}))
        tmp.replace(cache_path)
    events = log_events(work)
    for g in groups:      # the collector's own verdict on a group without rewards
        index = int(g['group'][1:]) if g['group'][1:].isdigit() else None
        if g['state'] == 'pending' and index is not None:
            if events['abandoned'].get(index):
                g['state'] = 'abandoned'
            elif events['scoring_failed'].get(index):
                g['state'] = 'scoring_failed'
    started = None
    try:
        started = json.loads((work / 'runs.jsonl').open().readline()).get('started')
    except (OSError, ValueError):
        pass
    return {'work': str(work), 'started': started, 'groups': groups, 'endpoint_dropped': events['endpoint_dropped']}


def _bucket():
    return {'groups': 0, 'tasks': set(), 'episodes': 0, 'scored': 0, 'successes': 0, 'guided': 0,
            'guided_successes': 0, 'categories': collections.Counter(), 'reasons': collections.Counter(),
            'groups_without_episodes': 0}


def aggregate(collections_):
    """Per family, per construct tag and overall: own episodes, scored, successes, pass rate, category and reason
    histograms (reasons counted once per failed own episode); tasks whose group yielded no scored episode."""
    families, constructs, total = collections.defaultdict(_bucket), collections.defaultdict(_bucket), _bucket()
    no_episodes = []
    endpoint_dropped = 0
    for coll in collections_:
        endpoint_dropped += coll.get('endpoint_dropped') or 0
        for g in coll['groups']:
            if g['state'] == 'pending':
                continue                    # still running: no verdict either way
            buckets = [families[g['family']], total, *(constructs[c] for c in g['constructs'])]
            own = [e for e in g['episodes'] if not e['guided']]
            scored = [e for e in own if e['category'] != 'unscored']
            for b in buckets:
                b['groups'] += 1
                b['tasks'].add(g['task_id'])
                if not scored:
                    b['groups_without_episodes'] += 1
            if not scored:
                reasons = sorted({r for e in own for r in e['reasons']}) or [f'group:{g["state"]}']
                no_episodes.append({'task_id': g['task_id'], 'family': g['family'], 'constructs': g['constructs'],
                                    'group': g['group'], 'work': coll['work'], 'state': g['state'],
                                    'reasons': reasons[:8]})
            for e in g['episodes']:
                for b in buckets:
                    if e['guided']:
                        b['guided'] += 1
                        b['guided_successes'] += e['category'] == 'success'
                        continue
                    b['episodes'] += 1
                    b['categories'][e['category']] += 1
                    if e['category'] != 'unscored':
                        b['scored'] += 1
                    if e['category'] == 'success':
                        b['successes'] += 1
                    else:
                        b['reasons'].update(e['reasons'])

    def finish(b):
        return {'groups': b['groups'], 'tasks': len(b['tasks']), 'episodes': b['episodes'], 'scored': b['scored'],
                'successes': b['successes'],
                'pass_rate': round(b['successes'] / b['scored'], 4) if b['scored'] else None,
                'guided': b['guided'], 'guided_successes': b['guided_successes'],
                'groups_without_episodes': b['groups_without_episodes'],
                'categories': dict(b['categories'].most_common()), 'reasons': dict(b['reasons'].most_common(25))}
    started = total['episodes'] + sum(1 for c in collections_ for g in c['groups'] if g['state'] in
                                      ('abandoned', 'scoring_failed'))
    infra = total['categories'].get('unscored', 0) + total['categories'].get('crash', 0)
    abandoned = sum(1 for c in collections_ for g in c['groups'] if g['state'] == 'abandoned')
    scoring_failed = sum(1 for c in collections_ for g in c['groups'] if g['state'] == 'scoring_failed')
    return {'overall': finish(total), 'families': {k: finish(v) for k, v in sorted(families.items())},
            'constructs': {k: finish(v) for k, v in sorted(constructs.items())},
            'infrastructure': {'unscored_episodes': total['categories'].get('unscored', 0),
                               'crashed_episodes': total['categories'].get('crash', 0),
                               'abandoned_groups': abandoned, 'scoring_failed_groups': scoring_failed,
                               'endpoint_dropped': endpoint_dropped,
                               'rate': round((infra + abandoned + scoring_failed) / started, 4) if started else None},
            'no_episodes': no_episodes,
            'pending_groups': sum(1 for c in collections_ for g in c['groups'] if g['state'] == 'pending')}


def _merge_rates(*aggregates):
    """{name: (scored, successes)} over several aggregates, for families and constructs."""
    out = {'families': collections.defaultdict(lambda: [0, 0]), 'constructs': collections.defaultdict(lambda: [0, 0])}
    for agg in aggregates:
        if not agg:
            continue
        for kind in out:
            for name, b in agg[kind].items():
                out[kind][name][0] += b['scored']
                out[kind][name][1] += b['successes']
    return out


def flags(current, previous=None, earlier=None, *, sfp=frozenset(), min_episodes=MIN_EPISODES,
          min_jump_episodes=MIN_JUMP_EPISODES, jump_ratio=JUMP_RATIO, jump_abs=JUMP_ABS, infra_rate=INFRA_RATE):
    """The flags of ``current`` (an ``aggregate``) against the previous round's and the lineage's earlier collections."""
    out = []
    history = _merge_rates(previous, earlier)
    have_history = any(history[k] for k in history)
    family_constructs = collections.defaultdict(set)
    for task in current.get('no_episodes') or []:
        family_constructs[task['family']] |= set(task['constructs'])

    for kind, label in (('families', 'family'), ('constructs', 'construct')):
        for name, b in current[kind].items():
            if b['scored'] < min_episodes or b['successes']:
                continue
            failing = b['reasons']
            past = history[kind].get(name)
            is_sfp = name in sfp if kind == 'constructs' else bool(sfp & set(b.get('construct_tags') or ()))
            record = {'scored': b['scored'], 'top_reasons': dict(list(failing.items())[:5]), 'sfp': is_sfp}
            if past and past[1] > 0:
                out.append({'flag': f'{label}_collapse', label: name, **record,
                            'before': {'scored': past[0], 'successes': past[1],
                                       'pass_rate': round(past[1] / past[0], 4) if past[0] else None}})
            elif not past and (have_history or kind == 'constructs' and name in sfp):
                out.append({'flag': f'{label}_never_passes', label: name, **record,
                            'note': 'new: in no compared collection' if have_history else 'no history to compare'})
            elif not have_history and kind == 'families' and any(r.startswith(('runtime:', 'job:', 'host:'))
                                                                  for r in failing):
                out.append({'flag': f'{label}_never_passes', label: name, **record, 'note': 'no history to compare'})

    baseline = previous if previous and previous['overall']['episodes'] else earlier
    episodes = current['overall']['episodes']
    if episodes:
        base_eps = (baseline or {}).get('overall', {}).get('episodes') or 0
        base_reasons = (baseline or {}).get('overall', {}).get('reasons') or {}
        for reason, n in current['overall']['reasons'].items():
            if n < min_jump_episodes:
                continue
            share = n / episodes
            before = (base_reasons.get(reason, 0) / base_eps) if base_eps else None
            if before is None:
                continue
            if share >= jump_ratio * before and share >= before + jump_abs:
                out.append({'flag': 'error_share_jump', 'reason': reason, 'share': round(share, 4),
                            'before': round(before, 4), 'episodes': n, 'sfp': False})

    infra = current['infrastructure']
    if infra['rate'] is not None and infra['rate'] > infra_rate:
        out.append({'flag': 'infrastructure', **infra, 'threshold': infra_rate, 'sfp': False})

    if current.get('no_episodes'):
        by_family = collections.Counter(t['family'] for t in current['no_episodes'])
        sfp_tasks = [t for t in current['no_episodes'] if sfp & set(t['constructs'])]
        reasons = collections.Counter(r for t in current['no_episodes'] for r in t['reasons'])
        out.append({'flag': 'no_episodes', 'tasks': len(current['no_episodes']), 'families': dict(by_family.most_common(10)),
                    'top_reasons': dict(reasons.most_common(6)), 'sfp': bool(sfp_tasks)})
    return out


def _annotate_family_constructs(agg, collections_):
    tags = collections.defaultdict(set)
    for coll in collections_:
        for g in coll['groups']:
            tags[g['family']] |= set(g['constructs'])
    for name, b in agg['families'].items():
        b['construct_tags'] = sorted(tags.get(name, ()))
    return agg


def report(current, previous=(), earlier=(), *, cache_dir=None, label='', sfp=None, **thresholds):
    """The health report of the ``current`` collection WORKs against ``previous`` (the previous round's) and
    ``earlier`` (the lineage's earlier collections)."""
    sfp = semantic_fp_constructs() if sfp is None else frozenset(sfp)
    started = time.time()
    cur_c = [collection_summary(w, cache_dir) for w in current]
    prev_c = [collection_summary(w, cache_dir) for w in previous]
    early_c = [collection_summary(w, cache_dir) for w in earlier]
    cur = _annotate_family_constructs(aggregate(cur_c), cur_c)
    prev = aggregate(prev_c) if prev_c else None
    early = aggregate(early_c) if early_c else None
    found = flags(cur, prev, early, sfp=sfp, **thresholds)
    compared = {}
    for kind in ('families', 'constructs'):
        rows = {}
        for name, b in cur[kind].items():
            rows[name] = {'pass_rate': b['pass_rate'], 'scored': b['scored'],
                          'previous': (prev or {}).get(kind, {}).get(name, {}).get('pass_rate') if prev else None,
                          'earlier': (early or {}).get(kind, {}).get(name, {}).get('pass_rate') if early else None}
        compared[kind] = rows
    return {'schema': SCHEMA, 'label': label, 'time': round(time.time(), 1), 'seconds': round(time.time() - started, 1),
            'collections': {'current': [str(w) for w in current], 'previous': [str(w) for w in previous],
                            'earlier': [str(w) for w in earlier]},
            'current': cur, 'previous': {'overall': prev['overall']} if prev else None,
            'earlier': {'overall': early['overall']} if early else None,
            'compared': compared, 'flags': found, 'sfp_flagged': any(f.get('sfp') for f in found)}


def summary_line(health):
    """One line for the drive log."""
    o = health['current']['overall']
    infra = health['current']['infrastructure']
    names = []
    for f in health['flags']:
        subject = f.get('family') or f.get('construct') or f.get('reason') or ''
        names.append(f'{f["flag"]}{"(" + subject + ")" if subject else ""}{"[sfp]" if f.get("sfp") else ""}')
    rate = f'{o["pass_rate"]:.3f}' if o['pass_rate'] is not None else 'n/a'
    irate = f'{infra["rate"]:.3f}' if infra['rate'] is not None else 'n/a'
    return (f'health {health.get("label", "")}: {o["episodes"]} own episodes, pass {rate}, '
            f'{len(health["current"]["families"])} families, infra {irate}, '
            f'{len(health["current"]["no_episodes"])} tasks without episodes; '
            f'{len(health["flags"])} flags' + (': ' + ', '.join(names[:12]) if names else ''))


def alert_lines(health, prefix):
    """Lines for the ops alerts log (``ops/storage-alerts.log`` format: ISO time, tag, message)."""
    stamp = time.strftime('%Y-%m-%dT%H:%M:%S%z')
    stamp = stamp[:-2] + ':' + stamp[-2:]
    out = []
    for f in health['flags']:
        detail = {k: v for k, v in f.items() if k not in ('flag',)}
        out.append(f'{stamp} COLLECTION HEALTH {prefix}: {f["flag"]} {json.dumps(detail, sort_keys=True)[:600]}')
    return out


def write(health, out):
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    tmp.write_text(json.dumps(health, indent=1, default=sorted) + '\n')
    tmp.replace(out)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('report')
    r.add_argument('--collect', action='append', required=True, help='a collection WORK of the current policy')
    r.add_argument('--previous', action='append', default=[], help="the previous round's collection WORK")
    r.add_argument('--earlier', action='append', default=[], help="the lineage's earlier collection WORKs")
    r.add_argument('--out', required=True)
    r.add_argument('--cache-dir')
    r.add_argument('--label', default='')
    r.add_argument('--min-episodes', type=int, default=MIN_EPISODES)
    r.add_argument('--infra-rate', type=float, default=INFRA_RATE)
    args = parser.parse_args()
    health = report(args.collect, args.previous, args.earlier, cache_dir=args.cache_dir, label=args.label,
                    min_episodes=args.min_episodes, infra_rate=args.infra_rate)
    write(health, args.out)
    print(summary_line(health))
    for f in health['flags']:
        print(json.dumps(f, sort_keys=True))


if __name__ == '__main__':
    main()
