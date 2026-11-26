#!/usr/bin/env python3
"""Engine A: call-level distillation from real model calls (plans/NATLANG_COMPILER.md §8.1, §9.3 ``call_level`` tier).

A site is one model call of one natural-language function: the request the model actually saw (the rendered system and
opening user message of the call, the call's tools, and what followed in its conversation: the host's argument scope,
tool output), keyed by the function's identity (``definition_id`` and signature, from the execution graph's invocation
event when the rollout has one, else the turn's ``function`` and the opening message's signature line).

    call_level_engine.py extract --out DIR [--collect-root ROOT | --collection DIR ...] [--authored-pool POOL ...]
                                 [--work WORK ...] [--kinds decision,readout,entry] [--max-per-site 64] [--workers N]
        sites from aei1's collected rollouts (every outcome: the policy's own answer and its episode's outcome are kept
        with the site), authored-trajectory pools (registry role ``ei-authored-teacher``: authored, replay-verified
        answers as the site's reference) and sft_sources WORK dirs (``all_episodes``). Held tasks and their siblings
        are out (``sft_sources.held_excluded_ids``: ``held_eval.training_exclusion_ids`` of every ``held-eval*`` role
        plus the held closure's source groups and ids). One row per (site, header sha, input sha); later observations of
        the same request add the policy's answer and outcome to it. DIR/sites.jsonl.zst, DIR/extract.json (funnel).
    call_level_engine.py teach --sites DIR --out DIR [--teacher MODEL_ID ...] [--samples 3] [--limit N] ...
        k teacher samples per site through the runtime's own chat client (call_level_teacher.mjs over ts-host
        model/chat-completion.ts) with teacher specs from ops/ei/teachers.json (teacher_gen.make_teacher: endpoint,
        key command, request controls). Resumable (DIR/answers.jsonl, one line per sample), paced (--concurrency,
        --min-interval-ms) and bounded (--max-requests, --max-total-tokens). An answer is verified when the samples agree
        on the parsed answer (a decision label, a readout value or return_result's value; --min-agreement over all k)
        and the value passes the runtime's type check against the call's declared return type where it parses
        (values.ts ``coerce``). A tool action other than return_result has no checker: it is kept, never verified.
        Teacher reasoning is zero-weight context (owner 2026-11-24), the verified answer the target.
        DIR/call-level.jsonl.zst, DIR/teach.json.

Kinds: ``decision`` (a no-tool readout whose opening asks for one of listed JSON values), ``readout`` (another no-tool
JSON reply), ``entry`` (the call's first action: everything after the opening is the host's scope and tool output),
``step`` (a later action, after the policy's own earlier actions; extracted only when asked, --kinds).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

import sft_sources as S  # noqa: E402
from natlang_neuralese.common import data_registry  # noqa: E402
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as sha  # noqa: E402
from natlang_neuralese.common.paths import resolve_str  # noqa: E402

SITE_SCHEMA = 'natlang.call-level-site/1'
ANSWER_SCHEMA = 'natlang.call-level-answer/1'
KINDS = ('decision', 'readout', 'entry', 'step')
DEFAULT_KINDS = ('decision', 'readout', 'entry')
VERIFIABLE = ('decision_label', 'readout_value', 'return_value')
COLLECT_ROOT = ('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'ei', 'aei1', 'collect')
TEACHERS_FILE = ('data_nvme', 'lfm3-rl-dgx-20261029', 'ops', 'ei', 'teachers.json')
CALL_OPENING = 'You are inside this call: '
LABELS = re.compile(r'Reply with exactly one of these JSON values and nothing else: (.*?)\.\s*$', re.S)
MAX_OBSERVATIONS = 16


# ---- the request a call saw --------------------------------------------------------------------------------------------
def header_split(messages):
    """Index past the call's opening: the leading system messages and the first user message (the rendered call)."""
    for i, m in enumerate(messages):
        if m.get('role') == 'user':
            return i + 1
        if m.get('role') != 'system':
            return i
    return len(messages)


def text_of(content):
    if isinstance(content, list):
        return ''.join(p.get('text', '') if isinstance(p, dict) else str(p) for p in content)
    return content or ''


def opening_text(messages):
    n = header_split(messages)
    return text_of(messages[n - 1].get('content')) if n and messages[n - 1].get('role') == 'user' else ''


def return_type(signature):
    """The declared return type of ``f(params): R`` (a rendered signature line) or ``(params) => R`` (an execution-graph
    signature), or None."""
    start = signature.find('(')
    if start < 0:
        return None
    depth, quote = 0, None
    for i in range(start, len(signature)):
        ch = signature[i]
        if quote:
            if ch == quote and signature[i - 1] != '\\':
                quote = None
            continue
        if ch in '"\'`':
            quote = ch
        elif ch in '([{<':
            depth += 1
        elif ch in ')]}>' and not (ch == '>' and signature[i - 1] == '='):
            depth -= 1
            if depth == 0:
                rest = signature[i + 1:].strip()
                for sep in ('=>', ':'):
                    if rest.startswith(sep):
                        return rest[len(sep):].strip() or None
                return None
    return None


def opening_signature(opening):
    """(function name, signature text) from a call's opening line ``You are inside this call: f(...): R``."""
    line = S.signature_line(opening)
    if not line.startswith(CALL_OPENING):
        return None, None
    sig = line[len(CALL_OPENING):].strip()
    return sig.split('(', 1)[0].strip() or None, sig


def declarations(opening):
    """The type declarations a call's opening renders (between its signature line and ``Instructions:``)."""
    body = opening.split('\n', 1)[1] if '\n' in opening else ''
    return body.split('\n\nInstructions:', 1)[0]


def decision_labels(messages):
    last = next((text_of(m.get('content')) for m in reversed(messages) if m.get('role') == 'user'), '')
    match = LABELS.search(last)
    if not match:
        return None
    try:
        labels = json.loads('[' + match.group(1) + ']')
    except ValueError:
        return None
    return labels or None


def native_state(messages):
    """Whether a request carries Neuralese state as non-text content parts (a ``native_state`` import): only a
    native-state server reads those; a hosted teacher refuses the request."""
    return any(isinstance(p, dict) and p.get('type', 'text') != 'text'
               for m in messages if isinstance(m.get('content'), list) for p in m['content'])


def call_kind(turn, n_header):
    if not turn.get('tools'):
        return 'decision' if decision_labels(turn['messages']) else 'readout'
    rest = turn['messages'][n_header:]
    own = [m for m in rest if m.get('role') == 'assistant' and not m.get('natlang_host_generated')]
    return 'step' if own else 'entry'


# ---- answers: the policy's sampled reply and a teacher's, parsed the same way ---------------------------------------------
def strip_fence(text):
    text = text.strip()
    match = re.fullmatch(r'```(?:json)?\s*(.*?)\s*```', text, re.S)
    return match.group(1) if match else text


def normalize_reply(text, calls):
    """(text, [(name, arguments)]) from a recorded target (OpenAI message) or the helper's turn (calls as pairs)."""
    out = []
    for call in calls or ():
        if isinstance(call, dict):
            fn = call.get('function') or {}
            name, args = fn.get('name'), fn.get('arguments')
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    pass
        else:
            name, args = call
        out.append((name, args))
    return text_of(text), out


def parse_answer(kind, text, calls, labels=None):
    """(basis, value) of a reply: ``decision_label``/``readout_value`` for a no-tool call's JSON reply,
    ``return_value`` for return_result with status success, ``return_failure`` for another status, ``action`` for any
    other tool action, ``invalid`` when a no-tool reply is no JSON value (or no listed label)."""
    if kind in ('decision', 'readout'):
        try:
            value = json.loads(strip_fence(text))
        except ValueError:
            return 'invalid', text.strip()
        if kind == 'decision':
            return ('decision_label', value) if labels is None or value in labels else ('invalid', value)
        return 'readout_value', value
    for name, args in calls:
        if name == 'return_result' and isinstance(args, dict):
            if args.get('status', 'success') == 'success':
                return 'return_value', args.get('value')
            return 'return_failure', args
    if calls:
        return 'action', [[n, a] for n, a in calls]
    return 'invalid', text.strip()


def answer_key(basis, value):
    return sha([basis, value])[:24]


# ---- extract -------------------------------------------------------------------------------------------------------------
def graph_invocations(row):
    """call id -> (definition id, signature, argument binding sha) from the rollout's execution graph."""
    out = {}
    for e in row.get('execution_graph') or ():
        if e.get('kind') == 'invocation' and e.get('phase', 'start') == 'start' and e.get('call_id'):
            occ = e.get('native_occurrence') or {}
            out[e['call_id']] = {'definition_id': occ.get('definition_id') or (e.get('definition') or {}).get('id'),
                                 'signature': occ.get('signature') or e.get('signature'),
                                 'argument_binding_sha256': occ.get('argument_binding_sha256')}
    return out


def site_rows(episode, turns, source, invocations=None, *, kinds=DEFAULT_KINDS, reference=None, counts=None):
    """Candidate site rows of one episode's model calls (``turns``: natlang.rollout/1 turns or sft_sources semantic
    turns). ``reference``: the source's answers are authored or verified (``authored``, ``teacher-success``)."""
    counts = counts if counts is not None else collections.Counter()
    invocations = invocations or {}
    functions = {}
    for turn in turns:
        inv = invocations.get(turn.get('invocation_id')) or {}
        functions.setdefault(turn.get('invocation_id'), inv.get('definition_id') or turn.get('definition_id')
                             or (turn.get('function') if turn.get('function') not in (None, 'unknown') else None))
    out = []
    for turn in turns:
        counts['calls'] += 1
        messages = turn.get('messages') or []
        n_header = header_split(messages)
        opening = opening_text(messages)
        if not opening:
            counts['calls_without_opening'] += 1
            continue
        kind = call_kind(turn, n_header)
        counts[f'calls:{kind}'] += 1
        if kind not in kinds:
            continue
        inv = invocations.get(turn.get('invocation_id')) or {}
        name, line_sig = opening_signature(opening)
        definition = functions.get(turn.get('invocation_id')) or name
        signature = inv.get('signature') or line_sig
        if not definition and not signature:   # a learned operator's view site: its whole opening names it
            definition = 'view:' + sha(S.signature_line(opening))[:16]
        labels = decision_labels(messages) if kind == 'decision' else None
        header = messages[:n_header]
        header_sha = sha({'header': [{k: m.get(k) for k in ('role', 'content')} for m in header],
                          'tools': turn.get('tools') or []})
        input_sha = sha(S.visible_key(messages[n_header:]))
        site_id = sha({'definition': definition, 'signature': signature})[:20]
        target = turn.get('target') or {}
        text, calls = normalize_reply(target.get('content'), target.get('tool_calls'))
        basis, value = parse_answer(kind, text, calls, labels)
        parent = turn.get('parent_invocation')
        exact = S.exact_tokens(turn) if turn.get('sampling') else (turn.get('exact') or None)
        decision = turn.get('decision_reasoning')
        if isinstance(decision, dict):
            decision = {k: v for k, v in decision.items() if not k.endswith('_ids')}
        out.append({
            'schema': SITE_SCHEMA, 'row_id': sha([site_id, header_sha, input_sha])[:32], 'site_id': site_id,
            'kind': kind, 'definition_id': definition, 'signature': signature,
            'check': {'returns': return_type(signature or ''), 'declarations': declarations(opening)},
            'labels': labels, 'role': 'root' if not parent else 'child',
            'parent_definition': functions.get(parent) if parent else None,
            'argument_binding_sha256': inv.get('argument_binding_sha256'),
            'header_sha': header_sha, 'input_sha': input_sha, 'header_messages': n_header,
            'messages': messages, 'tools': turn.get('tools') or [], 'native_state': native_state(messages),
            **({'prompt_ids': exact['prompt_ids']} if exact and exact.get('prompt_ids') else {}),
            'task_id': episode['task_id'], 'family': episode.get('family'),
            'source_groups': episode.get('source_groups') or [], 'source': source,
            'policy_answer': {'target': {k: target.get(k) for k in ('role', 'content', 'tool_calls', 'reasoning_content')
                                         if k in target},
                              'basis': basis, 'key': answer_key(basis, value)},
            **({'decision_reasoning': decision} if decision else {}),
            **({'reference': {'source': reference, 'basis': basis, 'key': answer_key(basis, value), 'value': value}}
               if reference else {}),
            'observations': [{'episode': episode['key'], 'outcome': episode.get('outcome'),
                              'success': episode.get('success'), 'policy': episode.get('policy'),
                              'basis': basis, 'key': answer_key(basis, value)}],
        })
    return out


def _held_out_units(excluded):
    global EXCLUDED
    EXCLUDED = excluded


EXCLUDED = frozenset()


def rollout_unit(unit, kinds):
    """Site rows and counts of one aei1 group (natlang.rollout/1 rows: every outcome)."""
    label, name, path, group_dir, archive_dir = unit
    counts, rows = collections.Counter(), []
    cases = S.group_cases(group_dir)
    # a live group compacted (rollouts.jsonl.zst) or archived since the listing
    path = Path(path) if Path(path).exists() else (
        S.group_rollouts(group_dir) or Path(archive_dir) / f'{name}.rollouts.jsonl.zst')
    for line in S.zstd_lines(path):
        if not line.strip():
            continue
        row = json.loads(line)
        counts['episodes'] += 1
        if row.get('unscored') or row.get('reward') is None:
            counts['episodes_unscored'] += 1
            continue
        if row.get('jolt_role', 'student') == 'teacher':   # a guided attempt: not the policy's own answer
            counts['episodes_guided'] += 1
            continue
        if row.get('split', 'train') != 'train':
            counts['episodes_not_train'] += 1
            continue
        if row['task_id'] not in cases:
            counts['episodes_without_task_row'] += 1
            continue
        if row['task_id'] in EXCLUDED:
            counts['episodes_held_excluded'] += 1
            continue
        success = S.episode_success(row)
        counts['episodes_success' if success else 'episodes_failure'] += 1
        episode = {'key': f"{label}:{name}:{row['sample']}", 'task_id': row['task_id'], 'family': row.get('family'),
                   'source_groups': row.get('source_groups'), 'outcome': row.get('outcome'), 'success': success,
                   'policy': (row.get('policy') or {}).get('checkpoint_sha256')}
        rows += site_rows(episode, row.get('turns') or [], label, graph_invocations(row), kinds=kinds, counts=counts)
    return label, rows, counts


def episode_unit(label, episode, kinds, reference):
    """Site rows of one sft_sources episode (an authored pool's or a WORK's): semantic turns, a task row."""
    counts = collections.Counter(episodes=1)
    if episode['task_id'] in EXCLUDED:
        counts['episodes_held_excluded'] += 1
        return label, [], counts
    if (episode.get('task') or {}).get('split', 'train') != 'train':
        counts['episodes_not_train'] += 1
        return label, [], counts
    counts['episodes_success'] += 1
    ep = {'key': episode['key'], 'task_id': episode['task_id'], 'family': episode.get('family'),
          'source_groups': (episode.get('task') or {}).get('source_groups'), 'outcome': S.SUCCESS, 'success': True,
          'policy': episode.get('policy')}
    return label, site_rows(ep, episode.get('turns') or [], label, kinds=kinds, reference=reference, counts=counts), counts


def authored_unit(unit, kinds):
    label, pool, line = unit
    out_rows, counts = [], collections.Counter()
    _, lines, c, _ = S._extract_teacher_record(label, pool, line)
    counts.update({f'source:{k}': v for k, v in c.items()})
    for episode_line in lines:
        _, rows, more = episode_unit(label, json.loads(episode_line), kinds, 'authored')
        out_rows += rows
        counts.update(more)
    return label, out_rows, counts


def run_unit(unit, kinds):
    return (rollout_unit if unit[0] == 'rl' else authored_unit)(unit[1:], kinds)


def run_units(units, kinds):
    return [run_unit(u, kinds) for u in units]


def collections_of(args):
    if args.collection:
        return [Path(c) for c in args.collection]
    root = Path(args.collect_root or resolve_str(*COLLECT_ROOT))
    # newest policy first: the per-site cap keeps the most recent policy's calls
    return sorted((p for p in root.glob('v*') if (p / 'groups').is_dir()), key=lambda p: p.name, reverse=True)


def authored_pools(args):
    if args.authored_pool is not None:
        return [Path(p) for p in args.authored_pool]
    import authored_trajectories
    return [Path(p).parent if Path(p).name == 'manifest.jsonl' else Path(p)
            for p in data_registry.role_paths(authored_trajectories.POOL_ROLE)]


def task_rows(units, works):
    """Every task row the sources name (each group's cases.jsonl; a WORK episode's task), by id."""
    tasks = {}
    for unit in units:
        directory = unit[4] if unit[0] == 'rl' else Path(json.loads(unit[3])['rollouts']).parent
        tasks.update(S.group_cases(directory))
    for work in works:
        for e in S.all_episodes(work, light=True):
            if e.get('task'):
                tasks.setdefault(e['task_id'], e['task'])
    return tasks


def extract(args):
    kinds = tuple(k for k in args.kinds.split(',') if k)
    if set(kinds) - set(KINDS):
        raise SystemExit(f'--kinds: unknown {sorted(set(kinds) - set(KINDS))}')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    held_manifest = args.held_manifest or data_registry.role_path('held-eval')
    colls, pools, works = collections_of(args), authored_pools(args), [Path(w) for w in args.work or ()]
    unregistered = []
    if not args.collection:   # a discovered collection no registry entry holds is left out (and listed), never read
        registry = data_registry.load()
        unregistered = [c for c in colls
                        if (registry.find(c) or {'admission': data_registry.INVENTORY}).get('admission') == data_registry.INVENTORY]
        colls = [c for c in colls if c not in unregistered]
    data_registry.require_registered([*map(str, colls), *map(str, pools), *map(str, works)])
    units = []
    for coll in colls:
        label = 'aei1:' + coll.name if 'aei1' in coll.parts else 'rl:' + '/'.join(coll.parts[-2:])
        units += [('rl', label, name, str(p), str(g), str(coll / 'archive'))
                  for name, p, g in S.rl_rollout_files(coll, coll / 'archive')]
    for pool in pools:
        label = 'authored:' + '/'.join(pool.parts[-2:])
        units += [('authored', label, str(pool), line) for line in (pool / 'manifest.jsonl').open() if line.strip()]
    if args.limit_units:
        units = units[:args.limit_units]
    started = time.time()
    tasks = task_rows(units, works)
    excluded = frozenset(S.held_excluded_ids(held_manifest, tasks, S.held_closure(held_manifest)))
    _held_out_units(excluded)
    funnel = collections.Counter(units=len(units), task_rows=len(tasks), held_excluded_tasks=len(excluded))
    by_source = collections.defaultdict(collections.Counter)
    rows, per_site, index = {}, collections.Counter(), {}
    partial = out / 'sites.partial.jsonl.zst'
    writer = S.ZstdWriter(partial)

    def admit(label, candidates, counts):
        by_source[label].update(counts)
        funnel.update(counts)
        for row in candidates:
            funnel[f'candidates:{row["kind"]}'] += 1
            existing = index.get(row['row_id'])
            if existing is not None:
                funnel[f'merged:{row["kind"]}'] += 1
                obs = rows[existing]
                obs['answer_counts'][row['policy_answer']['key']] += 1
                obs['n'] += 1
                if len(obs['observations']) < MAX_OBSERVATIONS:
                    obs['observations'] += row['observations']
                if row.get('reference') and not obs.get('reference'):
                    obs['reference'] = row['reference']
                continue
            bucket = (row['site_id'], row['header_sha'])
            if per_site[bucket] >= args.max_per_site:
                funnel[f'site_cap:{row["kind"]}'] += 1
                continue
            per_site[bucket] += 1
            index[row['row_id']] = len(rows)
            rows[len(rows)] = {'observations': list(row['observations']), 'n': 1,
                               'answer_counts': collections.Counter({row['policy_answer']['key']: 1}),
                               'reference': row.get('reference')}
            row.pop('observations')
            writer.write(row)
            funnel[f'admitted:{row["kind"]}'] += 1
            funnel[f'admitted_role:{row["role"]}'] += 1

    if args.workers > 1:
        import concurrent.futures
        import multiprocessing
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context('spawn'),
                                                    initializer=_held_out_units, initargs=(excluded,)) as pool:
            for batch in S.bounded_map(pool, run_units, S.batched(units, 4), 2 * args.workers, kinds):
                for label, candidates, counts in batch:
                    admit(label, candidates, counts)
    else:
        for unit in units:
            admit(*run_unit(unit, kinds))
    for work in works:
        label = 'work:' + work.name
        for episode in S.all_episodes(work):
            reference = 'teacher-success' if episode.get('source') == 'teacher' else 'own-success'
            admit(*episode_unit(label, episode, kinds, reference))
    writer.close()
    # second pass: each row with every observation of its request (answers and outcomes of all samples that sent it)
    final = S.ZstdWriter(out / 'sites.jsonl.zst')
    for i, line in enumerate(S.zstd_lines(partial)):
        row, obs = json.loads(line), rows[i]
        row['observations'], row['n_observations'] = obs['observations'], obs['n']
        row['answer_counts'] = dict(obs['answer_counts'])
        row['policy_success_any'] = any(o.get('success') for o in obs['observations'])
        if obs.get('reference'):
            row['reference'] = obs['reference']
        final.write(row)
    final.close()
    partial.unlink()
    summary = {'schema': 'natlang.call-level-extract/1', 'rows': final.rows, 'seconds': round(time.time() - started),
               'kinds': kinds, 'max_per_site': args.max_per_site, 'held_manifest': str(held_manifest),
               'distinct_sites': len({b[0] for b in per_site}), 'site_headers': len(per_site),
               'funnel': dict(sorted(funnel.items())),
               'sources': {k: dict(sorted(v.items())) for k, v in by_source.items()},
               'collections': [str(c) for c in colls], 'unregistered_collections': [str(c) for c in unregistered],
               'authored_pools': [str(p) for p in pools],
               'works': [str(w) for w in works]}
    (out / 'extract.json').write_text(json.dumps(summary, indent=1) + '\n')
    if not args.no_register:
        data_registry.record_dataset(
            data_registry.dataset_id(out, 'call-level-sites'), out, kind='trajectories',
            include=['sites.jsonl.zst', 'extract.json'], rows=final.rows,
            inputs=[*map(str, colls), *map(str, pools), *map(str, works), str(held_manifest)],
            format=f'{SITE_SCHEMA}: sites.jsonl.zst (one row per call site request: rendered messages, tools, exact '
                   'prompt ids, the policy answer and every observation of it), extract.json (funnel)',
            license=S.DERIVED_LICENSE, note='Engine A call-level sites (call_level_engine.py extract); held closure '
                                            'excluded (held_eval.training_exclusion_ids)')
    print(json.dumps({k: summary[k] for k in ('rows', 'seconds', 'distinct_sites', 'site_headers')}
                     | {'funnel': summary['funnel']}, indent=1))
    return summary


# ---- teach ---------------------------------------------------------------------------------------------------------------
class Helper:
    """call_level_teacher.mjs as a line-protocol subprocess."""

    def __init__(self, concurrency=1, min_interval_ms=0, node=None):
        from portable_paths import node_executable
        env = {**os.environ, 'CALL_LEVEL_CONCURRENCY': str(concurrency), 'CALL_LEVEL_MIN_INTERVAL_MS': str(min_interval_ms)}
        self.proc = subprocess.Popen([node or node_executable(), str(HERE / 'call_level_teacher.mjs')], env=env,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.lock = threading.Lock()

    def send(self, request):
        with self.lock:
            self.proc.stdin.write(json.dumps(request, ensure_ascii=False) + '\n')
            self.proc.stdin.flush()

    def answers(self):
        for line in self.proc.stdout:
            if line.strip():
                yield json.loads(line)

    def close(self):
        self.proc.stdin.close()


def check_values(helper, items):
    """{id: check} of (id, value, check) items through the runtime's type check."""
    pending = 0
    for i, value, check in items:
        helper.send({'op': 'check', 'id': i, 'value': value, 'check': check})
        pending += 1
    out = {}
    if pending:
        for answer in helper.answers():
            out[answer['id']] = answer.get('check')
            if len(out) == pending:
                break
    return out


def load_teachers(args):
    import teacher_gen
    specs = json.loads(Path(args.teachers_file or resolve_str(*TEACHERS_FILE)).read_text())
    specs += [json.loads(s) for s in args.teacher_spec or ()]
    chosen = []
    for model in args.teacher:
        spec = next((s for s in reversed(specs) if s.get('model_id') == model), None)
        if spec is None:
            raise SystemExit(f'--teacher {model}: no spec in the teachers file or --teacher-spec')
        if spec.get('provider'):
            raise SystemExit(f'--teacher {model}: a provider transport teacher (natlang Pi provider) answers whole '
                             'episodes through the collector only; call-level needs an HTTP endpoint teacher')
        chosen.append(teacher_gen.make_teacher(spec))
    return chosen


def teacher_key(teacher, cache):
    """A fresh API key (a key command's token is minted again after 30 minutes)."""
    import teacher_gen
    name = teacher.api_key_env
    if not name:
        return None
    stamp, key = cache.get(teacher.model_id, (0, None))
    if key and time.time() - stamp < 1800:
        return key
    env = teacher_gen.episode_env(teacher, dict(os.environ))
    key = env.get(name)
    cache[teacher.model_id] = (time.time(), key)
    return key


def select_sites(args, funnel):
    kinds = [k for k in args.kinds.split(',') if k]
    rank = {k: i for i, k in enumerate(kinds)}
    rng = random.Random(args.seed)
    picked = []
    for line in S.zstd_lines(Path(args.sites) / 'sites.jsonl.zst'):
        row = json.loads(line)
        funnel['sites_read'] += 1
        if row['kind'] not in rank:
            continue
        if native_state(row['messages']):
            funnel['native_state'] += 1
            continue
        size = len(json.dumps(row['messages'], ensure_ascii=False)) + len(json.dumps(row['tools'], ensure_ascii=False))
        if size > args.max_prompt_chars:
            funnel['too_long'] += 1
            continue
        row.pop('prompt_ids', None)
        # the frontier first: requests whose every observed policy answer came from a failed episode
        picked.append((rank[row['kind']], bool(row.get('policy_success_any')), rng.random(), row))
    picked.sort(key=lambda item: item[:3])
    per_site, out = collections.Counter(), []
    for *_, row in picked:
        if per_site[row['site_id']] >= args.per_site:
            funnel['batch_site_cap'] += 1
            continue
        per_site[row['site_id']] += 1
        out.append(row)
        if args.limit and len(out) >= args.limit:
            break
    return out


def read_answers(path):
    done = collections.defaultdict(list)
    if path.exists():
        for line in path.open():
            if line.strip():
                record = json.loads(line)
                done[(record['row_id'], record['teacher'], record['sample'])].append(record)
    return done


def teach(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    data_registry.require_registered([str(Path(args.sites))])
    funnel = collections.Counter()
    teachers = load_teachers(args)
    sites = select_sites(args, funnel)
    answers_path = out / 'answers.jsonl'
    done = read_answers(answers_path)
    spend = collections.defaultdict(collections.Counter)
    for records in done.values():
        for r in records:
            spend[r['teacher']]['requests'] += 1
            spend[r['teacher']]['prompt_tokens'] += (r.get('turn') or {}).get('prompt_tokens') or 0
            spend[r['teacher']]['completion_tokens'] += (r.get('turn') or {}).get('completion_tokens') or 0
    total = lambda: sum(c['prompt_tokens'] + c['completion_tokens'] for c in spend.values())   # noqa: E731
    requests = lambda: sum(c['requests'] for c in spend.values())   # noqa: E731
    keys, by_id = {}, {}
    todo = []
    for row in sites:
        for teacher in teachers:
            for k in range(args.samples):
                records = done.get((row['row_id'], teacher.model_id, k), [])
                if any(r.get('turn') for r in records) or len(records) >= args.retries:
                    continue
                todo.append((row, teacher, k))
    funnel['sites_selected'] = len(sites)
    funnel['requests_todo'] = len(todo)
    stop_reason = None
    if todo and not args.dry_run:
        helper = Helper(args.concurrency, args.min_interval_ms)
        sent, prior_requests = 0, requests()
        log = answers_path.open('a')
        inflight = threading.Semaphore(args.concurrency * 2)

        def reader():
            for answer in helper.answers():
                row, teacher, k = by_id.pop(answer['id'])
                turn = answer.get('turn')
                record = {'row_id': row['row_id'], 'teacher': teacher.model_id, 'sample': k, 'at': time.time(),
                          **({'turn': turn} if turn else {'error': answer.get('error'), 'status': answer.get('status')})}
                log.write(json.dumps(record, ensure_ascii=False) + '\n')
                log.flush()
                done[(row['row_id'], teacher.model_id, k)].append(record)
                c = spend[teacher.model_id]
                c['requests'] += 1
                c['errors'] += 0 if turn else 1
                c['prompt_tokens'] += (turn or {}).get('prompt_tokens') or 0
                c['completion_tokens'] += (turn or {}).get('completion_tokens') or 0
                inflight.release()

        thread = threading.Thread(target=reader, daemon=True)
        thread.start()
        for i, (row, teacher, k) in enumerate(todo):
            if args.max_requests and prior_requests + sent >= args.max_requests:
                stop_reason = 'max_requests'
                break
            if args.max_total_tokens and total() >= args.max_total_tokens:
                stop_reason = 'max_total_tokens'
                break
            if sum(c['errors'] for c in spend.values()) >= args.max_errors:
                stop_reason = 'max_errors'
                break
            inflight.acquire()
            rid = f'{i}'
            by_id[rid] = (row, teacher, k)
            helper.send({'op': 'chat', 'id': rid, 'endpoint': teacher.endpoint, 'model': teacher.model_id,
                         'chat_completions_url': teacher.chat_completions_url, 'api_key': teacher_key(teacher, keys),
                         'chat_config': teacher.chat_config, 'messages': row['messages'], 'tools': row['tools'],
                         'temperature': args.temperature, 'max_tokens': args.max_tokens})
            sent += 1
        helper.close()
        thread.join()
        log.close()
        funnel['requests_sent'] = sent
    return aggregate(args, sites, teachers, read_answers(answers_path), funnel, spend, stop_reason)


def aggregate(args, sites, teachers, done, funnel, spend, stop_reason):
    """Majority answer per site over its teacher samples, the runtime's type check, and DIR/call-level.jsonl.zst."""
    import teacher_gen
    out = Path(args.out)
    parsed, checks = {}, []
    for row in sites:
        samples = []
        for teacher in teachers:
            for k in range(args.samples):
                record = next((r for r in done.get((row['row_id'], teacher.model_id, k), []) if r.get('turn')), None)
                if record is None:
                    continue
                turn = record['turn']
                text, calls = normalize_reply(turn.get('text'), turn.get('calls'))
                basis, value = parse_answer(row['kind'], text, calls, row.get('labels'))
                if turn.get('truncated') or turn.get('malformed_call'):
                    basis = 'invalid'
                samples.append({'teacher': teacher.model_id, 'sample': k, 'turn': turn, 'basis': basis, 'value': value,
                                'key': answer_key(basis, value)})
                if basis in ('return_value', 'readout_value') and (row.get('check') or {}).get('returns'):
                    checks.append((f'{row["row_id"]}:{teacher.model_id}:{k}', value, row['check']))
        parsed[row['row_id']] = samples
    verdicts = {}
    if checks:
        helper = Helper(1, 0)
        verdicts = check_values(helper, checks)
        helper.close()
    writer = S.ZstdWriter(out / 'call-level.jsonl.zst')
    stats = collections.Counter()
    for row in sites:
        samples = parsed[row['row_id']]
        expected = args.samples * len(teachers)
        if not samples:
            stats['unanswered'] += 1
            continue
        stats['answered'] += 1
        for s in samples:
            s['check'] = verdicts.get(f'{row["row_id"]}:{s["teacher"]}:{s["sample"]}')
            if s['check'] and s['check'].get('checked') and not s['check'].get('ok'):
                s['basis'] = 'type_rejected'
                s['key'] = answer_key('type_rejected', s['value'])
        votes = collections.Counter(s['key'] for s in samples if s['basis'] != 'invalid')
        if not votes:
            stats['all_invalid'] += 1
            continue
        key, count = votes.most_common(1)[0]
        best = next(s for s in samples if s['key'] == key)
        agreement = count / expected
        verified = best['basis'] in VERIFIABLE and agreement >= args.min_agreement and len(samples) == expected
        stats[f'basis:{best["basis"]}'] += 1
        stats['verified' if verified else 'unverified'] += 1
        stats[f'{"verified" if verified else "unverified"}:{row["kind"]}'] += 1
        type_check = best.get('check') or ({'checked': True, 'ok': True, 'by': 'labels'} if best['basis'] == 'decision_label'
                                           else {'checked': False})
        policy_keys = row.get('answer_counts') or {}
        policy_agreement = policy_keys.get(key, 0) / max(1, sum(policy_keys.values()))
        if verified:
            stats['policy_agrees_with_verified'] += policy_agreement > 0.5
        turn = best['turn']
        target = {'role': 'assistant', 'content': turn.get('text') or ''}
        if turn.get('calls'):
            target['tool_calls'] = [{'id': f'call_{i}', 'type': 'function',
                                     'function': {'name': n, 'arguments': json.dumps(a, ensure_ascii=False)}}
                                    for i, (n, a) in enumerate(turn['calls'])]
        teacher = next(t for t in teachers if t.model_id == best['teacher'])
        writer.write({
            'schema': ANSWER_SCHEMA, 'row_id': row['row_id'], 'site_id': row['site_id'], 'kind': row['kind'],
            'definition_id': row.get('definition_id'), 'signature': row.get('signature'), 'role': row.get('role'),
            'task_id': row['task_id'], 'family': row.get('family'), 'source_groups': row.get('source_groups'),
            'source': row.get('source'), 'messages': row['messages'], 'tools': row['tools'],
            'target': target, 'teacher_reasoning': turn.get('reasoning'), 'teacher_reasoning_weight': 0.0,
            'teacher': teacher_gen.teacher_identity(teacher), 'basis': best['basis'], 'value': best['value'],
            'agreement': agreement, 'samples': len(samples), 'expected_samples': expected,
            'votes': dict(votes), 'type_check': type_check, 'verified': verified,
            'policy_agreement': policy_agreement, 'policy_success_any': row.get('policy_success_any'),
            **({'reference_agrees': (row['reference'].get('key') == key)} if row.get('reference') else {})})
    writer.close()
    usd = {m: round(c['prompt_tokens'] * args.usd_per_mtok_in / 1e6 + c['completion_tokens'] * args.usd_per_mtok_out / 1e6, 4)
           for m, c in spend.items()} if (args.usd_per_mtok_in or args.usd_per_mtok_out) else None
    summary = {'schema': 'natlang.call-level-teach/1', 'sites': str(args.sites), 'teachers': [t.model_id for t in teachers],
               'samples': args.samples, 'min_agreement': args.min_agreement, 'stop_reason': stop_reason,
               'funnel': dict(funnel), 'stats': dict(sorted(stats.items())), 'rows': writer.rows,
               'spend': {m: dict(c) for m, c in spend.items()}, **({'usd_estimate': usd} if usd is not None else {})}
    (out / 'teach.json').write_text(json.dumps(summary, indent=1) + '\n')
    if writer.rows and not args.no_register:
        data_registry.record_dataset(
            data_registry.dataset_id(out, 'call-level-answers'), out, kind='teacher-demos',
            include=['call-level.jsonl.zst', 'teach.json', 'answers.jsonl'], rows=writer.rows, closed=False,
            inputs=[str(args.sites)],
            format=f'{ANSWER_SCHEMA}: call-level.jsonl.zst (per site: rendered request, majority teacher answer as '
                   'target, teacher reasoning as zero-weight context, agreement, type check, verified), answers.jsonl '
                   '(every teacher sample), teach.json',
            license=S.DERIVED_LICENSE + '; teacher outputs: ' + ', '.join(t.model_id for t in teachers),
            note='Engine A call-level teacher answers (call_level_engine.py teach); train only rows with verified true')
    print(json.dumps({k: summary[k] for k in ('rows', 'stop_reason', 'stats', 'spend')}, indent=1))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('extract')
    p.add_argument('--out', required=True)
    p.add_argument('--collect-root', help='aei1 collect directory (default data_nvme:lfm3-rl-dgx-20261029/ops/ei/aei1/collect)')
    p.add_argument('--collection', action='append', help='one collection directory (repeatable; replaces --collect-root)')
    p.add_argument('--authored-pool', action='append', help='authored-trajectory pool (default: registry role '
                                                              'ei-authored-teacher); pass "" for none')
    p.add_argument('--work', action='append', help='sft_sources WORK directory (episodes*.jsonl.zst)')
    p.add_argument('--held-manifest', help='default: registry role held-eval (every held-eval* role is excluded too)')
    p.add_argument('--kinds', default=','.join(DEFAULT_KINDS))
    p.add_argument('--max-per-site', type=int, default=64, help='rows per (site, header) bucket')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--limit-units', type=int, default=0, help='first N groups/pool records only (smoke)')
    p.add_argument('--no-register', action='store_true')
    t = sub.add_parser('teach')
    t.add_argument('--sites', required=True, help='an extract output directory')
    t.add_argument('--out', required=True)
    t.add_argument('--teacher', action='append', required=True, help='model_id of a teachers-file spec (repeatable)')
    t.add_argument('--teachers-file', help='default data_nvme:lfm3-rl-dgx-20261029/ops/ei/teachers.json')
    t.add_argument('--teacher-spec', action='append', help='an extra teacher spec as JSON (teacher_gen form)')
    t.add_argument('--kinds', default='decision,readout,entry', help='kinds to teach, in priority order')
    t.add_argument('--samples', type=int, default=3)
    t.add_argument('--min-agreement', type=float, default=2 / 3)
    t.add_argument('--temperature', type=float, default=1.0)
    t.add_argument('--max-tokens', type=int, default=8192)
    t.add_argument('--limit', type=int, default=0, help='sites in this batch')
    t.add_argument('--per-site', type=int, default=4, help='requests per site id in this batch')
    t.add_argument('--max-prompt-chars', type=int, default=60000)
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--concurrency', type=int, default=2)
    t.add_argument('--min-interval-ms', type=int, default=0)
    t.add_argument('--max-requests', type=int, default=0, help='budget: requests over all launches (0: none)')
    t.add_argument('--max-total-tokens', type=int, default=0, help='budget: prompt + completion tokens (0: none)')
    t.add_argument('--max-errors', type=int, default=20, help='stop a launch after this many failed requests')
    t.add_argument('--retries', type=int, default=2, help='failed attempts per sample before it is left out')
    t.add_argument('--usd-per-mtok-in', type=float, default=0.0)
    t.add_argument('--usd-per-mtok-out', type=float, default=0.0)
    t.add_argument('--dry-run', action='store_true', help='select and aggregate only; send nothing')
    t.add_argument('--no-register', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'extract':
        if args.authored_pool == ['']:
            args.authored_pool = []
        return extract(args)
    return teach(args)


if __name__ == '__main__':
    main()
