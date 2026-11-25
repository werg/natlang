#!/usr/bin/env python3
"""Authored trajectories -> the expert-iteration teacher tier (owner 2026-11-25: "why don't we just author entire
trajectories directly for the IR without having to run it through a model? ... then all we need to do is collect the
reasoning in between the actions").

An authored trajectory is a task whose every model decision was written by its author, not sampled: the root call's
actions and the reply of every judgment (``.nl``) call, iteration step, folder reducer and decision readout. Its format,
``natlang.authored-trajectory/1`` (one JSON line each), is independent of who authored it:

    {"schema": "natlang.authored-trajectory/1", "id": ..., "source": "<producer>", "author": {...},
     "task": <natlang.program/2 row: the program, its inputs and its exact oracle>,
     "turns": [{"invocation_id", "function", "messages", "tools", "target"}, ...]   # optional, see below}

Without ``turns`` the task row's ``curriculum.reference`` (root actions, ``children`` answers by match fragments) is the
authored trajectory and the runtime renders it: ``scripts/rl_tasks/reference_episodes.mjs`` runs it through the
collector's own execution path (``curriculum.replayReference``), so each call's messages are what the model would see
(iteration, folder reducers and decision readouts included: the runtime issues those requests itself) and each target is
the authored reply (tool calls, or a decision readout's JSON text). With ``turns`` they are taken as given. Either way a
trajectory enters training only after ``sft_sources replay`` re-ran its task under the training pin's runtime with those
replies and the task's oracle accepted the result.

Neuralese is rendered as the serving engines read it by default (ts-host ``model/in-context-neuralese.ts``): a learned
compaction site's operator call (``<call>#operator-<n>``, function ``view``) is a turn of its own whose target is the
authored summary, and every consumer's messages carry each state's text in place (no sentinel, no markers). The replay
of these episodes (``sft_sources replay``, ``authored_episode``) reads Neuralese the same way.

Turns carry no reasoning. The reasoning between actions is the student's, filled by ``filled_reasoning.py`` (the
``filled`` tier, ``replaces=teacher``); until a fill exists the episode trains as a teacher episode: its empty think
block is zero-weight context and its actions are targets (sft_sources ThinkTreatment default).

    authored_trajectories.py render --cases CASES.jsonl ... --out AUTHORED.jsonl [--source S] [--runtime TS_HOST]
    authored_trajectories.py pool   --input AUTHORED.jsonl ... --out POOL
        a teacher pool (teacher_gen layout: POOL/manifest.jsonl, POOL/episodes/<e>/{cases,rollouts}.jsonl), the format
        ``sft_sources extract --teacher`` reads. Refuses rows outside the train split and every task
        ``held_eval.training_exclusion_ids`` names (held-v3/v2, held-inv and every held-eval* role, with siblings).
    authored_trajectories.py feed   [--root INVENTOR_ROOT] [--work WORK] [--no-replay] [--no-register] [--dry-run]
        the continuous flow for the program inventor: every admitted invented program of the train split that a review
        did not withhold and that has no export at its admitted digest -> render (its whole-program cases and stage
        tasks) -> ROOT/authored/<stamp>/ pool -> registry dataset ``natlang-sft-invented-authored-<stamp>`` (role
        ``ei-authored-teacher``: the async driver extracts every pool of that role into its WORK each round,
        ``expert_iteration.py async-drive --no-authored-teacher`` opts out) -> extracted and replayed into the fill
        WORK (``WORK``, default data_nvme:sft-invented-authored, role ``grpo-sd-teacher``: the reasoning fill draws
        its teacher episodes from that role). ``invented.py publish`` runs it after each batch (``--no-authored``).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / 'training/neuralese'), str(REPO), str(HERE)]

SCHEMA = 'natlang.authored-trajectory/1'
POOL_ROLE = 'ei-authored-teacher'
FILL_ROLE = 'grpo-sd-teacher'
TEACHER = {'model_id': 'authored-reference', 'provider': 'authored'}
PYTHON = REPO / '.venv-neuralese' / 'bin' / 'python'
REPLAY_MAX_MODEL_REQUESTS = 1000   # a task's own semantics.episode_model_calls still bounds it (collector.ts)


def read_jsonl(path):
    from natlang_neuralese.common.jsonio import read_jsonl as rows
    return list(rows(path))


def save_rows(path, rows):
    from natlang_neuralese.common.jsonio import write_bytes_durable
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_bytes_durable(path, ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows).encode())


def node():
    for candidate in sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True):
        return str(candidate)
    return 'node'


# ---- the format -------------------------------------------------------------------------------------------------------
def trajectory_row(task, *, source, author=None, turns=None):
    """A ``natlang.authored-trajectory/1`` row for ``task`` (``turns`` None: rendered from its reference)."""
    row = {'schema': SCHEMA, 'id': task['id'], 'source': source, 'author': author or {}, 'task': task}
    if turns is not None:
        row['turns'] = turns
    return row


def turn_problems(turn):
    problems = []
    if not isinstance(turn.get('messages'), list) or not turn['messages']:
        problems.append('no messages')
    target = turn.get('target')
    if not isinstance(target, dict) or target.get('role') != 'assistant':
        problems.append('target is not an assistant message')
    elif not target.get('tool_calls') and not (target.get('content') or '').strip():
        problems.append('empty target')
    elif target.get('reasoning_content'):
        problems.append('authored turns carry no reasoning (the student fills it)')
    return problems


def row_problems(row):
    """Why an authored-trajectory row cannot become a teacher episode (empty: it can)."""
    if row.get('schema') != SCHEMA:
        return [f'schema {row.get("schema")!r}']
    task = row.get('task') or {}
    problems = []
    if task.get('version') != 'natlang.program/2':
        problems.append('task is not a natlang.program/2 row')
    if task.get('split') != 'train':
        problems.append(f'task split {task.get("split")!r}: only train tasks train')
    turns = row.get('turns')
    if not turns:
        problems.append('no turns')
    for index, turn in enumerate(turns or ()):
        problems += [f'turn {index}: {p}' for p in turn_problems(turn)]
    return problems


# ---- render -----------------------------------------------------------------------------------------------------------
def render(cases, *, source, author=None, runtime=None, scratch):
    """Authored-trajectory rows of ``cases`` (natlang.program/2 rows carrying ``curriculum.reference``), rendered by the
    runtime; [(row, None)] or [(None, reason)] per case, in order."""
    runtime = Path(runtime or REPO / 'ts-host')
    scratch = Path(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    save_rows(scratch / 'cases.jsonl', cases)
    run = subprocess.run([node(), str(REPO / 'scripts/rl_tasks/reference_episodes.mjs'), str(runtime),
                          str(scratch / 'cases.jsonl'), str(scratch / 'episodes.jsonl')],
                         capture_output=True, text=True, timeout=7200)
    if run.returncode:
        raise SystemExit('reference_episodes.mjs failed: ' + (run.stderr or run.stdout)[-800:])
    rendered = {r['id']: r for r in read_jsonl(scratch / 'episodes.jsonl')}
    out = []
    for task in cases:
        r = rendered.get(task['id'])
        if r is None or not r.get('accepted'):
            out.append((None, (r or {}).get('error') or f'reference not accepted ({(r or {}).get("status")})'))
            continue
        out.append((trajectory_row(task, source=source, author=author, turns=r['turns']), None))
    return out


# ---- pool -------------------------------------------------------------------------------------------------------------
def episode_name(row):
    return 'a' + hashlib.sha256(row['id'].encode()).hexdigest()[:15]


# A decision readout's authored answer serves every request of the readout: the collector first samples its reasoning
# ``samples`` times on the readout's own messages (native/agent.ts decisionReasonings; the count is the run's option),
# then asks the answer. sft_sources.ReplayEpisode serves a reasoned group's repeats (at most ``samples``) from its action
# entry; unused repeats cost nothing.
READOUT_REPEATS = 16


def operator_turn(turn):
    """A learned operator's call (``<call>#operator-<n>``, learned compaction): its own per-call chain, the authored
    summary as its target, never a decision readout (it is asked once and its reply is captured in place)."""
    return bool(re.search(r'#operator-\d+$', turn.get('invocation_id') or ''))


def graph_calls(turns, renderer, stats):
    """The learner rows of an authored episode whose consumers read spliced states (an operator's summary, a
    ``compact_history`` note: ``captures`` on the producing turn), as ``lower_rollout_turns.episode_turns`` lowers an own
    episode: every call rendered by the serving engine's own rendering (``sft_sources.TemplateRenderer``), each state a
    pass-through import whose producer document the learner computes in the consumer's forward (``exp0_sft.row_inputs``),
    so no engine or store is needed. The document is the one ``Engine.capture_plan`` writes for a reply captured in its
    own context: the producing call's prompt, its reply up to the content (an empty thought closed), ``produce_open``,
    the content (``select``: that text inside the reply), ``produce_close``; the state is the content's rows. A consumer
    reads it between ``consume_open`` and ``consume_close``. A reply that is not the student's (all authored turns)
    trains after its think block (``sft_sources.think_context``). Returns the rows, or None when a state cannot be
    lowered (counted in ``stats``)."""
    import numpy as np
    from natlang_neuralese.model.shared_state import ROLE_MARKERS
    from natlang_neuralese.serve.targets import target_items
    from sft_sources import think_context
    tokenizer = renderer.tokenizer
    produce_open, produce_close, consume_open, consume_close = (int(tokenizer.convert_tokens_to_ids(m)) for m in ROLE_MARKERS)
    think_close = int(tokenizer.convert_tokens_to_ids('</think>'))
    stops = {int(tokenizer.convert_tokens_to_ids('<|im_end|>'))}
    rendered = []
    for turn in turns:
        target = {k: v for k, v in turn['target'].items() if v is not None}
        rendered.append(target_items(renderer, list(turn['messages']), turn.get('tools') or [], target))
    sources = {}
    for turn, (before, rest) in zip(turns, rendered):
        for capture in turn.get('captures') or ():
            if any(kind != 'tok' for kind, _ in before + rest):
                stats['authored_capture_after_import'] += 1
                return None
            prompt, reply = [i for _, i in before], [i for _, i in rest]
            end = next((i for i, token in enumerate(reply) if token in stops), len(reply))
            if capture.get('select') is not None:
                from natlang_neuralese.serve.engine import selected_span
                try:
                    start, stop = selected_span(lambda part: tokenizer.decode(part, skip_special_tokens=False),
                                                reply[:end], capture['select'])
                except Exception:
                    stats['authored_select_not_found'] += 1
                    return None
            else:
                start = max((i + 1 for i, token in enumerate(reply[:end]) if token == think_close), default=0)
                while start < end and not tokenizer.decode([reply[start]]).strip():
                    start += 1   # the whitespace after the thought is context, as the thinking-off operator prompt has it
                stop = end
            if start >= stop:
                stats['authored_capture_empty'] += 1
                return None
            prefix = [*prompt, *reply[:start], produce_open]
            span = reply[start:stop]
            sources[capture['id']] = {'ids': np.array(prefix + span, dtype=np.int32), 'start': len(prefix),
                                      'stop': len(prefix) + len(span)}
            stats['authored_captures'] += 1
    rows = []
    for index, (turn, (before, rest)) in enumerate(zip(turns, rendered)):
        ids, imports, used = [], [], {}
        for kind, value in before:
            if kind == 'tok':
                ids.append(int(value))
                continue
            block = value['id'] if isinstance(value, dict) else value
            source = sources.get(block)
            if source is None:
                stats['authored_import_without_producer'] += 1
                return None
            ids.append(consume_open)
            imports.append((len(ids), block))
            used[block] = source
            ids.extend(int(i) for i in source['ids'][source['start']:source['stop']])
            ids.append(consume_close)
        if any(kind != 'tok' for kind, _ in rest):
            stats['authored_state_in_reply'] += 1
            return None
        sample = np.array([i for _, i in rest], dtype=np.int32)
        row = {'invocation': turn.get('invocation_id') or '', 'index': index, 'prompt': np.array(ids, dtype=np.int32),
               'sample': sample, 'imports': imports, 'sources': used}
        context = think_context(renderer, 'teacher', turn['target'], sample)
        if context:
            row['weight'] = (np.arange(len(sample)) >= int(context)).astype(np.float32)
        stats['authored_consumer_calls' if imports else 'authored_calls'] += 1
        rows.append(row)
    return rows


def readout_turn(turn):
    """A tool-free request answered with text: a decision readout (or a generated value)."""
    return (not (turn.get('tools') or []) and not (turn.get('target') or {}).get('tool_calls')
            and not operator_turn(turn))


def rollout_row(row):
    """The natlang.rollout/1 row a teacher campaign would have written for this trajectory (one verified sample)."""
    task = row['task']
    turns = [{'messages': t['messages'], 'tools': t.get('tools') or [], 'target': t['target'],
              'invocation_id': t.get('invocation_id'), 'function': t.get('function') or 'unknown',
              **({'captures': t['captures']} if t.get('captures') else {}),
              'usage': {'completion_tokens': None},
              **({'decision_reasoning': t.get('decision_reasoning') or
                  {'index': 0, 'samples': READOUT_REPEATS, 'action': True}} if readout_turn(t) else {})}
             for t in row['turns']]
    return {'schema': 'natlang.rollout/1', 'round': 0, 'task_id': task['id'], 'family': task.get('family'),
            'group': 0, 'sample': 0, 'seed': None, 'split': task.get('split'),
            'source_ids': task.get('source_ids'), 'source_groups': task.get('source_groups'),
            'source_revisions': task.get('source_revisions'),
            'policy': {'temperature': None, 'checkpoint_sha256': None, 'endpoint': None},
            'authored': {'schema': SCHEMA, 'source': row.get('source'), 'author': row.get('author') or {}},
            'turns': turns, 'outcome': 'task_success', 'reward': {'quality': 1.0, 'graded': 1.0}, 'unscored': None}


def held_out(tasks):
    from natlang_neuralese.rl.held_eval import training_exclusion_ids
    return set(training_exclusion_ids(tasks))


def write_pool(rows, out):
    """Write ``rows`` (authored-trajectory rows) as a teacher pool at ``out``; returns its summary."""
    out = Path(out)
    refused, kept = {}, []
    for row in rows:
        problems = row_problems(row)
        if problems:
            refused[row.get('id') or '?'] = problems[:4]
        else:
            kept.append(row)
    excluded = held_out([r['task'] for r in kept]) if kept else set()
    for row in kept:
        if row['task']['id'] in excluded:
            refused[row['id']] = ['held: training_exclusion_ids names this task']
    kept = [r for r in kept if r['task']['id'] not in excluded]
    seen, manifest = set(), []
    for index, row in enumerate(kept):
        name = episode_name(row)
        if name in seen:
            refused[row['id']] = ['duplicate id']
            continue
        seen.add(name)
        directory = out / 'episodes' / name
        save_rows(directory / 'cases.jsonl', [row['task']])
        save_rows(directory / 'rollouts.jsonl', [rollout_row(row)])
        manifest.append({'index': index, 'task_id': row['task']['id'], 'family': row['task'].get('family'),
                         'base_rank': 'authored', 'root_mode': 'student', 'teacher': dict(TEACHER),
                         'authored': {'source': row.get('source'), 'author': row.get('author') or {}},
                         'episodes': [{'sample': 0, 'outcome': 'task_success', 'reward': 1.0,
                                       'turns': len(row['turns'])}],
                         'rollouts': str(directory / 'rollouts.jsonl'), 'provider_failure': False})
    save_rows(out / 'authored.jsonl', [r for r in kept if episode_name(r) in seen])
    save_rows(out / 'manifest.jsonl', manifest)
    summary = {'schema': 'natlang.authored-pool/1', 'episodes': len(manifest),
               'turns': sum(m['episodes'][0]['turns'] for m in manifest), 'refused': refused}
    (out / 'pool.json').write_text(json.dumps(summary, indent=1) + '\n')
    return summary


# ---- feed (program inventor) ------------------------------------------------------------------------------------------
def inventor_root():
    from natlang_neuralese.common.paths import resolve as data_path
    return Path(os.environ.get('NATLANG_INVENTOR_ROOT') or data_path('data_nvme', 'natlang-novel-programs'))


def default_work():
    from natlang_neuralese.common.paths import resolve as data_path
    return Path(data_path('data_nvme', 'sft-invented-authored'))


def withheld_reasons(admission, home):
    """invented.py's rule: a sampled review naming wrong expected answers of the admitted version withholds it."""
    for review_path in (Path(admission.get('submission') or '') / 'review.json', Path(home) / 'review.json'):
        if review_path.is_file():
            review = json.loads(review_path.read_text())
            if review.get('digest') == admission.get('digest') and review.get('wrong_expected'):
                return [str(item)[:400] for item in review['wrong_expected']]
    return []


def pending_programs(root):
    """[(admission, home)] of admitted train programs, not withheld, without an export at their admitted digest."""
    exported = exports(root)
    out = []
    for home in sorted((Path(root) / 'admitted').glob('*')):
        path = home / 'admission.json'
        if not path.exists():
            continue
        admission = json.loads(path.read_text())
        if admission.get('split') != 'train' or withheld_reasons(admission, home):
            continue
        if exported.get(admission['slug']) == admission.get('digest'):
            continue
        out.append((admission, home))
    return out


def exports(root):
    path = Path(root) / 'authored' / 'exported.jsonl'
    found = {}
    if path.exists():
        for r in read_jsonl(path):
            found[r['slug']] = r['digest']
    return found


def program_rows(admission, home, scratch, runtime=None):
    """Authored-trajectory rows of one admitted program (whole-program cases, then stage tasks) and the refusals."""
    # An authored program (natlang.authored-program/1, authored_programs.py admit) keeps every instance's authored
    # trajectory in trajectories.jsonl (cases.jsonl holds only its instances with an RL oracle)
    first = home / 'trajectories.jsonl' if (home / 'trajectories.jsonl').exists() else home / 'cases.jsonl'
    cases = [*read_jsonl(first), *(read_jsonl(home / 'stages.jsonl') if (home / 'stages.jsonl').exists() else [])]
    author = {'program': admission['slug'], 'digest': admission.get('digest'),
              'submission': admission.get('submission'), 'kind': 'invented-program reference'}
    rendered = render(cases, source='invented-reference', author=author, runtime=runtime, scratch=scratch)
    rows = [r for r, _ in rendered if r is not None]
    refused = {c['id']: why for c, (r, why) in zip(cases, rendered) if r is None}
    return rows, refused


def promote(dataset, role, role_file, no_check):
    """``data_registry.py promote`` run by the registry owner's checkout (the main one): promotion commits
    ``training/corpora/`` where it runs, and a pin (which renders and replays) takes no commits."""
    from natlang_neuralese.common import data_registry
    owner = Path(data_registry.registry_owner_repo())
    subprocess.run([str(owner / '.venv-neuralese' / 'bin' / 'python'), str(owner / 'scripts/data_registry.py'),
                    'promote', dataset, '--role', role, *(['--role-file', role_file] if role_file else []),
                    *(['--no-check'] if no_check else [])], check=True, cwd=owner, stdout=sys.stderr,
                   env={**os.environ, 'PYTHONPATH': f'{owner}/training/neuralese:{owner}'})


def register_pool(pool, stamp, summary, slugs, no_check=False):
    from natlang_neuralese.common import data_registry
    dataset = f'natlang-sft-invented-authored-{stamp}'
    data_registry.record_dataset(
        dataset, pool, kind='teacher-demos', include=['manifest.jsonl', 'authored.jsonl', 'pool.json',
                                                      'episodes/*/cases.jsonl', 'episodes/*/rollouts.jsonl'],
        format=f'{SCHEMA} teacher pool (teacher_gen layout: manifest.jsonl, episodes/<e>/{{cases,rollouts}}.jsonl): '
               f'{summary["episodes"]} authored reference trajectories, {summary["turns"]} model calls '
               '(scripts/neuralese_data/authored_trajectories.py)',
        license='invented (project-authored by frontier models)', rows=summary['episodes'],
        note='authored execution trajectories of admitted invented programs (train split): every judgment call with '
             'its runtime-rendered input and the scripted answer; no reasoning (filled_reasoning fills it); programs: '
             + ', '.join(slugs))
    promote(dataset, POOL_ROLE, 'manifest.jsonl', no_check)
    return dataset


def replay_into(work, pool, stamp, workers=6):
    """Extract the pool into the fill WORK and replay it under this checkout's runtime (the verified episodes the
    reasoning fill draws)."""
    env = {**os.environ, 'PYTHONPATH': f'{REPO}/scripts/neuralese_data:{REPO}/training/neuralese:{REPO}'}
    sft = [str(PYTHON), str(REPO / 'scripts/neuralese_data/sft_sources.py')]
    Path(work).mkdir(parents=True, exist_ok=True)
    with open(Path(work) / 'feed.log', 'a') as log:     # stdout stays the feed's JSON summary
        subprocess.run([*sft, 'extract', '--work', str(work), '--teacher', str(pool), '--out-name', f'episodes-{stamp}'],
                       check=True, env=env, cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        subprocess.run([*sft, 'replay', '--work', str(work), '--workers', str(workers), '--runtime-repo', str(REPO),
                        '--max-model-requests', str(REPLAY_MAX_MODEL_REQUESTS)], check=True, env=env, cwd=REPO,
                       stdout=log, stderr=subprocess.STDOUT)
    verdicts = [json.loads(line) for line in (Path(work) / 'replay.jsonl').read_text().splitlines() if line.strip()]
    keys = {f'teacher:{"/".join(Path(pool).parts[-2:])}:'}
    mine = [v for v in verdicts if any(str(v.get('key', '')).startswith(k) for k in keys)]
    import collections
    return dict(collections.Counter(str(v.get('outcome'))[:40] for v in mine))


def promote_fill_work(work, no_check=False):
    """The fill WORK's role (once): its sft-work dataset (sft_sources record_work) into ``grpo-sd-teacher``."""
    from natlang_neuralese.common import data_registry
    dataset = data_registry.dataset_id(work, 'sft-work')
    members = data_registry.role_paths(FILL_ROLE, repo=data_registry.registry_owner_repo())
    if Path(work).resolve().as_posix() in {Path(p).resolve().as_posix() for p in members}:
        return dataset, False
    promote(dataset, FILL_ROLE, None, no_check)
    return dataset, True


def feed(root=None, *, work=None, runtime=None, replay=True, register=True, dry_run=False, no_check=False,
         workers=6, resume=None):
    root = Path(root or inventor_root())
    work = Path(work or default_work())
    pending = pending_programs(root)
    summary = {'programs': [a['slug'] for a, _ in pending]}
    if dry_run or not pending:
        return summary
    base = root / 'authored'
    if resume:      # a pool this feed already rendered and wrote (a registration or replay that failed after it)
        pool = Path(resume)
        stamp = pool.name
        scratch = base / 'scratch' / stamp
        result = json.loads((pool / 'pool.json').read_text())
        exported = {r['task']['id'].split(':')[1] for r in read_jsonl(pool / 'authored.jsonl')}
        pending = [(a, h) for a, h in pending if a['slug'] in exported]
        summary['programs'] = [a['slug'] for a, _ in pending]
    else:
        stamp = time.strftime('%Y%m%d-%H%M%S')
        pool = base / stamp
        scratch = base / 'scratch' / stamp
        rows, refused = [], {}
        for admission, home in pending:
            mine, why = program_rows(admission, home, scratch / admission['slug'], runtime)
            rows += mine
            refused.update(why)
        result = write_pool(rows, pool)
        result['render_refused'] = refused
        (pool / 'pool.json').write_text(json.dumps(result, indent=1) + '\n')
    refused = result.get('render_refused') or {}
    summary.update(stamp=stamp, pool=str(pool), episodes=result['episodes'], turns=result['turns'],
                   refused=len(result['refused']) + len(refused))
    if register and result['episodes']:
        summary['dataset'] = register_pool(pool, stamp, result, [a['slug'] for a, _ in pending], no_check=no_check)
    if replay and register and result['episodes']:     # extract reads registered pools only
        summary['replay'] = replay_into(work, pool, stamp, workers=workers)
        if register:
            summary['fill_work'], summary['fill_work_promoted'] = promote_fill_work(work, no_check=no_check)
    with open(base / 'exported.jsonl', 'a') as handle:
        for admission, _ in pending:
            handle.write(json.dumps({'slug': admission['slug'], 'digest': admission.get('digest'), 'stamp': stamp,
                                     'at': time.time()}) + '\n')
    with open(base / 'feeds.jsonl', 'a') as handle:
        handle.write(json.dumps(summary) + '\n')
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('render')
    r.add_argument('--cases', action='append', required=True)
    r.add_argument('--out', required=True)
    r.add_argument('--source', default='reference')
    r.add_argument('--runtime')
    p = sub.add_parser('pool')
    p.add_argument('--input', action='append', required=True)
    p.add_argument('--out', required=True)
    f = sub.add_parser('feed')
    f.add_argument('--root')
    f.add_argument('--work')
    f.add_argument('--runtime')
    f.add_argument('--workers', type=int, default=6)
    f.add_argument('--no-replay', action='store_true')
    f.add_argument('--no-register', action='store_true')
    f.add_argument('--no-check', action='store_true', help='registry promotion without check-main --quick')
    f.add_argument('--dry-run', action='store_true')
    f.add_argument('--resume-pool', help='an already written ROOT/authored/<stamp> pool: register and replay it')
    args = parser.parse_args(argv)
    if args.command == 'render':
        cases = [c for path in args.cases for c in read_jsonl(path)]
        out = Path(args.out)
        rendered = render(cases, source=args.source, runtime=args.runtime, scratch=out.with_name(out.name + '.scratch'))
        save_rows(out, [row for row, _ in rendered if row is not None])
        print(json.dumps({'cases': len(cases), 'rendered': sum(r is not None for r, _ in rendered),
                          'refused': {c['id']: why for c, (r, why) in zip(cases, rendered) if r is None}}, indent=1))
    elif args.command == 'pool':
        rows = [row for path in args.input for row in read_jsonl(path)]
        print(json.dumps(write_pool(rows, args.out), indent=1))
    else:
        print(json.dumps(feed(args.root, work=args.work, runtime=args.runtime, replay=not args.no_replay,
                              register=not args.no_register, dry_run=args.dry_run, no_check=args.no_check,
                              workers=args.workers, resume=args.resume_pool), indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
