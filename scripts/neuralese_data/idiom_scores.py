#!/usr/bin/env python3
"""Per-decision idiom scores of natlang teacher decisions, for the LFM3 joint-SFT mixture (LFM3_RECIPE_20261028.md).

Reads native teacher turn rows (``natlang.teacher_training_turn.native/1``, one decision per row) or verified teacher
trajectories (``natlang.teacher_trajectory.native/1``, one row per program with its action turns) and writes one score
row per decision. The detectors are the idiom audit's (part B, 2026-10-28: regexes over the eval code of the decision's
tool calls) plus the recipe's target filters:

- categories (one primary per decision, rarest first): ``iteration`` (``iterateOn``), ``delegation`` (awaited ``nl``,
  per-item via ``map`` + ``Promise.all`` or an awaited ``for…of`` body), ``exact`` (sort/filter/reduce/count/join/
  slice/date arithmetic in eval, no ``nl``), ``scope`` (child invocations, typed ``return_result`` answers, declared
  input/service names used directly), ``broad`` (every other verified decision);
- positive-target filters: the root outcome is verified (``outcome.accepted`` and an accepted oracle), the decision is
  approved and not a failed action, it holds no un-awaited ``nl`` (an ``nl`` call neither awaited nor inside
  ``Promise.all``), it does not redeclare an input/service name in scope, and neither a family hold
  (training/source-reviews/family-holds.json) nor a pending item review covers its program or source groups.

The scores are classification heuristics over code text (regexes), recorded as such; they select and weight, they do
not admit. Usage:
  idiom_scores.py --source F [F ...] --corpus ID --out OUT.jsonl.zst [--kind turns|trajectories]
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts'))
sys.path.insert(0, str(REPO / 'training' / 'neuralese'))

SCHEMA = 'natlang.idiom-score/1'
NL_CALL = re.compile(r'\bnl\s*(?:<[^`(]*?>)?\s*(?:`|\()')
AWAITED_NL = re.compile(r'\bawait\s+nl\s*(?:<[^`(]*?>)?\s*(?:`|\()')
PROMISE_ALL = re.compile(r'\bPromise\.all\s*\(')
MAP_NL = re.compile(r'\.map\s*\(\s*(?:async\s*)?\(?[\w,\s{}\[\]]*\)?\s*=>[^;]*?\bnl\s*(?:<[^`(]*?>)?\s*(?:`|\()', re.S)
FOR_OF = re.compile(r'\bfor\s*\(\s*(?:const|let|var)\s+[^)]*\bof\b')
ITERATE = re.compile(r'\biterateOn\s*\(')
SORT = re.compile(r'\.sort\s*\(')
FILTER = re.compile(r'\.filter\s*\(')
EXACT = re.compile(r'\.(?:reduce|slice|join|every|some|find|findIndex|flatMap|localeCompare|toFixed)\s*\(|\bMath\.'
                   r'|\bnew\s+Date\b|\bDate\.(?:parse|UTC)\b|\.length\b')
DECIDE = re.compile(r'\bdecide\s*(?:<[^`(]*?>)?\s*(?:`|\()')
USE_LINE = re.compile(r'In eval you can use ([^;\n]+);')
CATEGORIES = ('iteration', 'delegation', 'exact', 'scope', 'broad')


def per_item_loop_nl(code: str) -> bool:
    for match in FOR_OF.finditer(code):
        if AWAITED_NL.search(code[match.end():match.end() + 600]):
            return True
    return False


def names_in_scope(messages) -> list[str]:
    for message in messages or []:
        if message.get('role') == 'user' and isinstance(message.get('content'), str):
            found = USE_LINE.search(message['content'])
            if found:
                return [n.strip() for n in found.group(1).replace(' and ', ',').split(',')
                        if re.fullmatch(r'\s*[A-Za-z_$][\w$]*\s*', n)]
    return []


def call_list(calls):
    for call in calls or []:
        fn = call.get('function', call) if isinstance(call, dict) else {}
        arguments = fn.get('arguments')
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except ValueError:
                arguments = {}
        yield fn.get('name'), arguments if isinstance(arguments, dict) else {}


def score_calls(calls, scope_names, *, child: bool) -> dict:
    s = collections.Counter()
    for name, arguments in call_list(calls):
        if name == 'return_result':
            s['return_result'] += 1
            if child:
                s['typed_child_answer'] += 1
        if name != 'eval' or not isinstance(arguments.get('code'), str):
            continue
        code = arguments['code']
        s['evals'] += 1
        n_nl, n_await = len(NL_CALL.findall(code)), len(AWAITED_NL.findall(code))
        s['nl_calls'] += n_nl
        s['nl_awaited'] += n_await
        if n_nl and n_nl > n_await and not PROMISE_ALL.search(code):
            s['unawaited_nl'] += 1
        if MAP_NL.search(code) and PROMISE_ALL.search(code):
            s['per_item_nl_promise_all'] += 1
        if per_item_loop_nl(code):
            s['per_item_nl_for_of'] += 1
        s['iterate'] += bool(ITERATE.search(code))
        s['sort'] += bool(SORT.search(code))
        s['filter'] += bool(FILTER.search(code))
        s['exact_ops'] += bool(EXACT.search(code))
        s['decide'] += bool(DECIDE.search(code))
        s['finish'] += arguments.get('finish') is True
        if scope_names:
            redeclared = any(re.search(r'\b(?:const|let|var)\s+' + re.escape(n) + r'\b', code) for n in scope_names)
            used = any(re.search(r'(?<![\w$.])' + re.escape(n) + r'\b', code) for n in scope_names)
            s['scope_redeclare'] += redeclared
            s['scope_direct'] += used and not redeclared
    return dict(s)


# Breadth domains (editing, files and folders, tool use, knowledge/search, proofs, agent and code work): a root decision
# of these families that shows none of the three execution idioms counts as breadth, not scope hygiene.
BROAD_FAMILY = re.compile(r'edit|folder|file|repo|code|tool|knowledge|qa\b|search|research|proof|agent|swe|contract|'
                          r'clef|bgkit|harness|writ', re.I)


def category(s: dict, *, child: bool, family: str | None = None) -> str:
    if s.get('iterate'):
        return 'iteration'
    if s.get('nl_awaited') or s.get('per_item_nl_promise_all') or s.get('per_item_nl_for_of'):
        return 'delegation'
    if (s.get('sort') or s.get('filter') or s.get('exact_ops')) and not s.get('nl_calls'):
        return 'exact'
    if not child and BROAD_FAMILY.search(family or ''):
        return 'broad'
    if child or s.get('typed_child_answer') or s.get('scope_direct'):
        return 'scope'
    return 'broad'


class Holds:
    """Family holds and pending item reviews (scripts/source_review.py), memoized per program."""
    def __init__(self):
        from source_review import pending_family_hold_reason, pending_unpinned_source_groups
        self.reason = pending_family_hold_reason
        self.pending_groups = pending_unpinned_source_groups()
        self.cache = {}

    def __call__(self, program: dict | None, groups) -> str | None:
        if set(groups or ()) & self.pending_groups:
            return 'pending item source review'
        if not isinstance(program, dict):
            return None
        key = program.get('id') or json.dumps(program.get('curriculum'), sort_keys=True)[:2000]
        if key not in self.cache:
            reason = self.reason(program)
            self.cache[key] = f'family hold: {reason[:120]}' if reason else None
        return self.cache[key]


def outcome_verified(outcome) -> bool | None:
    if not isinstance(outcome, dict):
        return None
    oracle = outcome.get('oracle')
    if outcome.get('accepted') is False or (isinstance(oracle, dict) and oracle.get('accepted') is False):
        return False
    if outcome.get('accepted') is True and (not isinstance(oracle, dict) or oracle.get('accepted') is True):
        return True
    return None


def finish(record: dict, s: dict, *, verified, approved, failed, held, child) -> dict:
    reasons = []
    if verified is not True:
        reasons.append('root outcome not verified' if verified is None else 'root outcome rejected')
    # Row-level approval is not a filter here: the renderer already admits a decision either by its own approval or
    # by its registry entry's admission (reviewed static batches), and only rendered decisions reach the mixture.
    if failed:
        reasons.append('failed action')
    if s.get('unawaited_nl'):
        reasons.append('un-awaited nl call')
    if s.get('scope_redeclare'):
        reasons.append('redeclares an input/service name')
    if held:
        reasons.append(held)
    family = ' '.join(str(record.get(k) or '') for k in ('family', 'curriculum_family'))
    record.update(reducer_features(record, record.get('task_modality'), record.get('task_kind'), s))
    record.update(scores=s, child=child, verified=verified, row_approved=approved, category=category(s, child=child, family=family),
                  positive=not reasons, exclusions=reasons)
    return record


def iter_rows(path: Path):
    if path.suffix == '.zst':
        proc = subprocess.Popen(['zstd', '-dcq', '--long=31', str(path)], stdout=subprocess.PIPE)
        stream = proc.stdout
    else:
        proc, stream = None, open(path, 'rb')
    for raw in stream:
        if raw.strip():
            yield json.loads(raw)
    if proc is not None and proc.wait():
        raise RuntimeError(f'zstd failed on {path}')


CALL_LINE = re.compile(r'You are inside this call:\s*([^\n]+)')
NEURALESE_MARK = re.compile(r'<\|neuralese\|>')
FOLDER = re.compile(r'\bfolders?\.[A-Za-z_$]|\bcallFolder\b|\.folder\s*\(')
FOLDER_APPLY = re.compile(r'\b\w*[fF]older\w*\.apply\s*\(')
FOLDER_PARAM = re.compile(r'\(\s*\w+\s*:\s*Folder\b')
NATLANG_TOOLS = frozenset(('eval', 'return_result', 'read_code', 'execution_plan'))


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return ''.join(part.get('text', '') if isinstance(part, dict) else str(part) for part in content)
    return ''


def call_identity(messages) -> dict:
    """The executing call's signature (function name, without the @eval:N site), an instruction digest (the
    function's identity across invocations), and a near-duplicate key of the task text (digits and whitespace
    normalized, first 400 characters)."""
    user = next((_text(m.get('content')) for m in messages or [] if m.get('role') == 'user'), '')
    found = CALL_LINE.search(user)
    signature = found.group(1).strip() if found else ''
    name = re.sub(r'@[^()]*', '', signature.split('(', 1)[0]).strip()
    body = user.split('Instructions:', 1)[-1] if 'Instructions:' in user else user
    instructions = body.split('In eval you can use', 1)[0].strip()
    normal = re.sub(r'\s+', ' ', re.sub(r'\d+', '0', instructions.lower()))[:400]
    import hashlib
    return {'function': name, 'signature': re.sub(r'@[^()]*', '', signature)[:200],
            'function_key': hashlib.sha256((name + '\n' + normal).encode()).hexdigest()[:16],
            'near_duplicate_key': hashlib.sha256(normal.encode()).hexdigest()[:16]}


def action_shape(calls, content) -> dict:
    """Character shares of the decision's output: natlang program actions (eval code, typed return_result answers,
    natlang runtime tools) versus generic tool use and plain chat text; text-emulated Neuralese markers and callable
    folder use (a character proxy for the assistant-token share, recorded as such)."""
    natlang = other = 0
    tools, markers, folders, applies, file_tools = [], 0, 0, 0, 0
    for name, arguments in call_list(calls):
        size = len(json.dumps(arguments, ensure_ascii=False))
        tools.append(name)
        code = arguments.get('code') if isinstance(arguments.get('code'), str) else ''
        markers += len(NEURALESE_MARK.findall(json.dumps(arguments, ensure_ascii=False)))
        folders += bool(FOLDER.search(code))
        applies += len(FOLDER_APPLY.findall(code))
        if name not in NATLANG_TOOLS and name:
            file_tools += 1
        if name in NATLANG_TOOLS:
            natlang += size
        else:
            other += size
    chat = len(_text(content))
    total = natlang + other + chat
    return {'tools': sorted(set(t for t in tools if t)), 'natlang_chars': natlang, 'generic_tool_chars': other,
            'chat_chars': chat, 'natlang_action_share': natlang / total if total else 0.,
            'text_neuralese_markers': markers + len(NEURALESE_MARK.findall(_text(content))), 'folder_calls': folders,
            'folder_apply_calls': applies, 'rooted_tool_calls': file_tools}


def reducer_features(record: dict, modality, kind, scores: dict) -> dict:
    """Directory-reducer use (kind/modality directory-reducer, a Folder first parameter, folder.apply commits, an
    iterateOn folder cursor carrying typed prior findings)."""
    reducer = (modality == 'directory-reducer' or kind == 'directory-reducer'
               or bool(FOLDER_PARAM.search(record.get('signature') or '')))
    return {'reducer': reducer, 'reducer_folder_param': bool(FOLDER_PARAM.search(record.get('signature') or '')),
            'reducer_iterate_cursor': bool(reducer and scores.get('iterate'))}


def turn_scores(row: dict, corpus: str, holds: Holds) -> dict:
    program = ((row.get('task') or {}).get('program_ir'))
    invocation = (row.get('source_ref') or {}).get('invocation_id') or ''
    child = invocation.startswith('task-') or bool(((row.get('decision') or {}).get('parent_invocation_id')))
    decision = row.get('decision') or {}
    s = score_calls((row.get('target') or {}).get('tool_calls'), names_in_scope(row.get('messages')), child=child)
    record = {'schema': SCHEMA, 'corpus': corpus, 'id': row['id'], 'trajectory': row.get('teacher_trajectory_id'),
              'program': row.get('program_id'), 'family': row.get('task_family') or row.get('family'),
              'curriculum_family': ((program or {}).get('curriculum') or {}).get('family'),
              'split': row.get('split'), 'source_groups': row.get('source_groups') or []}
    source_ref = row.get('source_ref') or {}
    capture = ((source_ref.get('host_result_capture') or {}).get('capture') or {})
    curriculum = (program or {}).get('curriculum') or {}
    target = row.get('target') or {}
    record.update(invocation=invocation or None, parent_invocation=capture.get('parent_call_id'),
                  invocation_status='done' if capture.get('complete') else None,
                  domain=curriculum.get('domain'), task_kind=row.get('task_kind'),
                  decision_index=decision.get('index'), **call_identity(row.get('messages')),
                  **action_shape(target.get('tool_calls'), target.get('content')))
    record['task_modality'] = row.get('task_modality')
    approved = (row.get('training_admission') or {}).get('approved')
    failed = bool(decision.get('failed_action'))
    return finish(record, s, verified=outcome_verified(row.get('outcome')), approved=approved, failed=failed,
                  held=holds(program, row.get('source_groups')), child=child)


def trajectory_scores(row: dict, corpus: str, holds: Holds):
    program = (row.get('task') or {}).get('program_ir') or {}
    outcome = row.get('outcome') or {}
    ledger = {item.get('invocation_id'): item for item in outcome.get('invocation_ledger') or []
              if isinstance(item, dict)}
    root = next((i for i, item in ledger.items() if item.get('parent_invocation_id') is None), None)
    verified = outcome_verified(outcome)
    for index, turn in enumerate(row.get('trajectory') or []):
        invocation = turn.get('invocation_id')
        child = invocation is not None and root is not None and invocation != root
        status = (ledger.get(invocation) or {}).get('completion_status')
        calls = (turn.get('model_response') or {}).get('raw_calls')
        s = score_calls(calls, names_in_scope(turn.get('context')), child=child)
        record = {'schema': SCHEMA, 'corpus': corpus, 'id': f'{row["id"]}#{index}', 'trajectory': row['id'],
                  'program': program.get('id'), 'family': program.get('family'),
                  'curriculum_family': (program.get('curriculum') or {}).get('family'),
                  'split': program.get('split'), 'source_groups': program.get('source_groups') or [],
                  'invocation_status': status, 'invocation': invocation,
                  'parent_invocation': (ledger.get(invocation) or {}).get('parent_invocation_id'),
                  'domain': (program.get('curriculum') or {}).get('domain'), 'task_kind': program.get('kind'),
                  'decision_index': index, **call_identity(turn.get('context')),
                  **action_shape(calls, ((turn.get('assistant') or {}).get('content')))}
        failed = status not in (None, 'done')  # a child invocation that did not complete is a failed child action
        yield finish(record, s, verified=verified, approved=None, failed=failed,
                     held=holds(program, program.get('source_groups')), child=child)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', type=Path, nargs='+', required=True)
    p.add_argument('--corpus', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--kind', choices=('turns', 'trajectories'), default='turns')
    a = p.parse_args(argv)
    holds = Holds()
    tmp = a.out.with_name(a.out.name + '.tmp')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    census = collections.Counter()
    with open(tmp, 'wb') as raw:
        zp = subprocess.Popen(['zstd', '-q', '-3', '-T2', '-c'], stdin=subprocess.PIPE, stdout=raw)
        for path in a.source:
            for row in iter_rows(path):
                records = ([turn_scores(row, a.corpus, holds)] if a.kind == 'turns'
                           else trajectory_scores(row, a.corpus, holds))
                for record in records:
                    census[f'{record["category"]}:{"positive" if record["positive"] else "excluded"}'] += 1
                    for reason in record['exclusions']:
                        census['exclusion:' + reason[:60]] += 1
                    zp.stdin.write(json.dumps(record, separators=(',', ':')).encode() + b'\n')
        zp.stdin.close()
        if zp.wait():
            raise RuntimeError('zstd failed')
    tmp.replace(a.out)
    summary = {'schema': SCHEMA + '#summary', 'corpus': a.corpus, 'sources': [str(s) for s in a.source],
               'census': dict(sorted(census.items()))}
    a.out.with_name(a.out.name.replace('.jsonl.zst', '') + '.summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
