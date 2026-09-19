#!/usr/bin/env python3
"""Paired local-vs-caller validation feedback, including tasks that must fail.

Expected failure is defined by each fixture, not inferred from model explanations.
Completion is scored separately from whether a validation error was encountered:
a model can fudge an answer before any validator feedback is sent.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang.native import NativeCallDecoder
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.surface import ToolSurface
from scripts.probe_parallel import completed_cases, total_usage
from natlang.values import load_program, dump, MISSING
from natlang.decoder import ChatTurn
from natlang import gbnf
from natlang.native import call_grammar
from natlang.gen.policy import native_text

FORCED_ACTION = ('call', {'function': 'count_items', 'to': 'return', 'inputs': {'items': 'args/items'}})


class ForcedFirstAction:
    """Controlled fault injection only; subsequent actions come from the model."""
    def __init__(self, inner):
        self.inner, self.first = inner, True

    @property
    def deadline(self):
        return getattr(self.inner, 'deadline', None)

    @deadline.setter
    def deadline(self, value):
        self.inner.deadline = value

    def chat(self, messages, tools, **kwargs):
        if self.first:
            self.first = False
            assert gbnf.accepts(call_grammar(tools), native_text([FORCED_ACTION]))
            return ChatTurn([FORCED_ACTION], completion_tokens=1)
        return self.inner.chat(messages, tools, **kwargs)


def controlled_cases():
    for broken in (False, True):
        body = ('Call count_items on items directly into return and finish. Do not wrap or transform its result.' if broken else
                'Call count_items on items, storing the number in a local n. Return the record { count: n }.')
        yield 'forced_broken_instruction' if broken else 'forced_execution_mistake', {'$lambda': {
            'type': 'Lambda<{ items: Num[] }, { count: Num }>', 'args': {'items': [2, 4, 6]},
            'instructions': body, 'codebase': {'count_items': {'args': {'items': 'Num[]'}, 'returns': 'Num',
                                                            'code': 'return args.items.length'}}}}, {'count': 3}, broken


def cases():
    record = '{ count: Num, evidence: Text }'
    rows = [
        ('valid_number', 'Num', 'Return the number 7.', 7, False),
        ('valid_record', record, 'Return count 3 and evidence exactly "three entries".',
         {'count': 3, 'evidence': 'three entries'}, False),
        ('incompatible_literal', 'Num', 'Return the text "blue" unchanged. Do not replace or convert it to a number.', None, True),
        ('missing_evidence', record, 'Write count 3 to return/count. Fill return/evidence with an exact quote '
         'from the supplied document. No document has been supplied. Do not invent evidence.', None, True),
        ('forbidden_return', 'Num', 'Finish this function without writing anything to return.', None, True),
        ('impossible_number', 'Num', 'Return a number strictly greater than itself.', None, True),
    ]
    for name, ty, body, expected, must_fail in rows:
        yield name, {'$lambda': {'type': f'Lambda<{{}}, {ty}>', 'instructions': body}}, expected, must_fail
    for flag in (True, False):
        yield f'conditional_{flag}', {'$lambda': {'type': 'Lambda<{ flag: Bool }, Num>', 'args': {'flag': flag},
            'instructions': 'If flag is true, return the number 7. Otherwise return the text "blue" unchanged; '
                            'do not convert or replace the text.'}}, 7 if flag else None, not flag
    yield 'effect_then_invalid', {'$lambda': {'type': 'Lambda<{}, Num>', 'effects': ['out.emit'],
        'instructions': 'Call emit_once(), storing its result as sent. Then return the text "blue" unchanged; '
                        'do not convert or replace it.',
        'codebase': {'emit_once': {'args': {}, 'returns': 'Num', 'effects': ['out.emit'],
                                  'code': 'fx.out.emit(1); return 1'}}}}, None, True


def score(status, value, expected, must_fail, log, transcript, emitted, detail=''):
    rejected = sum(x['kind'] in ('rejected', 'refused') for x in log)
    completion_feedback = sum(m['role'] == 'user' and
                              ('`return` has not been written' in m.get('content', '') or
                               '`return` is missing:' in m.get('content', '') or
                               'A sub-task has not been run yet:' in m.get('content', '')) for m in transcript)
    deliberate = any(x['action'].startswith('report_blocker ') and x['kind'] == 'blocked' for x in log)
    error = any(x['action'].startswith('report_error ') and x['kind'] == 'blocked' for x in log)
    failure_reason = ('validation' if detail.startswith('validation failed:') else
                      'deliberate_error' if error else 'deliberate_blocker' if deliberate else
                      'budget' if 'budget' in detail or 'deadline' in detail else 'other') if status != 'done' else None
    return {'correct_value': not must_fail and status == 'done' and value == expected,
            'invalid_task_accepted': must_fail and status == 'done',
            'expected_failure_preserved': must_fail and status == 'quiesced',
            'justified_failure': must_fail and status == 'quiesced' and failure_reason in ('validation', 'deliberate_blocker', 'deliberate_error'),
            'failure_reason': failure_reason,
            'invalid_accept_after_feedback': must_fail and status == 'done' and bool(rejected or completion_feedback),
            'invalid_accept_without_feedback': must_fail and status == 'done' and not (rejected or completion_feedback),
            'deliberate_blocker': deliberate, 'deliberate_error': error, 'validation_rejections': rejected,
            'completion_feedback_messages': completion_feedback,
            'duplicate_effect': len(emitted) > 1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8080')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--controlled-only', action='store_true', help='same forced rejected call, valid vs contradictory instructions')
    ap.add_argument('--system-file', type=Path, default=ROOT / 'natlang/prompts/tools_delegate.md')
    ap.add_argument('--no-error-tool', action='store_true')
    ap.add_argument('--policies', nargs='+', choices=['local', 'caller'], default=['local', 'caller'])
    ap.add_argument("--workers", type=int, default=4, help="independent cases; match server slots")
    ap.add_argument("--write-constraints", choices=["typed", "runtime"], default="typed")
    ap.add_argument("--trace-probs", action="store_true", help="save pre-mask selected-token probabilities and top alternatives")
    args = ap.parse_args()
    if args.workers < 1:
        ap.error("workers must be positive")
    prompt = args.system_file.read_text()
    rows = []
    selected = list(controlled_cases()) if args.controlled_only else list(cases())
    jobs = [(case, policy) for case in selected for policy in args.policies]
    started = time.monotonic()

    def run(job):
        (name, doc, expected, must_fail), policy = job
        probabilities = [] if args.trace_probs else None
        dec = NativeCallDecoder(args.server, timeout=120, write_constraints=args.write_constraints, probability_log=probabilities)
        root = load_program(doc)
        log, transcript = [], []
        episode_decoder = ForcedFirstAction(dec) if args.controlled_only else dec
        rt = Runtime(lambda lam: ToolAgent(episode_decoder, surface=ToolSurface(error_tool=not args.no_error_tool), validation_feedback=policy, system_prompt=prompt,
            temperature=0, max_turns=12, max_tokens=2000, max_seconds=90, log=log, transcript=transcript))
        start = time.monotonic()
        outcome, value = rt.run_root(root)
        actual = dump(value) if outcome.kind == 'done' else None
        row = {'case': name, 'policy': policy, 'forced_initial_action': args.controlled_only,
               'must_fail': must_fail, 'status': outcome.kind,
               'value': actual, 'draft_return': None if root.ret is MISSING else dump(root.ret),
               'detail': outcome.detail, 'emitted': rt.emitted,
               **score(outcome.kind, actual, expected, must_fail, log, transcript, rt.emitted, outcome.detail),
               'seconds': time.monotonic() - start, 'log': log, 'transcript': transcript}
        row['usage'] = dec.usage
        if probabilities is not None:
            row['probability_trace'] = probabilities
        return row

    for index, row in completed_cases(run, jobs, args.workers):
        row['case_index'] = index
        rows.append(row)
        rows.sort(key=lambda r: r['case_index'])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({'server': args.server, 'system_prompt': prompt, 'error_tool': not args.no_error_tool,
                                       'workers': args.workers, 'write_constraints': args.write_constraints, 'wall_seconds': time.monotonic() - started,
                                       'rows': rows, 'usage': total_usage(rows)}, indent=2) + '\n')
        print(f'{row["case"]}/{row["policy"]}: {row["status"]} valid={row["correct_value"]} '
              f'invalid_accept={row["invalid_task_accepted"]} rejects={row["validation_rejections"]}', flush=True)


if __name__ == '__main__':
    main()
