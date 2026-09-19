#!/usr/bin/env python3
"""Run the model interpreter on the same small programs under each marking instruction.

The programs use crisp callees to make their results independently checkable;
all statement selection, bindings and marking are still performed by the model.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang.decoder import LlamaServerDecoder
from natlang.native import NativeCallDecoder
from natlang.runtime import Runtime
from natlang.tool_agent import ToolAgent
from natlang.values import load_program, dump

STYLES = {
    'standalone': 'Mark completed lines with mark_done in a separate turn. Do not use done=.',
    'grouped': 'Group mark_done for previously completed lines with the next independent action. Do not use done=.',
    'en_passant': 'Prefer done= on write or call for the line it completes. Use mark_done for skipped lines or other completed work.',
    'mixed': 'Choose either mark_done or done=, whichever fits the current step.',
}


def cases(extended=False):
    inputs = [('nonempty', [True, False, True]), ('empty', [])]
    if extended:
        inputs.append(('all_false', [False, False]))
    for name, flags in inputs:
        body = ('function count(flags) -> Num\n\n'
                '  n = count_true(flags)\n'
                '  if n > 0:\n'
                '    return n\n'
                '  else:\n'
                '    return 0')
        yield name, {'$lambda': {
            'type': 'Lambda<{ flags: Bool[] }, Num>', 'instructions': body,
            'args': {'flags': flags}, 'codebase': {'count_true': {'args': {'flags': 'Bool[]'}, 'returns': 'Num',
            'description': 'Count the true flags exactly.', 'code': 'return args.flags.filter(Boolean).length'}}}}, sum(flags), {
                3: 'done', 4: 'done', 5: 'done' if any(flags) else 'skipped', 7: 'skipped' if any(flags) else 'done'}
    if extended:
        for early in (True, False):
            body = ('function choose(early) -> Num\n\n'
                    '  if early:\n'
                    '    return 11\n'
                    '  n = fallback()\n'
                    '  return n')
            yield f'early_return_{early}', {'$lambda': {
                'type': 'Lambda<{ early: Bool }, Num>', 'instructions': body,
                'args': {'early': early}, 'codebase': {'fallback': {'args': {}, 'returns': 'Num',
                'description': 'Return the fallback value 23.', 'code': 'return 23'}}}}, 11 if early else 23, {
                    3: 'done', 4: 'done' if early else 'skipped',
                    5: 'skipped' if early else 'done', 6: 'skipped' if early else 'done'}


def assess(marks, expected, log, style, transcript=None):
    """Score semantic lines separately from style. Headers/else syntax are not scored."""
    marks = {int(n): v for n, v in marks.items()}
    expected = {int(n): v for n, v in expected.items()}
    errors = [{'line': n, 'expected': v, 'actual': marks.get(n)}
              for n, v in expected.items() if marks.get(n) != v]
    calls = [(entry['action'].split(' ', 1)[0], json.loads(entry['action'].split(' ', 1)[1]))
             for entry in log]
    violations = []
    if style in ('standalone', 'grouped') and any('done' in a for _, a in calls):
        violations.append('used done= although this style forbids it')
    if transcript is not None:
        batches = [m['tool_calls'] for m in transcript if m.get('tool_calls')]
        grouped = [b for b in batches if any(c['function']['name'] == 'mark_done' for c in b)
                   and any(c['function']['name'] != 'mark_done' for c in b)]
        if style == 'standalone' and grouped:
            violations.append('grouped a mark with a work action')
        if style == 'grouped' and not grouped:
            violations.append('no mark was grouped with a work action')
    wrong_ranges = []
    for name, args in calls:
        span = args.get('done') if name in ('write', 'call', 'call_function') else None
        if name == 'mark_done' and not args.get('skipped'):
            span = [args['start'], args.get('end', args['start'])]
        if span is not None:
            first, last = span if isinstance(span, list) else (span, span)
            wrong_ranges.extend(n for n in range(first, last + 1) if expected.get(n) == 'skipped')
    return {'marks_correct': len(expected) - len(errors), 'marks_total': len(expected),
            'all_marks_correct': not errors, 'mark_errors': errors,
            'untaken_lines_marked_done': sorted(set(wrong_ranges)),
            'style_violations': violations,
            'style_compliant': not violations if transcript is not None or violations else None}


def summarize(rows):
    return {'episodes': len(rows), 'correct_results': sum(r['correct'] for r in rows),
            'correct_marking_episodes': sum(r['all_marks_correct'] for r in rows),
            'style_compliant_episodes': sum(r['style_compliant'] is True for r in rows),
            'style_unassessed_episodes': sum(r['style_compliant'] is None for r in rows)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--server', default='http://127.0.0.1:8081')
    ap.add_argument('--native', action='store_true')
    ap.add_argument('--extended', action='store_true')
    ap.add_argument('--system-file', type=Path, default=ROOT / 'natlang/prompts/tools_delegate.md')
    ap.add_argument('--rescore', type=Path, help='rescore a saved run without calling a model')
    ap.add_argument('--styles', nargs='+', choices=STYLES, default=list(STYLES)[:3])
    ap.add_argument('--out', type=Path, default=ROOT / 'runs/teacher-marking.json')
    args = ap.parse_args()
    if args.rescore:
        saved = json.loads(args.rescore.read_text())
        for row in saved['rows']:
            row.update(assess(row['marks'], row['expected_marks'], row['log'], row['style'], row.get('transcript')))
        saved['summary'] = summarize(saved['rows'])
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(saved, indent=2) + '\n')
        print(json.dumps(saved['summary']))
        return
    dec = NativeCallDecoder(args.server, timeout=180) if args.native else LlamaServerDecoder(args.server, timeout=180,
        chat_extra={'thinking_budget_tokens': 160}, tool_aliases={'call': 'call_function'})
    base = args.system_file.read_text()
    rows = []
    for style in args.styles:
        for name, doc, expected, marks in cases(args.extended):
            root = load_program(doc)
            log = []
            transcript = []
            rt = Runtime(lambda lam: ToolAgent(dec, system_prompt=base + '\n\n' + STYLES[style],
                                               temperature=0, log=log, transcript=transcript,
                                               max_turns=16, max_tokens=2400, max_seconds=240))
            start = time.monotonic()
            out, value = rt.run_root(root)
            row = {'style': style, 'case': name, 'status': out.kind, 'correct': out.kind == 'done' and dump(value) == expected,
                   **assess(root.marks, marks, log, style, transcript),
                   'marks': root.marks, 'expected_marks': marks, 'seconds': time.monotonic() - start,
                   'log': log, 'transcript': transcript}
            rows.append(row)
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({'server': args.server, 'rows': rows, 'usage': dec.usage,
                'system_prompt': base, 'surface_sha256': hashlib.sha256((ROOT / 'natlang/surface.py').read_bytes()).hexdigest(),
                'summary': summarize(rows)}, indent=2) + '\n')
            print(f"{style}/{name}: correct={row['correct']} marks={row['marks_correct']}/{len(marks)} "
                  f"{len(log)} actions {row['seconds']:.1f}s", flush=True)


if __name__ == '__main__':
    main()
