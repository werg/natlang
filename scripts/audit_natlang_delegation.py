#!/usr/bin/env python3
"""Stream native turns; count decisions, not repeated calls in their history.

Lexical eval candidates are not proof of successful execution. Child opening
contexts and return targets are counted independently to expose that distinction.
"""
import argparse
import collections
import json
import re
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('inputs', nargs='+', type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
c = collections.Counter()
runs = collections.defaultdict(lambda: collections.Counter())
families = collections.defaultdict(lambda: collections.Counter())
nl = re.compile(r'\bnl(?:\.with\([^)]*\))?(?:<[^`]*?>)?`')
for path in a.inputs:
    with path.open() as f:
        for line in f:
            if not line.strip(): continue
            r = json.loads(line)
            c['turns'] += 1
            run = str(r.get('source_ref', {}).get('trajectory_id', r['id']))
            task = r.get('task', {}).get('program_ir', {})
            sem = task.get('semantics', {})
            family = str(task.get('family', r.get('task_family', 'unknown')))
            def count(key):
                c[key] += 1; runs[run][key] += 1; families[family][key] += 1
            opening = next((m.get('content') for m in r.get('messages', []) if m['role'] == 'user'), '')
            call = re.match(r'You are inside this call: ([^\s(]+)\(', opening) if isinstance(opening, str) else None
            root = Path(sem.get('root', '')).stem
            child = bool(call and call[1] != root)
            if child:
                count('child_turns')
                if r.get('training_admission', {}).get('approved'): count('approved_child_turns')
                target = r.get('target', {})
                if not target.get('tool_calls') and isinstance(target.get('content'), str) and target['content']:
                    count('child_plain_reply_targets')
                    if r.get('training_admission', {}).get('approved'): count('approved_child_plain_reply_targets')
                    if len(target['content']) >= 16: count('long_child_plain_reply_targets')
            for t in r.get('target', {}).get('tool_calls', []):
                name = t['function']['name']
                try: args = json.loads(t['function']['arguments'])
                except (ValueError, TypeError): continue
                if name == 'return_result' and child:
                    count('child_return_targets')
                    if args.get('status') == 'success':
                        count('child_success_return_targets')
                        v = args.get('value')
                        if isinstance(v, (str, dict, list)) and len(v if isinstance(v, str) else json.dumps(v, separators=(',', ':'))) >= 16:
                            count('child_soft_eligible_return_targets')
                if name != 'eval': continue
                count('eval_targets')
                code = args.get('code', '')
                if not isinstance(code, str): continue
                inline = bool(nl.search(code))
                names = {Path(file).stem for file in sem.get('files', {}) if file.endswith('.nl')}
                named = any(re.search(r'\b' + re.escape(n) + r'\s*\(', code) for n in names)
                if inline: count('inline_nl_eval_candidates')
                if named: count('named_nl_eval_candidates')
                if inline or named: count('delegating_eval_candidates')
summary = {'schema': 'natlang.delegation-audit/1', 'counts': dict(c), 'runs': len(runs),
           'runs_with_delegating_eval_candidates': sum(bool(v['delegating_eval_candidates']) for v in runs.values()),
           'runs_with_child_turns': sum(bool(v['child_turns']) for v in runs.values()),
           'families': {k: dict(v) for k,v in sorted(families.items())},
           'caveats': ['Targets only; history occurrences are not counted again.',
                       'Eval lexical matches are candidates, not execution certificates.',
                       'All native turns counted; these counts do not grant training admission.']}
a.out.parent.mkdir(parents=True, exist_ok=True)
a.out.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps({k:v for k,v in summary.items() if k != 'families'}, indent=2))
