#!/usr/bin/env python3
"""Compare guided review orders on identical saved, labeled proposal contexts.

No actions are executed. This isolates review decisions from main-session drift.
Use a separate seed/group dataset from training; familiar templates are not a
novel-template evaluation. Incorrect decisions are retained, not retried.
"""
import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang.corpus import file_digest
from natlang.native import NativeCallDecoder
from natlang.tool_agent import review_tools, review_messages
from scripts.probe_parallel import completed_cases
from scripts.generate_agent_support import CHALLENGES


def summarize(rows):
    result = {}
    for variant, order in sorted({(r.get('prompt_variant', 'baseline'), r['order']) for r in rows}):
        subset = [r for r in rows if r['order'] == order and r.get('prompt_variant', 'baseline') == variant]
        matrix = Counter((r['expected'], r['decision']) for r in subset)
        result[variant + '/' + order] = {'total': len(subset), 'correct': sum(r['expected'] == r['decision'] for r in subset),
                         'false_rejections': sum(r['expected'] == 'approve' and r['decision'] != 'approve' for r in subset),
                         'bad_approvals': sum(r['expected'] != 'approve' and r['decision'] == 'approve' for r in subset),
                         'confusion': {f'{a}->{b}': n for (a,b),n in sorted(matrix.items())}}
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--server', default='http://127.0.0.1:8080')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--orders', nargs='+', choices=['reason_first','decision_first'], default=['reason_first','decision_first'])
    ap.add_argument('--prompts', nargs='+', choices=['baseline','repeat_instructions','checklist'], default=['baseline'])
    args = ap.parse_args()
    if args.out.exists():
        ap.error('refusing to overwrite existing results')
    examples = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    jobs = [(e, order, variant) for e in examples for order in args.orders for variant in args.prompts]
    def run(job):
        e, order, variant = job
        dec = NativeCallDecoder(args.server)
        prefix = e.get('pre_action_messages', e['messages'][:-1])
        messages = review_messages(copy.deepcopy(prefix), e['proposal'], e.get('check_call_index', 0), variant)
        challenge = next((c for c in CHALLENGES if e['messages'][-1]['content'].endswith(c)), None)
        if challenge:
            messages[-1]['content'] += '\n\n' + challenge
        answer = dec.review(messages, review_tools(order), temperature=0, seed=0, max_tokens=256)
        decision = 'invalid'
        if len(answer.calls) == 1 and answer.calls[0][0] == 'review_write':
            v = answer.calls[0][1]
            if v.get('decision') in ('approve','withdraw','error','blocker') and isinstance(v.get('reason'), str):
                decision = v['decision']
        return {'id': e['id'], 'expected': e['expected'], 'order': order, 'prompt_variant': variant, 'decision': decision,
                'challenge':challenge, 'calls': answer.calls, 'text': answer.text, 'usage': dec.usage}
    rows = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for index, row in completed_cases(run, jobs, args.workers):
        rows.append({'index': index, **row})
        rows.sort(key=lambda r: r['index'])
        doc = {'input': str(args.input), 'input_sha256': file_digest(args.input),
               'server': args.server, 'prompt_source_sha256': file_digest(ROOT / 'natlang/tool_agent.py'), 'rows': rows, 'summary': summarize(rows)}
        tmp = args.out.with_suffix('.tmp')
        tmp.write_text(json.dumps(doc, indent=2) + '\n')
        tmp.replace(args.out)
        print(f'{len(rows)}/{len(jobs)} {row["expected"]}->{row["decision"]}', flush=True)
    print(json.dumps(summarize(rows), indent=2))

if __name__ == '__main__':
    main()
