#!/usr/bin/env python3
"""Verified execution references and frozen proposal reviews, grouped against split leakage."""
import argparse
import copy
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from natlang import gbnf
from natlang.corpus import file_digest, digest
from natlang.gen.policy import ReferenceAgent, native_text
from natlang.gen.programs import Plan
from natlang.native import call_grammar, _strip_private
from natlang.runtime import Runtime
from natlang.surface import ToolSurface
from natlang.tool_agent import review_messages, review_tools
from natlang.values import load_program, dump
from scripts.generate_failures import matched_cases


def lesson_ids(case):
    name = case['name']
    family = name.split('_')[0]
    return {'literal':['quote_vs_semantics'], 'evidence':['missing_vs_empty'],
            'conditional':['branch_fidelity'], 'bounds':['honest_impossibility'],
            'binding':['exact_destination'], 'effect':['ordered_effects']}.get(
                family, ['count_vs_element', 'source_copy'])


CHALLENGES = (
    'Are you sure? Recheck the instructions and evidence. Stand by a correct proposal; withdraw a wrong one.',
    'Could this be a mistake? A repeated question is not new evidence. Explain briefly whether the exact proposal remains justified.',
    'Check whether this makes the task look successful by changing a requirement. If it does not, do not invent a problem merely because you were challenged.',
)


def support_cases(seed, group):
    # Retain matched impossibility/evidence controls, without injected recovery.
    for case in matched_cases(seed, group):
        case = copy.deepcopy(case)
        if case['name'] == 'binding_error':
            case['allowed_failure_call'] = dict(case['fault'][1])
        case['fault'] = None
        if case['name'] == 'binding_repair':
            case['steps'] = [
                ('call', {'function': 'size_of', 'to': 'let/size', 'inputs': {'items': 'args/items'}, 'done': 1}),
                ('write', {'path': 'return/size', 'type': 'Num', 'source': 'let/size', 'done': 2})]
        if case['name'].startswith('effect_'):
            case['steps'][0][1]['done'] = 1
            del case['steps'][1]
            if case['name'] == 'effect_success':
                case['steps'][1] = ('write', {'path': 'return', 'type': 'Num', 'source': 'let/sent', 'done': 2})
                del case['steps'][2:]
        yield case
    rng = random.Random(f'{seed}:{group}:support')
    # Different domains, scalar types, lengths, and record depths. Values differ
    # from lengths so a copied element cannot accidentally pass as a count.
    for text in (False, True):
        ty = 'Text' if text else 'Num'
        collection = 'tickets' if text else 'shipments'
        context = 'support triage' if text else 'shipment reporting'
        items = [rng.choice(['cedar', 'amber', 'harbor']) if text else rng.randint(20, 99)
                 for _ in range(rng.randint(1, 7))]
        for count in (False, True):
            fn = ('count_' if count else 'first_') + collection
            ret = 'Num' if count else ty
            expected = len(items) if count else items[0]
            field = 'total' if count else 'first'
            nested = group % 2 == 0
            rtype = f'{{ summary: {{ {field}: {ret} }} }}' if nested else f'{{ {field}: {ret} }}'
            dest = f'return/summary/{field}' if nested else f'return/{field}'
            instruction = rng.choice([
                f'Call {fn}({collection}), saving its result in local result.',
                f'Compute result by calling {fn} with {collection}.',
                f'Use {fn} on {collection} and store the answer in local result.'])
            body = f'For {context}: ' + instruction + f'\nCopy local result into {dest}.'
            doc = {'type': f'Lambda<{{ {collection}: {ty}[] }}, {rtype}>', 'instructions': body,
                   'args': {collection: items}, 'codebase': {fn: {'args': {'items': f'{ty}[]'},
                    'returns': ret, 'code': 'return args.items.length' if count else 'return args.items[0]'}}}
            value = {field: expected}
            if nested:
                value = {'summary': value}
            yield {'name': f'{ty}_{"count" if count else "element"}', 'root': {'$lambda': doc},
                   'steps': [('call', {'function': fn, 'to': 'let/result', 'inputs': {'items': f'args/{collection}'}, 'done': 1}),
                             ('write', {'path': dest, 'type': ret, 'source': 'let/result', 'done': 2})],
                   'expected': value, 'failure': None, 'effects': []}


def wrong_value(value):
    """A type-preserving near miss, rather than an obvious out-of-type sentinel."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "."
    if isinstance(value, dict):
        result = copy.deepcopy(value)
        key = next(iter(result))
        result[key] = wrong_value(result[key])
        return result
    raise ValueError(f"No contrast for {value!r}")


def generate_group(seed, group, *, state_view=False, include_reviews=True):
    execution, reviews = [], []
    for c in support_cases(seed, group):
        samples = []
        rt = Runtime(lambda lam: ReferenceAgent(Plan('calls', steps=c['steps']), samples, surface=ToolSurface(state_view=state_view),
                                                   system_prompt=(ROOT / "natlang/prompts/tools_small_support.md").read_text(),
                                                   terminal_tool=group % 2 == 0))
        out, value = rt.run_root(load_program(c['root']))
        assert rt.emitted == c['effects']
        if c['failure']:
            assert out.kind == 'quiesced' and out.detail.startswith(c['failure'] + ':')
        else:
            assert out.kind == 'done' and dump(value) == c['expected'], (c, out, value)
        family = f'support_{c["name"]}'
        pid = f'support:{seed}:{group}'
        for i, sample in enumerate(samples):
            execution.append({**sample, 'tools': _strip_private(sample['tools']),
                              'id': f'{pid}:{c["name"]}:{i}', 'program_id': pid, 'family': family})
            if not include_reviews:
                continue
            target = sample['target'].get('tool_calls', [])
            if not target:
                continue
            calls = [(t['function']['name'], json.loads(t['function']['arguments'])) for t in target]
            candidates = []
            name, args = calls[0]
            if name in ('write', 'call'):
                if name == 'call':
                    support = (f"The program requires {args['function']} into {args['to']}. "
                               f"The proposal uses that destination and the required input references {json.dumps(args.get('inputs', {}))}.")
                else:
                    required = c['expected']
                    for part in args['path'].split('/')[1:]:
                        required = required[part]
                    support = (f"The required value at {args['path']} is {json.dumps(required)}. " +
                               (f"The computed source {args['source']} holds that value; copying it preserves its type."
                                if 'source' in args else "This proposal writes that value without changing the requirement."))
                candidates.append((calls, 'approve', support))
                candidates.append(([('report_error', {'message':'The task cannot be completed as written.'})],
                                   'withdraw', 'This error report skips an earlier required operation. Execute that operation before reporting the later conflict.'
                                   if c['failure'] else 'The given inputs and requested operation permit this task. An unsupported failure claim should be withdrawn; continue the required work.'))
                if name == 'call':
                    candidates.append(([('mark_done', {'start': args.get('done', 1)})], 'withdraw',
                                       'The required call has not executed. Marking its line cannot perform its work.'))
                    wrong = {**args, 'to': 'return'}
                    if args['to'] != 'return':
                        candidates.append(([('call', wrong)], 'withdraw',
                                           'The instruction names a local destination. This proposal changes it; the task itself remains executable.'))
                else:
                    wrong = {k: v for k, v in args.items() if k != 'source'}
                    actual = c['expected']
                    for part in args['path'].split('/')[1:]:
                        actual = actual[part]
                    wrong['value'] = wrong_value(actual)
                    candidates.append(([('write', wrong)], 'withdraw',
                                       'The proposed value does not match the required result. The original task can still be completed.'))
            elif name in ('report_error', 'report_blocker'):
                candidates.append((calls, 'approve', args.get('message', args.get('missing'))))
                # A tempting write at the same pre-error state, not a corrected program.
                candidates.append(([('write', {'path': 'return', 'type': 'Num', 'value': 0})],
                                   'error' if name == 'report_error' else 'blocker',
                                   args.get('message', args.get('missing'))))
                if c['name'] == 'binding_error':
                    redirected = {**c['allowed_failure_call'], 'to':'return/size'}
                    candidates.append(([('call', redirected)], 'error',
                                       'The program explicitly requires the scalar call result directly in the record return. Appending a field changes that destination. The original task has a type conflict.'))
            for j, (candidate, decision, reason) in enumerate(candidates):
                tools = review_tools()
                verdict = [('review_write', {'reason': reason, 'decision': decision})]
                native = native_text(verdict)
                assert gbnf.accepts(call_grammar(tools, allow_reply=False, single_call=True), native)
                messages = review_messages(sample['messages'], candidate, 0,
                    ('baseline', 'repeat_instructions', 'checklist')[group % 3])
                # Every label sees the same challenge distribution. Pressure to
                # reconsider must not leak whether the proposal is correct.
                messages[-1]['content'] += '\n\n' + CHALLENGES[group % len(CHALLENGES)]
                reviews.append({'id': f'{pid}:{c["name"]}:{i}:review{j}', 'program_id': pid,
                                'contrast_group':f'{pid}:{c["name"]}:{i}',
                                'lesson_ids':lesson_ids(c) + ['honest_check_and_persistence'] +
                                    (['closure_is_not_execution'] if candidate[0][0]=='mark_done' else []),
                                'proposal_justified':decision == 'approve',
                                'task_feasible': None if c['failure']=='blocked' else not bool(c['failure']),
                                'label_source':'executed_reference_and_constructed_contrast',
                                'confidence':None,
                                'family': family, 'messages': messages,
                                'prompt_variant': ('baseline', 'repeat_instructions', 'checklist')[group % 3],
                                'tools': tools, 'proposal': candidate, 'expected': decision,
                                'pre_action_messages': sample['messages'], 'check_call_index': 0,
                                'target': {'role': 'assistant', 'content': '', 'tool_calls': [
                                    {'type': 'function', 'function': {'name': 'review_write',
                                     'arguments': json.dumps(verdict[0][1])}}]},
                                'native_target': native, 'skill': 'review_write', 'kind': 'review'})
    return execution, reviews


def generate_job(job):
    seed, group, state_view, include_reviews = job
    return generate_group(seed, group, state_view=state_view, include_reviews=include_reviews)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', type=int, default=100)
    ap.add_argument('--seed', type=int, default=104)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--reviews', type=Path, help='optional separate review-training corpus')
    ap.add_argument('--state-view', action='store_true', help='experimental full state in reference histories')
    args = ap.parse_args()
    if min(args.groups, args.workers) < 1 or (args.reviews and args.out.resolve() == args.reviews.resolve()):
        ap.error('positive groups and distinct output files required')
    for p in [args.out] + ([args.reviews] if args.reviews else []):
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            ap.error(f'refusing overwrite: {p}')
    counts = Counter()
    with args.out.open('x') as ex, (args.reviews.open('x') if args.reviews else nullcontext()) as re, ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = [(args.seed, group, args.state_view, bool(args.reviews)) for group in range(args.groups)]
        for rows, reviews in pool.map(generate_job, jobs):
            for f, data in ((ex, rows), (re, reviews)):
                for row in data:
                    f.write(json.dumps(row) + '\n')
            counts['execution_turns'] += len(rows)
            counts.update(r['expected'] for r in reviews)
    manifest = {'seed': args.seed, 'groups': args.groups, 'workers': args.workers, 'state_view': args.state_view, 'counts': dict(counts),
                'execution_sha256': file_digest(args.out), 'reviews_sha256': file_digest(args.reviews) if args.reviews else None,
                'generator_sha256': file_digest(Path(__file__)),
                'sources': digest([(str(p.relative_to(ROOT)), file_digest(p)) for p in
                                  sorted((ROOT / 'natlang').rglob('*.py')) + sorted((ROOT / 'natlang/prompts').glob('*.md'))])}
    args.out.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    main()
