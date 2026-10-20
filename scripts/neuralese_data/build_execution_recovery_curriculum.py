#!/usr/bin/env python3
"""Build source-only TRAIN candidates for short typed execution recovery.

This constructs fresh oracle-backed task inputs. It never calls a model and never
creates a teacher response or training-admission claim.
"""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from scripts.neuralese_data.build_fresh_recovery_teacher_worlds import DOMAINS, current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds

REVISION = 'authored-short-execution-recovery-20261019/1'
GRAPH_REVISION = 'authored-private-state-execution-recovery-20261019/1'
TERMINATION_REVISION = 'authored-verify-return-execution-recovery-20261020/1'
COMPUTED_REVISION = 'authored-computed-typed-result-recovery-20261020/1'
TYPES = ('export type Row = { id:string; category:string; value:number; active:boolean; '
         'blocked:boolean; waiver:boolean };\n'
         'export type Policy = { category:string; minimum:number };\n'
         'export type Summary = { ids:string[]; total:number; count:number };\n')
VARIANTS = ('page-range-recovery', 'reserved-service-binding', 'typed-result-recovery')


def computed_result_worlds(*, revision: str = COMPUTED_REVISION):
    """Fresh typed computations, saved-result verification and one-error recovery."""
    for kind in ('number', 'boolean', 'string', 'record'):
        rng = random.Random(f'{revision}:{kind}')
        a, b, cost = rng.randrange(10, 35), rng.randrange(40, 70), rng.randrange(2, 9)
        label = f'cargo-{rng.randrange(100, 999)}'
        facts = {'first': a, 'second': b, 'cost': cost, 'label': label}
        total = a + b - cost
        expected = {'number': total, 'boolean': a + cost < b,
                    'string': f'{label}:{total}',
                    'record': {'label': label, 'total': total, 'ready': a + cost < b}}[kind]
        expression = {'number': 'fact.first + fact.second - fact.cost',
                      'boolean': 'fact.first + fact.cost < fact.second',
                      'string': 'fact.label + ":" + (fact.first + fact.second - fact.cost)',
                      'record': '{label:fact.label,total:fact.first+fact.second-fact.cost,ready:fact.first+fact.cost<fact.second}'}[kind]
        result_type = {'record': '"Result"'}.get(kind, kind)
        types = ('export type Facts = { first:number; second:number; cost:number; label:string };\n'
                 'export type Result = { label:string; total:number; ready:boolean };\n')
        policy = {'number': 'Return first plus second minus cost as a number.',
                  'boolean': 'Return whether first plus cost is strictly less than second as a boolean.',
                  'string': 'Return the label, a colon, and first plus second minus cost as a string.',
                  'record': 'Return a record with label equal to the fact label, total equal to first plus second minus cost, and ready equal to whether first plus cost is strictly less than second.'}[kind]
        service = types + '/** Read the complete computation facts. */\nexport function read():Facts {return ' + json.dumps(facts, separators=(',', ':')) + ';}\n'
        saved = json.dumps(expected, separators=(',', ':')) + '\n'
        group = f'{revision}:{kind}'
        for variant in ('compute-return', 'verify-saved-return', 'recover-write-verify-return'):
            task_id = f'{group}:{variant}'
            instruction = 'Read facts.read(). ' + policy + ' Return the computed value as the declared type.'
            compute = 'const fact = await facts.read(); const result = ' + expression + '; '
            code = compute + 'return result;'
            semantics = {'root': 'compute.nl', 'inputs': {}, 'expected': expected, 'operation': 'exact',
                         'services': {'facts': service}, 'files': {'types.ts': types}}
            if variant == 'verify-saved-return':
                instruction += ' result.json already contains a saved result. Read it, verify it against the facts, and return its typed value once it agrees.'
                code = compute + "const saved = JSON.parse(await folder.file('result.json').readText()); if(JSON.stringify(saved)!==JSON.stringify(result))throw new Error('Saved result differs from computed facts'); return saved;"
                semantics.update(folder_files={'result.json': saved}, expected_files={'result.json': saved})
            elif variant == 'recover-write-verify-return':
                instruction += ' Continue from runtime type feedback. Save the computed value as JSON in result.json with a final newline, read it back, and return its typed value once verified.'
                # A compiler-rejected declaration provides genuine failure feedback
                # without embedding a wrong form in the model-facing instructions.
                semantics['failure_seed'] = {'code': 'const broken: number = "unavailable"; return broken;'}
                semantics.update(folder_files={'result.json': 'null\n'}, expected_files={'result.json': saved})
                code = compute + "await folder.file('result.json').writeText(JSON.stringify(result)+'\\n'); const saved = JSON.parse(await folder.file('result.json').readText()); if(JSON.stringify(saved)!==JSON.stringify(result))throw new Error('Saved result differs from computed facts'); return saved;"
            semantics['files']['compute.nl'] = ('---\nargs: {}\nreturns: ' + result_type +
                '\nkind: directory-reducer\n---\n' + instruction + '\n')
            yield {'version': 'natlang.program/2', 'id': task_id, 'kind': 'lambda_source',
                'family': 'curriculum_computed_typed_result_recovery',
                'source': 'natlang-authored-computed-typed-result-recovery', 'split': 'train',
                'source_ids': [task_id], 'source_groups': [group], 'source_revisions': [revision],
                'license': 'project-generated', 'gold_sources': ['constructed-world-oracle'],
                'generation': {'generator': revision, 'independent_world': group,
                    'authorship': 'fresh constructed TRAIN facts; no held input, answer, or trace used',
                    'provider_calls': 0, 'teacher_trajectory': False},
                'curriculum': {'version': 'natlang.inline_curriculum/1',
                    'family': 'computed_typed_result_recovery', 'family_version': 1,
                    'shape': kind, 'variant': variant, 'pair_group': None, 'split_group': group,
                    'slice': 'computed_result', 'domain': 'other', 'mode': 'single_call', 'inline': 'optional',
                    'evidence': {'world': ['fresh service facts'], 'retrieved': [], 'background': []},
                    'assumptions': [], 'decisive': [], 'plausible_actions': [],
                    'minimum_sequence': ['compute from actual facts', 'verify any saved result', 'return the typed value'],
                    'reference': {'root': [['eval', {'code': code}],
                        ['return_result', {'status': 'success', 'value': expected}]], 'children': []}},
                'semantics': semantics}


def termination_worlds(*, revision: str = TERMINATION_REVISION):
    """Twelve independent TRAIN tasks; ordinary correct returns remain eligible."""
    for world in range(4):
        rng = random.Random(f'{revision}:world-{world}')
        value = rng.randrange(21, 900)
        group = f'{revision}:world-{world}'
        saved = json.dumps({'value': value}, separators=(',', ':')) + '\n'
        for variant in ('verify-existing-return', 'write-readback-return', 'typed-feedback-return'):
            task_id = f'{group}:{variant}'
            semantics = {'root': 'finish.nl', 'inputs': {'value': value},
                         'expected': value, 'operation': 'exact',
                         'folder_files': {'result.json': saved if variant == 'verify-existing-return' else '{}\n'},
                         'expected_files': {'result.json': saved if variant != 'typed-feedback-return' else '{}\n'}}
            if variant == 'verify-existing-return':
                instruction = ('Read result.json and return its numeric value. The saved record already contains the requested value. '
                               'After verifying the saved value, return that number as the completed result.')
                code = "const saved = JSON.parse(await folder.file('result.json').readText()); return saved.value;"
            elif variant == 'write-readback-return':
                instruction = ('Save the supplied value in result.json as an object with one numeric value field and a final newline. '
                               'Read the saved record back. Once its value equals the supplied value, return that number as the completed result.')
                code = ("await folder.file('result.json').writeText(JSON.stringify({value})+'\\n'); "
                        "const saved = JSON.parse(await folder.file('result.json').readText()); return saved.value;")
            else:
                instruction = ('Read the numeric fact from facts.value() and return it as a number. '
                               'When runtime feedback describes a type contract, continue with the value of the declared type and complete the return.')
                semantics['services'] = {'facts': f'/** Read the numeric fact. */\nexport function value():number {{return {value};}}\n'}
                semantics['failure_seed'] = {'code': f'const result: number = {json.dumps(str(value))}; return result;'}
                code = 'return await facts.value();'
            semantics['files'] = {'finish.nl': '---\nargs:\n  value: number\nreturns: number\nkind: directory-reducer\n---\n' + instruction + '\n'}
            yield {'version': 'natlang.program/2', 'id': task_id, 'kind': 'lambda_source',
                   'family': 'curriculum_verify_return_execution_recovery',
                   'source': 'natlang-authored-verify-return-execution-recovery', 'split': 'train',
                   'source_ids': [task_id], 'source_groups': [group], 'source_revisions': [revision],
                   'license': 'project-generated', 'gold_sources': ['constructed-world-oracle'],
                   'generation': {'generator': revision, 'independent_world': group,
                       'authorship': 'fresh constructed TRAIN facts; no held input, answer, or trace used',
                       'provider_calls': 0, 'teacher_trajectory': False},
                   'curriculum': {'version': 'natlang.inline_curriculum/1', 'family': 'verify_return_execution_recovery',
                       'family_version': 1, 'shape': 'one-number-one-file', 'variant': variant,
                       'pair_group': None, 'split_group': group, 'slice': 'verify_return',
                       'domain': 'other', 'mode': 'single_call', 'inline': 'optional',
                       'evidence': {'world': ['fresh numeric record'], 'retrieved': [], 'background': []},
                       'assumptions': [], 'decisive': [], 'plausible_actions': [],
                       'minimum_sequence': ['perform the requested operation', 'verify the actual typed result',
                                            'return the completed result'],
                       'reference': {'root': [['eval', {'code': code}],
                           ['return_result', {'status': 'success', 'value': value}]], 'children': []}},
                   'semantics': semantics}


def short_worlds(*, revision: str = REVISION, domains=DOMAINS):
    """Eight fresh service domains; five rows/two pages; three exact recovery variants."""
    for domain, (namespace, category) in domains.items():
        rng = random.Random(f'{revision}:{domain}:world-0')
        minimum = 60 + rng.randrange(20, 90)
        rows = [
            {'id': f'{domain}-19-{i:02d}', 'category': category,
             'value': minimum + offset, 'active': active, 'blocked': blocked, 'waiver': waiver}
            for i, (offset, active, blocked, waiver) in enumerate((
                (17, True, False, True), (3, False, False, False),
                (0, True, False, False), (29, True, True, True),
                (minimum + 9, True, False, False)))
        ]
        # A fixed-but-seeded shuffle makes each domain's page boundaries distinct
        # while retaining an independently recomputable exact oracle.
        rng.shuffle(rows)
        page_size = 3
        page_count = (len(rows) + page_size - 1) // page_size
        policy = {'category': category, 'minimum': minimum}
        selected = [row for row in rows if row['category'] == category and not row['blocked'] and
                    (row['waiver'] or (row['active'] and row['value'] >= minimum))]
        expected = {'ids': sorted(row['id'] for row in selected),
                    'total': sum(row['value'] for row in selected), 'count': len(selected)}
        predicate = ('record.category === policy.category && !record.blocked && '
                     '(record.waiver || (record.active && record.value >= policy.minimum))')
        service = (TYPES + f'const ROWS:Row[] = {json.dumps(rows, separators=(",", ":"))};\n'
                   f'/** Read page n (1-based); use page_count() for the full source range. */\n'
                   f'export function page(n:number):Row[] {{if(!Number.isInteger(n)||n<1||n>{page_count}) '
                   f'throw new Error("Page number must be an integer from 1 through {page_count}; use page_count() to read the declared range."); '
                   f'return ROWS.slice((n-1)*{page_size},n*{page_size});}}\n'
                   f'/** Number of pages in this source. */\nexport function page_count():number {{return {page_count};}}\n')
        group = f'{revision}:{domain}:world-0'
        children = [{'match': ['select_record(', row['id']], 'calls': [
            ['eval', {'code': f'return {predicate};'}],
            ['return_result', {'status': 'success', 'value': row in selected}]]}
            for row in rows]
        root = (f'const pages = await Promise.all(Array.from({{length: await {namespace}.page_count()}}, '
                f'(_, index) => {namespace}.page(index + 1))); const rows = pages.flat(); '
                'const decisions = await Promise.all(rows.map(async record => ({record, keep: await select_record(record, policy)}))); '
                'const selected = decisions.filter(item => item.keep).map(item => item.record); '
                'const result = {ids:selected.map(item => item.id).sort(), total:selected.reduce((sum,item)=>sum+item.value,0), count:selected.length}; '
                "await folder.file('summary.json').writeText(JSON.stringify(result)+'\\n'); "
                "const saved = JSON.parse(await folder.file('summary.json').readText()); return saved;")
        for variant in VARIANTS:
            if variant == 'page-range-recovery': seed = f'return {namespace}.page(0);'
            elif variant == 'reserved-service-binding': seed = f'const {namespace} = {namespace}.page(1); return {namespace};'
            else: seed = 'const result: Summary = {ids:[],total:"not-a-number",count:0}; return result;'
            task_id = f'{group}:{variant}'
            yield {
                'version': 'natlang.program/2', 'id': task_id, 'kind': 'lambda_source',
                'family': 'curriculum_short_execution_recovery', 'source': 'natlang-authored-short-execution-recovery',
                'split': 'train', 'source_ids': [task_id], 'source_groups': [group],
                'source_revisions': [revision], 'license': 'project-generated',
                'gold_sources': ['constructed-world-oracle'],
                'generation': {'generator': revision, 'independent_world': group,
                    'authorship': 'fresh constructed TRAIN service facts; no held input or answer used',
                    'provider_calls': 0, 'teacher_trajectory': False},
                'curriculum': {'version': 'natlang.inline_curriculum/1',
                    'family': 'short_execution_recovery', 'family_version': 1,
                    'shape': f'{domain}-five-rows-two-pages', 'variant': variant,
                    'pair_group': None, 'split_group': group, 'slice': 'short_tool_recovery',
                    'domain': 'relational', 'mode': 'single_call', 'inline': 'optional',
                    'evidence': {'world': [f'fresh {domain} service pages'], 'retrieved': [], 'background': []},
                    'assumptions': [], 'decisive': [], 'plausible_actions': [],
                    'minimum_sequence': ['read both pages in the declared range',
                        'ask the typed selector about every row',
                        'retain every selected ID and value',
                        'write and read back the complete typed summary',
                        'use any runtime contract feedback to continue'],
                    'reference': {'root': [['eval', {'code': root}],
                        ['return_result', {'status': 'success', 'value': expected}]], 'children': children}},
                'semantics': {'root': 'summarize.nl', 'files': {'types.ts': TYPES,
                    'summarize.nl': ('---\nargs:\n  policy: "Policy"\nreturns: "Summary"\nkind: directory-reducer\n---\n'
                        f'Summarize the complete {domain} service source using the supplied policy.\n'
                        'Read every page in its declared range. Ask select_record about every complete row. '
                        'Preserve each selected id and value, then write the complete typed summary, read it back, and return it. '
                        'When an operation reports a runtime contract, continue from that feedback.\n'),
                    'summarize/select_record.nl': ('---\nargs:\n  record: "Row"\n  policy: "Policy"\nreturns: "boolean"\n---\n'
                        'Return true exactly when the category matches, the record is not blocked, and either its waiver is true '
                        'or it is active with value at least policy.minimum. Return the boolean directly.\n')},
                    'inputs': {'policy': policy}, 'expected': expected,
                    'folder_files': {'summary.json': '{}\n'},
                    'expected_files': {'summary.json': json.dumps(expected, separators=(',', ':')) + '\n'},
                    'services': {namespace: service}, 'failure_seed': {'code': seed}}}


def graph_rows(*, revision: str = GRAPH_REVISION):
    """Three authored root programs with real private producer/consumer edges."""
    tasks = [
        ('producer-consumer-copy', 'violet harbor',
         'const writer: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Read the supplied evidence and write its exact text: violet harbor.`; '
         'const reader: Neuralese<(note: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read the input note and return an exact copy of its text.`; return await reader(await writer());',
         [('Read the supplied evidence and write its exact text: violet harbor.', '<|neuralese|>violet harbor<|/neuralese|>'),
          ('Read the input note and return an exact copy of its text.', 'violet harbor')], 'string'),
        ('producer-consumer-typed-total', 83,
         'const writer: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Read the evidence note and write its exact text: item-count=4; total=83.`; '
         'const reader: Neuralese<(note: Neuralese<string>) => Promise<number>> = nl.with<number>({})`Read the note and return the number after total= as a number.`; return await reader(await writer());',
         [('Read the evidence note and write its exact text: item-count=4; total=83.', '<|neuralese|>item-count=4; total=83<|/neuralese|>'),
          ('Read the note and return the number after total= as a number.', 83)], 'number'),
        ('two-producers-consumer-join', 'copper ridge',
         'const first: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Read the first supplied evidence and write its exact text: copper.`; '
         'const second: Neuralese<() => Promise<Neuralese<string>>> = nl.with<Neuralese<string>>({})`Read the second supplied evidence and write its exact text: ridge.`; '
         'const reader: Neuralese<(left: Neuralese<string>, right: Neuralese<string>) => Promise<string>> = nl.with<string>({})`Read both notes and return the first note, one space, and the second note.`; return await reader(await first(), await second());',
         [('Read the first supplied evidence and write its exact text: copper.', '<|neuralese|>copper<|/neuralese|>'),
          ('Read the second supplied evidence and write its exact text: ridge.', '<|neuralese|>ridge<|/neuralese|>'),
          ('Read both notes and return the first note, one space, and the second note.', 'copper ridge')], 'string'),
    ]
    for name, expected, root, children, result_type in tasks:
        task_id = f'{revision}:{name}'
        calls = len(children)
        reference_children = [{'match': prompt, 'value': value} for prompt, value in children]
        yield {'version': 'natlang.program/2', 'id': task_id, 'kind': 'lambda_source',
            'family': 'curriculum_private_state_execution_recovery',
            'source': 'natlang-authored-private-state-execution-recovery', 'split': 'train',
            'source_ids': [task_id], 'source_groups': [task_id], 'source_revisions': [revision],
            'license': 'project-generated', 'gold_sources': ['constructed-world-oracle'],
            'generation': {'generator': revision, 'authorship': 'fresh constructed TRAIN evidence; no held content used',
                'provider_calls': 0, 'teacher_trajectory': False},
            'curriculum': {'version': 'natlang.inline_curriculum/1', 'family': 'private_state_execution_recovery',
                'family_version': 1, 'shape': name, 'variant': 'complete-private-graph',
                'pair_group': None, 'split_group': task_id, 'slice': 'producer_consumer',
                'domain': 'other', 'mode': 'single_call', 'inline': 'optional',
                'evidence': {'world': ['fresh constructed evidence'], 'retrieved': [], 'background': []},
                'assumptions': [], 'decisive': [], 'plausible_actions': [],
                'minimum_sequence': ['write source evidence into private Neuralese state',
                    'read that state from the consumer', 'return the exact public typed answer'],
                'reference': {'root': [['eval', {'code': root}],
                    ['return_result', {'status': 'success', 'value': expected}],
                    ['reply', {'text': 'done'}]], 'children': reference_children},
                'trajectory_contract': {'version': 'natlang.trajectory_contract/1',
                    'authored_root_eval': True,
                    'authored_root_sha256': __import__('hashlib').sha256(root.encode()).hexdigest(),
                    'min_child_invocations': calls, 'min_state_edges_from_children': calls - 1,
                    'min_iteration_steps': 0}},
            'semantics': {'root': 'main.nl', 'files': {'main.nl':
                f'---\nargs: {{}}\nreturns: {result_type}\n---\n'
                'Complete the assigned typed operation using a private writer and a private reader. '
                'The writer preserves the exact evidence in its state. The reader consumes that state and returns the requested public value. '
                'Return the final value with the declared type.\n'}, 'inputs': {}, 'expected': expected,
                'operation': 'exact'}}


def protection_check(rows, proof):
    groups = {group for row in rows for group in row.get('source_groups', [])}
    holds, _ = load_decision_holds(sorted((ROOT / 'training/decision-holds').glob('*.jsonl')))
    held_programs = {identifier.split(':decision:', 1)[0] for identifier in holds}
    ids = {row['id'] for row in rows}
    if any(row.get('split') != 'train' for row in rows):
        raise ValueError('curriculum candidates must be TRAIN only')
    if groups & (proof['held'] | proof['quality'] | proof['pending']):
        raise ValueError('fresh curriculum groups intersect current held, quality, or pending protection')
    if ids & held_programs:
        raise ValueError('fresh curriculum program identity intersects current decision holds')
    return {'groups': len(groups), 'held_intersections': 0, 'quality_intersections': 0,
            'pending_intersections': 0, 'decision_hold_intersections': 0,
            'current_protection': proof['proof']}


def write_packet(out: Path, name: str, rows: list[dict], *, root_mode: str, group: int,
                 selection_protection: dict, generator: Path):
    folder = out / name
    folder.mkdir(parents=True, exist_ok=False)
    cases = folder / 'cases.jsonl'
    cases.write_text(''.join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n'
                              for row in rows), encoding='utf-8', newline='\n')
    source_proof = folder / 'selection-proof.json'
    worlds = len({group for row in rows for group in row['source_groups']})
    write_json_durable(source_proof, {'schema': 'natlang.fresh-authored-teacher-packet/1',
        'cases': len(rows), 'cases_sha256': sha256_file_hex(cases), 'worlds': worlds,
        'revision': rows[0]['source_revisions'][0], 'profile': name,
        'domain_counts': {'ordinary': len(rows)} if root_mode == 'student' else {'private_state_graph': len(rows)},
        'family_shape_counts': {variant: sum(row['curriculum']['variant'] == variant for row in rows)
                                for variant in sorted({row['curriculum']['variant'] for row in rows})},
        'protection': selection_protection['current_protection'],
        'held_and_pending_intersections': 0, 'training_admission': False,
        'canonical_programs_sha256': canonical_json_sha256_hex(rows),
        'generator_sha256': sha256_file_hex(generator), 'source_corpora': [],
        'qualification': 'complete exact reference replay required before queue; actual teacher traces require independent admission'})
    proof = folder / 'cases-proof.json'
    write_json_durable(proof, {'schema': 'natlang.strong-teacher-catalog-packet/1',
        'cases': len(rows), 'cases_binding': {'path': str(cases), 'sha256': sha256_file_hex(cases), 'bytes': cases.stat().st_size},
        'family': rows[0]['curriculum']['family'],
        'expected_targets_visible': False, 'source_generator': {'path': str(generator),
            'sha256': sha256_file_hex(generator), 'bytes': generator.stat().st_size},
        'source_selection': {'path': str(source_proof), 'sha256': sha256_file_hex(source_proof), 'bytes': source_proof.stat().st_size},
        'source_kind': 'fresh independently constructed TRAIN task inputs with executable complete references',
        'scope': 'short structured tool use, preserve evidence through typed write/readback, recover from actual runtime feedback; '
                 'private-state packet additionally requires observed complete producer-consumer topology',
        'teacher_calls': 0, 'training_admission': False,
        'collection_sampling': {'group': group, 'root_mode': root_mode}})
    return {'candidate_packet': {'path': str(proof), 'sha256': sha256_file_hex(proof), 'bytes': proof.stat().st_size},
            'selection_proof': {'path': str(source_proof), 'sha256': sha256_file_hex(source_proof), 'bytes': source_proof.stat().st_size},
            'cases': {'path': str(cases), 'sha256': sha256_file_hex(cases), 'bytes': cases.stat().st_size},
            'rows': len(rows), 'collection_sampling': {'group': group, 'root_mode': root_mode}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--protection', type=Path, required=True)
    parser.add_argument('--profile', choices=('existing', 'verify-return', 'computed-result'), default='existing')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError('curriculum output must be a new immutable directory')
    out.mkdir(parents=True, exist_ok=True)
    protection = current_protection(args.protection.resolve(strict=True))
    if args.profile in ('verify-return', 'computed-result'):
        rows = list(computed_result_worlds() if args.profile == 'computed-result' else termination_worlds())
        if len(rows) != 12: raise ValueError('verify-return curriculum cardinality changed')
        packets = {args.profile: write_packet(out, args.profile, rows, root_mode='student', group=4,
            selection_protection=protection_check(rows, protection), generator=Path(__file__).resolve())}
    else:
        ordinary = list(short_worlds())
        graphs = list(graph_rows())
        ordinary_protection = protection_check(ordinary, protection)
        graph_protection = protection_check(graphs, protection)
        if len(ordinary) != 24 or len(graphs) != 3:
            raise ValueError('curriculum cardinality changed unexpectedly')
        packets = {
            'ordinary': write_packet(out, 'ordinary', ordinary, root_mode='student', group=2,
                selection_protection=ordinary_protection, generator=Path(__file__).resolve()),
            'private-state': write_packet(out, 'private-state', graphs, root_mode='authored', group=16,
                selection_protection=graph_protection, generator=Path(__file__).resolve()),
        }
    receipt = {'schema': 'natlang.lfm-execution-recovery-curriculum/1',
        'revision': COMPUTED_REVISION if args.profile == 'computed-result' else (TERMINATION_REVISION if args.profile == 'verify-return' else REVISION),
        'graph_revision': GRAPH_REVISION, 'packets': packets,
        'teacher_responses_generated': 0, 'model_calls': 0, 'training_admission': False,
        'held_content_used': False, 'current_protection': protection['proof']}
    write_json_durable(out / 'prepared.json', receipt)
    print(json.dumps(receipt, sort_keys=True, indent=2))

if __name__ == '__main__':
    main()
