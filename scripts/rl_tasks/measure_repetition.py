#!/usr/bin/env python3
"""Template, instruction and program-structure repetition of an RL task supply.

Reads one or more `natlang.program/2` case JSONL files (train rows) and reports how much of the supply is the same
instruction text, the same whole program and the same program structure, plus which natlang/Neuralese constructs the
programs declare. Instruction and program texts are compared after normalizing digits and quoted literals, so a
substituted number, name or symbol does not count as a new template. Effective counts are exp(Shannon entropy) of
the row distribution over distinct keys: the number of equally used templates that would give the same spread.
Structure is declared program shape only (files, folder depth, declared Neuralese results/arguments, iteration,
decision readout); it is not evidence that an episode actually used those constructs.

  measure_repetition.py --cases A.jsonl [--cases B.jsonl ...] [--label NAME] -o report.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
from pathlib import Path

DIGITS = re.compile(r'\d+(?:\.\d+)?')
QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|`[^`]*`')
SPACE = re.compile(r'\s+')
# Broad-builder source modules that are separate program generators (scripts/rl_tasks/sources/*.py).
OWN_GENERATOR_SOURCES = frozenset({'composed', 'decisions', 'codebase-ops'})
# Semantics that verify effects or state (files, services, function checks, repaired codebases), not only an answer.
EFFECT_KEYS = frozenset({'effects', 'expected_files', 'function_checks', 'operation', 'service_scopes', 'files_oracle',
                         'codebase_check', 'state_oracle'})


def normalize(text):
    return SPACE.sub(' ', DIGITS.sub('#', QUOTED.sub('"…"', text))).strip()


def split_file(text):
    if text.startswith('---'):
        parts = text.split('---', 2)
        if len(parts) == 3:
            return parts[1], parts[2]
    return '', text


def frontmatter_field(header, name):
    """The raw text of one top-level frontmatter field, including its indented lines."""
    lines, out, inside = header.splitlines(), [], False
    for line in lines:
        if re.match(rf'^{name}:', line):
            inside = True
            out.append(line[len(name) + 1:])
        elif inside and (line.startswith(' ') or line.startswith('\t')):
            out.append(line)
        elif inside:
            break
    return '\n'.join(out)


def curriculum_of(row):
    return row.get('curriculum') or {}


def program_features(row):
    semantics = row['semantics']
    files = {path: text for path, text in semantics['files'].items() if path.endswith('.nl')}
    root = semantics['root']
    bodies, neuralese_results, typed_results, neuralese_args, readouts = [], 0, 0, 0, 0
    for path, text in files.items():
        header, body = split_file(text)
        bodies.append(body)
        returns = frontmatter_field(header, 'returns')
        args = frontmatter_field(header, 'args')
        if 'Neuralese<' in returns:
            neuralese_results += 1
            if re.search(r'Neuralese<\s*(?!string\s*>)', returns):
                typed_results += 1
        if 'Neuralese' in args:
            neuralese_args += 1
        if re.search(r'^readout:\s*decision', header, re.M):
            readouts += 1
    code = '\n'.join(text for text in semantics['files'].values())
    oracle = semantics.get('oracle')
    explicit = (curriculum_of(row).get('external') or {}).get('oracle_depth')
    effect_state = (explicit == 'effect-state' if explicit in ('effect-state', 'answer') else
                    bool(EFFECT_KEYS & set(semantics)) or (isinstance(oracle, dict) and oracle.get('level') in
                                                           ('constraints', 'authoring', 'state', 'effects')))
    generator = (row.get('generation') or {}).get('generator') or 'unknown'
    external = curriculum_of(row).get('external') or {}
    source = str(external.get('source') or '')
    if generator == 'natlang.broad_external_rl_tasks/1':
        # The broad builder hosts several source modules; the newer ones are their own generators.
        kind = (source if source in OWN_GENERATOR_SOURCES else 'codebase' if 'codebase' in source else
                'composed' if external.get('framing') not in (None, 'plain', 'neuralese-helper') else 'dataset')
        generator = generator + ':' + kind
    wrapper = generator.endswith(':dataset') and len(files) == 1 and curriculum_of(row).get('mode') == 'single_call'
    depth = max((path.count('/') for path in files), default=0)
    root_header, root_body = split_file(semantics['files'][root])
    curriculum = row.get('curriculum') or {}
    return {
        'instruction': normalize(root_body),
        'program': normalize('\n'.join(f'{p}\n{split_file(t)[1]}' for p, t in sorted(files.items()))),
        'structure': json.dumps({'nl_files': min(len(files), 6), 'depth': min(depth, 4),
                                 'neuralese_results': min(neuralese_results, 3), 'typed_results': min(typed_results, 3),
                                 'neuralese_args': min(neuralese_args, 3), 'readouts': min(readouts, 3),
                                 'iterate': 'iterateOn' in code, 'decide': bool(re.search(r'\bdecide\(', code)),
                                 'services': sorted(semantics.get('services') or {}),
                                 'mode': curriculum.get('mode')}, sort_keys=True),
        'multi_function': len(files) >= 2, 'depth': depth, 'neuralese_result': neuralese_results > 0,
        'typed_neuralese_result': typed_results > 0, 'neuralese_argument': neuralese_args > 0,
        'neuralese_fan_in': neuralese_results >= 2, 'iterate': 'iterateOn' in code, 'readout_decision': readouts > 0,
        'decide_call': bool(re.search(r'\bdecide\(', code)), 'services': bool(semantics.get('services')),
        'single_call': curriculum.get('mode') == 'single_call',
        'generator': generator, 'dataset_wrapper': wrapper, 'effect_state_oracle': effect_state,
        'call_depth_ge_3': depth >= 2,
        'folder_reducer': bool(re.search(r'^kind:\s*directory-reducer', '\n'.join(
            split_file(t)[0] for t in files.values()), re.M)),
        'skills': any('/skills/' in path and path.endswith('SKILL.md') for path in semantics['files']),
        'neuralese_iterate': 'iterateOn' in code and neuralese_results + neuralese_args > 0,
        'family': (curriculum.get('rl') or {}).get('family') or row.get('family') or 'unspecified',
        'level': (curriculum.get('rl') or {}).get('level'),
    }


def effective(counter):
    total = sum(counter.values())
    if not total:
        return 0.
    return math.exp(-sum((n / total) * math.log(n / total) for n in counter.values()))


CONSTRUCTS = ('multi_function', 'call_depth_ge_3', 'neuralese_result', 'typed_neuralese_result',
              'neuralese_argument', 'neuralese_fan_in', 'iterate', 'neuralese_iterate', 'folder_reducer',
              'readout_decision', 'decide_call', 'skills', 'services', 'effect_state_oracle', 'dataset_wrapper',
              'single_call')


def report(paths, label=None):
    rows = 0
    instructions, programs, structures = collections.Counter(), collections.Counter(), collections.Counter()
    family_instructions = collections.defaultdict(set)
    family_rows = collections.Counter()
    family_programs = collections.defaultdict(set)
    constructs = collections.Counter()
    generators, levels = collections.Counter(), collections.Counter()
    for path in paths:
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                if row.get('split', 'train') != 'train':
                    continue
                rows += 1
                features = program_features(row)
                instructions[features['instruction']] += 1
                programs[features['program']] += 1
                structures[features['structure']] += 1
                family_instructions[features['family']].add(features['instruction'])
                family_rows[features['family']] += 1
                generators[features['generator']] += 1
                levels[str(features['level'])] += 1
                for key in CONSTRUCTS:
                    constructs[key] += bool(features[key])
                family_programs[features['family']].add(features['program'])
    narrow = [family for family, texts in family_instructions.items() if len(texts) <= 3]
    broad = [family for family, texts in family_programs.items() if len(texts) >= 20]
    top = sum(n for _, n in instructions.most_common(10))
    return {
        'schema': 'natlang.rl-task-repetition/1', 'label': label, 'cases': [str(Path(p).resolve()) for p in paths],
        'rows': rows, 'families': len(family_rows),
        'families_with_le3_instruction_texts': len(narrow),
        'rows_in_families_with_le3_instruction_texts': sum(family_rows[f] for f in narrow),
        'families_with_ge20_program_texts': len(broad),
        'share_of_families_with_ge20_program_texts': round(len(broad) / len(family_rows), 4) if family_rows else None,
        'distinct_instruction_texts': len(instructions), 'effective_instruction_texts': round(effective(instructions), 1),
        'top10_instruction_share': round(top / rows, 4) if rows else None,
        'distinct_programs': len(programs), 'effective_programs': round(effective(programs), 1),
        'distinct_structures': len(structures), 'effective_structures': round(effective(structures), 1),
        'top_structure_share': round(structures.most_common(1)[0][1] / rows, 4) if rows else None,
        'constructs': {key: {'rows': value, 'share': round(value / rows, 4) if rows else None}
                       for key, value in sorted(constructs.items())},
        'generators': {key: {'rows': value, 'share': round(value / rows, 4)} for key, value in generators.most_common()},
        'levels': dict(sorted(levels.items())),
        'scope': 'declared program text and structure of train cases; not observed episode behaviour',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--cases', action='append', required=True)
    parser.add_argument('--label')
    parser.add_argument('-o', '--out')
    args = parser.parse_args(argv)
    result = report(args.cases, args.label)
    text = json.dumps(result, indent=1, sort_keys=False)
    if args.out:
        Path(args.out).write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    sys.exit(main())
