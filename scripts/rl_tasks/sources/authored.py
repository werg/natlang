"""Authored RL tasks (`natlang.rl-authored-task/1`) -> ordinary `natlang.program/2` curriculum cases.

Authoring workers (plans/neuralese/RL_TASK_AUTHORING_KIT.md) write one JSON object per task: a complete natlang
program (root function plus helper functions in its callable folder), the invocation inputs, the exact expected
result, an oracle kind, meaningful wrong answers and type-valid alternative gold forms. This converter checks the
authored shape, assigns the split from the world identity (related tasks share a world and therefore a split),
records the author model as provenance and the scripted structure smoke, and emits the same case format the
collector, verify.mjs and the RL controller already use. It never invents gold, wrong answers or references.

  python scripts/rl_tasks/sources/authored.py IN.jsonl OUT.jsonl
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import CURRICULUM_VERSION, slug, text_digest, unit_split  # noqa: E402

SCHEMA = 'natlang.rl-authored-task/1'
GENERATOR = 'natlang.rl-authored-tasks/1'
CONSTRUCTS = frozenset({'call_depth_3', 'neuralese_relay', 'typed_neuralese', 'neuralese_fan_in', 'iterate',
                        'iterate_neuralese', 'folder_reducer', 'decision_readout', 'skills', 'multi_operation_codebase',
                        'effect_state'})
ORACLES = frozenset({'exact', 'normalized'})
NAME = re.compile(r'^[a-z][a-z0-9_]*$')
FILE = re.compile(r'^[a-z][a-z0-9_]*(/[a-z][a-z0-9_-]*)*(\.nl|\.ts|/SKILL\.md)$')
# World-folder items: relative paths of plain segments (no '..', no absolute paths, no hidden files).
FOLDER_FILE = re.compile(r'^[A-Za-z0-9_][A-Za-z0-9_.-]*(/[A-Za-z0-9_][A-Za-z0-9_.-]*)*$')


def folder_map(task, key):
    value = task.get(key)
    if value is None:
        return None
    if not isinstance(value, dict) or not value or not all(isinstance(text, str) for text in value.values()):
        fail(task, f'{key} must map relative file paths to text')
    for path in value:
        if not FOLDER_FILE.match(path) or '..' in path.split('/'):
            fail(task, f'{key} path {path!r} must be relative, without .. or hidden segments')
    return value


def reference_actions(task):
    reference = task.get('reference')
    if reference is None:
        return {'root': [['return_result', {'status': 'success', 'value': task['expected']}]]}
    if isinstance(reference, list):
        return {'root': reference}
    if (not isinstance(reference, dict) or not isinstance(reference.get('root'), list)
            or not isinstance(reference.get('children', []), list)):
        fail(task, 'reference is a root action list or {"root": [...], "children": [{"match": ..., "value": ...}]}')
    return {'root': reference['root'], **({'children': reference['children']} if reference.get('children') else {})}


def fail(task, message):
    raise ValueError(f'{task.get("id", "?")}: {message}')


def convert(task):
    if task.get('schema') != SCHEMA:
        fail(task, f'schema must be {SCHEMA}')
    for key in ('id', 'author', 'family', 'world', 'level', 'constructs', 'root', 'files', 'inputs', 'expected',
                'oracle', 'oracle_depth', 'wrong_answers', 'alias_answers', 'smoke'):
        if key not in task:
            fail(task, 'missing ' + key)
    if not NAME.match(task['family']):
        fail(task, 'family must be snake_case')
    if not isinstance(task['id'], str) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{2,80}', task['id']):
        fail(task, 'id must be lowercase letters, digits and dashes')
    if type(task['level']) is not int or not 0 <= task['level'] <= 3:
        fail(task, 'level must be 0, 1, 2 or 3')
    constructs = task['constructs']
    # An empty list is a correct task with no construct claim (a fake tag removed rather than simulated).
    if not isinstance(constructs, list) or not set(constructs) <= CONSTRUCTS:
        fail(task, 'constructs must be a list (empty when no construct applies) within ' + ', '.join(sorted(CONSTRUCTS)))
    if task['oracle'] not in ORACLES:
        fail(task, 'oracle must be exact or normalized')
    if task['oracle_depth'] not in ('answer', 'effect-state'):
        fail(task, 'oracle_depth must be answer or effect-state')
    if (task['oracle_depth'] == 'effect-state') != ('effect_state' in constructs):
        fail(task, 'oracle_depth effect-state and construct effect_state go together')
    files, root = task['files'], task['root']
    if root not in files or '/' in root or not root.endswith('.nl'):
        fail(task, 'root must be a top-level .nl file present in files')
    stem = root[:-3]
    for path in files:
        if not FILE.match(path) or (path != root and not path.startswith(stem + '/')):
            fail(task, f'file {path} must live in the root callable folder {stem}/')
        if path.endswith('.nl') and not files[path].startswith('---\n'):
            fail(task, f'{path} needs YAML frontmatter (args, returns)')
    folders, expected_folders = folder_map(task, 'folders'), folder_map(task, 'expected_folders')
    if folders is not None:
        if 'folder_reducer' not in constructs:
            fail(task, 'folders go with the folder_reducer construct')
        if not re.search(r'^kind:\s*directory-reducer\s*$', files[root].split('---', 2)[1], re.M):
            fail(task, 'folders need a root with `kind: directory-reducer` (the folder is the root\'s world)')
    if expected_folders is not None and folders is None:
        fail(task, 'expected_folders requires folders')
    if expected_folders is not None and task['oracle_depth'] != 'effect-state':
        fail(task, 'expected_folders is an effect-state oracle')
    wrong_code = task.get('wrong_code', [])
    if not isinstance(wrong_code, list) or not all(isinstance(code, str) and code for code in wrong_code):
        fail(task, 'wrong_code must be a list of root eval scripts')
    if wrong_code and task.get('reference') is None:
        fail(task, 'wrong_code replaces the last reference eval; give a reference')
    wrong, aliases = task['wrong_answers'], task['alias_answers']
    canonical = lambda value: json.dumps(value, sort_keys=True)  # noqa: E731
    if len(wrong) < 2 or any(canonical(w) == canonical(task['expected']) for w in wrong):
        fail(task, 'at least two wrong answers, none equal to expected')
    if not isinstance(aliases, list) or any(canonical(a) == canonical(w) for a in aliases for w in wrong):
        fail(task, 'alias_answers is a list of forms of the correct answer; none may equal a wrong answer')
    author = task['author']
    if not isinstance(author, dict) or not author.get('model') or not author.get('worker'):
        fail(task, 'author needs model and worker')
    smoke = task['smoke']
    if not isinstance(smoke, dict) or not isinstance(smoke.get('code'), str) or not isinstance(smoke.get('children'), dict):
        fail(task, 'smoke needs code (root eval script) and children (scripted helper answers by name)')
    world = 'authored:' + task['world']
    split = unit_split(world, 'train-carve', salt='rl-authored-20261103')
    family = 'auth_' + task['family']
    case_id = f'auth:{slug(task["family"], 40)}:{slug(task["id"], 80)}:{text_digest(json.dumps(task, sort_keys=True))[:8]}'
    root_body = files[root].split('---', 2)[-1]
    return {
        'version': 'natlang.program/2', 'id': case_id, 'kind': 'lambda_source', 'family': family,
        'source': 'authored-rl:' + author['model'], 'split': split, 'source_ids': ['authored:' + task['id']],
        'source_groups': sorted({world, *('authored-group:' + g for g in task.get('groups', []))}),
        'source_revisions': [GENERATOR], 'license': 'authored', 'gold_sources': ['authored-exact-oracle'],
        'generation': {'generator': GENERATOR, 'teacher_generation': True, 'author': author,
                       'method': 'program, inputs, exact expected result and controls authored by a model worker; '
                                 'qualified by reference, negative and alias replay; no sampled trajectory'},
        'curriculum': {
            'version': CURRICULUM_VERSION, 'family': family, 'family_version': 1, 'shape': task.get('shape', family),
            'variant': 'base', 'pair_group': None, 'split_group': world, 'slice': 'nested_scoped',
            'domain': 'other', 'mode': 'single_call', 'inline': 'optional',
            'evidence': {'world': [], 'retrieved': [], 'background': []}, 'assumptions': [], 'decisive': [],
            'plausible_actions': [], 'minimum_sequence': [],
            'reference': reference_actions(task),
            'rl': {'level': task['level'], 'family': family, 'basis': 'authored-exact-oracle-reference-verified'},
            **({'iterate': 'required'} if {'iterate', 'iterate_neuralese'} & set(constructs) else {}),
            **({'named': 'required'} if len([p for p in files if p.endswith('.nl')]) > 1 else {}),
            'external': {'source': 'authored', 'upstream': 'authored:' + author['model'], 'upstream_id': task['id'],
                         'upstream_split': 'train-carve', 'unit': world, 'shape': task.get('shape', family),
                         'instruction_sha256': text_digest(root_body)[:16], 'instruction_id': text_digest(root_body)[:12],
                         'neuralese_room': '', 'framing': 'authored', 'constructs': sorted(constructs),
                         'subject_domain': task.get('domain', 'other'),
                         'oracle_depth': task['oracle_depth'],
                         'checks': {'wrong_answers': wrong, 'alias_answers': aliases,
                                    **({'wrong_code': wrong_code} if wrong_code else {})},
                         'structure': {'name': family, 'constructs': sorted(constructs),
                                       'smoke': {'root': [['eval', {'code': smoke['code']}],
                                                          ['return_result', {'status': 'success',
                                                                             'value': task['expected']}]],
                                                 'children': smoke['children']}}}},
        'semantics': {'root': root, 'files': files, 'inputs': task['inputs'], 'expected': task['expected'],
                      # Declared aliases are alternate gold forms the oracle accepts (OracleSpec.alternates), in
                      # addition to what the oracle mode already ignores (record key order; case and spacing when
                      # normalized).
                      'oracle': {'level': task['oracle'], 'alternates': aliases} if aliases else task['oracle'],
                      # The runtime materializes folder_files as the directory reducer's world folder and, with
                      # expected_files, grades the folder the run leaves behind (collector.executeProgram).
                      **({'folder_files': folders} if folders is not None else {}),
                      **({'expected_files': expected_folders, 'files_oracle': {'compare': 'content', 'threshold': 1}}
                         if expected_folders is not None else {})},
    }


def main(argv=None):
    argv = argv or sys.argv[1:]
    if len(argv) != 2:
        raise SystemExit(__doc__)
    errors, out = [], []
    for number, line in enumerate(Path(argv[0]).read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(convert(json.loads(line)))
        except (ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
            errors.append(f'line {number}: {error}')
    ids = [row['id'] for row in out]
    if len(ids) != len(set(ids)):
        errors.append('duplicate task ids')
    Path(argv[1]).write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in out))
    print(json.dumps({'converted': len(out), 'errors': errors}))
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())
