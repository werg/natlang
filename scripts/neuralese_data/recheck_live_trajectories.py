#!/usr/bin/env python3
"""Recover explicitly checked decisions and measure the structure of observed live trajectories."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from live_teacher_sources import bound_file, load_episode, open_goal_fingerprint, same_json_value, validate_reassessment_check_roles, verify_original_rejection, verify_entry
from sft_sources import ZstdWriter, held_closure, held_hit


def read(path):
    path = Path(path)
    if path.suffix == '.gz':
        with gzip.open(path, 'rt') as stream:
            return json.load(stream)
    return json.loads(path.read_text())


def bind(root, name, value):
    content = (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode()
    (root / name).write_bytes(content)
    return {'path': name, 'sha256': hashlib.sha256(content).hexdigest()}


def recover(packet_path, pins_path, assessment_path, catalog_path, held_path, out):
    """Add a checked training scope while keeping the original capture and result intact."""
    packet, pins, assessment = map(read, (packet_path, pins_path, assessment_path))
    case_id = packet['test']['name']
    catalog = {row['scenario']: row for row in map(json.loads, Path(catalog_path).read_text().splitlines())}
    task = catalog[case_id]
    held = held_closure(held_path)
    if task['split'] != 'train' or held_hit(task['taskId'], [task['taskId']], task['sourceGroups'], held):
        raise ValueError('recovered task is held or outside the training split')
    original = packet['execution']['diagnostics']['openGoalVerification']
    exercise = packet['execution']['diagnostics']['natlangTeacher']
    digest = hashlib.sha256(Path(packet_path).read_bytes()).hexdigest()
    if (assessment.get('schema') != 'natlang.live-reassessment/1' or assessment.get('task_id') != task['taskId'] or
            assessment.get('original_sha256') != digest or not assessment.get('rationale') or
            not isinstance(packet['result'].get('passed'), bool)):
        raise ValueError('reassessment must bind the original failed training capture')
    checks = original.get('checks') or {}
    completed = assessment.get('completed_outcome')
    completed_proof = None
    if completed is not None:
        proof_path = bound_file(completed, Path(assessment_path).parent)
        completed_proof = read(proof_path)
        proof_checks = completed_proof.get('checks')
        if (completed_proof.get('schema') != 'natlang.live-outcome-reassessment/1' or
                completed_proof.get('original_sha256') != digest or completed_proof.get('task_id') != task['taskId'] or
                completed_proof.get('goal') != original.get('goal') or
                not same_json_value(completed_proof.get('entry'), exercise.get('entry')) or
                not same_json_value(completed_proof.get('actualResult'), exercise.get('report')) or
                completed_proof.get('passed') is not True or not isinstance(proof_checks, dict) or
                any(value is not True for value in proof_checks.values()) or
                not completed_proof.get('rationale') or not completed_proof.get('evidence') or
                not completed_proof.get('checker_sources')):
            raise ValueError('completed reassessment must independently attest the exact original whole outcome')
        if not verify_original_rejection(packet, completed_proof, proof_path.parent):
            raise ValueError('completed reassessment needs its original native failure or bound rejected review')
        validate_reassessment_check_roles(completed_proof, checks, proof_path.parent)
        for binding in completed_proof['evidence'] + completed_proof['checker_sources']:
            bound_file(binding, proof_path.parent)
    selected = assessment.get('invocations') or []
    if completed is not None and selected:
        raise ValueError('completed reassessment must not silently restrict its training scope')
    for row in selected:
        names = row.get('checks') or []
        if not isinstance(names, list) or not names or any(checks.get(name) is not True for name in names):
            raise ValueError('reassessment cites an absent or failed independent workspace check')
    if completed is None and packet['result'].get('passed') is not False:
        raise ValueError('partial reassessment requires its original native failure')
    if not selected and completed is None:
        raise ValueError('reassessment needs explicitly checked invocations')
    calls = exercise['calls']
    roots = exercise['rootCallIds']
    if len(roots) != 1 or len({call['callId'] for call in calls}) != len(calls):
        raise ValueError('reassessment requires one complete original call tree')
    root = next(call['record'] for call in calls if call['callId'] == roots[0])
    goal = original['goal']
    interaction_entry={'kind':'open-user-goal','goal':goal,'program':exercise['programPath']}
    if exercise.get('entry'):
        interaction_entry['entry']=exercise['entry']
    verify_entry(interaction_entry, root, partial=completed is None)
    if any(row.get('dirty') is not False for row in pins['repositories'].values()):
        raise ValueError('reassessment requires recorded clean code provenance')
    source_code = pins['repositories']['natlang']
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        raw = out / Path(packet_path).name
        shutil.copyfile(packet_path, raw)
        original_binding = bind(out, 'original-outcome.json', packet['result'])
        assessment_binding = bind(out, 'reassessment.json', assessment)
        assessor_binding = None
        if assessment.get('assessor'):
            source = bound_file(assessment['assessor'], Path(assessment_path).parent)
            target = out / 'assessor-source.mjs'
            shutil.copyfile(source, target)
            assessor_binding = {'path': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
        completed_original_binding = None
        if completed_proof is not None:
            original_proof_bytes = proof_path.read_bytes()
            (out / 'completed-outcome-original.json').write_bytes(original_proof_bytes)
            completed_original_binding = {'path': 'completed-outcome-original.json', 'sha256': hashlib.sha256(original_proof_bytes).hexdigest()}
            # Copy every independently bound witness/source into the derivative capture.
            for field in ('evidence', 'checker_sources'):
                copied = []
                for index, binding in enumerate(completed_proof[field]):
                    source = bound_file(binding, proof_path.parent)
                    target = out / f'reassessment-{field}-{index}{source.suffix}'
                    shutil.copyfile(source, target)
                    copied.append({**binding, 'path': target.name})
                completed_proof[field] = copied
            bind(out, 'completed-reassessment.json', completed_proof)
        proof_binding = {**bind(out, 'workspace.json', {'proof': completed_proof} if completed_proof is not None else original), 'kind': 'workspace'}
        invocation_scope = [{'call_id': row['call_id'], 'checks': {name: True for name in row['checks']},
                             'evidence': [proof_binding], **({'ordinals': row['ordinals']} if 'ordinals' in row else {})}
                            for row in selected]
        interaction = {'kind': 'open-user-goal', 'goal': goal, 'program': exercise['programPath']}
        if exercise.get('entry'):
            interaction['entry'] = exercise['entry']
        world = {'instance': pins['instance'], 'run_id': pins['run_id']}
        validator = bind(out, 'reassessment-validator.json', {
            'original_packet': {'path': raw.name, 'sha256': digest}, 'assessment': assessment_binding,
            'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            **({'completed_outcome': completed_original_binding} if completed_original_binding else {})})
        bound_calls = [bind(out, f'call-{index}.json', call) for index, call in enumerate(calls)]
        verification = {'schema': 'natlang.live-task-verification/1', 'task_id': task['taskId'],
                        'run_id': pins['run_id'], 'world': world, 'split': 'train', 'status': 'passed' if completed_proof is not None else 'partial',
                        'checks': {'originalCapturePreserved': True, 'independentInvocationEvidence': True},
                        'validator': validator, 'evidence': [proof_binding], 'interaction': interaction,
                        'root_call_ids': roots, 'calls': [call['summary'] for call in calls],
                        **({'reassessment': {'original_outcome': original_binding, 'assessment': assessment_binding}} if completed_proof is not None else
                           {'training_scope': {'schema': 'natlang.training-scope/1', 'original_outcome': original_binding,
                                              'invocations': invocation_scope, **({'assessor': assessor_binding} if assessor_binding else {})}})}
        spec = {'schema': 'natlang.live-teacher-task/1', 'id': task['taskId'], 'family': task['family'],
                'split': 'train', 'source_namespace': task['taskId'].rsplit(':', 1)[0],
                'source_ids': [task['taskId']], 'source_groups': task['sourceGroups'],
                'source_revisions': [row['git_commit'] for row in pins['repositories'].values()],
                'code': source_code, 'learner_kinds': ['natlang-executor'], 'world': world,
                'required_effect_kinds': ['workspace'], 'calls': bound_calls, 'interaction': interaction,
                'verification': bind(out, 'verification.json', verification)}
        manifest = out / 'manifest.json'
        manifest.write_text(json.dumps(spec, indent=2) + '\n')
        episode = load_episode(manifest)
        writer = ZstdWriter(out / 'trajectory.native.jsonl.zst')
        writer.write(episode['trajectory_ir'])
        writer.close()
        return {'manifest': str(manifest), 'decisions': len(episode['turns']), 'task_id': episode['task_id']}
    except BaseException:
        # An unsuccessful attempt remains visible for diagnosis and is never published as an export.
        (out / 'INCOMPLETE').touch()
        raise


def structural_coverage(manifests):
    entries, functions, domains, compositions, depths, widths = (Counter() for _ in range(6))
    rejected, episodes = [], 0
    for manifest in manifests:
        try:
            episode = load_episode(manifest)
        except (ValueError, KeyError, OSError) as error:
            rejected.append({'manifest': str(manifest), 'reason': str(error)})
            continue
        episodes += 1
        interaction = episode['task'].get('interaction') or {}
        entry = interaction.get('entry') or {'program': interaction.get('program'), 'callable': 'runUserGoal'}
        entries[str(entry.get('program'))] += 1
        ledger = (episode.get('trajectory_ir') or {}).get('outcome', {}).get('invocation_ledger') or []
        records = {row['call_id']: row for row in ledger}
        children = Counter(row['parent_call_id'] for row in ledger if row['parent_call_id'])
        learned = set(turn['invocation_id'] for turn in episode['turns'])
        used_domains = set()
        for call_id in learned:
            row = records.get(call_id)
            functions[next(turn['function'] for turn in episode['turns'] if turn['invocation_id'] == call_id)] += 1
            if row:
                current, depth = row, 0
                while current.get('parent_call_id') in records:
                    depth += 1
                    current = records[current['parent_call_id']]
                depths[depth] += 1
                widths[children[call_id]] += 1
                for effect in row.get('effects') or []:
                    if effect.get('service'):
                        used_domains.add(effect['service'])
        domains.update(used_domains)
        compositions[' + '.join(sorted(used_domains))] += 1
    return {'episodes': episodes, 'entry_points': dict(entries), 'learned_functions': dict(functions),
            'effect_services': dict(domains), 'service_compositions': dict(compositions),
            'learned_invocation_depths': dict(depths), 'learned_invocation_children': dict(widths),
            'unreadable_manifests': rejected}


def catalog_coverage(catalog_path, entries_path):
    """Count declared application entries separately from observed learned behavior."""
    entries = read(entries_path)
    programs, signatures, splits = Counter(), Counter(), Counter()
    missing = []
    goals = set()
    for row in map(json.loads, Path(catalog_path).read_text().splitlines()):
        kind = row.get('kind') or row.get('family', '').removeprefix('workspace-open-user-goal-')
        entry = row.get('entry') or entries.get(kind)
        splits[row.get('split', 'unknown')] += 1
        if not entry:
            missing.append(row['scenario'])
            continue
        programs[entry['program']] += 1
        signatures[entry.get('resultType', 'unknown')] += 1
        if row.get('goalSha256'):
            goals.add(row['goalSha256'])
    return {'declared_task_entries': dict(programs), 'declared_return_types': dict(signatures),
            'task_splits': dict(splits), 'distinct_goal_hashes': len(goals), 'unmapped_tasks': missing}


def audit_captures(captures, catalog_path):
    """Inventory every outcome without using whole-run failure as a data deletion rule."""
    catalog = {row['scenario']: row for row in map(json.loads, Path(catalog_path).read_text().splitlines())}
    findings, functions, domains, depths, widths = [], Counter(), Counter(), Counter(), Counter()
    for path in captures:
        packet = read(path)
        case = packet['test']['name']
        task = catalog.get(case)
        if not task or task['split'] != 'train':
            continue
        diagnostic = packet.get('execution', {}).get('diagnostics') or {}
        exercise = diagnostic.get('natlangTeacher') or {}
        proof = diagnostic.get('openGoalVerification') or {}
        calls = exercise.get('calls') or []
        records = {call['callId']: call['record'] for call in calls}
        children = Counter(row.get('parent_call_id') for row in records.values() if row.get('parent_call_id'))
        for call_id, record in records.items():
            functions[record.get('definition', {}).get('name', 'unknown')] += 1
            domains.update(effect['service'] for effect in record.get('effects') or [] if effect.get('service'))
            chain, parent = {call_id}, record.get('parent_call_id')
            while parent in records and parent not in chain:
                chain.add(parent)
                parent = records[parent].get('parent_call_id')
            depths[len(chain) - 1] += 1
            widths[children[call_id]] += 1
        checks = proof.get('checks') or {}
        failed = [key for key, value in checks.items() if value is not True]
        error = exercise.get('executionError') or (packet.get('execution') or {}).get('error')
        if not calls:
            reason = 'No original model invocation was captured. Diagnose the harness before model re-execution.'
        elif error:
            reason = 'Execution error recorded. Evaluate any completed effect against its independent evidence.'
        elif not checks:
            reason = 'No independent check map captured. Reassess original evidence before training admission.'
        else:
            reason = 'Review failed checks against the requested effects and returned values. Completed calls alone do not establish those outcomes.'
        findings.append({'task_id': task['taskId'], 'case': case, 'packet': str(path),
                         'original_sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                         'passed': packet['result'].get('passed'), 'goal': proof.get('goal'),
                         'passing_checks': [key for key, value in checks.items() if value is True],
                         'failed_checks': failed, 'execution_error': error,
                         'calls': len(calls), 'status': 'requires-independent-reassessment', 'reason': reason})
    return {'captures': findings, 'count': len(findings), 'observed_functions': dict(functions),
            'observed_effect_services': dict(domains), 'observed_depths': dict(depths),
            'observed_children': dict(widths)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    recheck = commands.add_parser('recover')
    for flag in ('packet', 'pins', 'assessment', 'catalog', 'held', 'out'):
        recheck.add_argument('--' + flag, type=Path, required=True)
    coverage = commands.add_parser('coverage')
    coverage.add_argument('manifests', type=Path, nargs='+')
    coverage.add_argument('--out', type=Path, required=True)
    coverage.add_argument('--catalog', type=Path)
    coverage.add_argument('--entries', type=Path)
    audit = commands.add_parser('audit')
    audit.add_argument('captures', type=Path, nargs='+')
    audit.add_argument('--catalog', type=Path, required=True)
    audit.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'recover':
        print(json.dumps(recover(args.packet, args.pins, args.assessment, args.catalog, args.held, args.out)))
    elif args.command == 'coverage':
        result = structural_coverage(args.manifests)
        if args.catalog or args.entries:
            if not args.catalog or not args.entries:
                parser.error('catalog coverage needs both --catalog and --entries')
            result['catalog'] = catalog_coverage(args.catalog, args.entries)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'episodes': result['episodes'], 'entries': len(result['entry_points']),
                          'functions': len(result['learned_functions']), 'unreadable': len(result['unreadable_manifests'])}))
    else:
        result = audit_captures(args.captures, args.catalog)
        args.out.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({'captures': result['count'], 'functions': len(result['observed_functions'])}))


if __name__ == '__main__':
    main()
