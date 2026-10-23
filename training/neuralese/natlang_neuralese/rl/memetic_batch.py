"""Group existing train invocations by exact natlang program and iterate every ready group.

The grouping and split mechanism is exact. Guidance revision remains the natural-
language author policy in memetic_program; this module never inspects held gold.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from collections import defaultdict
from pathlib import Path

from ..common.hashing import canonical_json_sha256_hex
from ..common.jsonio import canonical_json_bytes, write_bytes_durable, write_json_durable
from . import memetic_program


def program_identity(row: dict) -> tuple[str, dict]:
    semantics = row.get('semantics') or {}
    root, files = semantics.get('root'), semantics.get('files')
    if not isinstance(root, str) or not root or not isinstance(files, dict) or not files:
        raise ValueError('batch selector requires the exact natlang root and files')
    if root not in files or any(not isinstance(name, str) or not isinstance(value, str)
                                for name, value in files.items()):
        raise ValueError('batch selector found incomplete natlang program files')
    # Service implementations can close over per-invocation private world data.
    # Their stable contract is the declared scope and service names; bind the
    # exact implementation separately as a world input.
    program = {'root': root, 'files': files}
    if 'services' in semantics:
        program['service_names'] = sorted(semantics['services'])
    for key in ('service_scopes', 'service_contracts'):
        if key in semantics:
            program[key] = semantics[key]
    return 'natlang-program-sha256:' + canonical_json_sha256_hex(program), program


def world_bindings(row: dict) -> list[str]:
    bindings = {'source:' + value for field in ('source_groups', 'source_ids')
                for value in row.get(field) or [] if isinstance(value, str) and value}
    curriculum = row.get('curriculum') or {}
    evidence = curriculum.get('evidence') or {}
    for field in ('world', 'retrieved', 'background'):
        for item in evidence.get(field) or []:
            bindings.add('evidence:' + canonical_json_sha256_hex(item))
    services = (row.get('semantics') or {}).get('services')
    if services:
        bindings.add('service-world:' + canonical_json_sha256_hex(services))
    inputs = (row.get('semantics') or {}).get('inputs')
    if inputs not in (None, {}, []):
        bindings.add('inputs:' + canonical_json_sha256_hex(inputs))
    if not bindings:
        if inputs is None:
            raise ValueError('invocation has no source or world binding for a safe split')
        bindings.add('inputs:' + canonical_json_sha256_hex(inputs))
    return sorted(bindings)


def _components(rows: list[dict]):
    parents = list(range(len(rows)))
    def find(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    first = {}
    for index, row in enumerate(rows):
        for binding in row['memetic_source']['world_bindings']:
            if binding in first:
                parents[find(index)] = find(first[binding])
            else:
                first[binding] = index
    groups = defaultdict(list)
    for index in range(len(rows)):
        groups[find(index)].append(index)
    return sorted(groups.values(), key=lambda indices: min(rows[index]['id'] for index in indices))


def _allocation(rows: list[dict], components: list[list[int]], min_each: int):
    programs = sorted({row['program_id'] for row in rows})
    by_component = [{program: sum(rows[index]['program_id'] == program for index in part)
                     for program in programs} for part in components]
    def score(bits):
        totals = {program: [0, 0] for program in programs}
        for bit, counts in zip(bits, by_component):
            for program, count in counts.items():
                totals[program][bit] += count
        ready = [counts for counts in totals.values() if min(counts) >= min_each]
        return (len(ready), sum(min(counts) for counts in ready),
                -sum(abs(counts[0] - counts[1]) for counts in ready))
    if len(components) <= 18:
        candidates = ((0, *bits) for bits in itertools.product((0, 1), repeat=max(0, len(components)-1)))
        return max(candidates, key=score)
    bits = [0] * len(components)
    for index in range(1, len(bits)):
        bits[index] = max((0, 1), key=lambda bit: score([*bits[:index], bit, *([0] * (len(bits)-index-1))]))
    for _ in range(2):
        for index in range(1, len(bits)):
            previous = score(bits)
            bits[index] ^= 1
            if score(bits) < previous:
                bits[index] ^= 1
    return tuple(bits)


def select(source: Path, out: Path, *, min_each: int = 2) -> dict:
    if min_each < 2:
        raise ValueError('development and confirmation need at least two invocations each')
    raw = source.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    originals = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if len({row.get('id') for row in originals}) != len(originals):
        raise ValueError('source packet repeats invocation IDs')
    rows = []
    for index, original in enumerate(originals):
        if original.get('split') != 'train':
            raise ValueError('batch selector cannot repartition protected or non-train cases')
        program_id, program = program_identity(original)
        if original.get('program_id') not in (None, program_id):
            raise ValueError('existing program identity conflicts with exact root and files')
        row = dict(original)
        row['program_id'] = program_id
        row['memetic_source'] = {'schema': 'natlang.memetic-source/1', 'source_path': str(source.resolve()),
                                 'source_sha256': source_sha, 'row_index': index,
                                 'row_sha256': canonical_json_sha256_hex(original),
                                 'program_sha256': canonical_json_sha256_hex(program),
                                 'world_bindings': world_bindings(original)}
        rows.append(row)
    components = _components(rows)
    bits = _allocation(rows, components, min_each)
    by_program = defaultdict(lambda: {'development': [], 'confirmation': []})
    for bit, indices in zip(bits, components):
        slice_name = 'confirmation' if bit else 'development'
        for index in indices:
            by_program[rows[index]['program_id']][slice_name].append(rows[index])
    out.mkdir(parents=True, exist_ok=True)
    previous = out / 'batch.json'
    if previous.exists() and (json.loads(previous.read_text()).get('source_sha256') != source_sha or
                              json.loads(previous.read_text()).get('min_each') != min_each):
        raise ValueError('batch resume changed its source packet or split requirement')
    jobs = []
    for program_id, parts in sorted(by_program.items()):
        counts = {name: len(part) for name, part in parts.items()}
        entry = {'program_id': program_id, 'counts': counts, 'ready': min(counts.values()) >= min_each,
                 'source_ids': {name: sorted(row['id'] for row in part) for name, part in parts.items()}}
        if entry['ready']:
            directory = out / ('program-' + program_id.rsplit(':', 1)[1][:16]); directory.mkdir(exist_ok=True)
            for name, part in parts.items():
                path = directory / (name + '.jsonl')
                write_bytes_durable(path, b''.join(canonical_json_bytes(row) + b'\n' for row in part))
                entry[name] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        else:
            entry['pending'] = 'fewer than two independent train invocations in a split'
        jobs.append(entry)
    manifest = {'schema': 'natlang.memetic-batch-selection/1', 'source': str(source.resolve()),
                'source_sha256': source_sha, 'rows': len(rows), 'min_each': min_each,
                'world_components': len(components), 'jobs': jobs}
    write_json_durable(out / 'batch.json', manifest)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--min-each', type=int, default=2)
    parser.add_argument('--select-only', action='store_true')
    parser.add_argument('--baseline-only', action='store_true',
                        help='collect and bind unguided development/confirmation fitness without an author')
    parser.add_argument('--server')
    parser.add_argument('--model-id', choices=('lfm', 'mellum'))
    parser.add_argument('--author')
    parser.add_argument('--author-ferry', help='evaluator-local SDK ferry endpoint (loopback:8766)')
    parser.add_argument('--author-model')
    parser.add_argument('--author-provenance', type=Path)
    parser.add_argument('--recipe-base', type=Path)
    parser.add_argument('--recipe-graph-root', type=Path,
                        help='trainer-side root for each per-program native-guidance-graphs.jsonl')
    parser.add_argument('--foundation-id', default='native_full_sft')
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--case-seconds', type=int, default=900)
    parser.add_argument('--max-model-requests', type=int, default=32)
    parser.add_argument('--max-turns', type=int, default=20)
    parser.add_argument('--seed-base', type=int, default=13000)
    parser.add_argument('--min-gain-fraction', type=float, default=0.1)
    args = parser.parse_args(argv)
    manifest = select(args.source, args.out, min_each=args.min_each)
    if args.select_only:
        print(json.dumps({'manifest': str(args.out / 'batch.json'), 'ready': sum(job['ready'] for job in manifest['jobs'])}))
        return
    if any(getattr(args, name) is None for name in ('server', 'model_id')):
        parser.error('execution needs student server and model')
    if not args.baseline_only and any(getattr(args, name) is None for name in ('author', 'author_ferry', 'author_model', 'author_provenance')):
        parser.error('guided execution needs self-hosted author endpoint/model/provenance')
    # Each program owns a resumable journal. A failed job does not prevent other
    # ready programs from advancing; rerunning this command resumes each one.
    results = []
    for job in manifest['jobs']:
        if not job['ready']:
            continue
        program_id = job['program_id']
        directory = args.out / ('program-' + program_id.rsplit(':', 1)[1][:16])
        arguments = ['--program-id', program_id, '--development', job['development']['path'],
                     '--confirmation', job['confirmation']['path'], '--out', str(directory / 'learning'),
                     '--server', args.server, '--model-id', args.model_id,
                     '--foundation-id', args.foundation_id, '--generations', str(args.generations),
                     '--samples', str(args.samples), '--shards', str(args.shards),
                     '--temperature', str(args.temperature), '--case-seconds', str(args.case_seconds),
                     '--max-model-requests', str(args.max_model_requests), '--max-turns', str(args.max_turns),
                     '--seed-base', str(args.seed_base), '--min-gain-fraction', str(args.min_gain_fraction)]
        if args.baseline_only:
            arguments.append('--baseline-only')
        else:
            arguments.extend(['--author', args.author, '--author-ferry', args.author_ferry, '--author-model', args.author_model,
                              '--author-provenance', str(args.author_provenance)])
        if args.recipe_base:
            arguments.extend(['--recipe-base', str(args.recipe_base)])
        if args.recipe_graph_root:
            arguments.extend(['--recipe-graph-path', str(args.recipe_graph_root /
                ('program-' + program_id.rsplit(':', 1)[1][:16]) / 'learning/native-guidance-graphs.jsonl')])
        try:
            outcome = memetic_program.main(arguments)
            results.append({'program_id': program_id, 'status': outcome['status'],
                            'result': str(directory / 'learning' /
                                          ('baseline-result.json' if args.baseline_only else 'result.json')),
                            **({'binding': str(directory / 'learning/binding.json')}
                               if outcome['status'] == 'qualified' else {})})
        except Exception as error:
            results.append({'program_id': program_id, 'status': 'failed', 'error': repr(error)})
        write_json_durable(args.out / 'progress.json', {'schema': 'natlang.memetic-batch-progress/1', 'results': results})
    if any(result['status'] == 'failed' for result in results):
        raise RuntimeError('some memetic program jobs failed; inspect progress.json and rerun to resume')
    if args.baseline_only:
        print(json.dumps({'manifest': str(args.out / 'batch.json'), 'status': 'baseline_complete',
                          'results': results}))
        return
    if args.recipe_base:
        bindings = [json.loads(Path(result['binding']).read_text()) for result in results if result['status'] == 'qualified']
        if not bindings:
            print(json.dumps({'manifest': str(args.out / 'batch.json'), 'status': 'no_qualified_guidance',
                              'results': results}))
            return
        if len({canonical_json_bytes(binding['student']) for binding in bindings}) != 1:
            raise ValueError('batch guidance graphs come from different student checkpoints')
        graph_bytes = b''.join(Path(binding['native_training_graphs']['path']).read_bytes()
                               for binding in bindings)
        combined = args.out / 'native-guidance-graphs.jsonl'
        write_bytes_durable(combined, graph_bytes)
        graph_binding = {'path': str(combined), 'sha256': hashlib.sha256(graph_bytes).hexdigest(),
                         'rows': sum(binding['native_training_graphs']['rows'] for binding in bindings)}
        recipe = memetic_program.emit_recipe(args.recipe_base, args.out / 'native-guidance-recipe.json',
            graph_binding, foundation_id=args.foundation_id,
            graph_path=str(args.recipe_graph_root / combined.name) if args.recipe_graph_root else None)
        write_json_durable(args.out / 'training.json', {'schema': 'natlang.memetic-batch-training/1',
            'student': bindings[0]['student'], 'program_ids': [binding['program_id'] for binding in bindings],
            'graphs': graph_binding, 'recipe': recipe})
    print(json.dumps({'manifest': str(args.out / 'batch.json'), 'results': results}))


if __name__ == '__main__':
    main()
