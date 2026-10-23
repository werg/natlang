"""Iterate one program's stored native-state guidance on repeated train invocations.

The author sees only development failures. A disjoint train confirmation slice is
used solely to accept an improvement. Every candidate is written by the serving
student's native producer, then imported into actual executor model messages.
The output binding can be used by a program's guidance field; a changed student
checkpoint requires writing the text again and evaluating the new state.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import urllib.request
from pathlib import Path

from ..common.jsonio import write_json_durable
from ..eval.task_packet import read_jsonl
from . import memetic_task_runner as runner
from .task_rollouts import authored_root_row

STATE_ID = re.compile(r"nzs1_[a-z2-7]{52}\Z")
IDENTITY = ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256', 'architecture')


def request_json(server: str, path: str, value=None):
    request = urllib.request.Request(server.rstrip('/') + path,
        data=None if value is None else json.dumps(value).encode(),
        headers={'content-type': 'application/json'})
    with urllib.request.urlopen(request, timeout=900) as response:
        return json.load(response)


def verified_cases(path: Path, program_id: str):
    rows = [authored_root_row(row) for row in read_jsonl(path)]
    if len(rows) < 2 or any(row.get('program_id') != program_id for row in rows):
        raise ValueError('each slice needs multiple train invocations of exactly the named program')
    ids = [row['id'] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError('repeated invocation IDs in a slice')
    return rows


def prepare_slices(development: Path, confirmation: Path, program_id: str):
    dev = verified_cases(development, program_id)
    confirm = verified_cases(confirmation, program_id)
    if set(row['id'] for row in dev) & set(row['id'] for row in confirm):
        raise ValueError('development and confirmation invocation IDs overlap')
    return dev, confirm


def state_guidance(text: str, state_id: str) -> str:
    if not text.strip() or not STATE_ID.fullmatch(state_id):
        raise ValueError('native program guidance needs text and an exact state ID')
    # The model transport materializes this existing native-state sentinel into
    # a content part. No text approximation of the state is substituted.
    # Fitness must measure the native state itself. Showing the candidate text
    # would let a textual hint explain a gain without the state helping.
    return '\ue002' + state_id + '\ue003'


def write_state(server: str, text: str, expected_identity: dict):
    written = request_json(server, '/v1/neuralese/native-states/write-text', {'text': text, 'type': 'Neuralese<ProgramGuidance>'})
    state_id = written.get('id')
    if not isinstance(state_id, str) or not STATE_ID.fullmatch(state_id):
        raise ValueError('native writer returned no exact state ID')
    policy = (written.get('source_capture') or {}).get('policy') or {}
    metadata = written.get('native_state') or {}
    for key in IDENTITY:
        if policy.get(key) != expected_identity[key] or metadata.get(key) != expected_identity[key]:
            raise ValueError('guidance writer identity differs from actual serving student')
    if written.get('source_text_sha256') != hashlib.sha256(text.encode()).hexdigest():
        raise ValueError('guidance native producer changed its source text')
    segments = (written.get('source_capture') or {}).get('segments') or []
    if len(segments) != 3 or not all(segments[1].get('producing') or []):
        raise ValueError('guidance native producer omitted its actual produced token span')
    return written


def archive_state(server: str, out: Path, written: dict):
    """Retain the actual native tensor bytes, independently of server uptime."""
    from ..serve.store import decode_native_state
    state_id = written['id']
    path = out / 'states' / (state_id + '.safetensors')
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with urllib.request.urlopen(server.rstrip('/') + '/v1/neuralese/native-states/' + state_id,
                                    timeout=900) as response:
            raw = response.read()
        block, actual = decode_native_state(raw)
        if actual != state_id:
            raise ValueError('stored guidance bytes do not match the produced native state')
        block.validate()
        from ..common.jsonio import write_bytes_durable
        write_bytes_durable(path, raw)
    raw = path.read_bytes()
    _, actual = decode_native_state(raw)
    if actual != state_id:
        raise ValueError('archived guidance native state changed')
    return {'path': str(path), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def actor_identity(server: str):
    info = request_json(server, '/v1/neuralese/info')
    identity = info.get('native_identity') or {}
    if (not (info.get('native_state') or {}).get('enabled')
            or any(not isinstance(identity.get(key), str) or not identity[key] for key in IDENTITY)):
        raise ValueError('memetic guidance requires a native shared-state student server')
    return {key: identity[key] for key in IDENTITY}


def author_provenance(path: Path, model: str):
    """Require an independently prepared self-hosted teacher source receipt."""
    value = json.loads(path.read_text())
    if (value.get('schema') != 'natlang.memetic-author-provenance/1'
            or value.get('model_id') != model
            or value.get('hosting') != 'self-hosted-coreweave'
            or any(not isinstance(value.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', value[key])
                   for key in ('checkpoint_sha256', 'tokenizer_fingerprint', 'runtime_sha256'))):
        raise ValueError('memetic author needs exact self-hosted teacher checkpoint/tokenizer/runtime provenance')
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'receipt': value}


def evaluate(args, out: Path, name: str, guidance: str, rows: list[dict]):
    packet = out / (name + '-packet.jsonl')
    if not packet.exists():
        packet.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    target = out / 'evals' / name
    fit_path = out / (name + '-fit.json')
    if fit_path.exists():
        return json.loads(fit_path.read_text())
    if target.exists():
        shutil.rmtree(target)  # only this incomplete, invocation-specific evaluation
    options = argparse.Namespace(**vars(args))
    options.out = out
    options.rows = len(rows)
    options.shards = args.shards
    individual = {'id': name, 'guidance': guidance, 'adapter': None}
    fit = runner.evaluate(options, individual, packet, runner.default_system_prompt())
    if fit['scored'] != len(rows) * args.samples:
        raise ValueError('guidance evaluation lacks complete scored task outcomes')
    state_id = re.search(r'nzs1_[a-z2-7]{52}', guidance)
    if not state_id or not native_import_reached(target, state_id.group(0)):
        raise ValueError('guidance state did not reach the executing model as native state')
    fit['native_import_reached_model'] = True
    write_json_durable(fit_path, fit)
    return fit


def native_import_reached(evaluation: Path, state_id: str) -> bool:
    """Require a server import receipt, not merely a sentinel in rendered text."""
    paths = list(evaluation.glob('row-*/sample-*/results.jsonl'))
    if not paths:
        return False
    for path in paths:
        imported = False
        for result in read_jsonl(path):
            for step in result.get('trajectory') or []:
                response = step.get('model_response') or {}
                native = response.get('raw_response') or response
                receipts = (native.get('native_state') or {}).get('imports') or []
                if any(item.get('id') == state_id for item in receipts if isinstance(item, dict)):
                    imported = True
        if not imported:
            return False
    return True


def acceptance(previous: dict, candidate: dict):
    """Use paired seeds and complete oracle-scored outcomes, never author judgement."""
    if previous['scored'] != candidate['scored'] or not previous['scored']:
        return False
    return candidate['successes'] > previous['successes']


def guidance_training_graph(source: dict, *, program_id: str, written: dict, actor: dict):
    """Put a real producer and imported consumer in one native SFT graph.

    The producer uses the actual write-text token/role capture. The existing
    verified invocation graph supplies target tokens and weights. Position
    splicing, hybrid states and gradients are left to the shared graph executor.
    """
    from ..data.native_graph_schema import parse_graph, communication_classification
    if source.get('split') != 'train' or source.get('program_id') != program_id:
        raise ValueError('guidance training row is outside this program train split')
    if source.get('tokenizer_fingerprint') != actor['tokenizer_fingerprint']:
        raise ValueError('guidance writer and training graph tokenize differently')
    if not source.get('nodes'):
        raise ValueError('guidance training needs an actual consumer invocation')
    if source.get('external_states'):
        raise ValueError('guidance training needs self-contained source graphs; resolve external states first')
    segments = written['source_capture']['segments']
    ids = [token for segment in segments for token in segment['token_ids']]
    begin, end = written['source_capture']['payload_start'], written['source_capture']['payload_stop']
    if (not 0 < begin < end < len(ids) or not all(type(token) is int and token >= 0 for token in ids)
            or any(segment.get('producing') != [False] * len(segment['token_ids']) for segment in (segments[0], segments[2]))
            or segments[1].get('producing') != [True] * len(segments[1]['token_ids'])):
        raise ValueError('guidance source capture omitted actual distinct producer boundaries')
    result = copy.deepcopy(source)
    producer = 'memetic-guidance-producer'
    if any(node['name'] == producer for node in result['nodes']):
        raise ValueError('guidance producer occurrence collides with source graph')
    result['nodes'].insert(0, {'name': producer, 'token_ids': ids, 'producing': [[begin, end]],
                               'target_weights': [0.] + [1.] * (len(ids) - 1)})
    consumer = result['nodes'][1]
    consumer['imports'] = [{'at': 0, 'source': producer, 'start': begin, 'stop': end},
                           *(consumer.get('imports') or [])]
    result['id'] = 'memetic:' + written['id'] + ':' + source['id']
    result['source_groups'] = [*source['source_groups'], 'memetic-program:' + program_id]
    result['memetic_guidance'] = {'schema': 'natlang.memetic-graph-source/1',
        'program_id': program_id, 'source_id': source['id'],
        'source_sha256': hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        'native_state_id': written['id'], 'source_capture': written['source_capture']}
    example = parse_graph(result, tokenizer_fingerprint=actor['tokenizer_fingerprint'])
    if not communication_classification(example)['producer_consumer']:
        raise ValueError('guidance graph did not train a local producer/consumer dependency')
    return result


def export_training_graphs(path: Path, out: Path, *, program_id: str, confirmation_ids: set[str], written: dict, actor: dict):
    from ..common.jsonio import write_bytes_durable
    sources = read_jsonl(path)
    if len(sources) < 2 or len({row.get('id') for row in sources}) != len(sources):
        raise ValueError('native guidance training needs multiple distinct invocations')
    if any(row.get('id') in confirmation_ids for row in sources):
        raise ValueError('confirmation invocation was offered for training')
    graphs = [guidance_training_graph(row, program_id=program_id, written=written, actor=actor) for row in sources]
    output = out / 'native-guidance-graphs.jsonl'
    write_bytes_durable(output, ''.join(json.dumps(row, sort_keys=True) + '\n' for row in graphs).encode())
    return {'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'rows': len(graphs), 'producer_consumer_rows': len(graphs)}


def emit_recipe(base_path: Path, destination: Path, graph_binding: dict, *, graph_path: str | None = None):
    """Append an executable native SFT continuation with earned optimizer history."""
    from ..train.recipe import load_recipe
    recipe = copy.deepcopy(load_recipe(base_path))
    predecessor = recipe['stages'][-1]
    if predecessor['kind'] != 'native_state_sft' or 'held_native_graphs' not in predecessor['inputs']:
        raise ValueError('automatic memetic continuation requires a final native SFT stage with held graphs')
    stage_id = 'memetic_native_guidance'
    if any(stage['id'] == stage_id for stage in recipe['stages']):
        raise ValueError('memetic continuation stage already exists')
    key = 'memetic_native_train_graphs'
    recipe['id'] = recipe['id'] + '-memetic-native-guidance'
    recipe['description'] = (recipe.get('description') or '') + ' Native memetic guidance producer/consumer continuation.'
    recipe['input_bindings'][key] = {'path': graph_path or graph_binding['path'], 'sha256': graph_binding['sha256']}
    parameters = copy.deepcopy(predecessor['parameters'])
    parameters['epochs'] = 1.0
    recipe['stages'].append({'id': stage_id, 'kind': 'native_state_sft',
        'requires': [predecessor['id']],
        'inputs': {'native_graphs': key, 'held_native_graphs': predecessor['inputs']['held_native_graphs']},
        'parameters': parameters})
    write_json_durable(destination, recipe)
    load_recipe(destination)
    return {'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'stage_id': stage_id, 'predecessor': predecessor['id']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program-id', required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, required=True)
    parser.add_argument('--supervised-graphs', type=Path,
        help='verified same-program train native graphs; exports producer/consumer SFT graphs for the shared trainer')
    parser.add_argument('--recipe-base', type=Path,
        help='validated recipe ending in native_state_sft; emit an executable continuation stage')
    parser.add_argument('--recipe-graph-path',
        help='future trainer-side path for the exported graph bytes; SHA-256 stays exact')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--server', required=True)
    parser.add_argument('--model-id', choices=('lfm', 'mellum'), required=True)
    parser.add_argument('--author', required=True)
    parser.add_argument('--author-model', required=True)
    parser.add_argument('--author-provenance', type=Path, required=True)
    parser.add_argument('--generations', type=int, default=3)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--shards', type=int, default=2)
    parser.add_argument('--temperature', type=float, default=1.)
    parser.add_argument('--case-seconds', type=int, default=900)
    parser.add_argument('--max-model-requests', type=int, default=32)
    parser.add_argument('--max-turns', type=int, default=20)
    parser.add_argument('--seed-base', type=int, default=13000)
    args = parser.parse_args(argv)
    if args.recipe_base and not args.supervised_graphs:
        parser.error('--recipe-base requires --supervised-graphs')
    dev, confirm = prepare_slices(args.development, args.confirmation, args.program_id)
    teacher = author_provenance(args.author_provenance, args.author_model)
    identity = actor_identity(args.server)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    inputs = {'schema': 'natlang.memetic-program-inputs/1', 'program_id': args.program_id,
        'development': hashlib.sha256(args.development.read_bytes()).hexdigest(),
        'confirmation': hashlib.sha256(args.confirmation.read_bytes()).hexdigest(),
        'student': identity, 'author': teacher, 'seed_base': args.seed_base,
        'samples': args.samples, 'temperature': args.temperature,
        'supervised_graphs': (hashlib.sha256(args.supervised_graphs.read_bytes()).hexdigest()
                              if args.supervised_graphs else None),
        'recipe_base': (hashlib.sha256(args.recipe_base.read_bytes()).hexdigest()
                        if args.recipe_base else None),
        'recipe_graph_path': args.recipe_graph_path}
    input_path = out / 'inputs.json'
    if input_path.exists() and json.loads(input_path.read_text()) != inputs:
        raise ValueError('resume changed program, split, student or sampling identity')
    write_json_durable(input_path, inputs)
    state_path = out / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'generation': 0, 'best': None, 'history': []}
    if state['best'] is None:
        # A native block must contain actual text, including for the seed.
        seed_text = 'Follow the program instructions and verify the requested result before returning it.'
        seed_write = write_state(args.server, seed_text, identity)
        seed = {'text': seed_text, 'write': seed_write,
                'archive': archive_state(args.server, out, seed_write)}
        seed['development'] = evaluate(args, out, 'seed-dev', state_guidance(seed_text, seed_write['id']), dev)
        seed['confirmation'] = evaluate(args, out, 'seed-confirm', state_guidance(seed_text, seed_write['id']), confirm)
        state['best'] = seed
        write_json_durable(state_path, state)
    while state['generation'] < args.generations:
        generation = state['generation']
        best = state['best']
        pending = out / f'candidate-{generation:03d}-pending.json'
        if pending.exists():
            candidate = json.loads(pending.read_text())
        else:
            proposed = runner.propose_guidance(args, {'guidance': best['text']}, runner.failure_digest(best['development']))
            if proposed == best['text']:
                proposed += ' Check the type and output contract before returning.'
            written = write_state(args.server, proposed, identity)
            candidate = {'text': proposed, 'write': written,
                         'archive': archive_state(args.server, out, written)}
            write_json_durable(pending, candidate)
        proposed, written = candidate['text'], candidate['write']
        prefix = f'candidate-{generation:03d}'
        candidate['development'] = evaluate(args, out, prefix + '-dev', state_guidance(proposed, written['id']), dev)
        candidate['confirmation'] = evaluate(args, out, prefix + '-confirm', state_guidance(proposed, written['id']), confirm)
        accepted = (acceptance(best['development'], candidate['development'])
                    and acceptance(best['confirmation'], candidate['confirmation']))
        state['history'].append({'generation': generation, 'accepted': accepted, 'candidate': candidate})
        if accepted:
            state['best'] = candidate
        state['generation'] += 1
        write_json_durable(state_path, state)
        pending.unlink(missing_ok=True)
    best = state['best']
    binding = {'schema': 'natlang.memetic-program-guidance/1', 'program_id': args.program_id,
        'student': identity, 'text': best['text'], 'native_state_id': best['write']['id'],
        'native_state_artifact': best['archive'],
        'source_capture': best['write']['source_capture'], 'source_text_sha256': best['write']['source_text_sha256'],
        'program_guidance': state_guidance(best['text'], best['write']['id']),
        'development': {key: best['development'][key] for key in ('successes', 'scored')},
        'confirmation': {key: best['confirmation'][key] for key in ('successes', 'scored')},
        'inputs_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest()}
    if args.supervised_graphs is not None:
        binding['native_training_graphs'] = export_training_graphs(args.supervised_graphs, out,
            program_id=args.program_id, confirmation_ids={row['id'] for row in confirm},
            written=best['write'], actor=identity)
        if args.recipe_base is not None:
            binding['native_training_recipe'] = emit_recipe(args.recipe_base,
                out / 'native-guidance-recipe.json', binding['native_training_graphs'],
                graph_path=args.recipe_graph_path)
    write_json_durable(out / 'binding.json', binding)
    print(json.dumps({'binding': str(out / 'binding.json'), 'accepted': sum(h['accepted'] for h in state['history']),
                      'development': binding['development'], 'confirmation': binding['confirmation']}))


if __name__ == '__main__':
    main()
