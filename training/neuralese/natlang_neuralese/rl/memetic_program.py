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
    def bindings(rows):
        return {value for row in rows for value in (row.get('memetic_source') or {}).get('world_bindings', [])}
    if bindings(dev) & bindings(confirm):
        raise ValueError('development and confirmation world bindings overlap')
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


def author_provenance(path: Path, model: str, endpoint: str):
    """Require an independently prepared self-hosted teacher source receipt."""
    value = json.loads(path.read_text())
    if (value.get('schema') != 'natlang.memetic-author-provenance/1'
            or value.get('model_id') != model
            or value.get('hosting') != 'self-hosted-coreweave'
            or value.get('endpoint') != endpoint.rstrip('/')
            or not isinstance(value.get('sandbox_id'), str) or not value['sandbox_id']
            or any(not isinstance(value.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', value[key])
                   for key in ('checkpoint_sha256', 'tokenizer_fingerprint', 'runtime_sha256', 'source_receipt_sha256'))):
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
    individual = {'id': name, 'guidance': guidance}
    fit = runner.evaluate(options, individual, packet, runner.default_system_prompt())
    if fit['scored'] != len(rows) * args.samples:
        raise ValueError('guidance evaluation lacks complete scored task outcomes')
    if guidance:
        state_id = re.fullmatch(r'\ue002(nzs1_[a-z2-7]{52})\ue003', guidance)
        if not state_id or not native_import_reached(target, state_id.group(1)):
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
                receipts = response.get('native_state_imports') or []
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


def qualification(baseline: dict, best: dict, min_gain_fraction: float) -> bool:
    return all(best[split]['scored'] == baseline[split]['scored'] and best[split]['scored'] > 0 and
        (best[split]['successes'] - baseline[split]['successes']) / best[split]['scored'] >= min_gain_fraction
        for split in ('development', 'confirmation'))


def guidance_training_graph(rollout: dict, *, program_id: str, written: dict, actor: dict, roles, fetch_block):
    """Lower an actual successful guided whole episode into local producer/consumer SFT."""
    from ..data.native_graph_schema import parse_graph, communication_classification
    from ..model.native_graph_plan import Invocation, ImportSpan, compile_graph, invocation_dict
    from ..rl.episode_graph import whole_episode_turns
    from ..rl.native_replay import compile_replay
    from ..serve.store import native_state_id
    if rollout.get('split') != 'train' or (rollout.get('reward') or {}).get('quality') != 1.:
        raise ValueError('guidance SFT uses successful train episodes only')
    turns = whole_episode_turns(rollout)
    replay = compile_replay(turns, identity=actor, roles=roles, fetch_block=fetch_block)
    exact = [name for name, block in replay.external_blocks.items() if native_state_id(block) == written['id']]
    if len(exact) != 1 or len(replay.external_blocks) != 1 or replay.semantic_dependencies:
        raise ValueError('guided episode has unresolved external state or lacks its exact guidance import')
    external = exact[0]
    if any(node.name == 'memetic-guidance-producer' for node in replay.plan.invocations):
        raise ValueError('guidance producer occurrence collides with captured episode')
    segments = written['source_capture']['segments']
    if (len(segments) != 3 or [segment.get('kind') for segment in segments] != ['tokens'] * 3
            or segments[0].get('token_ids') != [roles.produce_open]
            or segments[2].get('token_ids') != [roles.produce_close]
            or segments[0].get('producing') != [False]
            or segments[2].get('producing') != [False]
            or not segments[1].get('token_ids')
            or segments[1].get('producing') != [True] * len(segments[1]['token_ids'])):
        raise ValueError('captured guidance producer markers or token roles differ from native writer')
    ids = [token for segment in segments for token in segment['token_ids']]
    begin, end = written['source_capture']['payload_start'], written['source_capture']['payload_stop']
    if (not 0 < begin < end < len(ids) or end-begin != len(replay.external_blocks[external].token_ids)
            or [token for token in ids[begin:end]] != replay.external_blocks[external].token_ids.tolist()
            or written['source_capture']['policy'] != actor):
        raise ValueError('captured guidance state differs from actual producer token span or actor')
    producer = Invocation('memetic-guidance-producer', tuple(ids), producing=((begin, end),))
    converted = [producer]
    consumed = []
    for node in replay.plan.invocations:
        edges = []
        for edge in node.imports:
            if edge.source == external:
                if edge.stop_gradient:
                    raise ValueError('guidance consumption was explicitly detached')
                consumed.append({'invocation': node.name, 'at': edge.at, 'start': edge.start, 'stop': edge.stop})
                edges.append(ImportSpan(edge.at, producer.name, begin+edge.start, begin+edge.stop,
                                        edge.logical, edge.stop_gradient))
            else:
                edges.append(edge)
        converted.append(Invocation(node.name, node.token_ids, tuple(edges), node.producing,
                                    node.logical_producing, node.cache_prefix))
    if not consumed or len(consumed) != len(replay.external_invocations.get(external) or []):
        raise ValueError('guidance consumer import occurrences differ from captured replay')
    plan = compile_graph(converted)
    weights = [[0.] * len(node.token_ids) for node in converted]
    # The independently useful guide is learned as a producer. Actual sampled
    # correct actions supply consumer targets, never a fabricated task answer.
    for local in range(begin, end):
        weights[0][local] = 0.2
    old_offsets = dict(zip((node.name for node in replay.plan.invocations), replay.plan.offsets))
    trained = 0
    for action in replay.actions:
        name = action['turn']
        node_index = next((index for index, node in enumerate(converted) if node.name == name), None)
        if node_index is None:
            raise ValueError('sampled action has no captured invocation')
        local = action['query'] - old_offsets[name] + 1
        node = converted[node_index]
        if local == len(node.token_ids):
            continue  # terminal sample not fed as an own token
        if not 0 < local < len(node.token_ids) or node.token_ids[local] != action['token_id']:
            raise ValueError('sampled action target differs from its captured own token')
        weights[node_index][local] = 1.
        trained += 1
    if trained == 0:
        raise ValueError('successful guided episode has no sampled consumer token targets')
    row = {'schema': 'natlang.neuralese-native-graph/1',
        'id': 'memetic:' + written['id'] + ':' + rollout['task_id'] + ':' + str(rollout['sample']),
        'program_id': program_id, 'split': 'train',
        'source_groups': ['memetic-program:' + program_id, 'task:' + rollout['task_id']],
        'tokenizer_fingerprint': actor['tokenizer_fingerprint'],
        'nodes': [{**json.loads(json.dumps(invocation_dict(node))), 'target_weights': weight}
                  for node, weight in zip(converted, weights)],
        'recorded_turn_bindings': [{key: turn.get(key) for key in ('graph_turn','written_nodes','read_bindings','external_read_bindings','written_occurrences','read_occurrences')}
                                   for turn in turns],
        'memetic_guidance': {'schema': 'natlang.memetic-graph-source/1', 'program_id': program_id,
            'source_task_id': rollout['task_id'], 'sample': rollout['sample'],
            'native_state_id': written['id'], 'source_capture': written['source_capture'],
            'consumed_occurrences': consumed,
            'rollout_sha256': hashlib.sha256(json.dumps(rollout, sort_keys=True).encode()).hexdigest()}}
    example = parse_graph(row, tokenizer_fingerprint=actor['tokenizer_fingerprint'])
    if not communication_classification(example)['producer_consumer'] or plan.token_count != example.plan.token_count:
        raise ValueError('captured guidance graph lost its producer/consumer dependency')
    return row


def export_training_graphs(evaluation: Path, out: Path, *, program_id: str, confirmation_ids: set[str],
                           written: dict, actor: dict, server: str):
    from ..common.jsonio import write_bytes_durable
    from ..model.shared_state import RoleTokens
    from ..rl.graph_replay import save_bundle, NativeBundleStore
    info = request_json(server, '/v1/neuralese/info')
    role_ids = (info.get('native_state') or {}).get('role_marker_ids')
    if (not isinstance(role_ids, list) or len(role_ids) != 4 or
            any(type(value) is not int for value in role_ids) or len(set(role_ids)) != 4):
        raise ValueError('native server omitted its four actual role marker IDs')
    rollouts = read_jsonl(evaluation / 'rollouts.jsonl')
    selected = [row for row in rollouts if (row.get('reward') or {}).get('quality') == 1.]
    if len({row['task_id'] for row in selected}) < 2:
        raise ValueError('native guidance training needs successes on multiple distinct invocations')
    if any(row.get('task_id') in confirmation_ids for row in selected):
        raise ValueError('confirmation invocation was offered for training')
    def fetch(ident):
        kind = ('native-states' if ident.startswith('nzs1_') else
                'native-tensor-artifacts' if ident.startswith('nnt1_') else 'blocks')
        with urllib.request.urlopen(server.rstrip('/') + '/v1/neuralese/' + kind + '/' + ident, timeout=900) as response:
            return response.read()
    bundle_dir = out / 'native-guidance-source-bundle'
    save_bundle(selected, bundle_dir, policy_id=actor['checkpoint_sha256'], fetch=fetch)
    reader = NativeBundleStore(bundle_dir / 'manifest.json', policy_id=actor['checkpoint_sha256'])
    graphs = [guidance_training_graph(row, program_id=program_id, written=written, actor=actor,
              roles=RoleTokens(*role_ids), fetch_block=reader.state) for row in selected]
    output = out / 'native-guidance-graphs.jsonl'
    write_bytes_durable(output, ''.join(json.dumps(row, sort_keys=True) + '\n' for row in graphs).encode())
    return {'path': str(output), 'sha256': hashlib.sha256(output.read_bytes()).hexdigest(),
            'rows': len(graphs), 'producer_consumer_rows': len(graphs),
            'source_bundle': {'path': str(bundle_dir / 'manifest.json'),
                              'sha256': hashlib.sha256((bundle_dir / 'manifest.json').read_bytes()).hexdigest()}}


def emit_recipe(base_path: Path, destination: Path, graph_binding: dict, *, foundation_id: str,
                graph_path: str | None = None):
    """Append an executable native SFT continuation with earned optimizer history."""
    from ..train.recipe import load_recipe
    from .native_raw_rounds import _supervised_stage_inventory
    recipe = copy.deepcopy(load_recipe(base_path))
    matches = [index for index, stage in enumerate(recipe['stages']) if stage['id'] == foundation_id]
    if len(matches) != 1:
        raise ValueError('memetic continuation needs the unique declared foundation stage')
    index = matches[0]
    foundation = recipe['stages'][index]
    if foundation['kind'] != 'native_state_sft' or 'held_native_graphs' not in foundation['inputs']:
        raise ValueError('declared foundation needs native SFT with held graphs')
    stage_id = 'memetic_native_guidance'
    if any(stage['id'] == stage_id for stage in recipe['stages']):
        raise ValueError('memetic continuation stage already exists')
    key = 'memetic_native_train_graphs'
    recipe['id'] = recipe['id'] + '-memetic-native-guidance'
    recipe['description'] = (recipe.get('description') or '') + ' Native memetic guidance producer/consumer continuation.'
    recipe['input_bindings'][key] = {'path': graph_path or graph_binding['path'], 'sha256': graph_binding['sha256']}
    parameters = copy.deepcopy(foundation['parameters'])
    parameters['epochs'] = 1.0
    precursor = next((stage for stage in reversed(recipe['stages'][:index])
                      if stage['kind'] == 'native_state_sft'), None)
    prerequisites = list(foundation['requires'])
    stage = {'id': stage_id, 'kind': 'native_state_sft',
        'requires': prerequisites,
        'inputs': {'native_graphs': key, 'held_native_graphs': foundation['inputs']['held_native_graphs']},
        'parameters': parameters}
    recipe['stages'].insert(index, stage)
    foundation['requires'] = [stage_id]
    write_json_durable(destination, recipe)
    load_recipe(destination)
    scheduled = _supervised_stage_inventory(recipe, foundation_id)
    if [item['id'] for item in scheduled].index(stage_id) >= [item['id'] for item in scheduled].index(foundation_id):
        raise ValueError('RAW supervisor did not schedule native guidance before foundation')
    return {'path': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'stage_id': stage_id, 'foundation_id': foundation_id,
            'predecessor': precursor['id'] if precursor else None}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--program-id', required=True)
    parser.add_argument('--development', type=Path, required=True)
    parser.add_argument('--confirmation', type=Path, required=True)
    parser.add_argument('--recipe-base', type=Path,
        help='validated native production recipe; emit captured producer/consumer SFT continuation')
    parser.add_argument('--foundation-id', default='native_full_sft')
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
    parser.add_argument('--min-gain-fraction', type=float, default=0.1,
                        help='minimum absolute success-rate gain on both disjoint slices')
    args = parser.parse_args(argv)
    if not 0 < args.min_gain_fraction <= 1:
        parser.error('--min-gain-fraction must be in (0, 1]')
    dev, confirm = prepare_slices(args.development, args.confirmation, args.program_id)
    teacher = author_provenance(args.author_provenance, args.author_model, args.author)
    identity = actor_identity(args.server)
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    inputs = {'schema': 'natlang.memetic-program-inputs/1', 'program_id': args.program_id,
        'development': hashlib.sha256(args.development.read_bytes()).hexdigest(),
        'confirmation': hashlib.sha256(args.confirmation.read_bytes()).hexdigest(),
        'student': identity, 'student_server': args.server, 'student_model_id': args.model_id,
        'author': teacher, 'author_server': args.author,
        'seed_base': args.seed_base, 'samples': args.samples, 'temperature': args.temperature,
        'min_gain_fraction': args.min_gain_fraction,
        'shards': args.shards, 'case_seconds': args.case_seconds,
        'max_model_requests': args.max_model_requests, 'max_turns': args.max_turns,
        'recipe_base': (hashlib.sha256(args.recipe_base.read_bytes()).hexdigest()
                        if args.recipe_base else None),
        'foundation_id': args.foundation_id,
        'recipe_graph_path': args.recipe_graph_path}
    input_path = out / 'inputs.json'
    if input_path.exists() and json.loads(input_path.read_text()) != inputs:
        raise ValueError('resume changed program, split, student or sampling identity')
    write_json_durable(input_path, inputs)
    state_path = out / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'generation': 0, 'best': None, 'baseline': None, 'history': []}
    if state.get('baseline') is None:
        state['baseline'] = {'development': evaluate(args, out, 'unguided-dev', '', dev),
                             'confirmation': evaluate(args, out, 'unguided-confirm', '', confirm)}
        write_json_durable(state_path, state)
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
    baseline = state['baseline']
    qualified = qualification(baseline, best, args.min_gain_fraction)
    outcome = {'schema': 'natlang.memetic-program-outcome/1', 'program_id': args.program_id,
               'status': 'qualified' if qualified else 'no_lift',
               'baseline': {split: {key: baseline[split][key] for key in ('successes', 'scored')}
                            for split in ('development', 'confirmation')},
               'best': {split: {key: best[split][key] for key in ('successes', 'scored')}
                        for split in ('development', 'confirmation')},
               'min_gain_fraction': args.min_gain_fraction,
               'accepted_revisions': sum(item['accepted'] for item in state['history'])}
    write_json_durable(out / 'result.json', outcome)
    if not qualified:
        (out / 'binding.json').unlink(missing_ok=True)
        print(json.dumps({'result': str(out / 'result.json'), 'status': 'no_lift'}))
        return outcome
    binding = {'schema': 'natlang.memetic-program-guidance/1', 'program_id': args.program_id,
        'student': identity, 'text': best['text'], 'native_state_id': best['write']['id'],
        'native_state_artifact': best['archive'],
        'source_capture': best['write']['source_capture'], 'source_text_sha256': best['write']['source_text_sha256'],
        'program_guidance': state_guidance(best['text'], best['write']['id']),
        'development': outcome['best']['development'],
        'confirmation': outcome['best']['confirmation'],
        'unguided_baseline': outcome['baseline'],
        'min_gain_fraction': args.min_gain_fraction,
        'inputs_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest()}
    if args.recipe_base is not None:
        binding['native_training_graphs'] = export_training_graphs(Path(best['development']['dir']), out,
            program_id=args.program_id, confirmation_ids={row['id'] for row in confirm},
            written=best['write'], actor=identity, server=args.server)
        binding['native_training_recipe'] = emit_recipe(args.recipe_base,
            out / 'native-guidance-recipe.json', binding['native_training_graphs'],
            foundation_id=args.foundation_id,
            graph_path=args.recipe_graph_path)
    write_json_durable(out / 'binding.json', binding)
    print(json.dumps({'binding': str(out / 'binding.json'), 'accepted': sum(h['accepted'] for h in state['history']),
                      'development': binding['development'], 'confirmation': binding['confirmation']}))
    return outcome


if __name__ == '__main__':
    main()
