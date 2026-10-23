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

from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex
from ..common.jsonio import write_json_durable
from ..common.runtime_inventory import python_runtime_binding, task_runtime_inventory
from ..eval.task_packet import read_jsonl
from . import memetic_task_runner as runner
from . import memetic_teacher_ferry as teacher_ferry
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


def actor_identity(server: str, info=None):
    info = info if info is not None else request_json(server, '/v1/neuralese/info')
    return teacher_ferry.served_native_identity(info)


def local_runtime_binding(runtime_root: Path | None = None) -> dict:
    """Bind the built collector, its deployment files, and the Python scorer."""
    import subprocess
    import sys
    root = Path(runtime_root).resolve(strict=True) if runtime_root is not None else Path(__file__).resolve().parents[4]
    if runtime_root is not None and root != Path(__file__).resolve().parents[4]:
        # Recompute a historical runtime with its own inventory implementation, not today's code.
        package_root = root / 'training' / 'neuralese'
        code = ('import json,sys;sys.path.insert(0,' + repr(str(package_root)) + ');'
                'from natlang_neuralese.rl.memetic_program import local_runtime_binding;'
                'print(json.dumps(local_runtime_binding(),sort_keys=True,separators=(",",":")))')
        result = subprocess.run([sys.executable, '-c', code], cwd=root, check=True, capture_output=True, text=True)
        return json.loads(result.stdout.strip().splitlines()[-1])
    inventory = task_runtime_inventory(root / 'ts-host')
    command = root / 'scripts/run_bonsai_queue.py'
    return {'schema': 'natlang.memetic-evaluator-runtime/1',
            'typescript_sha256': canonical_json_sha256_hex(inventory),
            'typescript_files': len(inventory),
            'python': python_runtime_binding(),
            'queue_worker_sha256': sha256_file_hex(command),
            'node_version': subprocess.check_output(['node', '--version'], text=True).strip(),
            'python_version': sys.version}


def _definition_bindings(root: Path) -> dict[str, str]:
    """Hash the exact Python evaluation functions and compiled task collector entrypoints."""
    import ast
    files = {
        'queue_worker': 'scripts/run_bonsai_queue.py',
        'task_classifier': 'training/neuralese/natlang_neuralese/eval/task_packet.py',
        'task_rollouts': 'training/neuralese/natlang_neuralese/rl/task_rollouts.py',
        'memetic_program': 'training/neuralese/natlang_neuralese/rl/memetic_program.py',
        'memetic_runner': 'training/neuralese/natlang_neuralese/rl/memetic_task_runner.py',
        'collector_cli': 'ts-host/dist/teacher/cli.js',
        'collector_core': 'ts-host/dist/teacher/collector.js',
    }
    result = {name: sha256_file_hex(root / relative) for name, relative in files.items()
              if name in {'queue_worker', 'task_classifier', 'collector_cli', 'collector_core'}}
    definitions = {
        'task_rollouts': ('build_queue', 'reward_of', 'rollout_turns'),
        'memetic_program': ('prepare_slices', 'evaluate'),
        'memetic_runner': ('default_system_prompt', 'evaluate'),
    }
    for module, names in definitions.items():
        path = root / files[module]
        text = path.read_text(encoding='utf-8')
        tree = ast.parse(text)
        found = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        if any(name not in found for name in names):
            raise ValueError(f'evaluator continuation source is missing {module} behavior bindings')
        for name in names:
            source = ast.get_source_segment(text, found[name])
            result[f'{module}.{name}'] = hashlib.sha256(source.encode()).hexdigest()
    return result


def _score_on_runtime(runtime_root: Path, evaluation: Path) -> dict:
    """Replay task outcomes read-only; the caller verifies complete actor capture separately.

    Empty result files are normal for verified request-budget prefixes. Their
    journal outcome is still scored; this function grants no capture admission.
    """
    import subprocess
    import sys
    code = r'''import json,shutil,sys,tempfile
from pathlib import Path
root=Path(sys.argv[1]); source=Path(sys.argv[2]); sys.path.insert(0,str(root/'training/neuralese'))
from natlang_neuralese.rl.task_rollouts import score,read_jsonl
with tempfile.TemporaryDirectory(prefix='natlang-score-replay-') as temp:
 target=Path(temp); packet=target/'packet.jsonl'; shutil.copyfile(source/'packet.jsonl',packet)
 for name in ('preparation.json','journal.jsonl'):
  shutil.copyfile(source/name,target/name)
 queue=read_jsonl(source/'queue.jsonl'); copied=[]
 for index,entry in enumerate(queue):
  original=Path(entry['output'])
  if not original.is_file(): raise RuntimeError('scorer replay requires saved result files')
  output=target/f'output-{index:06d}.jsonl'; shutil.copyfile(original,output)
  copied.append({**entry,'output':str(output),'jobs':str(target/f'jobs-{index:06d}')})
 (target/'queue.jsonl').write_text(''.join(json.dumps(row,sort_keys=True)+'\n' for row in copied),encoding='utf-8')
 report=score(target)
 print(json.dumps(report,sort_keys=True,separators=(',',':')))
'''
    process = subprocess.run([sys.executable, '-c', code, str(runtime_root), str(evaluation)],
                             cwd=runtime_root, check=True, capture_output=True, text=True)
    lines = process.stdout.strip().splitlines()
    if not lines:
        raise ValueError('isolated scorer replay produced no report')
    return json.loads(lines[-1])


def _manifest_digest(value: dict) -> str:
    unsigned = {key: item for key, item in value.items() if key != 'manifest_sha256'}
    return canonical_json_sha256_hex(unsigned)


def _load_runtime_continuation(out: Path) -> dict | None:
    path = out / 'evaluator-runtime-continuation.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    if value.get('schema') != 'natlang.memetic-evaluator-runtime-continuation/1' or \
            value.get('manifest_sha256') != _manifest_digest(value):
        raise ValueError('evaluator runtime continuation receipt is invalid')
    current, current_sha = value.get('origin_inputs'), value.get('origin_inputs_sha256')
    if not isinstance(current, dict) or current_sha != _pretty_json_sha256(current):
        raise ValueError('evaluator runtime continuation has an invalid origin binding')
    for row in value.get('history', []):
        if (row.get('from_inputs') != current or row.get('from_inputs_sha256') != current_sha or
                row.get('from_inputs_sha256') != _pretty_json_sha256(row.get('from_inputs')) or
                row.get('to_inputs_sha256') != _pretty_json_sha256(row.get('to_inputs')) or
                row.get('from_evaluator_runtime') != row.get('from_inputs', {}).get('evaluator_runtime') or
                row.get('to_evaluator_runtime') != row.get('to_inputs', {}).get('evaluator_runtime')):
            raise ValueError('evaluator runtime continuation chain is broken')
        current, current_sha = row['to_inputs'], row['to_inputs_sha256']
    if value.get('current_inputs') != current or value.get('current_inputs_sha256') != current_sha:
        raise ValueError('evaluator runtime continuation current input differs from its chain')
    return value


def _pretty_json_sha256(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, indent=2, sort_keys=True).encode('utf-8')).hexdigest()


def _verify_existing_fit_outputs(out: Path, actor: dict, previous_root: Path,
                                 current_root: Path, prior_inputs: set[str]) -> list[dict]:
    fits = []
    for fit_path in sorted(out.glob('*-fit.json')):
        name = fit_path.name[:-len('-fit.json')]
        fit = json.loads(fit_path.read_text())
        target = out / 'evals' / name
        if (fit.get('evaluation_inputs_sha256') not in prior_inputs or
                fit.get('actor_capture_reached_model') is not True or
                fit.get('scored') != len(read_jsonl(target / 'queue.jsonl')) or
                not student_capture_reached(target, actor)):
            raise ValueError(f'cached evaluation {name} is not a complete exact-actor measurement')
        # The rows already exist, so compare old and new score semantics without rerunning the collector.
        old_report = _score_on_runtime(previous_root, target)
        new_report = _score_on_runtime(current_root, target)
        if old_report != new_report or fit.get('successes') != new_report.get('successes') or \
                fit.get('scored') != new_report.get('scored'):
            raise ValueError(f'cached evaluation {name} changes under the current scorer')
        fits.append({'name': name, 'fit_sha256': sha256_file_hex(fit_path),
                     'evaluation_inputs_sha256': fit['evaluation_inputs_sha256'],
                     'score_sha256': canonical_json_sha256_hex(new_report),
                     'actor_sha256': canonical_json_sha256_hex(actor)})
    return fits


def _fit_receipt_continued(out: Path, name: str, fit_path: Path, fit: dict) -> bool:
    manifest = _load_runtime_continuation(out)
    if not manifest:
        return False
    digest = sha256_file_hex(fit_path)
    return any(row.get('name') == name and row.get('fit_sha256') == digest and
               row.get('evaluation_inputs_sha256') == fit.get('evaluation_inputs_sha256')
               for transition in manifest.get('history', [])
               for row in transition.get('verified_fits', []))


def evaluator_inputs(out: Path, inputs: dict, actor: dict, previous_runtime: Path | None) -> tuple[str, dict]:
    """Keep original input provenance immutable and chain an explicitly checked runtime continuation."""
    out.mkdir(parents=True, exist_ok=True)
    input_path = out / 'inputs.json'
    continuation_path = out / 'evaluator-runtime-continuation.json'
    if not input_path.exists():
        write_json_durable(input_path, inputs)
    origin_inputs = json.loads(input_path.read_text())
    origin_sha = sha256_file_hex(input_path)
    current_root = Path(__file__).resolve().parents[4]
    manifest = _load_runtime_continuation(out)
    previous_inputs = manifest['current_inputs'] if manifest else origin_inputs
    previous_sha = manifest['current_inputs_sha256'] if manifest else origin_sha
    if previous_inputs != inputs:
        old_nonruntime = {k: v for k, v in previous_inputs.items() if k != 'evaluator_runtime'}
        new_nonruntime = {k: v for k, v in inputs.items() if k != 'evaluator_runtime'}
        if old_nonruntime != new_nonruntime:
            raise ValueError('resume changed actor, task splits, packets or sampling; runtime continuation cannot authorize it')
        if previous_runtime is None:
            raise ValueError('evaluator inputs changed; a verified --runtime-continuation-from is required')
        previous_root = previous_runtime.resolve(strict=True)
        old_runtime = local_runtime_binding(previous_root)
        new_runtime = local_runtime_binding(current_root)
        if old_runtime != previous_inputs.get('evaluator_runtime') or new_runtime != inputs.get('evaluator_runtime'):
            raise ValueError('evaluator continuation runtime roots do not match their exact input receipts')
        if any(old_runtime.get(key) != new_runtime.get(key) for key in
               ('typescript_sha256', 'typescript_files', 'queue_worker_sha256', 'node_version', 'python_version')):
            raise ValueError('evaluator continuation changed the compiled TypeScript runtime, queue worker or language runtime')
        old_semantics = _definition_bindings(previous_root)
        new_semantics = _definition_bindings(current_root)
        exact_keys = ('queue_worker', 'task_classifier', 'collector_cli', 'collector_core',
                      'task_rollouts.build_queue', 'task_rollouts.reward_of', 'task_rollouts.rollout_turns',
                      'memetic_program.prepare_slices',
                      'memetic_runner.default_system_prompt', 'memetic_runner.evaluate')
        if any(old_semantics.get(key) != new_semantics.get(key) for key in exact_keys):
            raise ValueError('evaluator continuation changed a task collector or policy-scoring mechanism')
        prior_inputs = {origin_sha, previous_sha}
        if manifest:
            prior_inputs.update(row['to_inputs_sha256'] for row in manifest.get('history', []))
            prior_inputs.update(row['from_inputs_sha256'] for row in manifest.get('history', []))
        verified_fits = _verify_existing_fit_outputs(out, actor, previous_root, current_root, prior_inputs)
        if manifest is None:
            manifest = {'schema': 'natlang.memetic-evaluator-runtime-continuation/1',
                        'origin_inputs': origin_inputs, 'origin_inputs_sha256': origin_sha,
                        'current_inputs': previous_inputs, 'current_inputs_sha256': previous_sha, 'history': []}
        transition = {'from_inputs': previous_inputs, 'from_inputs_sha256': previous_sha,
                      'to_inputs': inputs, 'to_inputs_sha256': _pretty_json_sha256(inputs),
                      'from_runtime_root': str(previous_root), 'to_runtime_root': str(current_root),
                      'from_evaluator_runtime': old_runtime, 'to_evaluator_runtime': new_runtime,
                      'behavioral_bindings': new_semantics, 'verified_fits': verified_fits}
        manifest['history'].append(transition)
        manifest['current_inputs'] = inputs
        manifest['current_inputs_sha256'] = transition['to_inputs_sha256']
        manifest['manifest_sha256'] = _manifest_digest(manifest)
        write_json_durable(continuation_path, manifest)
        return manifest['current_inputs_sha256'], manifest
    if manifest is None:
        manifest = {'schema': 'natlang.memetic-evaluator-runtime-continuation/1',
                    'origin_inputs': origin_inputs, 'origin_inputs_sha256': origin_sha,
                    'current_inputs': inputs, 'current_inputs_sha256': origin_sha,
                    'history': []}
        manifest['manifest_sha256'] = _manifest_digest(manifest)
        write_json_durable(continuation_path, manifest)
    elif manifest.get('origin_inputs_sha256') != origin_sha:
        raise ValueError('evaluator continuation changed its immutable origin input receipt')
    return manifest['current_inputs_sha256'], manifest


def author_provenance(path: Path, model: str, endpoint: str):
    """Bind authoring to the existing self-hosted Qwen source and runtime."""
    value = json.loads(path.read_text())
    exact_sources = list(teacher_ferry.TEACHER_SOURCE_BINDINGS)
    if (value.get('schema') != 'natlang.memetic-author-provenance/1'
            or model != teacher_ferry.TEACHER_SERVE_ALIAS
            or value.get('model_id') != teacher_ferry.TEACHER_MODEL_ID
            or value.get('serve_alias') != teacher_ferry.TEACHER_SERVE_ALIAS
            or value.get('revision') != teacher_ferry.TEACHER_REVISION
            or value.get('hosting') != 'self-hosted-coreweave'
            or endpoint.rstrip('/') != teacher_ferry.TEACHER_URL
            or value.get('endpoint') != teacher_ferry.TEACHER_URL
            or value.get('sandbox_id') != teacher_ferry.TEACHER_SANDBOX_ID
            or value.get('source_bindings') != exact_sources
            or any(not isinstance(value.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', value[key])
                   for key in ('checkpoint_sha256', 'tokenizer_fingerprint', 'runtime_sha256', 'source_receipt_sha256'))):
        raise ValueError('memetic author needs the exact pinned self-hosted Qwen teacher checkpoint/tokenizer/runtime/source receipt')
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'receipt': value}


def evaluate(args, out: Path, name: str, guidance: str, rows: list[dict], *, base_system: str,
             actor: dict, server_info_sha256: str, evaluation_inputs_sha256: str | None = None):
    packet = out / (name + '-packet.jsonl')
    packet_bytes = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows).encode()
    if packet.exists() and packet.read_bytes() != packet_bytes:
        raise ValueError('cached evaluation packet differs from the bound invocation slice')
    if not packet.exists():
        from ..common.jsonio import write_bytes_durable
        write_bytes_durable(packet, packet_bytes)
    target = out / 'evals' / name
    fit_path = out / (name + '-fit.json')
    receipt = {'evaluation_inputs_sha256': evaluation_inputs_sha256 or sha256_file_hex(out / 'inputs.json'),
               'guidance_sha256': hashlib.sha256(guidance.encode()).hexdigest(),
               'packet_sha256': hashlib.sha256(packet_bytes).hexdigest()}
    if fit_path.exists():
        fit = json.loads(fit_path.read_text())
        exact_receipt = all(fit.get(key) == value for key, value in receipt.items()
                            if key != 'evaluation_inputs_sha256')
        same_input = fit.get('evaluation_inputs_sha256') == receipt['evaluation_inputs_sha256']
        continued_input = _fit_receipt_continued(out, name, fit_path, fit)
        if not exact_receipt or (not same_input and not continued_input):
            raise ValueError('cached guidance fitness is not bound to this student evaluation')
        if (fit.get('scored') != len(rows) * args.samples or fit.get('actor_capture_reached_model') is not True
                or not student_capture_reached(target, actor)):
            raise ValueError('cached guidance fitness lacks complete exact-actor rollout evidence')
        if guidance:
            state_id = re.fullmatch(r'\ue002(nzs1_[a-z2-7]{52})\ue003', guidance)
            if (not state_id or fit.get('native_import_reached_model') is not True
                    or not native_import_reached(target, state_id.group(1))):
                raise ValueError('cached guidance fitness lacks native state import evidence')
        return fit
    if target.exists():
        shutil.rmtree(target)  # only this incomplete, invocation-specific evaluation
    options = argparse.Namespace(**vars(args))
    options.out = out
    options.rows = len(rows)
    options.shards = args.shards
    options.policy_checkpoint_sha256 = actor['checkpoint_sha256']
    individual = {'id': name, 'guidance': guidance}
    fit = runner.evaluate(options, individual, packet, base_system)
    if fit['scored'] != len(rows) * args.samples:
        raise ValueError('guidance evaluation lacks complete scored task outcomes')
    if not student_capture_reached(target, actor):
        raise ValueError('scored evaluation lacks exact student actor capture on every invocation')
    if canonical_json_sha256_hex(request_json(args.server, '/v1/neuralese/info')) != server_info_sha256:
        raise ValueError('serving student info changed during evaluation')
    if guidance:
        state_id = re.fullmatch(r'\ue002(nzs1_[a-z2-7]{52})\ue003', guidance)
        if not state_id or not native_import_reached(target, state_id.group(1)):
            raise ValueError('guidance state did not reach the executing model as native state')
        fit['native_import_reached_model'] = True
    fit['actor_capture_reached_model'] = True
    fit.update(receipt)
    write_json_durable(fit_path, fit)
    return fit


def captured_invocations(evaluation: Path):
    """Read complete results or the shared verifier's exact budget prefix."""
    from .task_rollouts import _budget_terminated_prefix
    packet = evaluation / 'packet.jsonl'
    queue = evaluation / 'queue.jsonl'
    if not queue.exists():
        return []
    entries = read_jsonl(queue)
    tasks = read_jsonl(packet)
    preparation = json.loads((evaluation / 'preparation.json').read_text())
    packet_sha = hashlib.sha256(packet.read_bytes()).hexdigest()
    if preparation.get('packet_sha256') != packet_sha:
        return []
    finishes = {row['key']: row for row in read_jsonl(evaluation / 'journal.jsonl')
                if row.get('event') == 'finish'}
    results = []
    for entry in entries:
        path = Path(entry['output'])
        rows = read_jsonl(path) if path.exists() else []
        if not rows:
            prefix = _budget_terminated_prefix(entry, tasks[entry['index']], finishes.get(entry['key']),
                packet_source_path=packet, packet_source_sha256=packet_sha)
            rows = [{'trajectory': prefix['trajectory']}] if prefix else []
        results.append(rows)
    return results


def student_capture_reached(evaluation: Path, actor: dict) -> bool:
    invocations = captured_invocations(evaluation)
    if not invocations:
        return False
    for rows in invocations:
        captured = False
        for result in rows:
            for step in result.get('trajectory') or []:
                sampling = (step.get('model_response') or {}).get('rl_sampling') or {}
                if not sampling:
                    continue
                if any(sampling.get('policy', {}).get(key) != actor[key] for key in IDENTITY):
                    return False
                captured = True
        if not captured:
            return False
    return True


def native_import_reached(evaluation: Path, state_id: str) -> bool:
    """Require a server import receipt, not merely a sentinel in rendered text."""
    invocations = captured_invocations(evaluation)
    if not invocations:
        return False
    for rows in invocations:
        imported = False
        for result in rows:
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
    parser.add_argument('--baseline-only', action='store_true')
    parser.add_argument('--runtime-continuation-from', type=Path,
                        help='exact prior frozen evaluator runtime root for a verified coordinator-only continuation')
    parser.add_argument('--author')
    parser.add_argument('--author-ferry', help='durable evaluator-local teacher transport on loopback:8766')
    parser.add_argument('--author-model')
    parser.add_argument('--author-provenance', type=Path)
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
    info = request_json(args.server, '/v1/neuralese/info')
    identity = actor_identity(args.server, info)
    server_info_sha256 = canonical_json_sha256_hex(info)
    base_system = runner.default_system_prompt()
    out = args.out.resolve(); out.mkdir(parents=True, exist_ok=True)
    inputs = {'schema': 'natlang.memetic-student-evaluation-inputs/1', 'program_id': args.program_id,
        'development': hashlib.sha256(args.development.read_bytes()).hexdigest(),
        'confirmation': hashlib.sha256(args.confirmation.read_bytes()).hexdigest(),
        'student': identity, 'student_server': args.server, 'student_info_sha256': server_info_sha256,
        'student_model_id': args.model_id,
        'system_prompt_sha256': hashlib.sha256(base_system.encode()).hexdigest(),
        'evaluator_runtime': local_runtime_binding(),
        'seed_base': args.seed_base, 'samples': args.samples, 'temperature': args.temperature,
        'shards': args.shards, 'case_seconds': args.case_seconds,
        'max_model_requests': args.max_model_requests, 'max_turns': args.max_turns}
    input_path = out / 'inputs.json'
    current_inputs_sha256, _ = evaluator_inputs(
        out, inputs, identity, args.runtime_continuation_from)
    state_path = out / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'generation': 0, 'best': None, 'baseline': None, 'history': []}
    measured_baseline = {'development': evaluate(args, out, 'unguided-dev', '', dev,
                             base_system=base_system, actor=identity, server_info_sha256=server_info_sha256,
                             evaluation_inputs_sha256=current_inputs_sha256),
                         'confirmation': evaluate(args, out, 'unguided-confirm', '', confirm,
                             base_system=base_system, actor=identity, server_info_sha256=server_info_sha256,
                             evaluation_inputs_sha256=current_inputs_sha256)}
    if state.get('baseline') is not None and state['baseline'] != measured_baseline:
        raise ValueError('resumed baseline fitness differs from its exact cached evaluation receipt')
    if state.get('baseline') is None:
        state['baseline'] = measured_baseline
        write_json_durable(state_path, state)
    if args.baseline_only:
        result = {'schema': 'natlang.memetic-baseline-result/1', 'status': 'baseline_complete',
                  'program_id': args.program_id, 'student': identity,
                  'inputs_sha256': sha256_file_hex(input_path),
                  'current_inputs_sha256': current_inputs_sha256,
                  'runtime_continuation_sha256': sha256_file_hex(out / 'evaluator-runtime-continuation.json'),
                  'development': {key: measured_baseline['development'][key] for key in ('successes', 'scored')},
                  'confirmation': {key: measured_baseline['confirmation'][key] for key in ('successes', 'scored')}}
        write_json_durable(out / 'baseline-result.json', result)
        print(json.dumps(result))
        return result
    if any(value is None for value in (args.author, args.author_model, args.author_provenance)):
        parser.error('guided iteration requires --author, --author-model and --author-provenance')
    if args.author_ferry != 'http://127.0.0.1:8766':
        parser.error('--author-ferry must be the evaluator-local http://127.0.0.1:8766 endpoint')
    teacher = author_provenance(args.author_provenance, args.author_model, args.author)
    author_inputs = {'schema': 'natlang.memetic-author-inputs/1', 'author': teacher,
                     'author_server': args.author, 'author_ferry': args.author_ferry, 'min_gain_fraction': args.min_gain_fraction,
                     'recipe_base': (hashlib.sha256(args.recipe_base.read_bytes()).hexdigest()
                                     if args.recipe_base else None),
                     'foundation_id': args.foundation_id, 'recipe_graph_path': args.recipe_graph_path}
    author_path = out / 'author.json'
    if author_path.exists() and json.loads(author_path.read_text()) != author_inputs:
        raise ValueError('resume changed author or guidance training recipe identity')
    write_json_durable(author_path, author_inputs)
    if state['best'] is None:
        # A native block must contain actual text, including for the seed.
        seed_text = 'Follow the program instructions and verify the requested result before returning it.'
        seed_write = write_state(args.server, seed_text, identity)
        seed = {'text': seed_text, 'write': seed_write,
                'archive': archive_state(args.server, out, seed_write)}
        seed['development'] = evaluate(args, out, 'seed-dev', state_guidance(seed_text, seed_write['id']), dev,
                                       base_system=base_system, actor=identity, server_info_sha256=server_info_sha256,
                                       evaluation_inputs_sha256=current_inputs_sha256)
        seed['confirmation'] = evaluate(args, out, 'seed-confirm', state_guidance(seed_text, seed_write['id']), confirm,
                                        base_system=base_system, actor=identity, server_info_sha256=server_info_sha256,
                                        evaluation_inputs_sha256=current_inputs_sha256)
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
        candidate['development'] = evaluate(args, out, prefix + '-dev', state_guidance(proposed, written['id']), dev,
                                             base_system=base_system, actor=identity, server_info_sha256=server_info_sha256,
                                             evaluation_inputs_sha256=current_inputs_sha256)
        candidate['confirmation'] = evaluate(args, out, prefix + '-confirm', state_guidance(proposed, written['id']), confirm,
                                              base_system=base_system, actor=identity, server_info_sha256=server_info_sha256,
                                              evaluation_inputs_sha256=current_inputs_sha256)
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
               'accepted_revisions': sum(item['accepted'] for item in state['history']),
               'inputs_sha256': sha256_file_hex(input_path),
               'current_inputs_sha256': current_inputs_sha256,
               'runtime_continuation_sha256': sha256_file_hex(out / 'evaluator-runtime-continuation.json')}
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
        'inputs_sha256': sha256_file_hex(input_path), 'current_inputs_sha256': current_inputs_sha256,
        'runtime_continuation_sha256': sha256_file_hex(out / 'evaluator-runtime-continuation.json'),
        'author_sha256': sha256_file_hex(author_path)}
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
