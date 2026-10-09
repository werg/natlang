"""Declared, gated training stages with frozen code and explicit artifact handoff.

Stage implementations are shared Python modules. Extend HANDLERS with a typed
handler and declare its dependencies in a JSON recipe. No shell command strings.
An unsuccessful stage cannot advance its dependents or issue a certificate.
"""
import argparse
import hashlib
import json
import re
import shutil
import signal
import subprocess
import sys
import os
from pathlib import Path

from .output_embedding_projection import sha


HANDLERS = {
    'core_text_warmup': {'module':'natlang_neuralese.train.text_warmup', 'required_inputs':{'records','pieces'}, 'optional_inputs':{'text_data','continue_from','heads','student_checkpoint'},
                        'parameters':{'text_data','student_checkpoint','batch','steps','tokens','prefix_tokens','cutoff','group_size',
                          'backbone_training','rank','optimizer','lr','sketch_lr','embedding_weight','sketch_weight','text_weight',
                          'projection_patience','projection_min_evals','projection_min_improvement',
                          'backbone_ramp_evals','pass_ramp_evals','checkpoint_every','checkpoint_minutes','eval_every','held_documents','seed','checkpoint_layers',
                          'max_ce_delta','max_relative_mse','min_agreement','consecutive_gates','neuralese_input',
                          'input_map_kernel','input_map_rank','rollout_passes','rollout_start_passes',
                          'max_sequence_passes'},'result':'heads.pt'},
    'text_warmup_runtime': {'module':'natlang_neuralese.eval.text_warmup_runtime', 'required_inputs':{'records'}, 'optional_inputs':{'heads'},
                            'parameters':set(),'result':'report.json'},
    'raw_recurrence_training': {'module': 'natlang_neuralese.train.trajectories', 'required_inputs':{'records','pieces'}, 'optional_inputs':{'heads'},
                                'parameters': {'steps', 'batch', 'lr', 'rank', 'lora_lr', 'backbone_lr', 'backbone_training', 'max_tokens',
                                               'train', 'eval', 'handover', 'write_curriculum', 'max_writes',
                                               'write_depth', 'tokens_per_vector', 'heads_lr', 'distill',
                                               'crisp_weight', 'memory_gb', 'backward_policy', 'graph_memory_gb',
                                               'graph_headroom_gb', 'checkpoint_layers', 'checkpoint_attention_only',
                                               'staged_checkpoint_attention_only', 'checkpoint_elide_rng', 'producer_batch_size',
                                               'producer_batch_memory_gb', 'ffn_chunk_tokens',
                                               'optimizer', 'checkpoint_every', 'eval_every', 'seed', 'writer_text_weight',
                                               'max_write_vectors', 'content_transport', 'writer_length_policy', 'writer_supervision', 'stop_supervision',
                                               'sketch_gradient', 'sketch_target_weight', 'sketch_target_backbone_scale',
                                               'projection_anchor_weight', 'projection_anchor_backbone_scale',
                                               'local_stage_batch_size', 'train_control_rows', 'token_cache_mib'},
                                'result': 'checkpoint.pt'},
    'raw_runtime_qualification': {'module': 'natlang_neuralese.eval.raw_port_handoff', 'required_inputs':{'records'}, 'optional_inputs':set(),
                                  'parameters': {'limit', 'max_length'}, 'result': 'heads.pt'},
    'verified_heads_handoff': {'module':'natlang_neuralese.train.verified_heads_handoff',
                               'required_inputs':{'warmup_checkpoint','warmup_manifest','warmup_heads','warmup_runtime_report'},
                               'optional_inputs':set(),
                               'parameters':{'max_ce_delta','max_relative_mse','min_agreement'},'result':'report.json'},
    'token_identity': {'module': 'natlang_neuralese.eval.foundation', 'required_inputs':{'records'}, 'optional_inputs':set(),
                       'parameters': {'limit', 'max_tokens'}, 'result': 'identity.json'},
    'causal_embedding_distillation': {'module': 'natlang_neuralese.train.causal_bootstrap', 'required_inputs':{'records'}, 'optional_inputs':{'pieces'},
                                     'parameters': {'cutoff', 'steps', 'batch', 'lr', 'tokens', 'contexts',
                                                    'context_tokens', 'eval_every', 'checkpoint_every',
                                                    'seed', 'agreement_gate', 'kl_gate', 'source_fraction',
                                                    'argmax_weight', 'continue_from', 'stop_on_gate'}, 'result': 'best-checkpoint.pt'},
}
DIRECT_STAGE_SCHEMA = 'natlang.neuralese-declared-direct-stage/1'

INPUT_BINDING_NAME = re.compile(r'[a-z][a-z0-9_.-]*')


def validate_input_bindings(recipe):
    """Validate the optional, content-pinned dataset catalog and stage role map."""
    catalog = recipe.get('input_bindings', {})
    if not isinstance(catalog, dict):
        raise ValueError('input_bindings must be an object')
    for name, binding in catalog.items():
        if not isinstance(name, str) or not INPUT_BINDING_NAME.fullmatch(name) or not isinstance(binding, dict):
            raise ValueError('invalid named input binding')
        if set(binding) != {'path', 'sha256'} or not isinstance(binding['path'], str) or not binding['path']:
            raise ValueError('each input binding requires only path and sha256')
        if not isinstance(binding['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', binding['sha256']):
            raise ValueError('input binding sha256 must be lowercase hex')
    referenced = set()
    for stage in recipe['stages']:
        if 'inputs' not in stage:
            continue
        selected = stage['inputs']
        if not isinstance(selected, dict):
            raise ValueError('stage inputs must map input roles to named bindings')
        kind = stage['kind']
        allowed = HANDLERS[kind]['required_inputs'] | HANDLERS[kind]['optional_inputs']
        if not set(selected) <= allowed or not HANDLERS[kind]['required_inputs'] <= set(selected):
            raise ValueError('stage input roles do not satisfy handler contract: ' + stage['id'])
        if any(not isinstance(name, str) or name not in catalog for name in selected.values()):
            raise ValueError('stage refers to an unknown named input binding')
        referenced.update(selected.values())
    if set(catalog) - referenced:
        raise ValueError('unused named input binding(s): ' + ', '.join(sorted(set(catalog) - referenced)))


def parse_input_binding_overrides(values):
    overrides = {}
    for value in values or []:
        name, sep, path = value.partition('=')
        if not sep or not INPUT_BINDING_NAME.fullmatch(name) or not path:
            raise ValueError('--input-binding must be NAME=PATH')
        if name in overrides:
            raise ValueError('duplicate --input-binding: ' + name)
        overrides[name] = path
    return overrides


def resolve_stage_inputs(recipe, stage, defaults, overrides=None):
    """Return hashed, exact inputs for one stage; overrides change location, never identity."""
    selected = stage.get('inputs')
    if selected is None:
        result = {}
        supported = HANDLERS[stage['kind']]['required_inputs'] | HANDLERS[stage['kind']]['optional_inputs']
        for role, path in defaults.items():
            if path is not None and role in supported:
                resolved = Path(path).resolve()
                result[role] = {'binding': None, 'path': str(resolved), 'sha256': sha(resolved)}
        missing = HANDLERS[stage['kind']]['required_inputs'] - set(result)
        if missing:
            raise ValueError('missing required stage input(s): ' + ', '.join(sorted(missing)))
        return result
    catalog = recipe.get('input_bindings', {})
    overrides = overrides or {}
    unknown = set(overrides) - set(catalog)
    if unknown:
        raise ValueError('override for undeclared input binding: ' + ', '.join(sorted(unknown)))
    result = {}
    for role, name in selected.items():
        spec = catalog[name]
        path = Path(overrides.get(name, spec['path'])).resolve()
        actual_sha = sha(path)
        if actual_sha != spec['sha256']:
            raise ValueError('input binding content hash mismatch: ' + name)
        result[role] = {'binding': name, 'path': str(path), 'sha256': actual_sha}
    return result


def stage_input_args(resolved, kind):
    """Serialize already-validated file roles to the shared handler CLIs."""
    args = []
    supported = HANDLERS[kind]['required_inputs'] | HANDLERS[kind]['optional_inputs']
    for role in ('records', 'pieces', 'text_data', 'continue_from', 'student_checkpoint', 'warmup_checkpoint',
                 'warmup_manifest', 'warmup_heads', 'warmup_runtime_report'):
        if role in resolved and role in supported:
            args += ['--' + role.replace('_', '-'), resolved[role]['path']]
    return args


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if recipe.get('schema') == DIRECT_STAGE_SCHEMA:
        validate_direct_stage_recipe(recipe)
        return recipe
    if recipe.get('schema') != 'natlang.neuralese-training-recipe/1' or not recipe.get('stages'):
        raise ValueError('invalid or empty training recipe')
    declared, complete, identity_stages, embedding_stages, runtime_stages = set(), set(), set(), set(), set()
    for stage in recipe['stages']:
        name, kind = stage.get('id'), stage.get('kind')
        if not isinstance(name, str) or not re.fullmatch(r'[a-z][a-z0-9_-]*', name) or name in declared:
            raise ValueError('stage id must be unique and safe')
        if kind not in HANDLERS:
            raise ValueError('unknown stage implementation: ' + str(kind))
        required = stage.get('requires')
        if not isinstance(required, list) or len(set(required)) != len(required) or not set(required) <= complete:
            raise ValueError('stage dependencies must precede the stage')
        parameters = stage.get('parameters')
        if not isinstance(parameters, dict) or not set(parameters) <= HANDLERS[kind]['parameters']:
            raise ValueError('unknown stage parameters')
        if 'text_data' in parameters and 'text_data' in stage.get('inputs', {}):
            raise ValueError('text_data must be declared either as a named stage input or a legacy parameter')
        if kind == 'core_text_warmup':
            neuralese_input = parameters.get('neuralese_input')
            if neuralese_input not in {'map', 'sketch'}:
                raise ValueError('core text warm-up must declare neuralese_input as map or sketch')
            rollout_passes = parameters.get('rollout_passes', 0)
            if type(rollout_passes) is not int or rollout_passes < 0 or rollout_passes == 1:
                raise ValueError('invalid core text warm-up rollout_passes')
            max_sequence_passes = parameters.get('max_sequence_passes', 3)
            if type(max_sequence_passes) is not int or max_sequence_passes < 3:
                raise ValueError('max_sequence_passes must be an integer of at least 3')
            if neuralese_input == 'map' and rollout_passes != 0:
                raise ValueError('mapped core text warm-up requires rollout_passes=0')
            if neuralese_input == 'map' and max_sequence_passes != 3:
                raise ValueError('max_sequence_passes applies only to the sketch sequence schedule')
            if rollout_passes and max_sequence_passes != 3:
                raise ValueError('choose either rollout_passes or max_sequence_passes above 3')
            if 'rollout_start_passes' in parameters:
                start_passes = parameters['rollout_start_passes']
                if (type(start_passes) is not int or start_passes < 2 or rollout_passes == 0 or
                        start_passes > rollout_passes):
                    raise ValueError('rollout_start_passes must be an integer from 2 through rollout_passes')
            for option_name in ('input_map_kernel', 'input_map_rank'):
                value = parameters.get(option_name, 4 if option_name == 'input_map_kernel' else 64)
                if type(value) is not int or value < 1:
                    raise ValueError('invalid core text warm-up ' + option_name)
        if kind == 'token_identity':
            identity_stages.add(name)
        if kind == 'causal_embedding_distillation' and not set(required) & identity_stages:
            raise ValueError('embedding distillation requires an explicit token identity gate')
        if kind == 'causal_embedding_distillation':
            embedding_stages.add(name)
        if kind == 'raw_runtime_qualification' and not set(required) & embedding_stages:
            raise ValueError('runtime qualification requires an explicit embedding foundation')
        if kind == 'raw_runtime_qualification':
            runtime_stages.add(name)
        if kind == 'verified_heads_handoff' and not set(required) & runtime_stages:
            raise ValueError('verified heads handoff requires the raw runtime qualification stage')
        if kind == 'core_text_warmup' and 'continue_from' in stage.get('inputs', {}):
            handoffs = [s for s in recipe['stages'] if s['id'] in required and s['kind']=='verified_heads_handoff']
            if (len(handoffs) != 1 or
                    handoffs[0].get('inputs', {}).get('warmup_checkpoint') != stage['inputs']['continue_from'] or
                    handoffs[0].get('inputs', {}).get('warmup_heads') != stage['inputs'].get('heads')):
                raise ValueError('continuation checkpoint and heads must be the exact files validated by a required verified_heads_handoff')
        if kind == 'core_text_warmup' and 'heads' in stage.get('inputs', {}) and 'continue_from' not in stage.get('inputs', {}):
            raise ValueError('stage-specific heads input requires a verified full-state continuation checkpoint')
        if kind == 'text_warmup_runtime' and 'heads' not in stage.get('inputs', {}):
            if not any(s['id'] in required and s['kind'] == 'core_text_warmup' for s in recipe['stages']):
                raise ValueError('adapted runtime requires a core warm-up predecessor or exact stage heads binding')
        if kind == 'raw_recurrence_training':
            runtime_dependencies = [s for s in recipe['stages']
                                    if s['id'] in required and s['kind'] == 'text_warmup_runtime']
            if not runtime_dependencies:
                raise ValueError('recurrence training requires an adapted warm-up runtime gate')
            if 'heads' not in stage.get('inputs', {}) and not any(
                    s['id'] in required and s['kind'] == 'core_text_warmup' for s in recipe['stages']):
                raise ValueError('recurrence requires a core warm-up predecessor or exact stage heads binding')
        declared.add(name)
        complete.add(name)
    if not identity_stages or not any(s['kind'] == 'causal_embedding_distillation' for s in recipe['stages']):
        raise ValueError('neuralese recipe must declare identity and embedding distillation stages')
    validate_input_bindings(recipe)
    return recipe


def validate_direct_stage_recipe(recipe):
    """Validate a single shared handler stage with exact external lineage pins.

    This path is for a narrowly scoped repair that reuses previously qualified
    weights and therefore must not rerun unrelated foundation stages. It uses
    the same handler registry, input-role contracts, and typed parameter CLI as
    a full recipe.
    """
    if recipe.get('execution') != 'direct_shared_stage' or not isinstance(recipe.get('id'), str):
        raise ValueError('direct stage must declare its execution mode and recipe id')
    stage = recipe.get('stage')
    if not isinstance(stage, dict) or set(stage) != {'id', 'kind'}:
        raise ValueError('direct stage needs exactly an id and registered handler kind')
    if not re.fullmatch(r'[a-z][a-z0-9_-]*', stage['id']) or stage['kind'] not in HANDLERS:
        raise ValueError('invalid direct stage identity or handler')
    handler = HANDLERS[stage['kind']]
    inputs = recipe.get('inputs')
    if not isinstance(inputs, dict):
        raise ValueError('direct stage requires pinned input bindings')
    roles = set(inputs)
    if not handler['required_inputs'] <= roles or not roles <= handler['required_inputs'] | handler['optional_inputs']:
        raise ValueError('direct stage input roles do not satisfy the shared handler contract')
    for role, binding in inputs.items():
        if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'} or
                not isinstance(binding['path'], str) or not binding['path'] or
                not isinstance(binding['sha256'], str) or
                not re.fullmatch(r'[0-9a-f]{64}', binding['sha256'])):
            raise ValueError('invalid direct stage input binding: ' + role)
    parameters = recipe.get('parameters')
    if not isinstance(parameters, dict) or not set(parameters) <= handler['parameters']:
        raise ValueError('direct stage contains unsupported handler parameters')
    runtime = recipe.get('runtime')
    if (not isinstance(runtime, dict) or not re.fullmatch(r'sha256:[0-9a-f]{64}', runtime.get('image', '')) or
            not isinstance(runtime.get('package_code'), dict) or not runtime['package_code']):
        raise ValueError('direct stage requires a pinned image and shared package code inventory')
    lineage = recipe.get('lineage')
    if not isinstance(lineage, dict) or not lineage.get('scope'):
        raise ValueError('direct stage requires an explicit lineage scope')
    evidence = lineage.get('evidence', [])
    if not isinstance(evidence, list):
        raise ValueError('direct-stage lineage evidence must be a list')
    for item in evidence:
        if (not isinstance(item, dict) or set(item) != {'role', 'path', 'sha256'} or
                not isinstance(item['role'], str) or not item['role'] or
                not isinstance(item['path'], str) or not item['path'] or
                not isinstance(item['sha256'], str) or not re.fullmatch(r'[0-9a-f]{64}', item['sha256'])):
            raise ValueError('invalid direct-stage lineage evidence binding')
    outputs = recipe.get('outputs')
    if not isinstance(outputs, dict) or not isinstance(outputs.get('directory'), str):
        raise ValueError('direct stage requires a declared output directory')
    return recipe


def require_gate(report, kind):
    if kind == 'token_identity':
        if report.get('token_aligned_reference_passed') is not True:
            raise ValueError('token identity gate failed')
    elif kind == 'causal_embedding_distillation':
        if report.get('feedback_gate_passed') is not True:
            raise ValueError('embedding distillation gate failed')
    elif kind == 'raw_runtime_qualification':
        if report.get('runtime_transport_passed') is not True:
            raise ValueError('raw runtime transport gate failed')
    elif kind == 'verified_heads_handoff':
        if report.get('handoff_verified') is not True:
            raise ValueError('exact trained heads handoff gate failed')
    elif kind == 'raw_recurrence_training':
        if report.get('training_stage_completed') is not True or report.get('errors') != 0:
            raise ValueError('raw recurrence training stage incomplete or errored')
    elif kind=='core_text_warmup':
        if report.get('qualified') is not True:raise ValueError('text warm-up alignment goal not met')
    elif kind=='text_warmup_runtime':
        if report.get('runtime_qualified') is not True or report.get('output_reference_qualified') is not True:
            raise ValueError('adapted warm-up runtime/output reference gate failed')
    else:
        raise ValueError('no gate adapter for stage')


def write_json(path, value):
    path = Path(path)
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(value, indent=2) + '\n')
    pending.replace(path)


class _ChildSignalForwarder:
    """Forward container termination signals to the active training child.

    Container PID 1 can ignore the default SIGTERM disposition. Recipe runners
    therefore install a handler and explicitly notify their child, allowing the
    trainer to finish its current update and write its resumable checkpoint.
    The same object is used by direct and multi-stage recipes.
    """
    def __init__(self):
        self.stopped = [False]
        self.child = [None]
        self._previous = {}

    def _handle(self, sig, _frame):
        self.stopped[0] = True
        child = self.child[0]
        self._send_if_running(child, sig)

    @staticmethod
    def _send_if_running(child, sig):
        if child is None:
            return
        try:
            if child.poll() is None:
                child.send_signal(sig)
        except ProcessLookupError:
            # The child can exit between poll() and send_signal().
            pass

    def install(self):
        for sig in (signal.SIGINT, signal.SIGTERM):
            self._previous[sig] = signal.getsignal(sig)
            signal.signal(sig, self._handle)

    def attach(self, child):
        self.child[0] = child
        # A signal can arrive after the handler is installed but before Popen
        # returns. Deliver it once the process handle becomes available.
        if self.stopped[0]:
            self._send_if_running(child, signal.SIGTERM)
        return child

    def detach(self, child):
        if self.child[0] is child:
            self.child[0] = None

    def close(self):
        for sig, previous in self._previous.items():
            signal.signal(sig, previous)
        self._previous.clear()

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, _type, _value, _traceback):
        self.close()
        return False

    def run(self, command, *, env=None):
        child = self.attach(subprocess.Popen(command, env=env))
        try:
            return subprocess.CompletedProcess(command, child.wait())
        finally:
            self.detach(child)


def stage_parameter_args(parameters):
    """Serialize a validated stage's typed options into CLI arguments."""
    command = []
    for key, value in parameters.items():
        if isinstance(value, bool):
            command += ['--' + ('' if value else 'no-') + key.replace('_', '-')]
        else:
            command += ['--' + key.replace('_', '-'), str(value)]
    return command


def run_declared_direct_stage(recipe, recipe_path, output, launch_metadata, device):
    """Freeze, bind, and run one explicitly declared shared stage.

    Direct stages are only appropriate when lineage evidence already qualifies
    their prerequisites and rerunning those independent stages would change the
    experiment. The declaration still uses the shared handler registry and
    records exact input/code/runtime pins before the child starts.
    """
    validate_direct_stage_recipe(recipe)
    output = Path(output).resolve()
    declared_output = Path(recipe['outputs']['directory']).resolve()
    if output != declared_output:
        raise ValueError('direct-stage output differs from its declaration')
    if output.exists():
        raise ValueError('fresh direct-stage output directory required')
    metadata_path = Path(launch_metadata).resolve()
    if metadata_path.exists() or output == metadata_path or output in metadata_path.parents:
        raise ValueError('direct-stage launch metadata must be a fresh external path')

    inputs = {}
    for role, binding in recipe['inputs'].items():
        path = Path(binding['path']).resolve()
        if sha(path) != binding['sha256']:
            raise ValueError('direct-stage input hash mismatch: ' + role)
        inputs[role] = {'path': str(path), 'sha256': binding['sha256']}
    lineage_evidence = []
    for item in recipe['lineage'].get('evidence', []):
        path = Path(item['path']).resolve()
        if sha(path) != item['sha256']:
            raise ValueError('direct-stage lineage evidence hash mismatch: ' + item['role'])
        lineage_evidence.append({'role': item['role'], 'path': str(path), 'sha256': item['sha256']})

    package = Path(__file__).parents[1]
    actual_code = {str(path.relative_to(package)): sha(path)
                   for path in sorted(package.rglob('*.py'))}
    if actual_code != recipe['runtime']['package_code']:
        raise ValueError('direct-stage shared package differs from its frozen code inventory')

    output.mkdir(parents=True)
    frozen = output / 'runtime' / 'natlang_neuralese'
    shutil.copytree(package, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    frozen_code = {str(path.relative_to(frozen)): sha(path)
                   for path in sorted(frozen.rglob('*.py'))}
    if frozen_code != actual_code:
        raise ValueError('direct-stage runtime snapshot differs from reviewed package bytes')
    stage = recipe['stage']
    stage_output = output / stage['id']
    stage_output.mkdir()
    handler = HANDLERS[stage['kind']]
    if 'heads' not in inputs:
        raise ValueError('direct stage requires an exact heads input')
    command = [sys.executable, '-m', handler['module'], '--heads', inputs['heads']['path']]
    command += stage_input_args(inputs, stage['kind'])
    command += ['--out', str(stage_output), '--device', device]
    parameters = dict(recipe['parameters'])
    if 'text_data' in inputs:
        parameters.pop('text_data', None)
    command += stage_parameter_args(parameters)
    plan = {'schema': 'natlang.neuralese-direct-stage-plan/1', 'recipe_id': recipe['id'],
            'recipe_path': str(Path(recipe_path).resolve()), 'recipe_sha256': sha(recipe_path),
            'stage': stage, 'inputs': inputs, 'device': device,
            'runtime_image': recipe['runtime']['image'], 'frozen_code': frozen_code,
            'parameters': parameters, 'child_argv': command, 'stage_output': str(stage_output),
            'lineage': recipe['lineage'], 'lineage_evidence': lineage_evidence}
    write_json(output / 'direct-stage-plan.json', plan)
    write_json(metadata_path, {'schema': 'natlang.neuralese-direct-stage-launch-intent/1',
                               'recipe_id': recipe['id'], 'recipe_sha256': plan['recipe_sha256'],
                               'plan_path': str(output / 'direct-stage-plan.json'),
                               'plan_sha256': sha(output / 'direct-stage-plan.json'),
                               'output_path': str(output), 'device': device,
                               'input_sha256': {role: value['sha256'] for role, value in inputs.items()},
                               'lineage_evidence_sha256': {item['role']: item['sha256'] for item in lineage_evidence},
                               'runtime_image': recipe['runtime']['image'],
                               'frozen_code_count': len(frozen_code),
                               'frozen_code_manifest_sha256': hashlib.sha256(
                                   json.dumps(frozen_code, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                               'child_argv': command})
    environment = dict(os.environ)
    environment['PYTHONPATH'] = str(frozen.parent) + os.pathsep + environment.get('PYTHONPATH', '')
    print(json.dumps({'stage': stage['id'], 'command': command}), flush=True)
    with _ChildSignalForwarder() as signal_forwarder:
        result = signal_forwarder.run(command, env=environment)
    stage_report = stage_output / 'report.json'
    report = json.loads(stage_report.read_text()) if stage_report.is_file() else None
    qualified = False
    if result.returncode == 0 and report is not None:
        try:
            require_gate(report, stage['kind'])
            qualified = True
        except ValueError:
            qualified = False
    outputs = {str(path.relative_to(output)): {'bytes': path.stat().st_size, 'sha256': sha(path)}
               for path in sorted(stage_output.rglob('*')) if path.is_file()}
    write_json(output / 'direct-stage-result.json',
               {'schema': 'natlang.neuralese-direct-stage-result/1', 'recipe_id': recipe['id'],
                'stage_id': stage['id'], 'process_exit_code': result.returncode,
                'interrupted': signal_forwarder.stopped[0],
                'completed': result.returncode == 0 and not signal_forwarder.stopped[0], 'qualified': qualified,
                'gate_scope': recipe['lineage'].get('qualification_scope'),
                'report': report, 'outputs': outputs})
    if signal_forwarder.stopped[0]:
        print(json.dumps({'status':'interrupted','stage':stage['id'],
                          'child_exit_code':result.returncode}),flush=True)
    return result.returncode


def require_foundation(certificate, *, heads, checkpoint):
    """Downstream API: validate exact handoff, not an unrelated passed report.

    This validates foundation only. A runtime must separately qualify its own
    transport/gradient replay, then requalify when its backbone changes.
    """
    proof = json.loads(Path(certificate).read_text())
    if proof.get('schema') != 'natlang.neuralese-foundation-certificate/1' or proof.get('qualified') is not True:
        raise ValueError('qualified foundation certificate required')
    if proof['heads_sha256'] != sha(heads) or proof['feedback_checkpoint_sha256'] != sha(checkpoint):
        raise ValueError('foundation certificate belongs to different weights')
    for stage in proof['stages']:
        # Registered snapshots may relocate a recipe directory across machines.
        # Resolve its adjacent reports first; exact hashes still bind all bytes.
        adjacent = Path(certificate).parent / Path(stage['report']).name
        report_path = adjacent if adjacent.is_file() else Path(stage['report'])
        if sha(report_path) != stage['report_sha256']:
            raise ValueError('foundation stage report changed')
        report = json.loads(report_path.read_text())
        require_gate(report['gate'], report['kind'])
    kinds = {stage['kind'] for stage in proof['stages']}
    if not {'token_identity', 'causal_embedding_distillation'} <= kinds:
        raise ValueError('foundation stages missing')
    return proof


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--heads', type=Path,
                        help='required by multi-stage recipes; direct stages pin heads in their input manifest')
    parser.add_argument('--records', type=Path, help='legacy shared fallback; prefer stage inputs in declared recipes')
    parser.add_argument('--pieces', type=Path, help='legacy shared fallback; prefer stage inputs in declared recipes')
    parser.add_argument('--text-data', type=Path, help='legacy shared fallback for text warm-up stages')
    parser.add_argument('--input-binding', action='append', default=[], metavar='NAME=PATH',
                        help='relocate a declared named input while preserving its recipe-pinned SHA-256')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--launch-metadata', type=Path,
                        help='write an immutable launch-intent receipt outside --out before running')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--until', help='run through a declared stage, retaining resumable state')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    if recipe.get('schema') == DIRECT_STAGE_SCHEMA:
        if args.inspect:
            print(json.dumps(recipe, indent=2))
            return 0
        if args.input_binding or args.records or args.pieces or args.text_data or args.until or args.heads:
            parser.error('direct-stage recipes use only their pinned inputs and declared output')
        if args.launch_metadata is None:
            parser.error('direct-stage execution requires --launch-metadata')
        return run_declared_direct_stage(recipe, args.recipe, args.out, args.launch_metadata, args.device)
    if args.heads is None:
        parser.error('multi-stage training recipes require --heads')
    if args.until and args.until not in {stage['id'] for stage in recipe['stages']}:
        parser.error('unknown stopping stage')
    if args.inspect:
        print(json.dumps(recipe, indent=2))
        return
    args.out = args.out.resolve()
    args.heads = args.heads.resolve()
    overrides = parse_input_binding_overrides(args.input_binding)
    defaults = {'records': args.records, 'pieces': args.pieces, 'text_data': args.text_data}
    stage_inputs = {}
    for stage in recipe['stages']:
        stage_defaults = dict(defaults)
        if 'text_data' in stage['parameters']:
            stage_defaults['text_data'] = stage['parameters']['text_data']
        stage_inputs[stage['id']] = resolve_stage_inputs(recipe, stage, stage_defaults, overrides)
    inputs = {str(args.heads): sha(args.heads)}
    for path in (args.records, args.pieces, args.text_data):
        if path is not None:
            resolved = path.resolve()
            inputs[str(resolved)] = sha(resolved)
    for resolved in stage_inputs.values():
        for value in resolved.values():
            inputs[value['path']] = value['sha256']
    package = Path(__file__).parents[1]
    frozen = args.out / 'runtime' / 'natlang_neuralese'
    plan_path = args.out / 'recipe-plan.json'
    plan = {'schema': 'natlang.neuralese-recipe-plan/1', 'recipe': recipe,
            'recipe_sha256': sha(args.recipe), 'inputs': inputs, 'device': args.device}
    if any('inputs' in stage for stage in recipe['stages']):
        plan['stage_inputs'] = stage_inputs
    if plan_path.exists():
        existing = json.loads(plan_path.read_text())
        if any(existing[key] != value for key, value in plan.items()):
            raise ValueError('recipe or input identity changed; use a new stage lineage')
        if {str(path.relative_to(frozen)): sha(path) for path in frozen.rglob('*.py')} != existing['code']:
            raise ValueError('frozen recipe runtime changed')
        plan = existing
    else:
        if args.out.exists():
            raise ValueError('fresh recipe directory required')
        shutil.copytree(package, frozen, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
        plan['code'] = {str(path.relative_to(frozen)): sha(path) for path in frozen.rglob('*.py')}
        write_json(plan_path, plan)
    if args.launch_metadata is not None:
        metadata = args.launch_metadata.resolve()
        if metadata == args.out or args.out in metadata.parents:
            raise ValueError('--launch-metadata must be outside the recipe output directory')
        if metadata.exists():
            raise ValueError('launch metadata path already exists; choose a fresh immutable sidecar')
        write_json(metadata, {'schema': 'natlang.neuralese-recipe-launch-intent/1',
                              'recipe_path': str(args.recipe.resolve()),
                              'recipe_sha256': sha(args.recipe),
                              'output_path': str(args.out),
                              'heads_path': str(args.heads),
                              'heads_sha256': sha(args.heads),
                              'device': args.device,
                              'until': args.until,
                              'recipe_plan_path': str(plan_path),
                              'recipe_plan_sha256': sha(plan_path),
                              'frozen_runtime_sha256': plan.get('code', {}),
                              'stage_inputs': stage_inputs})
    with _ChildSignalForwarder() as signal_forwarder:
        stopped = signal_forwarder.stopped
        reports = []
        feedback_checkpoint = None
        for stage in recipe['stages']:
            if stopped[0]:
                print(json.dumps({'status': 'interrupted_before_child_start', 'stage': stage['id']}), flush=True)
                return
            directory = args.out / stage['id']
            report_path = args.out / (stage['id'] + '-report.json')
            kind = stage['kind']
            if report_path.exists():
                report = json.loads(report_path.read_text())
                require_gate(report['gate'], kind)
                if report['recipe_sha256'] != plan['recipe_sha256'] or report['inputs'] != inputs:
                    raise ValueError('stage report belongs to another recipe/input lineage')
                if sha(report['artifact']) != report['artifact_sha256']:
                    raise ValueError('qualified artifact changed')
            else:
                for dependency in stage['requires']:
                    predecessor = next(report for report in reports if report['id'] == dependency)
                    require_gate(predecessor['gate'], predecessor['kind'])
                output = directory / HANDLERS[kind]['result']
                stage_heads = args.heads
                if kind == 'core_text_warmup' and 'heads' in stage_inputs[stage['id']]:
                    stage_heads = stage_inputs[stage['id']]['heads']['path']
                if kind in ('core_text_warmup','text_warmup_runtime','raw_recurrence_training'):
                    has_exact_stage_heads = 'heads' in stage_inputs[stage['id']]
                    if has_exact_stage_heads:
                        stage_heads = stage_inputs[stage['id']]['heads']['path']
                    elif kind == 'core_text_warmup':
                        predecessor_kind = 'raw_runtime_qualification'
                        predecessor = next(r for r in reversed(reports)
                                           if r['id'] in stage['requires'] and r['kind'] == predecessor_kind)
                        stage_heads = predecessor['artifact']
                    else:
                        predecessor_kind = 'core_text_warmup'
                        predecessor=next(r for r in reversed(reports) if r['id'] in stage['requires'] and r['kind']==predecessor_kind)
                        stage_heads=predecessor['artifact']
                command = [sys.executable, '-m', HANDLERS[kind]['module'], '--heads',
                           str(stage_heads)] + stage_input_args(stage_inputs[stage['id']], kind) + ['--out',
                           str(output if kind == 'token_identity' else directory), '--device', args.device]
                if kind == 'raw_runtime_qualification':
                    command += ['--checkpoint', feedback_checkpoint, '--certificate', str(args.out / 'foundation-certificate.json')]
                if kind == 'verified_heads_handoff':
                    raw = next(r for r in reversed(reports) if r['id'] in stage['requires'] and r['kind']=='raw_runtime_qualification')
                    command += ['--raw-runtime-report', str(Path(raw['artifact']).parent / 'runtime-report.json')]
                stage_parameters = stage['parameters']
                if 'text_data' in stage_inputs[stage['id']]:
                    stage_parameters = {key: value for key, value in stage_parameters.items() if key != 'text_data'}
                command += stage_parameter_args(stage_parameters)
                environment = dict(os.environ)
                environment['PYTHONPATH'] = str(frozen.parent) + os.pathsep + environment.get('PYTHONPATH', '')
                print(json.dumps({'stage': stage['id'], 'command': command}), flush=True)
                result = signal_forwarder.run(command, env=environment)
                code = result.returncode
                if stopped[0]:
                    print(json.dumps({'status': 'interrupted', 'stage': stage['id'],
                                      'child_exit_code': code}), flush=True)
                    return
                if code:
                    raise RuntimeError('stage failed: ' + stage['id'])
                if kind == 'token_identity':
                    gate = json.loads(output.read_text())
                elif kind in ('raw_runtime_qualification','text_warmup_runtime'):
                    gate = json.loads((directory / ('runtime-report.json' if kind=='raw_runtime_qualification' else 'report.json')).read_text())
                elif kind=='verified_heads_handoff':
                    gate=json.loads((directory/'report.json').read_text())
                elif kind=='core_text_warmup':
                    gate=json.loads((directory/'report.json').read_text())
                else:
                    import torch
                    state = torch.load(output, mmap=True, weights_only=False, map_location='cpu')
                    if kind == 'raw_recurrence_training':
                        if state.get('schema') != 'natlang.neuralese_recurrence_checkpoint/1':
                            raise ValueError('wrong recurrence artifact schema')
                        if state['identity']['files'].get(str(Path(stage_heads).resolve())) != sha(stage_heads):
                            raise ValueError('recurrence artifact runtime handoff differs')
                        gate = {'training_stage_completed': state['step'] >= stage['parameters'].get('steps', 500),
                                'step': state['step'], 'errors': state['errors'], 'semantic_channel_qualified': False}
                    else:
                        gate = state['best']
                        if state['identity']['inputs'].get(str(args.heads)) != sha(args.heads):
                            raise ValueError('bootstrap checkpoint backbone identity differs')
                report = {'id': stage['id'], 'kind': kind, 'recipe_sha256': plan['recipe_sha256'],
                          'inputs': inputs, 'gate': gate, 'artifact': str(output), 'artifact_sha256': sha(output)}
                write_json(args.out / (stage['id'] + '-attempt.json'), report)
                require_gate(gate, kind)
                write_json(report_path, report)
            reports.append(report)
            if kind == 'causal_embedding_distillation':
                feedback_checkpoint = report['artifact']
            if kind == 'causal_embedding_distillation':
                certificate = {'schema': 'natlang.neuralese-foundation-certificate/1', 'qualified': True,
                               'runtime_qualified': False, 'heads_sha256': sha(args.heads),
                               'feedback_checkpoint': feedback_checkpoint, 'feedback_checkpoint_sha256': sha(feedback_checkpoint),
                               'recipe_sha256': plan['recipe_sha256'],
                               'stages': [{'kind': report['kind'], 'report': str(args.out / (report['id'] + '-report.json')),
                                           'report_sha256': sha(args.out / (report['id'] + '-report.json'))} for report in reports if report['kind'] in {'token_identity', 'causal_embedding_distillation'}]}
                write_json(args.out / 'foundation-certificate.json', certificate)
            if args.until == stage['id']:
                return
    print(json.dumps({'status': 'recipe_completed', 'foundation_runtime_qualified': any(r['kind'] == 'raw_runtime_qualification' for r in reports)}), flush=True)


if __name__ == '__main__':
    main()
