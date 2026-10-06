"""Declared, gated training stages with frozen code and explicit artifact handoff.

Stage implementations are shared Python modules. Extend HANDLERS with a typed
handler and declare its dependencies in a JSON recipe. No shell command strings.
An unsuccessful stage cannot advance its dependents or issue a certificate.
"""
import argparse
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
    'core_text_warmup': {'module':'natlang_neuralese.train.text_warmup',
                        'parameters':{'text_data','student_checkpoint','steps','tokens','prefix_tokens','cutoff','group_size',
                          'backbone_training','rank','optimizer','lr','sketch_lr','embedding_weight','sketch_weight','text_weight',
                          'aligned_steps','ramp_steps','checkpoint_every','eval_every','held_documents','seed','checkpoint_layers',
                          'max_ce_delta','max_relative_mse','min_agreement','consecutive_gates'},'result':'heads.pt'},
    'text_warmup_runtime': {'module':'natlang_neuralese.eval.text_warmup_runtime',
                            'parameters':set(),'result':'report.json'},
    'raw_recurrence_training': {'module': 'natlang_neuralese.train.trajectories',
                                'parameters': {'steps', 'batch', 'lr', 'rank', 'lora_lr', 'max_tokens',
                                               'train', 'eval', 'handover', 'write_curriculum', 'max_writes',
                                               'write_depth', 'tokens_per_vector', 'heads_lr', 'distill',
                                               'crisp_weight', 'memory_gb', 'backward_policy', 'graph_memory_gb',
                                               'graph_headroom_gb', 'checkpoint_layers', 'checkpoint_attention_only',
                                               'staged_checkpoint_attention_only', 'checkpoint_elide_rng', 'producer_batch_size',
                                               'producer_batch_memory_gb', 'ffn_chunk_tokens',
                                               'optimizer', 'checkpoint_every', 'eval_every', 'seed', 'writer_text_weight',
                                               'max_write_vectors', 'content_transport', 'writer_length_policy', 'writer_supervision', 'stop_supervision',
                                               'sketch_gradient', 'sketch_target_weight', 'sketch_target_backbone_scale',
                                               'local_stage_batch_size', 'train_control_rows', 'token_cache_mib'},
                                'result': 'checkpoint.pt'},
    'raw_runtime_qualification': {'module': 'natlang_neuralese.eval.raw_port_handoff',
                                  'parameters': {'limit', 'max_length'}, 'result': 'heads.pt'},
    'token_identity': {'module': 'natlang_neuralese.eval.foundation',
                       'parameters': {'limit', 'max_tokens'}, 'result': 'identity.json'},
    'causal_embedding_distillation': {'module': 'natlang_neuralese.train.causal_bootstrap',
                                     'parameters': {'cutoff', 'steps', 'batch', 'lr', 'tokens', 'contexts',
                                                    'context_tokens', 'eval_every', 'checkpoint_every',
                                                    'seed', 'agreement_gate', 'kl_gate', 'source_fraction',
                                                    'argmax_weight', 'continue_from', 'stop_on_gate'}, 'result': 'best-checkpoint.pt'},
}


def load_recipe(path):
    recipe = json.loads(Path(path).read_text())
    if recipe.get('schema') != 'natlang.neuralese-training-recipe/1' or not recipe.get('stages'):
        raise ValueError('invalid or empty training recipe')
    declared, complete, identity_stages, embedding_stages, runtime_stages = set(), set(), set(), set(), set()
    warmup_stages, warmed_runtime_stages=set(),set()
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
        if kind=='core_text_warmup':
            if not set(required)&runtime_stages:raise ValueError('text warm-up requires qualified raw foundation/runtime')
            warmup_stages.add(name)
        if kind=='text_warmup_runtime':
            if not set(required)&warmup_stages:raise ValueError('adapted runtime requires qualified text warm-up')
            warmed_runtime_stages.add(name)
        if kind == 'raw_recurrence_training' and not (set(required)&warmed_runtime_stages and set(required)&warmup_stages):
            raise ValueError('Natlang trajectories require mandatory text warm-up and adapted runtime qualification')
        declared.add(name)
        complete.add(name)
    if not identity_stages or not any(s['kind'] == 'causal_embedding_distillation' for s in recipe['stages']):
        raise ValueError('neuralese recipe must declare identity and embedding distillation stages')
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
    elif kind == 'raw_recurrence_training':
        if report.get('training_stage_completed') is not True or report.get('errors') != 0:
            raise ValueError('raw recurrence training stage incomplete or errored')
    elif kind=='core_text_warmup':
        if report.get('qualified') is not True:raise ValueError('mandatory text warm-up alignment gate failed')
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
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--pieces', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--until', help='run through a declared stage, retaining resumable state')
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args(argv)
    recipe = load_recipe(args.recipe)
    if args.until and args.until not in {stage['id'] for stage in recipe['stages']}:
        parser.error('unknown stopping stage')
    if args.inspect:
        print(json.dumps(recipe, indent=2))
        return
    args.out = args.out.resolve()
    args.heads, args.records, args.pieces = (path.resolve() for path in (args.heads, args.records, args.pieces))
    inputs = {str(path): sha(path) for path in (args.heads, args.records, args.pieces)}
    package = Path(__file__).parents[1]
    frozen = args.out / 'runtime' / 'natlang_neuralese'
    plan_path = args.out / 'recipe-plan.json'
    plan = {'schema': 'natlang.neuralese-recipe-plan/1', 'recipe': recipe,
            'recipe_sha256': sha(args.recipe), 'inputs': inputs, 'device': args.device}
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
    stopped = [False]
    child = [None]
    def interrupt(*_):
        stopped[0] = True
        if child[0] is not None:
            child[0].send_signal(signal.SIGTERM)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, interrupt)
    reports = []
    feedback_checkpoint = None
    for stage in recipe['stages']:
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
            if kind in ('core_text_warmup','text_warmup_runtime','raw_recurrence_training'):
                predecessor_kind='raw_runtime_qualification' if kind=='core_text_warmup' else 'core_text_warmup'
                predecessor=next(r for r in reversed(reports) if r['id'] in stage['requires'] and r['kind']==predecessor_kind)
                stage_heads=predecessor['artifact']
            command = [sys.executable, '-m', HANDLERS[kind]['module'], '--heads',
                       str(stage_heads), '--records', str(args.records), '--out',
                       str(output if kind == 'token_identity' else directory), '--device', args.device]
            if kind in {'causal_embedding_distillation', 'raw_recurrence_training','core_text_warmup'}:
                command += ['--pieces', str(args.pieces)]
            if kind == 'raw_runtime_qualification':
                command += ['--checkpoint', feedback_checkpoint, '--certificate', str(args.out / 'foundation-certificate.json')]
            if kind=='raw_recurrence_training':
                adapted_runtime=next(r for r in reversed(reports) if r['id'] in stage['requires'] and r['kind']=='text_warmup_runtime')
                command += ['--warmup-runtime-report',adapted_runtime['artifact']]
            for key, value in stage['parameters'].items():
                if isinstance(value, bool):
                    command += ['--' + ('' if value else 'no-') + key.replace('_', '-')]
                else:
                    command += ['--' + key.replace('_', '-'), str(value)]
            environment = dict(os.environ)
            environment['PYTHONPATH'] = str(frozen.parent) + os.pathsep + environment.get('PYTHONPATH', '')
            print(json.dumps({'stage': stage['id'], 'command': command}), flush=True)
            child[0] = subprocess.Popen(command, env=environment)
            code = child[0].wait()
            child[0] = None
            if stopped[0]:
                print(json.dumps({'status': 'checkpointed_on_signal', 'stage': stage['id']}), flush=True)
                return
            if code:
                raise RuntimeError('stage failed: ' + stage['id'])
            if kind == 'token_identity':
                gate = json.loads(output.read_text())
            elif kind in ('raw_runtime_qualification','text_warmup_runtime'):
                gate = json.loads((directory / ('runtime-report.json' if kind=='raw_runtime_qualification' else 'report.json')).read_text())
            elif kind=='core_text_warmup':
                from .warmup_admission import require_text_warmup
                gate=require_text_warmup(output)
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
