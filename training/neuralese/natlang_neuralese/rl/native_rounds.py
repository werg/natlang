"""Parallel sealed native RL rounds followed by one joint semantic refresh.

Learners update different participants of ONE immutable parent concurrently.
This is synchronous policy-lag-zero execution; no stale/off-policy KV reuse.
Commands in the existing per-model controller may run locally or through the
qualified sandbox execution bridge. No cloud resources are allocated here.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path

from ..common.hashing import sha256_file_hex_cached
from ..common.jsonio import write_json_durable
from .production import run, run_command, RunControl


def bootstrap_configs(plan_path, completion_path, out):
    """Bind actual completed shared weights and admitted inputs, then configure both learners."""
    from types import SimpleNamespace
    from ..train.native_readiness import _bound, validate_completion
    from .production import configure_native
    from ..common.hashing import canonical_json_sha256_hex
    plan_path, completion_path, out = Path(plan_path), Path(completion_path), Path(out)
    plan = json.loads(plan_path.read_text())
    if plan.get('schema') != 'natlang.native-rl-bootstrap/1' or len(plan.get('models') or []) < 2:
        raise ValueError('native RL bootstrap requires a typed plan for at least two student models')
    proof = json.loads(completion_path.read_text())
    adapter_path = _bound(proof['adapter'])
    adapter = json.loads(adapter_path.read_text())
    key = (proof.get('participants_signature'), proof.get('curriculum_signature'))
    validate_completion(completion_path, {key: {'execution_signature': 'bootstrap-completion-verifier',
                                              'models': list(adapter['models'])}})
    binding = {'plan_sha256': sha256_file_hex_cached(plan_path),
               'completion_sha256': sha256_file_hex_cached(completion_path),
               'adapter_sha256': sha256_file_hex_cached(adapter_path)}
    (out/'configs').mkdir(parents=True, exist_ok=True)
    receipt = out/'bootstrap.json'
    reuse_configs = False
    if receipt.exists():
        prior = json.loads(receipt.read_text())
        if prior['binding'] == binding and all(Path(path).is_file() for path in prior['configs']):
            reuse_configs = True  # retain useful live operator tweaks after artifact verification
    refresh_config = _bound(plan.get('semantic_refresh_config') or proof['curriculum'])
    task_sources = [declaration.get('task_source') for declaration in plan['models']]
    if any(task_sources):
        if any(source is None for source in task_sources) or any(source != task_sources[0] for source in task_sources[1:]):
            raise ValueError('both native RL learners must bind the same admitted task-source manifest')
        from .task_source import resolve_task_source
        source = task_sources[0]
        resolved_source = resolve_task_source(source['manifest'])
        if (resolved_source['manifest_sha256'] != source['manifest_sha256']
                or resolved_source['cases_sha256'] != source['cases_sha256']
                or resolved_source['rows'] != source['rows']
                or resolved_source['audit_sha256'] != source['audit_sha256']
                or resolved_source['selection_proof_sha256'] != source['selection_proof_sha256']):
            raise ValueError('native RL bootstrap task-source manifest binding changed')
    results, participants = [], set()
    for declaration in plan['models']:
        model = declaration['model']
        participant = declaration.get('native_participant')
        if participant is None:
            architectures = {'lfm': {'lfm2', 'lfm2_moe'}, 'mellum': {'mellum', 'maple'}}[model]
            candidates = [name for name, entry in adapter['models'].items()
                          if entry['native_identity']['architecture'] in architectures]
            if len(candidates) != 1:
                raise ValueError('native RL bootstrap must name its unambiguous actual shared participant')
            participant = candidates[0]
        if participant in participants or participant not in adapter['models']:
            raise ValueError('native RL bootstrap participants must be distinct actual exported models')
        participants.add(participant)
        inputs = {name: _bound(declaration[name]) for name in
                  ('cases', 'production_recipe', 'seed_native_graphs', 'held_native_graphs')}
        if task_sources[0] is not None and (inputs['cases'] != resolved_source['cases']
                or sha256_file_hex_cached(inputs['cases']) != resolved_source['cases_sha256']):
            raise ValueError('native RL cases differ from the exact shared task-source manifest')
        from ..common.jsonio import iter_jsonl_rows
        cases = [row for _, row in iter_jsonl_rows(inputs['cases'])]
        if not cases or any(row.get('split') != 'train' for row in cases):
            raise ValueError('native RL bootstrap requires admitted nonempty train-only source cases')
        runtime = Path(declaration['runtime']).resolve(strict=True)
        if not (runtime/'dist/teacher/cli.js').is_file():
            raise ValueError('native RL bootstrap collector runtime is not built')
        config_path = out/'configs'/(model + '.json')
        if reuse_configs:
            results.append(config_path)
            continue
        args = SimpleNamespace(**inputs, model=model, native_participant=participant,
            semantic_adapter=adapter_path, semantic_refresh_config=refresh_config,
            checkpoint=Path(adapter['checkpoint']['path']), base=Path(declaration['base']),
            production_stage=declaration['production_stage'], runtime=runtime,
            out=out/model, config=config_path, port=declaration['port'],
            coefficient_policy=_bound(declaration['coefficient_policy']) if declaration.get('coefficient_policy') else None,
            coefficient_optimizer=declaration.get('coefficient_optimizer'),
            coefficient_task_eval=declaration.get('coefficient_task_eval'))
        config = configure_native(args)
        overrides = declaration.get('commands') or {}
        if set(overrides) - {'actor_command', 'learner_command', 'seed_command', 'round_command',
                             'refresh_command', 'before_collect', 'before_update', 'after_update'}:
            raise ValueError('native RL bootstrap has an unsupported provider command hook')
        for name, argv in overrides.items():
            if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
                raise ValueError('native RL provider bridge commands must be explicit nonempty argv lists')
            config[name] = argv
        if plan.get('refresh_command'):
            argv = plan['refresh_command']
            if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
                raise ValueError('native RL refresh provider must be an explicit argv list')
            config['refresh_command'] = argv
        if declaration.get('provider'):
            config['provider'] = declaration['provider']
        if declaration.get('provider_queue'):
            import sys
            config['round_command'] = [sys.executable, '-m', 'natlang_neuralese.rl.native_bridge',
                '--queue', declaration['provider_queue'], '--request', '{request}',
                '--result', '{result}', '--stop-file', '{stop_file}']
        if plan.get('refresh_provider'):
            config['refresh_provider'] = plan['refresh_provider']
        config['bootstrap_binding'] = binding
        config['source_admission'] = declaration.get('source_admission')
        if declaration.get('task_source'):
            config['task_source'] = declaration['task_source']
        write_json_durable(config_path, config)
        results.append(config_path)
    write_json_durable(receipt, {'schema': 'natlang.native-rl-bootstrap-receipt/1',
        'binding': binding, 'configs': [str(path.resolve()) for path in results],
        'participants': sorted(participants), 'provider_commands_executed': False})
    return results


def bridge_result(command, request, stop):
    from .native_bridge import binding, verify_result
    from .production import RunControl
    request = Path(request)
    result = request.with_suffix('.result.json')
    stop_file = request.with_suffix('.stop.json')
    payload = json.loads(request.read_text())
    values = {'request': str(request), 'result': str(result), 'stop_file': str(stop_file),
              'out': payload['out'], 'native_participant': payload.get('native_participant', ''),
              '_run_control': RunControl(stop), '_bridge_stop_file': stop_file,
              '_bridge_request': binding(request)}
    run_command(command, values)
    if not result.is_file():
        raise RuntimeError('provider bridge returned without draining its exact remote worker/result')
    value = verify_result(request, result)
    if value['status'] == 'failed':
        raise RuntimeError('provider native worker failed: ' + value.get('error', 'see durable result'))
    if value['status'] == 'stopped' and not stop.requested:
        raise RuntimeError('provider native worker stopped unexpectedly; resume its preserved state')
    return value


def round_job(config_path, config, directory, stop):
    if not config.get('round_command'):
        return run(config_path, rounds=1, defer_native_refresh=True, stop=stop)
    if stop.requested:
        return {'phase': 'requested_stop'}
    from .native_bridge import write_request
    from ..train.native_readiness import _bound
    state = Path(config['out'])/'state.json'
    request = write_request(directory, kind='round', out=config['out'], config=config_path,
                            state=state if state.exists() else None)
    result = bridge_result(config['round_command'], request, stop)
    if result.get('state'):
        return json.loads(_bound(result['state']).read_text())
    if stop.requested:
        return {'phase': 'requested_stop'}
    raise RuntimeError('remote sealed round returned no exact durable state')


def refresh_job(configs, adapter, updates, semantic_config, directory, device, stop):
    from .native_refresh import refresh
    commands = [config['refresh_command'] for config in configs if config.get('refresh_command')]
    if not commands:
        return refresh(adapter=adapter, updates=updates, config=semantic_config,
                       out=directory/'semantic-refresh', device=device, stop=stop)
    if any(command != commands[0] for command in commands):
        raise ValueError('native learners declared different joint refresh providers')
    from .native_bridge import write_request
    from ..train.native_readiness import _bound
    request = write_request(directory/'bridge-refresh', kind='refresh',
        out=directory/'semantic-refresh', adapter=adapter, updates=updates,
        semantic_config=semantic_config, device=device, provider=configs[0].get('refresh_provider'))
    result = bridge_result(commands[0], request, stop)
    if result.get('deployment'):
        return json.loads(_bound(result['deployment']).read_text())
    if stop.requested:
        raise InterruptedError('remote semantic refresh checkpointed; no actor publication')
    raise RuntimeError('remote semantic refresh returned no exact deployment')


def apply_deployment(config, deployment):
    """Idempotent publication bookkeeping, even after a crash between actors."""
    out = Path(config['out']); out.mkdir(parents=True, exist_ok=True)
    path = out/'state.json'
    with (out/'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from .production import initial_controller_state
        state = json.loads(path.read_text()) if path.exists() else initial_controller_state(config)
        name = config['native_participant']
        from .native_semantic_route import accept_policy_deployment
        accept_policy_deployment(config,state,deployment)
        pending = state.get('pending_native_checkpoint')
        if pending:
            accepted = deployment.get('updated_participants', {}).get(name)
            if accepted and accepted['sha256'] == sha256_file_hex_cached(pending):
                state.update(round=state['round'] + 1, trained_rounds=state['trained_rounds'] + 1,
                             collection_attempt=0, phase='collect')
                state.pop('pending_native_checkpoint', None)
                state.pop('pending_native_round_dir', None)
            else:
                # Pending learner belongs to the current sealed parent and
                # must still join the refresh, not be replaced by another actor.
                if state['checkpoint'] != deployment['checkpoint']['path']:
                    raise ValueError('pending native RL update was not merged into the published joint checkpoint')
                return
        if state.get('checkpoint') != deployment['checkpoint']['path']:
            state['group_stats'] = []
        state.update(checkpoint=deployment['checkpoint']['path'],
                     semantic_adapter=deployment['semantic_adapter']['path'])
        write_json_durable(path, state)


def _run_rounds(config_paths, out, *, rounds=None, device='cuda', stop):
    from .native_refresh import refresh
    from ..train.recipe import resolve_recipe_data
    out = Path(out).resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out/'joint-controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        latest = out/'latest.json'
        completed = 0
        deployment=None
        coefficient_updates={}
        while rounds is None or completed < rounds:
            if stop.requested:
                return {'phase': 'requested_stop', 'production_complete': False}
            configs = [resolve_recipe_data(path) for path in config_paths]
            names = [config.get('native_participant') for config in configs]
            if (len(configs) < 2 or len(set(names)) != len(names)
                    or not all(config.get('native_state') for config in configs)):
                raise ValueError('joint RL requires distinct native participant configurations')
            if latest.exists():
                transaction = json.loads(latest.read_text())
                deployment, cycle = transaction['deployment'], transaction['next_round']
                for config in configs:
                    apply_deployment(config, deployment)
                adapter = deployment['semantic_adapter']['path']
            else:
                cycle = 0
                adapters = {str(Path(config['semantic_adapter']).resolve()) for config in configs}
                parents = {sha256_file_hex_cached(config['checkpoint']) for config in configs}
                if len(adapters) != 1 or len(parents) != 1:
                    raise ValueError('parallel native learners must start from the same immutable joint parent')
                adapter = next(iter(adapters))
            coefficient_latest=out/'coefficient-latest.json'
            if coefficient_latest.exists():
                coefficient_cycle=json.loads(coefficient_latest.read_text())
                if coefficient_cycle.get('schema')!='natlang.native-coefficient-rounds/1':
                    raise ValueError('coefficient joint cycle receipt changed its actual schema')
                cycle=max(cycle,coefficient_cycle['next_round'])
            # Real production configs declare the full recipe. Minimal direct
            # round fixtures may omit it; they exercise only the sealed RL job.
            if any(config.get('production_recipe') for config in configs):
                if not all(config.get('production_recipe') for config in configs):
                    raise ValueError('all native learners must declare their mandatory production recipe')
                pending_rl=any((Path(config['out'])/'state.json').exists() and
                    (lambda state:state.get('pending_native_checkpoint') or state.get('pending_semantic_source'))(
                        json.loads((Path(config['out'])/'state.json').read_text()))
                    for config in configs)
                if not pending_rl:
                    from .native_supervised import interleave
                    adapter=interleave(config_paths,configs,adapter,out/f'round-{cycle:06d}'/'mandatory-sft',
                        cycle=cycle,latest=latest,device=device,stop=stop)
            semantic_configs = {str(Path(config['semantic_refresh_config']).resolve()) for config in configs}
            if len(semantic_configs) != 1:
                raise ValueError('both learners must share one actual semantic refresh curriculum')
            # Each existing controller durably seals its packet and finishes
            # its complete shared-trainer epoch before returning its update.
            with ThreadPoolExecutor(max_workers=len(configs)) as pool:
                futures = [pool.submit(round_job, path, config, out/f'round-{cycle:06d}'/'bridge-rounds'/name, stop)
                           for path, config, name in zip(config_paths, configs, names)]
                states, failures = [], []
                for future in futures:
                    try:
                        states.append(future.result())
                    except Exception as error:
                        failures.append(error)
                        states.append(None)
            if failures:
                raise failures[0]  # successful sibling checkpoints remain resumable/pending
            if stop.requested:
                return {'phase': 'requested_stop', 'states': states, 'production_complete': False}
            updates = {name: state['pending_native_checkpoint'] for name, state in zip(names, states)
                       if state.get('pending_native_checkpoint')}
            coefficient_updates={name:{'policy':{'path':state['coefficient_policy'],
                'sha256':sha256_file_hex_cached(state['coefficient_policy'])},'trained_rounds':state['trained_rounds']}
                for config,name,state in zip(configs,names,states)
                if config.get('coefficient_policy') and state.get('phase')=='collect'
                and (state.get('last_learner_progress') or {}).get('schema')=='natlang.native-coefficient-training-report/1'
                and state['last_learner_progress'].get('qualification_complete') is True}
            policy_sources=[state['pending_semantic_source'] for state in states if state.get('pending_semantic_source')]
            semantic_pending=(json.loads(Path(adapter).read_text()).get('semantic_obligations') or {}).get('complete') is False
            semantic_pending=semantic_pending or bool(policy_sources)
            if not updates and not semantic_pending and coefficient_updates:
                write_json_durable(coefficient_latest,{'schema':'natlang.native-coefficient-rounds/1',
                    'next_round':cycle+1,'policies':coefficient_updates,
                    'base_checkpoint_updated':False,'semantic_refresh_executed':False,
                    'unchanged_semantic_adapter':{'path':str(Path(adapter).resolve()),'sha256':sha256_file_hex_cached(adapter)}})
                completed+=1
                continue
            if not updates and not semantic_pending and rounds is None:
                import time
                for _ in range(300):
                    if stop.requested:
                        return {'phase': 'requested_stop', 'states': states, 'production_complete': False}
                    time.sleep(.2)
                continue
            if not updates and not semantic_pending:
                return {'phase': 'await_useful_rollouts_or_data', 'states': states, 'production_complete': False}
            directory = out/f'round-{cycle:06d}'
            semantic_config=next(iter(semantic_configs))
            if policy_sources:
                from .native_semantic_route import policy_curriculum
                semantic_config=policy_curriculum(semantic_config,policy_sources,directory/'policy-curriculum.json')
            deployment = refresh_job(configs, adapter, updates, semantic_config, directory, device, stop)
            # All external actors receive the whole joint publication, not
            # only the model which happened to finish its learner first.
            for config in configs:
                if config.get('after_update'):
                    run_command(config['after_update'], {**config, '_run_control': RunControl(stop),
                        'checkpoint': deployment['checkpoint']['path'],
                        'policy_id': deployment['checkpoint']['sha256'],
                        'semantic_adapter': deployment['semantic_adapter']['path'],
                        'semantic_adapter_sha256': deployment['semantic_adapter']['sha256'],
                        'deployment': str(directory/'semantic-refresh/deployment.json')})
            write_json_durable(latest, {'schema': 'natlang.native-rl-joint-controller/1',
                                      'deployment': deployment, 'next_round': cycle + 1})
            for config in configs:
                apply_deployment(config, deployment)
            completed += 1
        return {'phase': 'collect', 'deployment': deployment,
                **({'coefficient_policies':coefficient_updates} if coefficient_updates else {}),'production_complete': False}


def run_rounds(config_paths, out, *, rounds=None, device='cuda', stop=None):
    from ..train.loop import StopSignal
    owned = stop is None
    if owned:
        stop = StopSignal().install()
    try:
        return _run_rounds(config_paths, out, rounds=rounds, device=device, stop=stop)
    except InterruptedError:
        if not stop.requested:
            raise
        return {'phase': 'requested_stop', 'production_complete': False}
    finally:
        if owned:
            stop.restore()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--config', action='append', type=Path)
    source.add_argument('--bootstrap', type=Path)
    parser.add_argument('--shared-completion', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    if args.bootstrap and not args.shared_completion:
        parser.error('--bootstrap requires actual --shared-completion')
    configs = bootstrap_configs(args.bootstrap, args.shared_completion, args.out) if args.bootstrap else args.config
    return run_rounds(configs, args.out, rounds=args.rounds, device=args.device)


if __name__ == '__main__':
    main()
