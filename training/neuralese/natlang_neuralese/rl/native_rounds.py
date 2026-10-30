"""Sealed native RL rounds: provider command contract, bridge results and the raw bootstrap entry.

Commands in the existing per-model controller may run locally or through the
qualified sandbox execution bridge. No cloud resources are allocated here.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ..common.jsonio import write_json_durable
from .production import run, run_command, RunControl


def configure_provider(config,declaration,plan,binding,config_path,*,write=True):
    """One exact provider/actor command contract for the raw bootstrap."""
    overrides = declaration.get('commands') or {}
    if set(overrides) - {'actor_command', 'learner_command', 'seed_command', 'round_command',
                         'before_collect', 'after_actor_ready', 'before_update', 'after_update'}:
        raise ValueError('native RL bootstrap has an unsupported provider command hook')
    for name, argv in overrides.items():
        if not isinstance(argv, list) or not argv or any(not isinstance(arg, str) for arg in argv):
            raise ValueError('native RL provider bridge commands must be explicit nonempty argv lists')
        config[name] = list(argv)
    if declaration.get('actor_runtime'):
        from .production import actor_runtime_configuration,bind_actor_runtime_command
        repository=Path(declaration['actor_runtime']).resolve(strict=True)
        config.update(actor_runtime_configuration(repository))
        if config.get('actor_command') is not None:
            config['actor_command']=bind_actor_runtime_command(config['actor_command'],repository)
    if declaration.get('provider'):
        config['provider'] = declaration['provider']
    if declaration.get('provider_queue'):
        import sys
        config['round_command'] = [sys.executable, '-m', 'natlang_neuralese.rl.native_bridge',
            '--queue', declaration['provider_queue'], '--request', '{request}',
            '--result', '{result}', '--stop-file', '{stop_file}']
    config['bootstrap_binding'] = binding
    config['source_admission'] = declaration.get('source_admission')
    if declaration.get('task_source'):
        config['task_source'] = declaration['task_source']
    if 'native_checkpoint_control_directory' in declaration:
        directory = declaration['native_checkpoint_control_directory']
        if (not isinstance(directory, str) or not Path(directory).is_absolute()
                or '..' in Path(directory).parts or str(Path(directory).resolve()) != directory):
            raise ValueError('checkpoint control publication needs an explicit canonical directory')
        config['native_checkpoint_control_directory'] = directory
    if 'native_checkpoint_publication' in declaration:
        from ..model.native_checkpoint_control import validate_publication_selection,publication_trainer_parameters
        selection=validate_publication_selection(declaration['native_checkpoint_publication'])
        config['native_checkpoint_publication']=selection
        for key in ('trainer_parameters','seed_trainer_parameters'):
            if key=='trainer_parameters' or key in config:
                config.setdefault(key,{}).update(publication_trainer_parameters(selection))
    from ..model.native_checkpoint_control import publication_owner_configuration
    config.update(publication_owner_configuration(declaration, model=config.get('model')or config.get('native_participant')or declaration.get('model')))
    if write:write_json_durable(config_path, config)
    return config

def bridge_result(command, request, stop):
    from .native_bridge import binding, verify_result, queue_client_command
    from .production import RunControl
    request = Path(request)
    result = request.with_suffix('.result.json')
    stop_file = request.with_suffix('.stop.json')
    payload = json.loads(request.read_text())
    values = {'request': str(request), 'result': str(result), 'stop_file': str(stop_file),
              'out': payload['out'], 'native_participant': payload.get('native_participant', ''),
              '_run_control': RunControl(stop), '_bridge_stop_file': stop_file,
              '_bridge_request': binding(request)}
    # Only this known queue client is followed immediately by our complete
    # verifier. Avoid decoding the same full checkpoint in both.
    command = queue_client_command(command)
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
        return run(config_path, rounds=1, stop=stop)
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-bootstrap', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--rounds', type=int)
    args = parser.parse_args(argv)
    from .native_raw_rounds import run as run_raw
    return run_raw(args.raw_bootstrap, args.out, cycles=args.rounds)


if __name__ == '__main__':
    main()
