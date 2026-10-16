"""Adaptive collect→verify→GRPO/RFT→continue driver for both production models.

One controller per model; endpoint replicas may live on different sandboxes.
Model/optimizer processes are launched by explicit commands through the existing
host/cloud supervisor. The controller never silently changes provider resources.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

from ..common.jsonio import read_jsonl, write_json_durable
from ..common.paths import root
from .task_rollouts import prepare, score


def validate_actor_receipts(receipts, *, policy_id, neuralese_server, expected_native=None, expected_coefficient=None):
    """Attest native and legacy actors explicitly, without projected-head aliases."""
    import re
    if not receipts or any(not receipt or receipt.get('checkpoint_sha256') != policy_id
                           for receipt in receipts.values()):
        raise ValueError('RL actor did not attest the requested immutable checkpoint')
    modes = set()
    for receipt in receipts.values():
        mode = receipt.get('communication', 'legacy-projected' if receipt.get('neuralese') else 'text')
        modes.add(mode)
        if mode == 'native-state':
            identity = receipt.get('native_identity') or {}
            if (receipt.get('schema') != 'natlang.rl-policy-binding/2'
                    or identity.get('checkpoint_sha256') != policy_id
                    or any(not isinstance(identity.get(key), str) or not re.fullmatch(r'[0-9a-f]{64}', identity[key])
                           for key in ('checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256'))
                    or not isinstance(identity.get('architecture'), str) or not identity['architecture']):
                raise ValueError('native RL actor lacks exact checkpoint/tokenizer/embedding provenance')
        elif neuralese_server and mode != 'legacy-projected':
            raise ValueError('RL actor did not restore the requested Neuralese communication interface')
    if len(modes) != 1:
        raise ValueError('RL replicas disagree on their communication interface')
    if 'native-state' in modes:
        identities = [receipt['native_identity'] for receipt in receipts.values()]
        if any(identity != identities[0] for identity in identities[1:]):
            raise ValueError('native RL replicas disagree on their exact policy identity')
    if expected_native is not None:
        if modes != {'native-state'} or any(receipt.get('native_identity') != expected_native['native_identity'] or
                receipt.get('native_participant') != expected_native['native_participant'] or
                receipt.get('semantic_adapter_sha256') != expected_native['semantic_adapter_sha256']
                for receipt in receipts.values()):
            raise ValueError('native RL actor differs from the requested shared participant or semantic adapter')
    if expected_coefficient is not None and any(receipt.get('coefficient_policy')!=expected_coefficient
            or receipt.get('coefficient_policy_sha256')!=expected_coefficient['policy_sha256'] for receipt in receipts.values()):
        raise ValueError('native RL actor differs from its exact declared coefficient policy')
    return receipts


class RunControl:
    """Share one stop request across explicit child processes and thread workers."""
    def __init__(self, stop):
        self.stop = stop

    def wait(self, child, *, stop_file=None, stop_request=None):
        forwarded = False
        while True:
            if self.stop.requested and not forwarded:
                if stop_file is not None:
                    write_json_durable(stop_file, {'schema': 'natlang.native-rl-bridge-stop/1',
                        'checkpoint_requested': True, 'request': stop_request})
                elif child.poll() is None:
                    child.terminate()  # exact child PID, trainer checkpoints before exiting
                forwarded = True
            try:
                return child.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                pass


def collect(*, cases, out, endpoints, policy_id, rows, offset=0, group=4, workers=4,
            root_mode='student', runtime=None, seed_base=700000, neuralese_server=True,
            case_seconds=600, max_model_requests=128, max_turns=160, control=None, expected_native=None, coefficient_policy=None):
    """Bounded subprocess fanout; existing collector and exact task verifier."""
    if control is not None and control.stop.requested:
        raise InterruptedError('collection stop requested')
    if not endpoints or workers < 1:
        raise ValueError('RL needs served endpoints and positive worker concurrency')
    from ..common.hashing import sha256_file_hex
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    runtime = Path(runtime or root('repo') / 'ts-host')
    binding = {'schema': 'natlang.rl-collection-binding/1',
        'cases_sha256': sha256_file_hex(Path(cases)), 'policy_id': policy_id,
        'rows': rows, 'offset': offset, 'group': group, 'root_mode': root_mode,
        'seed_base': seed_base, 'temperature': 1., 'guided': False,
        'case_seconds': case_seconds, 'max_model_requests': max_model_requests, 'max_turns': max_turns,
        'endpoints': endpoints, 'neuralese_server': neuralese_server,
        'runtime': {str(p.relative_to(runtime)): sha256_file_hex(p)
                    for p in sorted((runtime / 'dist').rglob('*.js'))},
        'collector_sha256': sha256_file_hex(root('repo') / 'scripts/run_bonsai_queue.py'),
        **({'expected_native_actor': expected_native} if expected_native is not None else {})}
    coefficient_runtime=None
    if coefficient_policy is not None:
        from ..model.native_coefficient_policy import runtime_binding
        coefficient_policy=Path(coefficient_policy).resolve(strict=True)
        coefficient_runtime=runtime_binding(json.loads(coefficient_policy.read_text()),coefficient_policy)
        binding['coefficient_policy']={'path':str(coefficient_policy),'sha256':sha256_file_hex(coefficient_policy),'runtime':coefficient_runtime}
    receipt = out / 'collection-binding.json'
    if receipt.exists():
        if json.loads(receipt.read_text()) != binding:
            raise ValueError('RL collection resume policy/cases/sampling/runtime changed')
    elif (out / 'score.json').exists():
        raise ValueError('completed RL collection has no provenance binding')
    else:
        write_json_durable(receipt, binding)
    if (out / 'score.json').exists():
        saved = json.loads((out / 'score.json').read_text())
        if saved.get('collection_complete'):
            return saved
    from .grpo import Server
    def actor_receipts():
        receipts = {url: json.loads(Server(url).get_bytes('/v1/neuralese/info'))
                    .get('rl_policy_binding') for url in endpoints}
        return validate_actor_receipts(receipts, policy_id=policy_id, neuralese_server=neuralese_server, expected_native=expected_native,expected_coefficient=coefficient_runtime)
    actors = actor_receipts()
    preparation = prepare(SimpleNamespace(cases=Path(cases), out=out, rows=rows, offset=offset,
        group=group, temperature=1., case_seconds=case_seconds,
        max_model_requests=max_model_requests, max_turns=max_turns,
        seed_base=seed_base, root_mode=root_mode, match_seeds_from=None,
        policy_checkpoint_sha256=policy_id))
    entries = list(read_jsonl(out / 'queue.jsonl'))
    directory = out / 'workers'
    directory.mkdir(exist_ok=True)
    config = out / 'sampling-config.json'
    write_json_durable(config, {'guidance': False, 'x_natlang_rl_capture': True,
        **({'x_natlang_native_coefficient_policy':coefficient_runtime} if coefficient_runtime is not None else {})})
    runtime = Path(runtime or root('repo') / 'ts-host')
    if not (runtime / 'dist/teacher/cli.js').is_file():
        raise FileNotFoundError('RL requires the built natlang collector runtime')
    def worker(index):
        queue = directory / f'queue-{index}.jsonl'
        journal = directory / f'journal-{index}.jsonl'
        endpoint = endpoints[index % len(endpoints)].rstrip('/')
        for entry in entries[index::workers]:
            entry['rl_endpoint'] = endpoint
        queue.write_text(''.join(json.dumps(e) + '\n' for e in entries[index::workers]))
        command = [sys.executable, str(root('repo') / 'scripts/run_bonsai_queue.py'), str(queue), str(journal),
            '--runtime', str(runtime), '--server', endpoint, '--chat-completions-url', endpoint + '/v1/chat/completions',
            '--model-id', 'natlang-rl', '--model-concurrency', '1', '--chat-request-config', str(config)]
        if neuralese_server:
            command.append('--neuralese-server')
        with (directory / f'worker-{index}.log').open('a') as log:
            if control is None:
                code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
            else:
                process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                code = control.wait(process)
        return {'worker': index, 'returncode': code, 'journal': str(journal)}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(worker, range(workers)))
    (out / 'journal.jsonl').write_text(''.join(Path(r['journal']).read_text()
        for r in results if Path(r['journal']).exists()))
    (out / 'queue.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in entries))
    report = score(out, out / 'rollouts.jsonl')
    from .graph_replay import block_ids, save_bundle
    rollouts = list(read_jsonl(out / 'rollouts.jsonl'))
    if any(block_ids(rollouts)):
        from .grpo import Server
        servers = [Server(url) for url in endpoints]
        origins = {}
        def fetch(block_id):
            from .graph_replay import verified_bundle_bytes
            previous = verified_bundle_bytes(out / 'blocks/manifest.json', block_id, policy_id=policy_id)
            if previous is not None:
                origins[block_id] = 'retained-verified-prior-actor-lease'
                return previous
            errors = []
            for server in servers:
                try:
                    kind = ('native-states' if block_id.startswith('nzs1_') else
                            'native-tensor-artifacts' if block_id.startswith('nnt1_') else 'blocks')
                    raw = server.get_bytes('/v1/neuralese/' + kind + '/' + block_id)
                    origins[block_id] = server.base
                    return raw
                except Exception as error:
                    errors.append(str(error))
            raise RuntimeError('RL block unavailable on every endpoint: ' + block_id + ': ' + '; '.join(errors))
        manifest = save_bundle(rollouts, out / 'blocks', policy_id=policy_id, fetch=fetch)
        manifest['actor_origins'] = origins
        manifest['collection_binding'] = binding
        write_json_durable(out / 'blocks/manifest.json', manifest)
        report['block_bundle'] = str(out / 'blocks/manifest.json')
    if actor_receipts() != actors:
        raise ValueError('RL actor policy changed during collection')
    report.update(**({'coefficient_policy':binding['coefficient_policy']} if coefficient_runtime is not None else {}),
                  collector_workers=results, preparation=preparation,
                  actor_receipts=actors, collection_complete=not (control and control.stop.requested))
    write_json_durable(out / 'score.json', report)
    return report


def run_command(template, values):
    if not isinstance(template, list) or not template or any(not isinstance(v, str) for v in template):
        raise ValueError('RL deployment commands must be nonempty argv lists')
    command = []
    for part in template:
        if part.startswith('{') and part.endswith('}') and isinstance(values.get(part[1:-1]), list):
            command.extend(values[part[1:-1]])
            continue
        for name, value in values.items():
            part = part.replace('{' + name + '}', str(value))
        if part == '' and command and command[-1] == '--rl-block-bundle':
            command.pop()
            continue
        command.append(part)
    control = values.get('_run_control')
    if control is None:
        subprocess.run(command, check=True)
    else:
        child = subprocess.Popen(command)
        code = control.wait(child, stop_file=values.get('_bridge_stop_file'),
                            stop_request=values.get('_bridge_request'))
        if code and not control.stop.requested:
            raise subprocess.CalledProcessError(code, command)


def _bind_collected_coefficient_policy(values,kwargs,binding,identity):
    """Verify exact actor publication before a live or recovered collection."""
    from ..common.hashing import sha256_file_hex_cached
    from ..model.native_coefficient_policy import verify_base_transition
    if not isinstance(binding,dict) or set(binding)!={'path','sha256'}:
        raise RuntimeError('managed coefficient actor omitted its actual policy binding')
    try:
        target=Path(binding['path']).resolve(strict=True)
        if sha256_file_hex_cached(target)!=binding['sha256']:
            raise ValueError('coefficient actor policy bytes changed')
        verify_base_transition(kwargs['coefficient_policy'],target,identity=identity,
            cancel_stale_reason=values.get('coefficient_cancel_stale_reason'))
    except (OSError,ValueError,KeyError,TypeError) as error:
        raise RuntimeError('managed coefficient actor publication failed exact verification: '+str(error)) from error
    kwargs['coefficient_policy']=str(target)
    values.update(coefficient_policy=str(target),coefficient_policy_sha256=binding['sha256'])


def collect_with_actor(config, values, **kwargs):
    """Optional colocated actor lifecycle; its parent is the ledger/sandbox unit.

    The entire actor lease ends only AFTER collect has durably archived blocks.
    External fleets use the same collect function and explicit supervisor hooks.
    """
    completed = Path(kwargs['out']) / 'score.json'
    if completed.is_file() and json.loads(completed.read_text()).get('collection_complete'):
        saved=json.loads(completed.read_text())
        if kwargs.get('coefficient_policy') is not None:
            receipt=next(iter(saved['actor_receipts'].values()))
            binding=saved.get('coefficient_policy') or {}
            _bind_collected_coefficient_policy(values,kwargs,{key:binding.get(key) for key in ('path','sha256')},receipt['native_identity'])
        return collect(**kwargs)  # sealed packet already owns exact durable blocks
    control = kwargs.get('control')
    template = config.get('actor_command')
    if not template:
        return collect(**kwargs)
    if len(kwargs['endpoints']) != 1:
        raise ValueError('managed colocated actor requires one endpoint; external fleets use supervisor hooks')
    import time
    command = []
    for part in template:
        for key, value in values.items():
            part = part.replace('{' + key + '}', str(value))
        command.append(part)
    from .grpo import Server
    from ..common.hashing import sha256_file_hex_cached
    with (Path(values['out']) / 'actor.log').open('a') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                if control is not None and control.stop.requested:
                    raise InterruptedError('actor stop requested before collection')
                if process.poll() is not None:
                    raise RuntimeError('managed RL actor exited; see actor.log')
                try:
                    info = json.loads(Server(kwargs['endpoints'][0], timeout=2).get_bytes('/v1/neuralese/info'))
                    if (info.get('rl_policy_binding') or {}).get('checkpoint_sha256') == values['policy_id']:
                        if kwargs.get('coefficient_policy') is not None:
                            actor=info['rl_policy_binding']
                            _bind_collected_coefficient_policy(values,kwargs,actor.get('coefficient_policy_binding'),actor['native_identity'])
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(.2)
            return collect(**kwargs)
        finally:
            if process.poll() is None:
                process.terminate()  # explicit child PID; no pattern/process-wide kill
            # Keep polling so Python handles stop signals while the actor exits.
            if control is not None:
                control.wait(process)
            else:
                from ..train.loop import wait_forwarding_signals
                wait_forwarding_signals(process)


def select_tasks(cases, stats, budget, *, command=None):
    """S7 selectTasks(groupStats,budget)->{row,samples}[], crisp/NL boundary.

    A command may execute the existing natlang runtime and return the same typed
    JSON. This mechanism validates identities/counts; policy wording is in S7.
    """
    if command:
        response = subprocess.run(command, input=json.dumps({'groupStats': stats,
            'budget': budget, 'tasks': [{'row': i, 'id': row['id']} for i, row in enumerate(cases)]}),
            text=True, capture_output=True, check=True)
        chosen = json.loads(response.stdout)
    else:
        retired = {g['task_id'] for g in stats if g.get('scored', 0) and g.get('mean') == 1.}
        order = list(range(budget['offset'], len(cases))) + list(range(budget['offset']))
        chosen = [{'row': i, 'samples': budget['group']} for i in order
                  if cases[i]['id'] not in retired][:budget['rows']]
    if not isinstance(chosen, list) or any(not isinstance(item, dict) or
            type(item.get('row')) is not int or not 0 <= item['row'] < len(cases) or
            type(item.get('samples')) is not int or item['samples'] < 2 for item in chosen):
        raise ValueError('selectTasks must return distinct valid {row,samples>=2} entries')
    if len({item['row'] for item in chosen}) != len(chosen):
        raise ValueError('selectTasks returned duplicate task rows')
    if len({item['samples'] for item in chosen}) > 1:
        raise ValueError('current collector fanout requires one shared group size per round')
    return chosen


def trained_counts(directory, progress):
    """Committed update exposure across process resumes, from shared train.jsonl.

    this_run alone loses partial-process updates and becomes zero on a no-op
    resume. Keep the last record per committed update/cursor, not log duplicates.
    """
    log = Path(directory) / 'train.jsonl'
    if progress.get('train_exposure'):
        return progress['train_exposure']
    if not log.is_file():
        return {key: (progress.get('this_run') or {}).get(key, 0)
                for key in ('supervised_tokens', 'target_tokens')}
    rows = {}
    for row in read_jsonl(log):
        if row.get('step', 0) <= progress.get('step', progress.get('steps', 0)):
            rows[(row.get('step'), row.get('kind', 'direct'), row.get('e2e_cursor'))] = row
    return {key: sum(row.get(key, 0) for row in rows.values())
            for key in ('supervised_tokens', 'target_tokens')}


def expected_native_actor(adapter_path, participant, policy_id):
    """Bind the actual immutable adapter's named participant, not only shared file SHA."""
    from ..model.native_composite import resolve_composite_artifact
    from ..common.hashing import sha256_file_hex_cached
    adapter_path = Path(adapter_path)
    export = json.loads(adapter_path.read_text())
    if export.get('schema') != 'natlang.native-semantic-adapter/1' or participant not in export.get('models', {}):
        raise ValueError('native RL actor needs its actual exported shared participant')
    resolve_composite_artifact(export['checkpoint'],adapter_path)
    model = export['models'][participant]
    if (export['checkpoint']['sha256'] != policy_id or model['native_identity']['checkpoint_sha256'] != policy_id
            or model['checkpoint'].get('participant') != participant or model['checkpoint']['sha256'] != policy_id):
        raise ValueError('native RL actor adapter and participant differ from the current shared policy')
    return {'native_participant': participant, 'native_identity': model['native_identity'],
            'semantic_adapter_sha256': sha256_file_hex_cached(adapter_path)}


def initial_controller_state(config):
    return {'schema':'natlang.rl-production-state/1','round':0,'offset':0,
        'checkpoint':config['checkpoint'],**({'coefficient_policy':config['coefficient_policy']} if config.get('coefficient_policy') else {}),'trained_rounds':0,'collected_task_ids':[],
        'attempted_task_ids':[],'trained_task_ids':[],'trained_supervised_tokens':0,
        'trained_target_tokens':0,'production_complete':False}


def run(config_path, *, rounds=None, defer_native_refresh=False, stop=None):
    """Hot-reload configuration at round boundaries; preserve actual state/exposure."""
    from ..common.hashing import sha256_file_hex
    config_path = Path(config_path)
    from ..train.recipe import resolve_recipe_data
    initial = resolve_recipe_data(config_path)
    control = RunControl(stop) if stop is not None else None
    out = Path(initial['out']).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # One owner for this controller's exact state; no double learner updates.
    with (out / 'controller.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = out / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else initial_controller_state(initial)
        write_json_durable(state_path, state)
        if initial.get('mandatory_sft_only'):
            if not initial.get('native_state'):
                raise ValueError('mandatory supervised continuation requires native shared training')
            from .native_supervised import execute
            return execute(initial,state,state_path,control)
        completed = 0
        while rounds is None or completed < rounds:
            if control is not None and control.stop.requested:
                state['phase'] = 'requested_stop'
                write_json_durable(state_path, state)
                return state
            config = resolve_recipe_data(config_path)
            if not config.get('after_update') and not config.get('actor_command'):
                raise ValueError('configure after_update before learning: every endpoint needs new weights')
            cases = list(read_jsonl(config['cases']))
            if any(r.get('split') != 'train' for r in cases):
                raise ValueError('RL source must contain only train-split tasks')
            if not cases:
                state.update(phase='await_new_train_tasks', reason='train task source is empty')
                write_json_durable(state_path, state)
                return state
            if state['offset'] >= len(cases):
                state['offset'] = 0
            attempt = state.get('collection_attempt', 0)
            round_dir = out / (f"round-{state['round']:06d}" + (f'-attempt-{attempt}' if attempt else ''))
            selection_file = out / f"round-{state['round']:06d}-selection.json"
            if selection_file.exists() and not json.loads(selection_file.read_text()).get('selection'):
                selection_file.unlink()  # repair pre-fix empty snapshots; no active collection owns them
            if selection_file.exists():
                snapshot = json.loads(selection_file.read_text())
                selection, selected = snapshot['selection'], snapshot['cases']
            else:
                selection = select_tasks(cases, state.get('group_stats', []),
                    {'rows': config.get('rows_per_round', 32), 'group': config.get('group', 4), 'offset': state['offset']},
                    command=config.get('select_tasks_command'))
                selected = [cases[item['row']] for item in selection]
                if selection:
                    write_json_durable(selection_file, {'source_sha256': sha256_file_hex(Path(config['cases'])),
                        'selection': selection, 'cases': selected})
            if not selection:
                state.update(phase='await_new_train_tasks', reason='current train tasks solved; continue production SFT/new task admission')
                write_json_durable(state_path, state)
                return state
            round_dir.mkdir(parents=True, exist_ok=True)
            selected_path = round_dir / 'selected-cases.jsonl'
            selected_path.write_text(''.join(json.dumps(row) + '\n' for row in selected))
            policy = sha256_file_hex(Path(state['checkpoint']))
            values = {**{k: v for k, v in config.items() if isinstance(v, (str, int, float, list))},
                      'checkpoint': state['checkpoint'], 'policy_id': policy, 'out': str(round_dir),
                      'rollouts': str(round_dir / 'rollouts.jsonl'), 'mode': 'grpo',
                      'learner_out': str(round_dir / 'learner'), 'block_bundle': '',
                      'continue_identity': str(Path(state['checkpoint']).parent / 'identity-check.json'),
                      'continue_monitor_reference': str(Path(state['checkpoint']).parent / 'monitor-reference.pt')}
            if control is not None:
                values['_run_control'] = control
            from ..train.recipe import stage_parameter_args
            expected_native = None
            if config.get('native_state'):
                values['semantic_adapter'] = state.get('semantic_adapter', config['semantic_adapter'])
                values['semantic_adapter_sha256'] = sha256_file_hex(Path(values['semantic_adapter']))
                export = json.loads(Path(values['semantic_adapter']).read_text())
                expected_native = expected_native_actor(values['semantic_adapter'], config['native_participant'], policy)
                values['rl_tokenizer'] = str(Path(values['semantic_adapter']).parent /
                    export['tokenizers'][config['native_participant']]['path'])
            values['trainer_args'] = stage_parameter_args(config.get('trainer_parameters', {}))
            values['seed_trainer_args']=stage_parameter_args(config.get('seed_trainer_parameters',config.get('trainer_parameters',{})))
            if config.get('coefficient_policy'):
                policy_path=Path(state.get('coefficient_policy',config['coefficient_policy'])).resolve(strict=True)
                # An already sealed round keeps its original collection actor
                # while the learner/publication is resumed. The current policy
                # may already have advanced; never recollect or relabel that row.
                if state.get('active_update_round')==str(round_dir) and state.get('update_coefficient_policy'):
                    sealed=state['update_coefficient_policy']
                    policy_path=Path(sealed['path']).resolve(strict=True)
                    if sha256_file_hex(policy_path)!=sealed['sha256']:
                        raise ValueError('sealed coefficient round source policy changed during publication recovery')
                values.update(coefficient_policy=str(policy_path),coefficient_policy_sha256=sha256_file_hex(policy_path),
                    coefficient_cancel_stale_reason=config.get('coefficient_cancel_stale_reason'))
            if config.get('heads'):
                values['heads_sha256'] = sha256_file_hex(Path(config['heads']))
            if config.get('before_collect'):
                run_command(config['before_collect'], values)
            try:
                report = collect_with_actor(config, values, cases=selected_path, out=round_dir,
                    endpoints=config['endpoints'], policy_id=policy,
                    rows=len(selected), offset=0,
                    group=selection[0]['samples'], workers=config.get('workers', 4),
                    root_mode=config.get('root_mode', 'student'), runtime=config.get('runtime'),
                    seed_base=700000 + state['round'] * 1000000,
                    neuralese_server=config.get('neuralese_server', True),
                    coefficient_policy=values.get('coefficient_policy'),
                    case_seconds=config.get('case_seconds', 600),
                    max_model_requests=config.get('max_model_requests', 128),
                    max_turns=config.get('max_turns', 160), control=control, expected_native=expected_native)
            except InterruptedError:
                state['phase'] = 'requested_stop'
                write_json_durable(state_path, state)
                return state
            if control is not None and control.stop.requested:
                state.update(phase='stopped_collecting', latest_reward=report)
                write_json_durable(state_path, state)
                return state
            values['block_bundle'] = report.get('block_bundle', '')
            if report.get('coefficient_policy'):
                state['coefficient_policy']=report['coefficient_policy']['path']
                values.update(coefficient_policy=state['coefficient_policy'],
                              coefficient_policy_sha256=report['coefficient_policy']['sha256'])
            rows = list(read_jsonl(round_dir / 'rollouts.jsonl'))
            collected = set(state.get('collected_task_ids', [])) | {r['task_id'] for r in rows if r.get('turns')}
            attempted = set(state.get('attempted_task_ids', [])) | {r['id'] for r in selected}
            state.update(collected_task_ids=sorted(collected), attempted_task_ids=sorted(attempted), available_tasks=len(cases),
                collected_task_fraction=len(collected & {r['id'] for r in cases}) / len(cases),
                latest_reward=report, phase='select_update')
            grouped = {g['task_id']: g for g in state.get('group_stats', [])}
            grouped.update({g['task_id']: g for g in report.get('groups', [])})
            state['group_stats'] = list(grouped.values())
            write_json_durable(state_path, state)
            if report.get('scored', 0) == 0:
                state.update(phase='infrastructure_repair', collection_attempt=attempt + 1,
                             reason='all rollout outcomes unscored; no learning signal')
                write_json_durable(state_path, state)
                raise RuntimeError('RL collector infrastructure failed; repair and retry retained round with fresh attempt')
            if report.get('scored') == report.get('successes'):
                state.update(round=state['round'] + 1, offset=(selection[-1]['row'] + 1) % len(cases),
                             phase='collect', collection_attempt=0)
                write_json_durable(state_path, state)
                completed += 1
                continue  # all-success groups have no useful gradient
            rft_stages = ['all']
            if report['groups_with_spread']:
                mode = 'grpo'
            else:
                if config.get('native_state'):
                    if any((row.get('reward') or {}).get('quality') == 1 for row in rows):
                        mode = 'rft'
                    elif config.get('seed_command'):
                        mode = 'seed'
                    else:
                        raise RuntimeError('native RL flat failures require the declared verified whole-graph seed source')
                else:
                    # Reuse existing verified successful teachers when the student
                    # has no spread; never convert infrastructure failures to zero.
                    from .stage import select_rft_documents
                    seeds = [r for path in config.get('teacher_rollouts', []) for r in read_jsonl(path)]
                    if any(r.get('split') != 'train' for r in seeds):
                        raise ValueError('production RFT seeds must explicitly bind a train split')
                    merged = rows + seeds
                    docs, _ = select_rft_documents(merged, max_success_rate=.99, held_fraction=0. if config.get('held_text_data') else .125)
                    if any(d['split'] == 'train' for d in docs) and (config.get('held_text_data') or any(d['split'] == 'test' for d in docs)):
                        mode = 'rft'
                        from .stage import document_has_values
                        kinds = {document_has_values(doc) for doc in docs if doc['split'] == 'train'}
                        rft_stages = (['text', 'value'] if len(kinds) > 1 else ['value' if True in kinds else 'text'])
                        rft_file = round_dir / 'rft-rollouts.jsonl'
                        rft_file.write_text(''.join(json.dumps(r) + '\n' for r in merged))
                        values['rollouts'] = str(rft_file)
                    elif config.get('seed_command'):
                        mode = 'seed'
                    else:
                        state.update(phase='needs_verified_seed_data', reason='no usable policy-gradient spread or RFT successes')
                        write_json_durable(state_path, state)
                        raise RuntimeError('RL needs verified successful seed trajectories; continue declared SFT seed stage')
            if state.get('active_update_round') == str(round_dir):
                mode = state['update_mode']
                values['rollouts'] = state['update_rollouts']
                rft_stages = state.get('update_stages', ['all'])
            values['mode'] = mode
            if config.get('coefficient_policy') and state.get('active_update_round')!=str(round_dir):
                state['update_coefficient_policy']={'path':values['coefficient_policy'],'sha256':values['coefficient_policy_sha256']}
            state.update(phase='updating', update_mode=mode, active_update_round=str(round_dir),
                         update_rollouts=values['rollouts'], update_stages=rft_stages)
            write_json_durable(state_path, state)
            base_learner_out = Path(values['learner_out'])
            pending = False
            for kind in rft_stages:
                stage_out = base_learner_out if len(rft_stages) == 1 else base_learner_out.parent / ('learner-' + kind)
                values.update(learner_out=str(stage_out), rft_document_kind=kind)
                learner_report = stage_out / 'report.json'
                saved_progress=json.loads(learner_report.read_text()) if learner_report.is_file() else {}
                already_complete=bool(saved_progress.get('completed') and
                    (not (config.get('coefficient_policy') and mode!='seed') or saved_progress.get('qualification_complete') is True))
                if config.get('before_update') and not already_complete:
                    run_command(config['before_update'], values)
                if not already_complete:
                    run_command(config['seed_command'] if mode == 'seed' else config['learner_command'], values)
                checkpoint = stage_out / 'checkpoint.pt'
                if (not checkpoint.is_file() or not learner_report.is_file()) and control is not None and control.stop.requested:
                    state['phase'] = 'requested_stop'
                    write_json_durable(state_path, state)
                    return state
                if not checkpoint.is_file() or not learner_report.is_file():
                    raise RuntimeError('learner produced no recoverable full checkpoint/progress report')
                progress = json.loads(learner_report.read_text())
                coefficient_update=bool(config.get('coefficient_policy') and mode!='seed')
                if coefficient_update:
                    if (progress.get('schema')!='natlang.native-coefficient-training-report/1'
                            or progress.get('base_checkpoint_unchanged') is not True):
                        raise ValueError('coefficient learner did not preserve its actual full-backbone parent')
                    if progress.get('completed') and not progress.get('qualification_complete'):
                        raise ValueError('coefficient learner completed data but lacks actual current-policy held/task qualification')
                counts = trained_counts(stage_out, progress)
                per_stage = state.setdefault('trained_stage_counts', {})
                previous_counts = per_stage.get(str(stage_out), {})
                for key in ('supervised_tokens', 'target_tokens'):
                    state['trained_' + key] = state.get('trained_' + key, 0) + counts[key] - previous_counts.get(key, 0)
                per_stage[str(stage_out)] = counts
                write_json_durable(state_path, state)
                if not progress.get('completed'):
                    state.update(phase='learner_pending', partial_checkpoint=str(checkpoint), last_learner_progress=progress)
                    write_json_durable(state_path, state)
                    pending = True
                    break  # shared trainer resumes this exact substage/output on retry
                trained_ids = set(state.get('trained_task_ids', []))
                data_path = stage_out / 'rl-objective.json'
                if mode == 'grpo' and data_path.exists():
                    trained_ids.update(d['task_id'] for d in json.loads(data_path.read_text())['documents'] if d['split'] == 'train')
                if mode == 'rft' or (config.get('native_state') and mode == 'grpo'):
                    receipt = json.loads((stage_out / 'rl-data-receipt.json').read_text())
                    trained_ids.update(d['task_id'] for d in receipt['tasks_selected'] if d['split'] == 'train')
                accounted = set(state.get('accounted_stages', []))
                if str(stage_out) not in accounted:
                    accounted.add(str(stage_out))
                    state.update(trained_task_ids=sorted(trained_ids), accounted_stages=sorted(accounted),
                        last_learner_progress=progress, phase='publishing')
                    write_json_durable(state_path, state)
                # Next full substage continues exact P/TCP + optimizer from this one.
                if coefficient_update:
                    published=progress['policy']
                    if sha256_file_hex(Path(published['path']))!=published['sha256']:
                        raise ValueError('coefficient learner publication bytes changed')
                    state['coefficient_policy']=published['path']
                    values.update(coefficient_policy=published['path'],coefficient_policy_sha256=published['sha256'])
                    checkpoint=Path(state['checkpoint'])  # actual unchanged backbone, never the coefficient state slot
                values.update(checkpoint=str(checkpoint), continue_identity=str(stage_out / 'identity-check.json'),
                              continue_monitor_reference=str(stage_out / 'monitor-reference.pt'))
            if pending:
                continue
            if config.get('native_state') and not (config.get('coefficient_policy') and mode!='seed'):
                state.update(phase='awaiting_semantic_refresh', pending_native_checkpoint=str(checkpoint),
                             pending_native_round_dir=str(round_dir))
                write_json_durable(state_path, state)
                if defer_native_refresh or (control is not None and control.stop.requested):
                    return state
                from .native_refresh import refresh
                deployment = refresh(adapter=values['semantic_adapter'],
                    updates={config['native_participant']: str(checkpoint)},
                    config=config['semantic_refresh_config'], out=round_dir/'semantic-refresh',
                    device=config.get('refresh_device', 'cuda'), stop=stop)
                checkpoint = Path(deployment['checkpoint']['path'])
                state['semantic_adapter'] = deployment['semantic_adapter']['path']
                values.update(semantic_adapter=state['semantic_adapter'],
                    semantic_adapter_sha256=deployment['semantic_adapter']['sha256'],
                    deployment=str(round_dir/'semantic-refresh/deployment.json'))
            if control is not None and control.stop.requested:
                return state
            # Publishing weights is a required boundary, not optional fake
            # success: collection must not silently keep using the old policy.
            values['checkpoint'] = str(checkpoint)
            values['policy_id'] = sha256_file_hex(checkpoint)
            if config.get('after_update'):
                run_command(config['after_update'], values)
            else:
                # Colocated actor is drained/stopped; the next lease loads this
                # checkpoint and attests it before any rollout is admitted.
                state['actor_policy_pending_load'] = values['policy_id']
                if config.get('coefficient_policy'):state['coefficient_policy_pending_load']=values['coefficient_policy_sha256']
            state.update(checkpoint=str(checkpoint), round=state['round'] + 1,
                         offset=(selection[-1]['row'] + 1) % len(cases),
                         trained_rounds=state['trained_rounds'] + 1, phase='collect',
                         production_complete=False, collection_attempt=0, group_stats=[])
            write_json_durable(state_path, state)
            state.pop('pending_native_checkpoint', None)
            state.pop('pending_native_round_dir', None)
            write_json_durable(state_path, state)
            completed += 1
        return state


def configure_native(args):
    """Declare both cloud/local native learners using the same trainer and refresh."""
    from ..train.recipe import resolve_recipe_data, effective_stage_parameters, HANDLERS
    from ..common.hashing import sha256_file_hex
    for name in ('native_participant', 'semantic_adapter', 'semantic_refresh_config',
                 'seed_native_graphs', 'held_native_graphs'):
        if getattr(args, name, None) is None:
            raise ValueError('native RL configure requires --' + name.replace('_', '-'))
    adapter = json.loads(args.semantic_adapter.read_text())
    if (adapter.get('schema') != 'natlang.native-semantic-adapter/1'
            or adapter['checkpoint']['sha256'] != sha256_file_hex(args.checkpoint)
            or args.native_participant not in adapter['models']):
        raise ValueError('native RL parent participant and semantic adapter must bind the same actual shared checkpoint')
    recipe = resolve_recipe_data(args.production_recipe)
    stages = [stage for stage in recipe['stages'] if stage['id'] == args.production_stage]
    if len(stages) != 1 or stages[0]['kind'] not in {'native_state_sft', 'native_state_rl'}:
        raise ValueError('native RL configure requires the actual declared native production stage')
    parameters = effective_stage_parameters(stages[0], [], recipe)
    parameters = {key: value for key, value in parameters.items()
                  if key in HANDLERS['native_state_sft']['parameters'] and value is not None}
    parameters.pop('memory_qualification',None)  # one-time source-capacity probe already bound to the SFT parent
    parameters.update(base=str(args.base.resolve()))
    parameters.setdefault('epochs', 1)
    if not parameters.get('task_eval') or not parameters.get('task_eval_every'):
        raise ValueError('native production must retain real task evaluation and its cadence')
    repo, python = str(root('repo')), sys.executable
    endpoint = 'http://127.0.0.1:' + str(args.port)
    common = ['--continue-from', '{checkpoint}', '--native-participant', '{native_participant}',
              '--held-native-graphs', '{held_native_graphs}', '--out', '{learner_out}', '{trainer_args}']
    config = {'schema': 'natlang.rl-production-config/1', 'id': 'native-rl-' + args.model,
        'native_state': True, 'model': args.model, 'out': str(args.out.resolve()),
        'checkpoint': str(args.checkpoint.resolve()), 'native_participant': args.native_participant,
        'semantic_adapter': str(args.semantic_adapter.resolve()),
        'semantic_refresh_config': str(args.semantic_refresh_config.resolve()),
        'seed_native_graphs': str(args.seed_native_graphs.resolve()),
        'held_native_graphs': str(args.held_native_graphs.resolve()), 'base': str(args.base.resolve()),
        'cases': str(args.cases.resolve()), 'runtime': str(args.runtime.resolve()),
        'endpoints': [endpoint], 'rows_per_round': 32, 'group': 4, 'workers': 4,
        'neuralese_server': True, 'trainer_parameters': parameters,
        'production_recipe': str(args.production_recipe.resolve()), 'production_stage': args.production_stage,
        'actor_command': [python, repo + '/scripts/neuralese_data/serve_direct_checkpoint.py',
            '--native-state', '--store-dir', '{out}/actor-store',
            '--checkpoint', '{checkpoint}', '--checkpoint-sha256', '{policy_id}',
            '--native-participant', '{native_participant}', '--base', '{base}',
            '--semantic-adapter', '{semantic_adapter}', '--semantic-adapter-sha256', '{semantic_adapter_sha256}',
            '--port', str(args.port)],
        'learner_command': [python, '-m', 'natlang_neuralese.rl.native_stage',
            '--rollouts', '{rollouts}', '--rl-tokenizer', '{rl_tokenizer}', '--rl-mode', '{mode}',
            '--rl-block-bundle', '{block_bundle}', *common],
        'seed_command': [python, '-m', 'natlang_neuralese.train.native_state_sft',
            '--native-graphs', '{seed_native_graphs}', '--objective-continuation', *common]}
    if parameters.get('communication_foundation'):
        from ..train.recipe import resolve_stage_inputs
        sources=resolve_stage_inputs(recipe,stages[0],{})
        foundation=sources.get('foundation_native_graphs')
        if not foundation:
            raise ValueError('native production foundation requires its exact declared foundation_native_graphs source')
        config['foundation_native_graphs']=foundation['path']
        config['seed_command'] += ['--foundation-native-graphs','{foundation_native_graphs}']
    coefficient_policy=getattr(args,'coefficient_policy',None)
    if coefficient_policy is not None:
        from ..model.native_coefficient_policy import policy_core
        core=policy_core(json.loads(Path(coefficient_policy).read_text()))
        if core['base_native_identity']!=adapter['models'][args.native_participant]['native_identity']:
            raise ValueError('coefficient production policy must bind the actual declared shared participant')
        coefficient_optimizer=getattr(args,'coefficient_optimizer',None)
        if not isinstance(coefficient_optimizer,dict) or coefficient_optimizer.get('name') not in {'adam','sgd'}:
            raise ValueError('coefficient production requires its explicit shared named Adam/SGD settings')
        config['coefficient_policy']=str(Path(coefficient_policy).resolve(strict=True))
        config['seed_trainer_parameters']=dict(parameters)
        config['trainer_parameters']={key:value for key,value in parameters.items()
            if key in HANDLERS['native_coefficient_rl']['parameters']}
        config['trainer_parameters']['coefficient_optimizer']=coefficient_optimizer
        if getattr(args,'coefficient_task_eval',None) is not None:
            config['trainer_parameters']['task_eval']=args.coefficient_task_eval
        from ..eval.task_cadence import parse_task_eval
        if (parse_task_eval(config['trainer_parameters'].get('task_eval')) or {}).get('async'):
            raise ValueError('coefficient task evaluation currently requires explicit synchronous task_eval; preserve actual actor bindings instead of using a base-only async snapshot')
        config['actor_command'] += ['--coefficient-policy','{coefficient_policy}',
            '--coefficient-policy-sha256','{coefficient_policy_sha256}',
            '--coefficient-rebase-out','{out}/coefficient-rebase']
        config['coefficient_cancel_stale_reason']='Actual backbone advanced at a safe whole-policy update boundary; retain completed credits and moments/RNG, cancel untrained stale behavior and recollect on the new base.'
        config['actor_command'] += ['--coefficient-cancel-stale-reason','{coefficient_cancel_stale_reason}']
        config['learner_command'] += ['--coefficient-policy','{coefficient_policy}']
        config['seed_command']=[part.replace('{trainer_args}','{seed_trainer_args}') for part in config['seed_command']]
    write_json_durable(args.config, config)
    return config


def configure(args):
    if getattr(args, 'native_state', False):
        return configure_native(args)
    if args.heads is None or args.held_text_data is None:
        raise ValueError('legacy projected configuration requires --heads; new production uses --native-state')
    """Inherit actual production trainer options, never invent a new optimizer."""
    from ..train.checkpoint_policy import CheckpointPolicy
    from ..common.hashing import sha256_file_hex
    from ..train.direct_sft import SCHEMA, OPERATIONAL
    state = CheckpointPolicy.load(args.checkpoint)
    if state.get('schema') != SCHEMA:
        raise ValueError('RL requires a direct_sft/mixed checkpoint; standalone trajectories needs explicit conversion')
    from ..train.checkpoint_offload import require_optimizer
    from ..train.recipe import resolve_recipe_data, effective_stage_parameters
    require_optimizer(state, args.checkpoint)
    if state.get('optimizer') is None:
        raise ValueError('RL continuation requires the full optimizer state')
    recipe = resolve_recipe_data(args.production_recipe)
    stages = [stage for stage in recipe['stages'] if stage['id'] == args.production_stage]
    if len(stages) != 1 or stages[0]['kind'] != 'neuralese_direct':
        raise ValueError('production stage must name an existing neuralese_direct stage')
    declared = effective_stage_parameters(stages[0], [], recipe)
    options = (state.get('identity') or {}).get('options') or {}
    if options.get('family'):
        raise ValueError('RL family-member KD unsupported: declare a full-model continuation first')
    if state.get('neuralese_projection') is None:
        raise ValueError('value RL requires the declared direct mixed stage first (existing e2e_mix addition)')
    removed = OPERATIONAL | {'heads', 'out', 'records', 'pieces', 'text_data', 'continued_text_data',
        'steps', 'epochs', 'e2e_mix', 'rl_objective', 'objective_continuation', 'family', 'recovery_selection'}
    parameters = {key: value for key, value in options.items() if key not in removed and value is not None}
    # Identity excludes operational settings: inherit quality/runtime controls from the actual recipe.
    excluded = (removed - OPERATIONAL) | {'base', 'out', 'steps', 'continue_from', 'continue_identity', 'continue_monitor_reference',
        'stop_at', 'plan_only', 'plan_out', 'handoff', 'fresh_monitor_reference'}
    parameters.update({key: value for key, value in declared.items()
                       if key in OPERATIONAL and key not in excluded and value is not None})
    if not parameters.get('task_eval') or not parameters.get('task_eval_every'):
        raise ValueError('production recipe must declare real task_eval and task_eval_every')
    parameters.update(epochs=1, window_execution='sequential', batch=1, sdpa_backend='flash',
                      save_state=True, base=str(args.base.resolve()))
    repo = str(root('repo'))
    python = sys.executable
    endpoint = f'http://127.0.0.1:{args.port}'
    config = {'schema': 'natlang.rl-production-config/1', 'id': 'rl-' + args.model,
        'production_recipe': str(args.production_recipe.resolve()),
        'production_recipe_sha256': sha256_file_hex(args.production_recipe),
        'production_stage': args.production_stage,
        'model': args.model, 'out': str(args.out.resolve()), 'checkpoint': str(args.checkpoint.resolve()),
        'heads': str(args.heads.resolve()), 'base': str(args.base.resolve()),
        'cases': str(args.cases.resolve()), 'held_text_data': str(args.held_text_data.resolve()),
        'runtime': str(args.runtime.resolve()), 'endpoints': [endpoint], 'rows_per_round': 32,
        'group': 4, 'workers': 4, 'neuralese_server': True, 'trainer_parameters': parameters,
        'teacher_rollouts': [str(root('repo') / 'runs/rl-verified-cross-model-seed-20261015-v1/mellum-rollouts.jsonl')],
        'actor_command': [python, repo + '/scripts/neuralese_data/serve_direct_checkpoint.py',
            '--checkpoint', '{checkpoint}', '--checkpoint-sha256', '{policy_id}',
            '--heads', '{heads}', '--heads-sha256', '{heads_sha256}', '--base', '{base}',
            '--neuralese', '--prompt-rendering', 'full', '--sdpa-backend', 'flash', '--port', str(args.port)],
        'learner_command': [python, '-m', 'natlang_neuralese.rl.stage', '--rollouts', '{rollouts}',
            '--rl-tokenizer', '{base}', '--rl-mode', '{mode}', '--rft-document-kind', '{rft_document_kind}',
            '--rl-policy-id', '{policy_id}', '--rl-prompt-rendering', 'full',
            '--rl-block-bundle', '{block_bundle}', '--rl-held-text-data', '{held_text_data}',
            '--heads', '{heads}', '--continue-from', '{checkpoint}', '--continue-identity', '{continue_identity}',
            '--continue-monitor-reference', '{continue_monitor_reference}', '--out', '{learner_out}', '{trainer_args}']}
    source_args = []
    for key in ('text_data', 'records', 'pieces'):
        if options.get(key):
            source_args += ['--' + key.replace('_', '-'), str(options[key])]
    if source_args:
        config['seed_command'] = [python, '-m', 'natlang_neuralese.train.direct_sft', *source_args,
            '--heads', '{heads}', '--continue-from', '{checkpoint}', '--continue-identity', '{continue_identity}',
            '--continue-monitor-reference', '{continue_monitor_reference}', '--out', '{learner_out}',
            '--fresh-monitor-reference', '{trainer_args}']
        if options.get('e2e_mix'):
            config['seed_command'] += ['--e2e-mix', json.dumps(options['e2e_mix'])]
    state = None
    write_json_durable(args.config, config)
    print(json.dumps({'config': str(args.config), 'model': args.model, 'launched': False}))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--config', type=Path, required=True)
    r.add_argument('--rounds', type=int)
    c = sub.add_parser('collect')
    c.add_argument('--cases', type=Path, required=True)
    c.add_argument('--out', type=Path, required=True)
    c.add_argument('--endpoint', action='append', required=True)
    c.add_argument('--policy-id', required=True)
    c.add_argument('--rows', type=int, default=32)
    c.add_argument('--group', type=int, default=4)
    c.add_argument('--workers', type=int, default=4)
    c.add_argument('--runtime', type=Path)
    c.add_argument('--coefficient-policy',type=Path)
    g = sub.add_parser('configure', help='bind shared local/CoreWeave loop to an exact mixed production checkpoint')
    g.add_argument('--model', choices=['lfm', 'mellum'], required=True)
    for key in ('checkpoint', 'base', 'cases', 'runtime', 'out', 'config'):
        g.add_argument('--' + key, type=Path, required=True)
    g.add_argument('--held-text-data', type=Path)
    g.add_argument('--heads', type=Path)
    g.add_argument('--native-state', action='store_true')
    g.add_argument('--native-participant')
    g.add_argument('--coefficient-policy',type=Path)
    g.add_argument('--coefficient-optimizer',type=json.loads)
    g.add_argument('--coefficient-task-eval',type=json.loads)
    for key in ('semantic-adapter', 'semantic-refresh-config', 'seed-native-graphs', 'held-native-graphs'):
        g.add_argument('--' + key, type=Path)
    g.add_argument('--production-recipe', type=Path, required=True)
    g.add_argument('--production-stage', required=True)
    g.add_argument('--port', type=int, default=8097)
    a = p.parse_args(argv)
    if a.command == 'configure':
        configure(a)
        return
    if a.command == 'run':
        from ..train.loop import StopSignal
        stop = StopSignal().install()
        try:
            run(a.config, rounds=a.rounds, stop=stop)
        finally:
            stop.restore()
    else:
        print(json.dumps(collect(cases=a.cases, out=a.out, endpoints=a.endpoint, policy_id=a.policy_id,
            rows=a.rows, group=a.group, workers=a.workers, runtime=a.runtime,coefficient_policy=a.coefficient_policy)))


if __name__ == '__main__':
    main()
