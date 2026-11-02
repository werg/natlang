"""Native RL admission adapter; updates use the existing full native trainer.

This module only prepares exact sampled graphs/objective receipts. It contains
no forward, optimizer, checkpoint loop, synthetic reward or projection path.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ..common.jsonio import write_json_durable, write_bytes_durable
from ..common.hashing import sha256_file_hex_cached


def prepare(rollouts, *, checkpoint, tokenizer, bundle=None, mode='grpo', coefficient_policy=None,
            expected_history_reasoning_policy=None, expected_history_binding_sha256=None,
            expected_history_binding_path=None, policy_loss=None, jolt_cases=None, actors=None):
    from transformers import AutoTokenizer
    from ..model.shared_state import RoleTokens, ROLE_MARKERS
    from ..data.text_corpus import tokenizer_fingerprint
    from .native_replay import prepare_objective
    tokenizer_path = Path(tokenizer).resolve()
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    fingerprint = tokenizer_fingerprint(tokenizer)

    def bind_history_profile(identity):
        from .native_raw_policy import bind_saved_history_reasoning
        return bind_saved_history_reasoning(tokenizer, tokenizer_path=tokenizer_path,
            checkpoint_sha256=identity['checkpoint_sha256'], tokenizer_fingerprint=fingerprint,
            expected_history_reasoning_policy=expected_history_reasoning_policy,
            expected_binding_sha256=expected_history_binding_sha256,
            expected_binding_path=expected_history_binding_path)

    if coefficient_policy is not None:
        from ..model.native_coefficient_policy import policy_core
        core=policy_core(json.loads(Path(coefficient_policy).read_text()))
        identity=core['base_native_identity']
        if identity['checkpoint_sha256']!=sha256_file_hex_cached(checkpoint) or identity['tokenizer_fingerprint']!=fingerprint:
            raise ValueError('coefficient-policy admission changed the exact base checkpoint or tokenizer')
        bind_history_profile(identity)
        marker_ids=tokenizer.convert_tokens_to_ids(list(ROLE_MARKERS))
        if len(set(marker_ids))!=4 or any(value==tokenizer.unk_token_id for value in marker_ids):
            raise ValueError('native RL tokenizer lacks its four distinct role markers')
        if bundle is None:raise ValueError('coefficient-policy admission requires its actual collector bundle')
        from .graph_replay import verified_bundle_bytes,NativeBundleStore
        from .native_replay import invocation_behavior_policy
        from ..common.hashing import canonical_json_sha256_hex
        source=Path(bundle).resolve(strict=True)
        original=json.loads(source.read_text())
        if original.get('schema')!='natlang.rl-block-bundle/1' or original.get('policy_id')!=identity['checkpoint_sha256']:
            raise ValueError('coefficient-policy collector bundle changed its actual base-runtime owner')
        reader=NativeBundleStore(source,policy_id=identity['checkpoint_sha256'],manifest=original)
        fetch=reader.state
        scored=[row for row in rollouts if row.get('reward') is not None]
        behavior=invocation_behavior_policy(scored,identity=identity,roles=RoleTokens(*marker_ids),fetch_block=fetch)
        owner=canonical_json_sha256_hex(behavior)
        # Rebind only a new immutable archive descriptor. Original collection
        # manifest and artifact bytes remain unchanged and exactly verified.
        for ident in original['blocks']:
            if verified_bundle_bytes(source,ident,policy_id=identity['checkpoint_sha256']) is None:
                raise ValueError('coefficient-policy collector archive lost a declared artifact')
        rebound={**original,'policy_id':owner,'source_collection_manifest':{'path':source.name,'sha256':sha256_file_hex_cached(source)}}
        destination=source.parent/('invocation-'+owner+'.json')
        if destination.exists() and json.loads(destination.read_text())!=rebound:
            raise ValueError('immutable invocation admission archive changed its source binding')
        write_json_durable(destination,rebound)
        return prepare_objective(rollouts,identity=identity,roles=RoleTokens(*marker_ids),fetch_block=fetch,
                                 block_bundle=destination,mode=mode,behavior_policy=behavior,policy_loss=policy_loss,
                                 privileged=_jolt_source(policy_loss,mode,rows=rollouts,tokenizer=tokenizer,cases=jolt_cases,
                                                         roles=RoleTokens(*marker_ids),tokenizer_fingerprint=fingerprint,
                                                         fetch_text=lambda state_id: fetch(state_id).text))
    policies = [turn['sampling']['policy'] for row in rollouts if row.get('reward') is not None
                for turn in row.get('turns', []) if (turn.get('sampling') or {}).get('policy')]
    actor_decl = json.loads(Path(actors).read_text()) if actors is not None else None
    if actor_decl is not None:
        # Declared stale/quantized actors (rl.actors): the learner identity is theirs with its own checkpoint.
        from .actors import actor_declaration
        learner_sha = sha256_file_hex_cached(checkpoint)
        known = set((actor_decl.get('actors') or {}) if isinstance(actor_decl, dict) else ())
        policies = [{**policy, 'checkpoint_sha256': learner_sha} if policy.get('checkpoint_sha256') in known
                    else policy for policy in policies]
    if not policies or any(policy != policies[0] for policy in policies):
        raise ValueError('native RL requires one exact captured actor identity')
    identity = policies[0]
    if actor_decl is not None:
        actor_decl = actor_declaration(actor_decl, learner_identity=identity)
    if any((turn.get('sampling') or {}).get('adapter_binding') for row in rollouts for turn in row.get('turns', [])):
        raise ValueError('adapted actors require native coefficient-policy learning; full-backbone trainer export/rebinding is not yet supported')
    if (identity['checkpoint_sha256'] != sha256_file_hex_cached(checkpoint)
            or identity['tokenizer_fingerprint'] != fingerprint):
        raise ValueError('native RL checkpoint/tokenizer differs from actual sampled policy')
    bind_history_profile(identity)
    marker_ids = tokenizer.convert_tokens_to_ids(list(ROLE_MARKERS))
    if len(set(marker_ids)) != 4 or any(value == tokenizer.unk_token_id for value in marker_ids):
        raise ValueError('native RL tokenizer lacks its four distinct role markers')
    roles = RoleTokens(*marker_ids)
    archive = json.loads(Path(bundle).read_text()) if bundle else None
    owners = {identity['checkpoint_sha256'], *((actor_decl or {}).get('actors') or ())}
    if archive and (archive.get('schema') != 'natlang.rl-block-bundle/1' or archive.get('policy_id') not in owners):
        raise ValueError('native RL archive does not bind the sampled policy')
    from .graph_replay import NativeBundleStore
    reader=NativeBundleStore(bundle,policy_id=archive['policy_id'],manifest=archive) if bundle else None
    def fetch(block_id):
        if reader is None:raise ValueError('native RL state requires its durable actor bundle')
        return reader.state(block_id)
    privileged = _jolt_source(policy_loss, mode, rows=rollouts, tokenizer=tokenizer, cases=jolt_cases, roles=roles,
                              tokenizer_fingerprint=fingerprint, fetch_text=lambda state_id: fetch(state_id).text)

    return prepare_objective(rollouts, identity=identity, roles=roles, fetch_block=fetch,
        block_bundle=Path(bundle).resolve() if bundle else None, mode=mode, policy_loss=policy_loss,
        privileged=privileged, actors=actor_decl)


def _jolt_source(policy_loss, mode, **options):
    from .policy_math import policy_loss_declaration
    declared = policy_loss_declaration(policy_loss)
    if declared is None or declared['schema'] != 'jolt-plus/1' or mode != 'grpo':
        return None
    return jolt_privileged_source(**options)


def jolt_privileged_source(*, rows, tokenizer, cases, roles, tokenizer_fingerprint, registry=None, fetch_text=None):
    """Verified captured guidance for JOLT+; custody fails closed per episode.


    Without readable protected packets or bound training cases every student episode is refused a privileged
    teacher (recorded per row) and keeps its outcome term; nothing is dropped from RL.
    """
    from types import SimpleNamespace
    from ..common.jsonio import iter_jsonl_rows
    from ..data.render import Renderer
    from ..serve.chat import tokenizer_model_type
    from .jolt import privileged_source
    from ..common.custody import CustodyError, ProtectedWorlds
    from .privileged_context import PrivilegedRendering
    try:
        if cases is None:
            raise CustodyError('no training cases are bound to this learner (--jolt-cases)')
        custody = ProtectedWorlds.load(registry=registry)
    except CustodyError as error:
        reason = error
        def refused(rollout, replay, turns):
            raise CustodyError(str(reason))
        return refused
    worlds = {}
    for _, case in iter_jsonl_rows(Path(cases)):
        if case.get('id') in worlds:
            raise ValueError('JOLT+ training cases repeat a world id')
        worlds[case.get('id')] = case
    # Raw-policy tokenizers are saved separately from the base config.json. The
    # checkpoint-bound history sidecar was already verified by prepare(); use
    # its backbone rather than guessing from the exported tokenizer directory.
    model_type=(getattr(tokenizer,'natlang_history_reasoning',None) or {}).get('backbone') or tokenizer_model_type(tokenizer)
    if not model_type:
        raise ValueError('JOLT replay requires its authenticated tokenizer backbone')
    renderer = Renderer(tokenizer, SimpleNamespace(open_id=roles.produce_open, close_id=roles.produce_close),model_type=model_type)
    rendering = PrivilegedRendering.from_renderer(renderer, template=model_type,
                                                  tokenizer_fingerprint=tokenizer_fingerprint,
                                                  consume_open=roles.consume_open, consume_close=roles.consume_close)
    return privileged_source(rows, cases=worlds, rendering=rendering, custody=custody, fetch_text=fetch_text)


def main(argv=None):
    from ..train.hardware_profile import apply_environment_early
    apply_environment_early()  # allocator settings must precede CUDA initialization
    import sys
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', action='append', type=Path, required=True)
    parser.add_argument('--rl-tokenizer', type=Path, required=True)
    parser.add_argument('--rl-mode', choices=('grpo', 'rft'), default='grpo')
    # Declared grpo policy loss (rl.policy_math.policy_loss_declaration), e.g.
    # '{"schema":"score-centered/1","k":128,"importance":"tis"}'. Omitted: clipped ratio loss.
    parser.add_argument('--rl-policy-loss')
    # JOLT+ privileged context: the exact train-split cases of the rollouts; custody against every registered
    # declared protected source (common.custody.ProtectedWorlds, training/protected-worlds.json).
    parser.add_argument('--jolt-cases', type=Path)
    parser.add_argument('--rl-actors', type=Path, help='natlang.rl-actors/1 declaration admitting stale or quantized '
                        'actors of earlier lineage checkpoints (importance-corrected)')
    parser.add_argument('--rl-block-bundle', type=Path)
    parser.add_argument('--coefficient-policy',type=Path)
    parser.add_argument('--history-reasoning-policy', choices=('keep', 'last_turn_only'))
    parser.add_argument('--history-reasoning-binding-sha256')
    parser.add_argument('--history-reasoning-binding-path', type=Path)
    parser.add_argument('--continue-from', type=Path, required=True)
    parser.add_argument('--held-native-graphs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    # These describe the parent's supervised schedule, not sampled policy rows.
    # Mandatory CE/foundation credits remain in its carried continuation frame.
    parser.add_argument('--communication-foundation', action='store_true')
    parser.add_argument('--foundation-native-graphs', type=Path)
    parser.add_argument('--foundation-communication-batches', type=int)
    parser.add_argument('--refresh-foundation-readiness', action='store_true')
    # Scheduling policy remains applicable to policy rounds; unlike the parent's
    # cohort and qualification request, it must reach the actual nested trainer.
    parser.add_argument('--foundation-qualification-request', type=Path)
    parser.add_argument('--resident-data-directory', type=Path)
    parser.add_argument('--resident-geometry-qualification', type=Path)
    # A resumed supervised controller can carry its original capacity request.
    # It describes the parent's CE source/objective, never the sampled RL graph.
    # _admit_and_train builds the mandatory policy-capacity request below.
    parser.add_argument('--memory-qualification')
    # Optional owner-follow metadata is a control-plane pointer. The RL stage
    # passes it to the real nested trainer boundary below; it never changes the
    # rollout objective or trainer arguments.
    parser.add_argument('--publication-owner-pointer', type=Path)
    parser.add_argument('--publication-role', choices=('lfm', 'mellum'))
    # Learner forward-token bound per episode graph (measured on its exact causal-prefix packing, the learner's
    # default): episodes above it are deferred, kept whole in deferred-long-graphs.jsonl and recorded in the
    # data receipt, never silently dropped.
    parser.add_argument('--max-graph-forward-tokens', type=int)
    args, trainer = parser.parse_known_args(argv)
    from .production import validate_native_policy_trainer_cli
    validate_native_policy_trainer_cli(trainer)
    if any((args.communication_foundation, args.foundation_native_graphs,
            args.foundation_communication_batches, args.refresh_foundation_readiness,
            args.resident_data_directory, args.resident_geometry_qualification)):
        raise ValueError('SFT source controls cannot be applied to a native RL stage')
    # Coefficient checkpoints have their own authenticated state and resume path.
    args.learner_resume = (None if args.coefficient_policy is not None or
                           (args.out / 'native-rl-completed-operation.json').exists()
                           else _partial_learner_resume(args))
    operation_request = _completed_operation_request(args, trainer, raw_argv)
    from contextlib import nullcontext
    from .native_output_lease import coefficient_output_lease
    with coefficient_output_lease(args.out) if args.coefficient_policy is not None else nullcontext():
        if _completed_operation_replay(args, operation_request):
            return 0
        return _admit_and_train(args,trainer, operation_request=operation_request)


def _partial_learner_resume(args):
    """Authenticate an earned rolling learner separately from behavior policy."""
    out = Path(args.out)
    checkpoint, report_path, receipt_path = (out / 'checkpoint.pt', out / 'report.json',
                                             out / 'native-training-receipt.json')
    if not checkpoint.exists() and not report_path.exists() and not receipt_path.exists():
        return None
    if not all(path.is_file() for path in (checkpoint, report_path, receipt_path, out / 'rl-data-receipt.json')):
        raise ValueError('partial learner lacks its complete checkpoint, report and admission custody')
    report, receipt, admission = (json.loads(path.read_text()) for path in
                                   (report_path, receipt_path, out / 'rl-data-receipt.json'))
    actual = sha256_file_hex_cached(checkpoint)
    behavior = sha256_file_hex_cached(args.continue_from)
    expected_rollouts = {str(Path(p).resolve(strict=True)): sha256_file_hex_cached(p) for p in args.rollouts}
    boundary = report.get('continuation_boundary') or {}
    if (report.get('schema') != 'natlang.neuralese-native-sft-report/1'
            or report.get('completed') is not False
            or report.get('checkpoint_sha256') != actual
            or Path(report.get('checkpoint', '')).resolve() != checkpoint.resolve()
            or receipt.get('schema') != 'natlang.native-training-readiness/1'
            or receipt.get('completed') is not False
            or (receipt.get('checkpoint') or {}).get('sha256') != actual
            or Path((receipt.get('checkpoint') or {}).get('path', '')).resolve() != checkpoint.resolve()
            or receipt.get('continuation') != report.get('continuation')
            or receipt.get('continuation_boundary') != report.get('continuation_boundary')
            or boundary.get('checkpoint_sha256') != actual
            or admission.get('schema') != 'natlang.native-rl-admission/1'
            or (admission.get('policy') or {}).get('checkpoint_sha256') != behavior
            or admission.get('policy_loss') != getattr(args, 'rl_policy_loss', None)
            or admission.get('rollouts') != expected_rollouts
            or admission.get('native_graphs_sha256') != sha256_file_hex_cached(out / 'rl-native-graphs.jsonl')
            or admission.get('objective_sha256') != sha256_file_hex_cached(out / 'rl-native-objective.json')):
        raise ValueError('partial learner differs from its checkpoint, behavior admission or saved report')
    return {'path':str(checkpoint.resolve()), 'sha256':actual}


def _completed_operation_request(args, trainer, raw_argv):
    """Bind one exact RL operation's command, inputs, and code before its update."""
    from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached

    def path_binding(path):
        path=Path(path).resolve(strict=True)
        if path.is_file():
            files=[{'path':'.','sha256':sha256_file_hex_cached(path)}]
        elif path.is_dir():
            files=[{'path':item.relative_to(path).as_posix(),'sha256':sha256_file_hex_cached(item)}
                   for item in sorted(path.rglob('*')) if item.is_file()]
        else:
            raise ValueError('RL operation input is neither a regular file nor a directory')
        return {'path':str(path),'files':files,'tree_sha256':canonical_json_sha256_hex(files)}

    inputs={
        'rollouts':[path_binding(path) for path in args.rollouts],
        'tokenizer':path_binding(args.rl_tokenizer),
        'continue_from':path_binding(args.continue_from),
        'held_native_graphs':path_binding(args.held_native_graphs),
    }
    if getattr(args, 'learner_resume', None) is not None:
        inputs['learner_resume'] = args.learner_resume
    for key in ('jolt_cases','rl_actors','rl_block_bundle','coefficient_policy','history_reasoning_binding_path',
                'foundation_qualification_request','publication_owner_pointer'):
        value=getattr(args,key,None)
        if value is not None:inputs[key]=path_binding(value)
    if args.memory_qualification:
        try:memory_ref=json.loads(args.memory_qualification)
        except (TypeError,json.JSONDecodeError):memory_ref=None
        if isinstance(memory_ref,dict) and isinstance(memory_ref.get('path'),str):
            bound=path_binding(memory_ref['path'])
            if memory_ref.get('sha256') is not None and memory_ref['sha256']!=bound['files'][0]['sha256']:
                raise ValueError('RL parent memory qualification input changed before operation binding')
            inputs['memory_qualification']=bound
    # Native trainer switches can carry policy, qualification or model files.
    # Bind any existing path-valued option so an exact retry cannot hide a
    # mutated external artifact behind identical argv text.
    trainer_paths=[];index=0
    while index<len(trainer):
        token=trainer[index];value=None
        if token.startswith('--') and '=' in token:
            value=token.split('=',1)[1]
        elif token.startswith('--') and index+1<len(trainer) and not trainer[index+1].startswith('--'):
            index+=1;value=trainer[index]
        if value:
            try:
                parsed=json.loads(value)
            except (TypeError,json.JSONDecodeError):parsed=None
            candidate=parsed.get('path') if isinstance(parsed,dict) else value
            if isinstance(candidate,str):
                try:path=Path(candidate)
                except (TypeError,ValueError):path=None
                if path is not None and path.exists():
                    bound=path_binding(path)
                    if isinstance(parsed,dict) and parsed.get('sha256') not in (None,bound['files'][0]['sha256']):
                        raise ValueError('RL trainer qualification input changed before operation binding')
                    trainer_paths.append({'option':token.split('=',1)[0],'binding':bound})
        index+=1
    if trainer_paths:inputs['trainer_path_inputs']=trainer_paths
    package=Path(__file__).resolve().parents[1]
    code={name:sha256_file_hex_cached(package/name) for name in (
        'rl/native_stage.py','train/native_state_sft.py','rl/native_replay.py',
        'rl/policy_source_continuation.py')}
    request={'schema':'natlang.native-rl-operation-request/1','argv':raw_argv,
        'trainer_arguments':list(trainer),'output':str(args.out.resolve()),
        'input_bindings':inputs,'code_bindings':code}
    return {**request,'request_sha256':canonical_json_sha256_hex(request)}


def _completed_operation_replay(args, request):
    """Accept only an exact retry of an already committed policy operation."""
    from ..common.hashing import canonical_json_sha256_hex, sha256_file_hex_cached
    from ..train.native_resident import binding
    marker=args.out/'native-rl-completed-operation.json'
    if not marker.exists():return False
    saved=json.loads(marker.read_text())
    saved_request=saved.get('request')
    hash_payload={key:value for key,value in saved_request.items() if key!='request_sha256'} if isinstance(saved_request,dict) else None
    if (saved.get('schema')!='natlang.native-rl-completed-operation/1'
            or not isinstance(saved_request,dict)
            or {**saved_request, 'input_bindings': {k:v for k,v in saved_request.get('input_bindings', {}).items() if k != 'learner_resume'}, 'request_sha256':request['request_sha256']} != request
            or saved.get('request_sha256')!=saved_request.get('request_sha256')
            or canonical_json_sha256_hex(hash_payload)!=saved_request['request_sha256']):
        raise ValueError('RL output directory is sealed to a different completed operation')
    report_ref,receipt_ref,checkpoint_ref=saved.get('report'),saved.get('training_receipt'),saved.get('checkpoint')
    for ref in (report_ref,receipt_ref,checkpoint_ref):
        if not isinstance(ref,dict) or binding(ref.get('path'))!=ref:
            raise ValueError('completed RL operation artifact changed; refusing replay')
    report=json.loads(Path(report_ref['path']).read_text())
    receipt=json.loads(Path(receipt_ref['path']).read_text())
    if (report.get('completed') is not True
            or report.get('checkpoint_sha256')!=checkpoint_ref['sha256']
            or receipt.get('checkpoint',{}).get('sha256')!=checkpoint_ref['sha256']
            or receipt.get('continuation',{}).get('checkpoint_sha256')
                !=request['input_bindings']['continue_from']['files'][0]['sha256']
            or (saved_request['input_bindings'].get('learner_resume') is not None
                and (receipt.get('operation_parent') or {}).get('sha256') != saved_request['input_bindings']['learner_resume']['sha256'])
            or Path(checkpoint_ref['path']).resolve()!= (args.out/'checkpoint.pt').resolve()
            or sha256_file_hex_cached(Path(checkpoint_ref['path']))!=checkpoint_ref['sha256']):
        raise ValueError('completed RL operation receipt does not prove its exact committed update')
    for key,filename in (('data_receipt','rl-data-receipt.json'),('objective','rl-native-objective.json'),
                         ('graphs','rl-native-graphs.jsonl')):
        ref=saved.get(key)
        if not isinstance(ref,dict) or binding(ref.get('path'))!=ref or Path(ref['path']).resolve()!=(args.out/filename).resolve():
            raise ValueError('completed RL operation source artifact changed; refusing replay')
    return True


def skip_zero_signal_jolt(graphs, objective):
    """Omit only complete episodes whose JOLT outcome and privileged terms are both zero."""
    import math
    if (objective.get('mode')!='grpo' or (objective.get('policy_loss') or {}).get('schema')!='jolt-plus/1'):
        return graphs,objective,None
    rows=objective['rows'];ids=[graph['id'] for graph in graphs]
    if len(ids)!=len(set(ids)) or set(ids)!=set(rows):
        raise ValueError('JOLT graph/objective identity differs before zero-signal projection')
    omitted=[]
    for ident in ids:
        row=rows[ident];info=row.get('jolt');advantage=row.get('advantage')
        if (isinstance(info,dict) and info.get('role') in ('student','teacher')
                and info.get('privileged') is None and type(advantage) in (int,float)
                and math.isfinite(advantage) and advantage==0):
            omitted.append(ident)
    if not omitted:return graphs,objective,None
    omitted_set=set(omitted);kept=[graph for graph in graphs if graph['id'] not in omitted_set]
    projected={**objective,'rows':{key:value for key,value in rows.items() if key not in omitted_set}}
    disposition={'schema':'natlang.jolt-zero-signal-projection/1','omitted_graph_ids':omitted,
        'omitted_count':len(omitted),'active_count':len(kept),'omitted_graph_visits':0,
        'reason':'zero centered outcome advantage and no privileged twin; no active decision KL or writer term',
        'normalization':'active episodes and their sampled actions only; explicit change from retaining zero-loss episodes',
        'coverage':'original scored rollouts and complete execution graphs remain immutable evidence'}
    projected['zero_signal_projection']=disposition
    return kept,projected,disposition


def _admit_and_train(args,trainer,*,operation_request=None):
    from ..common.jsonio import iter_jsonl_rows
    from ..common.hashing import canonical_json_sha256_hex
    rows = [row for path in args.rollouts for _, row in iter_jsonl_rows(path)]
    graphs, objective = prepare(rows, checkpoint=args.continue_from, tokenizer=args.rl_tokenizer,
        bundle=args.rl_block_bundle, mode=args.rl_mode,coefficient_policy=args.coefficient_policy,
        expected_history_reasoning_policy=args.history_reasoning_policy,
        expected_history_binding_sha256=args.history_reasoning_binding_sha256,
        expected_history_binding_path=args.history_reasoning_binding_path,
        policy_loss=args.rl_policy_loss, jolt_cases=args.jolt_cases, actors=args.rl_actors)
    if not graphs:
        raise ValueError('no useful native policy episodes: route flat failures to verified graph SFT seeding')
    graphs,objective,zero_signal = skip_zero_signal_jolt(graphs,objective)
    if not graphs and zero_signal is not None:
        args.out.mkdir(parents=True,exist_ok=True)
        write_json_durable(args.out/'zero-signal-report.json',{'schema':'natlang.native-rl-no-signal/1',
            'policy_sha256':sha256_file_hex_cached(args.continue_from),'policy_loss':args.rl_policy_loss,
            'rollouts':{str(path.resolve()):sha256_file_hex_cached(path) for path in args.rollouts},
            'projection':zero_signal,'optimizer_updates':0,'graph_visits':0})
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    graph_path, objective_path = args.out/'rl-native-graphs.jsonl', args.out/'rl-native-objective.json'
    # Deferral runs on the admitted graphs in memory, so the (large) graph and objective files are written once.
    deferred = _defer_long_graphs(args, graphs, objective, share='--no-share-causal-prefixes' not in trainer)
    if not graphs:
        raise ValueError('every admitted native policy episode exceeds --max-graph-forward-tokens')
    from .episode_order import mix_episodes, order_receipt
    mix_episodes(graphs,objective)
    from .policy_source_continuation import arrange_and_seal, finish_seal
    transition = arrange_and_seal(args.out, graphs, objective) if args.coefficient_policy is None else None
    if transition is not None:
        # Seal the complete successor before replacing either predecessor. A retry
        # can recover even if interrupted between the two atomic replacements.
        import os
        next_graph=args.out/'rl-native-graphs.next.jsonl'
        next_objective=args.out/'rl-native-objective.next.json'
        write_bytes_durable(next_graph, ''.join(json.dumps(row) + '\n' for row in graphs).encode())
        write_json_durable(next_objective, objective)
        transition_path=finish_seal(args.out, transition, next_graph, next_objective,
                                    final_graph=graph_path, final_objective=objective_path)
        os.replace(next_graph,graph_path); os.replace(next_objective,objective_path)
        directory=os.open(args.out,os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    else:
        write_bytes_durable(graph_path, ''.join(json.dumps(row) + '\n' for row in graphs).encode())
        write_json_durable(objective_path, objective)
        transition_path=None
    episode_order=order_receipt(graphs,objective)
    write_json_durable(args.out/'rl-episode-order.json',episode_order)
    write_json_durable(args.out/'rl-data-receipt.json', {
        'episode_order':episode_order,
        'schema': 'natlang.native-rl-admission/1',
        **({'base_native_identity':objective['base_native_identity'],'behavior_policy_sha256':objective['behavior_policy_sha256'],
               'coefficient_policy':{'path':str(args.coefficient_policy.resolve(strict=True)),
                                     'sha256':sha256_file_hex_cached(args.coefficient_policy)}}
           if args.coefficient_policy is not None else {'policy':objective['policy']}), 'mode': args.rl_mode,
        **({'policy_loss': objective['policy_loss']} if 'policy_loss' in objective else {}),
        'rollouts': {str(path.resolve()): sha256_file_hex_cached(path) for path in args.rollouts},
        'tasks_selected': [{'task_id': row['task_id'], 'split': 'train'} for row in objective['rows'].values()],
        **({'zero_signal_projection':zero_signal} if zero_signal is not None else {}),
        'omissions':{'unscored':[{'task_id':row.get('task_id'),'sample':row.get('sample')} for row in rows if row.get('reward') is None],
                     **({'deferred_long_graphs':deferred} if deferred is not None else {}),
                     },
        'episodes': len(graphs), 'sampled_tokens': sum(len(row['actions']) for row in objective['rows'].values()),
        'neuralese_bonus_episodes': sum(bool(row.get('neuralese_bonus')) for row in objective['rows'].values()),
        'native_graphs_sha256': sha256_file_hex_cached(graph_path),
        'objective_sha256': sha256_file_hex_cached(objective_path), 'execution': 'one-packed-graph-forward'})
    if args.coefficient_policy is not None:
        if args.publication_owner_pointer is not None:
            raise ValueError('coefficient-policy RL publication requires its own exact policy-writer boundary')
        from .native_coefficient_stage import main as train_coefficients
        coefficient_args=['--coefficient-policy',str(args.coefficient_policy),'--graphs',str(graph_path),
            '--objective',str(objective_path),'--continue-from',str(args.continue_from),
            '--held-native-graphs',str(args.held_native_graphs),'--out',str(args.out),*trainer]
        return train_coefficients(coefficient_args)
    from ..data.native_sources import NativeGraphSource
    from .native_replay import NativePolicyObjective
    from ..eval.native_memory import policy_capacity_request
    policy_objective=NativePolicyObjective(objective_path)
    source=NativeGraphSource(graph_path,policy_objective.identity['tokenizer_fingerprint'],required_split='train')
    request_path=args.out/'rl-policy-capacity-request.json'
    request=policy_capacity_request(source,policy_objective)
    write_json_durable(request_path,request)
    from ..train.native_resident import binding
    trainer += ['--policy-capacity-request',json.dumps(binding(request_path),sort_keys=True)]
    from ..train.native_state_sft import main as train
    trainer_args=['--native-graphs', str(graph_path), '--held-native-graphs', str(args.held_native_graphs),
        '--objective-spec', str(objective_path), '--continue-from', str(args.learner_resume['path'] if getattr(args, 'learner_resume', None) else args.continue_from),
        '--objective-continuation', '--out', str(args.out), *trainer]
    if transition_path is not None:
        trainer_args += ['--policy-source-continuation', str(transition_path)]
    from .diloco_contract import register_admitted_stage
    register_admitted_stage(args,trainer_args)
    # The trainer reads the admitted graphs/objective back from disk; release the preparation copies (parsed
    # rollouts, graphs, objective: several GB of Python objects) so they do not sit in host memory, which on
    # unified memory is also GPU memory, for the whole update.
    del rows, graphs, objective, policy_objective, source, request
    _release_host_memory()
    result=_invoke_raw_trainer(args, trainer, trainer_args, train)
    if result==0 and operation_request is not None and args.coefficient_policy is None:
        from ..train.native_resident import binding
        report_path=args.out/'report.json';receipt_path=args.out/'native-training-receipt.json'
        checkpoint_path=args.out/'checkpoint.pt'
        data_path=args.out/'rl-data-receipt.json';objective_path=args.out/'rl-native-objective.json'
        graph_path=args.out/'rl-native-graphs.jsonl'
        if all(path.is_file() for path in (report_path,receipt_path,checkpoint_path,data_path,objective_path,graph_path)):
            report=json.loads(report_path.read_text());training_receipt=json.loads(receipt_path.read_text())
            checkpoint=binding(checkpoint_path)
            parent_sha=operation_request['input_bindings']['continue_from']['files'][0]['sha256']
            learner_parent=operation_request['input_bindings'].get('learner_resume')
            boundary=report.get('continuation_boundary') or {}
            if (report.get('completed') is True and report.get('checkpoint_sha256')==checkpoint['sha256']
                    and boundary.get('new_optimizer_updates',0)>0
                    and training_receipt.get('checkpoint',{}).get('sha256')==checkpoint['sha256']
                    and training_receipt.get('continuation',{}).get('checkpoint_sha256')==parent_sha
                    and (learner_parent is None or (training_receipt.get('operation_parent') or {}).get('sha256')==learner_parent['sha256'])):
                payload={key:value for key,value in operation_request.items() if key!='request_sha256'}
                write_json_durable(args.out/'native-rl-completed-operation.json',{
                    'schema':'natlang.native-rl-completed-operation/1','request':operation_request,
                    'request_sha256':canonical_json_sha256_hex(payload),
                    'report':binding(report_path),'training_receipt':binding(receipt_path),
                    'checkpoint':checkpoint,'data_receipt':binding(data_path),
                    'objective':binding(objective_path),'graphs':binding(graph_path)})
    return result


def _release_host_memory():
    import ctypes, gc
    gc.collect()
    try:
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    except OSError:
        pass


def _defer_long_graphs(args, graphs, objective, *, share=True):
    """Apply --max-graph-forward-tokens to the admitted graphs in place (before they are written): keep deferred
    episodes whole beside the round, return the record.

    The limit applies to the forward tokens the learner actually runs: the exact causal-prefix packing of the
    episode's turns (on by default) when the objective supports it (NativePolicyObjective.shares_causal_prefixes:
    no per-invocation behavior actors, no archived external states), else every turn's full rendered rows."""
    limit = args.max_graph_forward_tokens
    if limit is None:
        return None
    if type(limit) is not int or limit < 1:
        raise ValueError('--max-graph-forward-tokens must be a positive integer')
    from ..data.native_graph_schema import parse_graph
    from ..train.native_prefix_packing import causal_prefix_geometry
    behavior = objective.get('schema') == 'natlang.native-policy-objective/2'
    fingerprint = (objective['base_native_identity'] if behavior else objective['policy'])['tokenizer_fingerprint']
    sizes = {}
    for row in graphs:
        example = parse_graph(row, tokenizer_fingerprint=fingerprint)
        supported = share and not behavior and not getattr(example, 'external_states', None)
        sizes[example.id] = (causal_prefix_geometry(example, share=supported)[0], example.plan.token_count)
    long = {key for key, (size, _) in sizes.items() if size > limit}
    record = {'reason': 'packed episode graph exceeds the learner forward-token limit',
              'max_graph_forward_tokens': limit, 'path': None, 'share_causal_prefixes': share,
              'episodes': [{'id': key, 'task_id': objective['rows'][key].get('task_id'),
                            'sample': objective['rows'][key].get('sample'), 'forward_tokens': sizes[key][0],
                            'original_tokens': sizes[key][1],
                            'sampled_tokens': len(objective['rows'][key]['actions'])} for key in sorted(long)]}
    if not long:
        return record
    deferred_path = args.out / 'deferred-long-graphs.jsonl'
    write_bytes_durable(deferred_path, ''.join(json.dumps({'graph': row, 'objective_row': objective['rows'][row['id']]})
                                               + '\n' for row in graphs if row['id'] in long).encode())
    record['path'] = {'path': str(deferred_path.resolve()), 'sha256': sha256_file_hex_cached(deferred_path)}
    graphs[:] = [row for row in graphs if row['id'] not in long]
    objective['rows'] = {key: row for key, row in objective['rows'].items() if key not in long}
    return record


def _publication_values(args, trainer):
    """Build the exact selected publication contract from forwarded trainer flags."""
    if args.publication_owner_pointer is None:
        if args.publication_role is not None:
            raise ValueError('publication role requires an owner pointer')
        return None
    if args.publication_role is None:
        raise ValueError('owner publication requires its exact participant role')
    def one(flag, *, required=True):
        found=[index for index,value in enumerate(trainer) if value==flag]
        if not found:
            if required:raise ValueError('owner publication trainer omitted '+flag)
            return None
        if len(found)!=1 or found[0]+1>=len(trainer):
            raise ValueError('owner publication trainer has ambiguous '+flag)
        return trainer[found[0]+1]
    if '--checkpoint-publication' not in trainer:
        raise ValueError('owner publication trainer disabled checkpoint publication')
    from ..model.native_checkpoint_control import publication_selection, validate_publication_selection
    namespace=one('--checkpoint-publication-namespace')
    roots=json.loads(one('--checkpoint-publication-readonly-input-roots'))
    if not isinstance(roots,list) or not all(isinstance(value,str) for value in roots):
        raise ValueError('owner publication roots must be an explicit JSON string list')
    selection=publication_selection(namespace=namespace,readonly_input_roots=roots)
    # The verifier canonicalizes roots internally; the owner manifest must still
    # preserve the exact list forwarded to the real, unchanged trainer command.
    return validate_publication_selection({**selection,'readonly_input_roots':roots})


def _invoke_raw_trainer(args, forwarded_trainer_args, actual_trainer_args, train):
    """Publish the true inner SFT call immediately before invoking its writer."""
    selection=_publication_values(args,forwarded_trainer_args)
    if selection is not None:
        import sys
        from .native_publication_owner import emit_current
        module='natlang_neuralese.train.native_state_sft'
        command=[sys.executable,'-u','-m',module,*actual_trainer_args]
        job_output=Path(args.out).resolve().parent
        config={'native_participant':args.publication_role,
            'native_checkpoint_publication_owner_pointer':str(args.publication_owner_pointer),
            'native_checkpoint_publication':selection,
            'runtime':str(Path(__file__).resolve().parents[4]),
            'production_stage':'native_rl', 'request_id':None}
        emit_current(config,command,out=job_output,selected_checkpoint=(args.learner_resume['path'] if getattr(args, 'learner_resume', None) else args.continue_from),
            runtime_root=config['runtime'])
    return train(actual_trainer_args)


if __name__ == '__main__':
    raise SystemExit(main())
