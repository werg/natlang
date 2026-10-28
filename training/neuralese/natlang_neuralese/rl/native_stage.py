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
    renderer = Renderer(tokenizer, SimpleNamespace(open_id=roles.produce_open, close_id=roles.produce_close))
    rendering = PrivilegedRendering.from_renderer(renderer, template=tokenizer_model_type(tokenizer),
                                                  tokenizer_fingerprint=tokenizer_fingerprint,
                                                  consume_open=roles.consume_open, consume_close=roles.consume_close)
    return privileged_source(rows, cases=worlds, rendering=rendering, custody=custody, fetch_text=fetch_text)


def main(argv=None):
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
    parser.add_argument('--foundation-readiness-policy', choices=('at-boundary', 'on-shared-request'))
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
    args, trainer = parser.parse_known_args(argv)
    from contextlib import nullcontext
    from .native_output_lease import coefficient_output_lease
    with coefficient_output_lease(args.out) if args.coefficient_policy is not None else nullcontext():
        return _admit_and_train(args,trainer)


def _admit_and_train(args,trainer):
    from ..common.jsonio import iter_jsonl_rows
    rows = [row for path in args.rollouts for _, row in iter_jsonl_rows(path)]
    graphs, objective = prepare(rows, checkpoint=args.continue_from, tokenizer=args.rl_tokenizer,
        bundle=args.rl_block_bundle, mode=args.rl_mode,coefficient_policy=args.coefficient_policy,
        expected_history_reasoning_policy=args.history_reasoning_policy,
        expected_history_binding_sha256=args.history_reasoning_binding_sha256,
        expected_history_binding_path=args.history_reasoning_binding_path,
        policy_loss=args.rl_policy_loss, jolt_cases=args.jolt_cases, actors=args.rl_actors)
    if not graphs:
        raise ValueError('no useful native policy episodes: route flat failures to verified graph SFT seeding')
    args.out.mkdir(parents=True, exist_ok=True)
    graph_path, objective_path = args.out/'rl-native-graphs.jsonl', args.out/'rl-native-objective.json'
    write_bytes_durable(graph_path, ''.join(json.dumps(row) + '\n' for row in graphs).encode())
    write_json_durable(objective_path, objective)
    write_json_durable(args.out/'rl-data-receipt.json', {
        'schema': 'natlang.native-rl-admission/1',
        **({'base_native_identity':objective['base_native_identity'],'behavior_policy_sha256':objective['behavior_policy_sha256']}
           if args.coefficient_policy is not None else {'policy':objective['policy']}), 'mode': args.rl_mode,
        **({'policy_loss': objective['policy_loss']} if 'policy_loss' in objective else {}),
        'rollouts': {str(path.resolve()): sha256_file_hex_cached(path) for path in args.rollouts},
        'tasks_selected': [{'task_id': row['task_id'], 'split': 'train'} for row in objective['rows'].values()],
        'omissions':{'unscored':[{'task_id':row.get('task_id'),'sample':row.get('sample')} for row in rows if row.get('reward') is None]},
        'episodes': len(graphs), 'sampled_tokens': sum(len(row['actions']) for row in objective['rows'].values()),
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
    if args.foundation_readiness_policy is not None:
        trainer += ['--foundation-readiness-policy', args.foundation_readiness_policy]
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
        '--objective-spec', str(objective_path), '--continue-from', str(args.continue_from),
        '--objective-continuation', '--out', str(args.out), *trainer]
    return _invoke_raw_trainer(args, trainer, trainer_args, train)


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
        emit_current(config,command,out=job_output,selected_checkpoint=args.continue_from,
            runtime_root=config['runtime'])
    return train(actual_trainer_args)


if __name__ == '__main__':
    raise SystemExit(main())
