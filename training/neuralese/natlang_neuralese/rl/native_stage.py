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


def prepare(rollouts, *, checkpoint, tokenizer, bundle=None, mode='grpo', coefficient_policy=None):
    from transformers import AutoTokenizer
    from ..model.shared_state import RoleTokens, ROLE_MARKERS
    from ..data.text_corpus import tokenizer_fingerprint
    from ..serve.store import decode_native_state
    from .native_replay import prepare_objective
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    fingerprint = tokenizer_fingerprint(tokenizer)
    if coefficient_policy is not None:
        from ..model.native_coefficient_policy import policy_core
        core=policy_core(json.loads(Path(coefficient_policy).read_text()))
        identity=core['base_native_identity']
        if identity['checkpoint_sha256']!=sha256_file_hex_cached(checkpoint) or identity['tokenizer_fingerprint']!=fingerprint:
            raise ValueError('coefficient-policy admission changed the exact base checkpoint or tokenizer')
        marker_ids=tokenizer.convert_tokens_to_ids(list(ROLE_MARKERS))
        if len(set(marker_ids))!=4 or any(value==tokenizer.unk_token_id for value in marker_ids):
            raise ValueError('native RL tokenizer lacks its four distinct role markers')
        if bundle is None:raise ValueError('coefficient-policy admission requires its actual collector bundle')
        from .graph_replay import verified_bundle_bytes
        from .native_replay import invocation_behavior_policy
        from ..common.hashing import canonical_json_sha256_hex
        source=Path(bundle).resolve(strict=True)
        original=json.loads(source.read_text())
        if original.get('schema')!='natlang.rl-block-bundle/1' or original.get('policy_id')!=identity['checkpoint_sha256']:
            raise ValueError('coefficient-policy collector bundle changed its actual base-runtime owner')
        def fetch(block_id):
            raw=verified_bundle_bytes(source,block_id,policy_id=identity['checkpoint_sha256'])
            if raw is None:raise ValueError('coefficient-policy collector bundle omitted an actual raw observation')
            block,actual=decode_native_state(raw)
            if actual!=block_id:raise ValueError('coefficient-policy collector archive changed native content identity')
            return block
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
                                 block_bundle=destination,mode=mode,behavior_policy=behavior)
    policies = [turn['sampling']['policy'] for row in rollouts if row.get('reward') is not None
                for turn in row.get('turns', []) if (turn.get('sampling') or {}).get('policy')]
    if not policies or any(policy != policies[0] for policy in policies):
        raise ValueError('native RL requires one exact captured actor identity')
    identity = policies[0]
    if any((turn.get('sampling') or {}).get('adapter_binding') for row in rollouts for turn in row.get('turns', [])):
        raise ValueError('adapted actors require native coefficient-policy learning; full-backbone trainer export/rebinding is not yet supported')
    if (identity['checkpoint_sha256'] != sha256_file_hex_cached(checkpoint)
            or identity['tokenizer_fingerprint'] != fingerprint):
        raise ValueError('native RL checkpoint/tokenizer differs from actual sampled policy')
    marker_ids = tokenizer.convert_tokens_to_ids(list(ROLE_MARKERS))
    if len(set(marker_ids)) != 4 or any(value == tokenizer.unk_token_id for value in marker_ids):
        raise ValueError('native RL tokenizer lacks its four distinct role markers')
    roles = RoleTokens(*marker_ids)
    archive = json.loads(Path(bundle).read_text()) if bundle else None
    if archive and (archive.get('schema') != 'natlang.rl-block-bundle/1' or
                    archive.get('policy_id') != identity['checkpoint_sha256']):
        raise ValueError('native RL archive does not bind the sampled policy')
    def fetch(block_id):
        if archive is None:
            raise ValueError('native RL state requires its durable actor bundle')
        entry = archive['blocks'][block_id]
        root = Path(bundle).resolve().parent
        path = (root / entry['path']).resolve(strict=True)
        if (path.parent != root or path.stat().st_size != entry['bytes']
                or sha256_file_hex_cached(path) != entry['sha256']):
            raise ValueError('native RL archive path/size/hash changed')
        block, actual = decode_native_state(path.read_bytes())
        if actual != block_id:
            raise ValueError('native RL archive content identity changed')
        return block
    return prepare_objective(rollouts, identity=identity, roles=roles, fetch_block=fetch,
        block_bundle=Path(bundle).resolve() if bundle else None, mode=mode)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollouts', action='append', type=Path, required=True)
    parser.add_argument('--rl-tokenizer', type=Path, required=True)
    parser.add_argument('--rl-mode', choices=('grpo', 'rft'), default='grpo')
    parser.add_argument('--rl-block-bundle', type=Path)
    parser.add_argument('--coefficient-policy',type=Path)
    parser.add_argument('--continue-from', type=Path, required=True)
    parser.add_argument('--held-native-graphs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    # These describe the parent's supervised schedule, not sampled policy rows.
    # Mandatory CE/foundation credits remain in its carried continuation frame.
    parser.add_argument('--communication-foundation', action='store_true')
    parser.add_argument('--foundation-native-graphs', type=Path)
    args, trainer = parser.parse_known_args(argv)
    from ..common.jsonio import iter_jsonl_rows
    rows = [row for path in args.rollouts for _, row in iter_jsonl_rows(path)]
    graphs, objective = prepare(rows, checkpoint=args.continue_from, tokenizer=args.rl_tokenizer,
        bundle=args.rl_block_bundle, mode=args.rl_mode,coefficient_policy=args.coefficient_policy)
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
        'rollouts': {str(path.resolve()): sha256_file_hex_cached(path) for path in args.rollouts},
        'tasks_selected': [{'task_id': row['task_id'], 'split': 'train'} for row in objective['rows'].values()],
        'omissions':{'unscored':[{'task_id':row.get('task_id'),'sample':row.get('sample')} for row in rows if row.get('reward') is None]},
        'episodes': len(graphs), 'sampled_tokens': sum(len(row['actions']) for row in objective['rows'].values()),
        'native_graphs_sha256': sha256_file_hex_cached(graph_path),
        'objective_sha256': sha256_file_hex_cached(objective_path), 'execution': 'one-packed-graph-forward'})
    if args.coefficient_policy is not None:
        from .native_coefficient_stage import main as train_coefficients
        return train_coefficients(['--coefficient-policy',str(args.coefficient_policy),'--graphs',str(graph_path),
            '--objective',str(objective_path),'--continue-from',str(args.continue_from),
            '--held-native-graphs',str(args.held_native_graphs),'--out',str(args.out),*trainer])
    from ..train.native_state_sft import main as train
    return train(['--native-graphs', str(graph_path), '--held-native-graphs', str(args.held_native_graphs),
        '--objective-spec', str(objective_path), '--continue-from', str(args.continue_from),
        '--objective-continuation', '--out', str(args.out), *trainer])


if __name__ == '__main__':
    raise SystemExit(main())
