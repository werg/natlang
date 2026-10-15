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


def prepare(rollouts, *, checkpoint, tokenizer, bundle=None, mode='grpo'):
    from transformers import AutoTokenizer
    from ..model.shared_state import RoleTokens, ROLE_MARKERS
    from ..data.text_corpus import tokenizer_fingerprint
    from ..serve.store import decode_native_state
    from .native_replay import prepare_objective
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    fingerprint = tokenizer_fingerprint(tokenizer)
    policies = [turn['sampling']['policy'] for row in rollouts if row.get('reward') is not None
                for turn in row.get('turns', []) if (turn.get('sampling') or {}).get('policy')]
    if not policies or any(policy != policies[0] for policy in policies):
        raise ValueError('native RL requires one exact captured actor identity')
    identity = policies[0]
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
    parser.add_argument('--continue-from', type=Path, required=True)
    parser.add_argument('--held-native-graphs', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args, trainer = parser.parse_known_args(argv)
    from ..common.jsonio import iter_jsonl_rows
    rows = [row for path in args.rollouts for _, row in iter_jsonl_rows(path)]
    graphs, objective = prepare(rows, checkpoint=args.continue_from, tokenizer=args.rl_tokenizer,
        bundle=args.rl_block_bundle, mode=args.rl_mode)
    if not graphs:
        raise ValueError('no useful native policy episodes: route flat failures to verified graph SFT seeding')
    args.out.mkdir(parents=True, exist_ok=True)
    graph_path, objective_path = args.out/'rl-native-graphs.jsonl', args.out/'rl-native-objective.json'
    write_bytes_durable(graph_path, ''.join(json.dumps(row) + '\n' for row in graphs).encode())
    write_json_durable(objective_path, objective)
    write_json_durable(args.out/'rl-data-receipt.json', {
        'schema': 'natlang.native-rl-admission/1', 'policy': objective['policy'], 'mode': args.rl_mode,
        'rollouts': {str(path.resolve()): sha256_file_hex_cached(path) for path in args.rollouts},
        'tasks_selected': [{'task_id': row['task_id'], 'split': 'train'} for row in objective['rows'].values()],
        'episodes': len(graphs), 'sampled_tokens': sum(len(row['actions']) for row in objective['rows'].values()),
        'native_graphs_sha256': sha256_file_hex_cached(graph_path),
        'objective_sha256': sha256_file_hex_cached(objective_path), 'execution': 'one-packed-graph-forward'})
    from ..train.native_state_sft import main as train
    return train(['--native-graphs', str(graph_path), '--held-native-graphs', str(args.held_native_graphs),
        '--objective-spec', str(objective_path), '--continue-from', str(args.continue_from),
        '--objective-continuation', '--out', str(args.out), *trainer])


if __name__ == '__main__':
    raise SystemExit(main())
