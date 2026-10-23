#!/usr/bin/env python3
"""Serve one exact native shared-state checkpoint as a rollout actor."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.http import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--base', type=Path, help='exact source snapshot for native restoration')
    parser.add_argument('--native-participant', help='named model within a shared native checkpoint')
    parser.add_argument('--native-actor-view', type=Path, help='optional exact source-bound compact actor receipt')
    parser.add_argument('--native-actor-view-sha256')
    parser.add_argument('--semantic-adapter', type=Path, help='checkpoint-bound shared semantic adapter receipt')
    parser.add_argument('--semantic-adapter-sha256')
    parser.add_argument('--coefficient-policy', type=Path)
    parser.add_argument('--coefficient-policy-sha256')
    parser.add_argument('--coefficient-rebase-out', type=Path)
    parser.add_argument('--coefficient-cancel-stale-reason')
    parser.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='full')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--sdpa-backend', choices=['default', 'flash', 'math'], default='default')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--store-dir', type=Path, help='durable actor block store retained across interrupted leases')
    args = parser.parse_args()
    if bool(args.native_actor_view) != bool(args.native_actor_view_sha256):
        parser.error('native actor view requires its exact receipt SHA-256')
    if bool(args.semantic_adapter) != bool(args.semantic_adapter_sha256):
        parser.error('semantic adapter requires its exact SHA-256')
    if bool(args.coefficient_policy) != bool(args.coefficient_policy_sha256):
        parser.error('coefficient policy requires its exact SHA-256')

    import torch

    torch.set_num_threads(args.threads)
    from natlang_neuralese.common.attention_backend import configure_sdpa_backend
    configure_sdpa_backend(args.sdpa_backend)
    checkpoint = args.checkpoint.resolve(strict=True)
    bindings = [('checkpoint', checkpoint, args.checkpoint_sha256)]
    if args.semantic_adapter:
        bindings.append(('semantic adapter', args.semantic_adapter.resolve(strict=True), args.semantic_adapter_sha256))
    if args.coefficient_policy:
        bindings.append(('coefficient policy', args.coefficient_policy.resolve(strict=True), args.coefficient_policy_sha256))
    if args.native_actor_view:
        bindings.append(('actor view', args.native_actor_view.resolve(strict=True), args.native_actor_view_sha256))
    for label, path, expected in bindings:
        actual = sha256_file_hex_cached(path)
        if actual != expected:
            raise ValueError(f'{label} SHA-256 mismatch: {actual} != {expected}')

    view_options = ({'actor_view': {'path': str(args.native_actor_view.resolve(strict=True)),
        'sha256': args.native_actor_view_sha256, 'bytes': args.native_actor_view.stat().st_size}}
        if args.native_actor_view else {})
    engine = load_native_engine(str(checkpoint), base=str(args.base) if args.base else None,
        device=args.device, prompt_rendering=args.prompt_rendering, participant=args.native_participant,
        semantic_adapter=str(args.semantic_adapter) if args.semantic_adapter else None, **view_options)
    if args.store_dir is not None:
        from natlang_neuralese.serve.store import TensorStore
        engine.store = TensorStore(args.store_dir)
    if args.coefficient_policy:
        from natlang_neuralese.model.native_coefficient_policy import policy_core, rebase_policy
        from natlang_neuralese.rl.native_coefficient_stage import bind_policy
        source = args.coefficient_policy.resolve(strict=True)
        source_core = policy_core(json.loads(source.read_text()))
        actual = source
        if source_core['base_native_identity'] != engine.native_identity:
            if args.coefficient_rebase_out is None:
                raise ValueError('coefficient policy changed base; explicitly configure its preserved-value rebase output')
            actual = rebase_policy(engine, source, args.coefficient_rebase_out,
                                   cancel_stale_reason=args.coefficient_cancel_stale_reason)
        bind_policy(engine, actual)
        engine.rl_policy_binding['coefficient_policy_binding'] = {
            'path': str(actual), 'sha256': sha256_file_hex_cached(actual)}
        engine.rl_policy_binding['coefficient_policy_source'] = {
            'path': str(source), 'sha256': sha256_file_hex_cached(source)}
    engine.start()
    server = serve(engine, args.host, args.port)
    print(json.dumps({'event': 'native_training_checkpoint_serving',
        'native_state': True,
        'checkpoint_sha256': args.checkpoint_sha256,
        'rl_policy_binding': engine.rl_policy_binding,
        'prompt_rendering': args.prompt_rendering,
        'listening': f'http://{server.server_address[0]}:{server.server_address[1]}'}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.stop()


if __name__ == '__main__':
    main()
