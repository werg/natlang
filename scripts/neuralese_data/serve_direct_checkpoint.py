#!/usr/bin/env python3
"""Serve an exact direct-stage training checkpoint for train-only task rollouts.

This uses the same restore_engine path as the in-run task evaluator. It is an
experiment entry point, not an export or a qualification of the checkpoint.
The caller supplies immutable hashes to avoid accidentally serving a newer
rotating checkpoint or mismatched clean heads.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'training/neuralese'))

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.eval.task_cadence import restore_engine
from natlang_neuralese.serve.http import serve


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--checkpoint-sha256', required=True)
    parser.add_argument('--heads', type=Path)
    parser.add_argument('--base', type=Path, help='relocated pinned base; same inventory required')
    parser.add_argument('--heads-sha256')
    parser.add_argument('--native-state', action='store_true', help='serve the native-state checkpoint without legacy heads')
    parser.add_argument('--native-participant', help='named model within a shared native checkpoint')
    parser.add_argument('--semantic-adapter', type=Path, help='checkpoint-bound shared semantic adapter receipt')
    parser.add_argument('--semantic-adapter-sha256')
    parser.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='full')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--neuralese', action='store_true', help='restore the mixed checkpoint writer/readout too')
    parser.add_argument('--sdpa-backend', choices=['default', 'flash', 'math'], default='default')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    if args.native_state:
        if args.heads is not None or args.heads_sha256 is not None:
            parser.error('native-state actors restore native checkpoints without legacy heads')
    elif args.heads is None or args.heads_sha256 is None:
        parser.error('direct diagnostic actors require --heads and --heads-sha256')
    if not args.native_state and (args.native_participant or args.semantic_adapter):
        parser.error('native participants and semantic adapters require --native-state')
    if bool(args.semantic_adapter) != bool(args.semantic_adapter_sha256):
        parser.error('semantic adapter needs its exact --semantic-adapter-sha256')

    import torch

    torch.set_num_threads(args.threads)
    from natlang_neuralese.common.attention_backend import configure_sdpa_backend
    configure_sdpa_backend(args.sdpa_backend)
    checkpoint = args.checkpoint.resolve(strict=True)
    heads = args.heads.resolve(strict=True) if args.heads else None
    bindings = [('checkpoint', checkpoint, args.checkpoint_sha256)]
    if heads:
        bindings.append(('heads', heads, args.heads_sha256))
    if args.semantic_adapter:
        bindings.append(('semantic adapter', args.semantic_adapter.resolve(strict=True), args.semantic_adapter_sha256))
    for label, path, expected in bindings:
        actual = sha256_file_hex(path)
        if actual != expected:
            raise ValueError(f'{label} SHA-256 mismatch: {actual} != {expected}')

    with tempfile.TemporaryDirectory(prefix='natlang-direct-serve-') as temp:
        if args.native_state:
            from natlang_neuralese.serve import load_native_engine
            engine = load_native_engine(str(checkpoint), base=str(args.base) if args.base else None,
                device=args.device, prompt_rendering=args.prompt_rendering,
                participant=args.native_participant,
                semantic_adapter=str(args.semantic_adapter) if args.semantic_adapter else None)
        else:
            (Path(temp) / 'request.json').write_text(json.dumps({'weights': str(checkpoint)}))
            engine = restore_engine({'dir': temp, 'heads': str(heads), 'device': args.device,
                                     'prompt_rendering': args.prompt_rendering, 'neuralese': args.neuralese,
                                     'base': str(args.base) if args.base else None})
            engine.rl_policy_binding = {'checkpoint_sha256': args.checkpoint_sha256,
                                        'heads_sha256': args.heads_sha256, 'neuralese': args.neuralese}
        engine.start()
        server = serve(engine, args.host, args.port)
        print(json.dumps({'event': 'direct_training_checkpoint_serving',
                          'checkpoint_sha256': args.checkpoint_sha256,
                          'heads_sha256': args.heads_sha256,
                          'native_state': args.native_state,
                          'rl_policy_binding': engine.rl_policy_binding,
                          'neuralese_binding': getattr(engine, 'neuralese_checkpoint_binding', None),
                          'prompt_rendering': args.prompt_rendering,
                          'listening': f'http://{server.server_address[0]}:{server.server_address[1]}'}),
              flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
            engine.stop()


if __name__ == '__main__':
    main()
