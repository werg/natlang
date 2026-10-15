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
    parser.add_argument('--heads', type=Path, required=True)
    parser.add_argument('--base', type=Path, help='relocated pinned base; same inventory required')
    parser.add_argument('--heads-sha256', required=True)
    parser.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='full')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--neuralese', action='store_true', help='restore the mixed checkpoint writer/readout too')
    parser.add_argument('--sdpa-backend', choices=['default', 'flash', 'math'], default='default')
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    from natlang_neuralese.common.attention_backend import configure_sdpa_backend
    configure_sdpa_backend(args.sdpa_backend)
    checkpoint = args.checkpoint.resolve(strict=True)
    heads = args.heads.resolve(strict=True)
    for label, path, expected in (
        ('checkpoint', checkpoint, args.checkpoint_sha256),
        ('heads', heads, args.heads_sha256),
    ):
        actual = sha256_file_hex(path)
        if actual != expected:
            raise ValueError(f'{label} SHA-256 mismatch: {actual} != {expected}')

    with tempfile.TemporaryDirectory(prefix='natlang-direct-serve-') as temp:
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
