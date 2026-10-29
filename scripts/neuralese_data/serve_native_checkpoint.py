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
    parser.add_argument('--transport-receipt',type=Path)
    parser.add_argument('--transport-receipt-sha256')
    parser.add_argument('--transport-view-receipt',type=Path)
    parser.add_argument('--transport-view-receipt-sha256')
    parser.add_argument('--transport-recovery-receipt',type=Path)
    parser.add_argument('--transport-recovery-receipt-sha256')
    parser.add_argument('--training-source-sha256')
    parser.add_argument('--transport-scope',choices=('independent-recovery','diagnostic-local-pin'))
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
    parser.add_argument('--engine', choices=('native', 'sglang'), default='native',
                        help='sglang: ordinary-token turns generate on the SGLang fork, the rest on the native engine (serve/sglang_actor.py)')
    parser.add_argument('--sglang-python', help="the fork's Python (default: training/neuralese/sglang-fork.json python)")
    parser.add_argument('--sglang-port', type=int, help='port of the SGLang server (default: --port + 1000)')
    # SGLang's pools are sized from need: requests x context, radix-shared; the memory fraction is only a ceiling.
    parser.add_argument('--sglang-max-running', type=int, default=32, help='concurrent sequences on SGLang')
    parser.add_argument('--sglang-context', type=int, default=32768, help='longest prompt + reply in tokens')
    parser.add_argument('--sglang-max-total-tokens', type=int,
                        help='KV pool tokens shared by all sequences (default: 4 x --sglang-context; prompts share prefixes)')
    parser.add_argument('--sglang-mem-fraction', type=float, default=0.3, help='ceiling, not the allocation')
    parser.add_argument('--sglang-arg', action='append', default=[], help='extra sglang.launch_server argument')
    parser.add_argument('--hf-export-dir', type=Path, help='HF export cache (default: <store-dir or checkpoint dir>/sglang-hf/<sha>)')
    args = parser.parse_args()
    if args.engine == 'sglang' and (args.semantic_adapter or args.coefficient_policy or args.transport_receipt):
        parser.error('the SGLang actor serves ordinary checkpoints only (no semantic adapter, coefficient policy or transport view)')
    if bool(args.native_actor_view) != bool(args.native_actor_view_sha256):
        parser.error('native actor view requires its exact receipt SHA-256')
    transport_values=(args.transport_receipt,args.transport_receipt_sha256,args.transport_view_receipt,
        args.transport_view_receipt_sha256,args.training_source_sha256,args.transport_scope)
    recovery_values=(args.transport_recovery_receipt,args.transport_recovery_receipt_sha256)
    if any((*transport_values,*recovery_values)) and (not all(transport_values) or args.native_actor_view or args.native_participant
            or (args.transport_scope=='independent-recovery' and not all(recovery_values))
            or (args.transport_scope=='diagnostic-local-pin' and any(recovery_values))):
        parser.error('transported actor requires all exact receipt bindings and no full-source actor view or participant')
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
    if args.transport_receipt:
        bindings.extend([('transport receipt',args.transport_receipt.resolve(strict=True),args.transport_receipt_sha256),
            ('transport view receipt',args.transport_view_receipt.resolve(strict=True),args.transport_view_receipt_sha256)])
        if args.transport_recovery_receipt:
            bindings.append(('transport recovery receipt',args.transport_recovery_receipt.resolve(strict=True),args.transport_recovery_receipt_sha256))
    for label, path, expected in bindings:
        actual = sha256_file_hex_cached(path)
        if actual != expected:
            raise ValueError(f'{label} SHA-256 mismatch: {actual} != {expected}')

    view_options = ({'actor_view': {'path': str(args.native_actor_view.resolve(strict=True)),
        'sha256': args.native_actor_view_sha256, 'bytes': args.native_actor_view.stat().st_size}}
        if args.native_actor_view else {})
    if args.transport_receipt:
        def ref(path,sha):return {'path':str(path.resolve(strict=True)),'sha256':sha,'bytes':path.stat().st_size}
        view_options={'transported_actor':{'transport':ref(args.transport_receipt,args.transport_receipt_sha256),
            'view_receipt':ref(args.transport_view_receipt,args.transport_view_receipt_sha256),
            'recovery_receipt':ref(args.transport_recovery_receipt,args.transport_recovery_receipt_sha256) if args.transport_recovery_receipt else None,
            'source_sha256':args.training_source_sha256,'scope':args.transport_scope}}
    if args.engine == 'sglang':
        from natlang_neuralese.serve.sglang_actor import SglangEngine
        view_options['engine_class'] = SglangEngine
    engine = load_native_engine(str(checkpoint), base=str(args.base) if args.base else None,
        prompt_rendering=args.prompt_rendering, participant=args.native_participant,
        semantic_adapter=str(args.semantic_adapter) if args.semantic_adapter else None, device=args.device,
        **view_options)
    if args.engine == 'sglang':
        launch_sglang(engine, args, checkpoint)
        # The native model is loaded again only when a turn SGLang cannot serve arrives (serve/sglang_actor.py).
        engine.release_native({'checkpoint': str(checkpoint), 'base': str(args.base) if args.base else None,
                               'device': args.device, 'participant': args.native_participant,
                               **({'actor_view': view_options['actor_view']} if 'actor_view' in view_options else {})})
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
        'checkpoint_sha256': args.training_source_sha256 if args.transport_receipt else args.checkpoint_sha256,
        **({'transport_view_sha256':args.checkpoint_sha256,'inference_only':True} if args.transport_receipt else {}),
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


def launch_sglang(engine, args, checkpoint):
    """Export the checkpoint for SGLang (cached by SHA-256), start the fork's server and bind the front to it."""
    import os
    import subprocess
    import time
    import urllib.request
    from natlang_neuralese.export.native_hf import export_loaded
    root = Path(__file__).resolve().parents[2]
    fork = json.loads((root / 'training/neuralese/sglang-fork.json').read_text())
    python = args.sglang_python or fork['python']
    export_dir = args.hf_export_dir or ((args.store_dir or checkpoint.parent) / 'sglang-hf' / args.checkpoint_sha256[:16])
    manifest = export_loaded(engine.shared_model, engine.tokenizer, engine.role_tokens, engine.model_config,
                             checkpoint=checkpoint, checkpoint_sha256=args.checkpoint_sha256, out_dir=export_dir)
    port = args.sglang_port or args.port + 1000
    context = min(args.sglang_context, engine.context)
    total = args.sglang_max_total_tokens or 4 * context
    if total < context:
        raise ValueError('--sglang-max-total-tokens must hold one full-context sequence')
    command = [python, '-m', 'sglang.launch_server', '--model-path', str(export_dir), '--host', '127.0.0.1',
               '--port', str(port), '--dtype', 'bfloat16', '--mem-fraction-static', str(args.sglang_mem_fraction),
               '--context-length', str(context), '--max-total-tokens', str(total),
               '--max-running-requests', str(args.sglang_max_running),
               '--max-mamba-cache-size', str(8 * args.sglang_max_running),
               '--cuda-graph-max-bs-decode', str(args.sglang_max_running),
               '--skip-tokenizer-init', '--trust-remote-code', *args.sglang_arg]
    print(json.dumps({'event': 'sglang_launch', 'command': command, 'export': manifest}), flush=True)
    # FlashInfer JIT-compiles kernels: the fork venv's ninja and the CUDA toolkit's nvcc on PATH.
    cuda = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    path = os.pathsep.join([str(Path(python).parent), cuda + '/bin', os.environ.get('PATH', '')])
    process = subprocess.Popen(command, env={**os.environ, 'PYTHONPATH': '', 'PATH': path, 'CUDA_HOME': cuda},
                               start_new_session=True)
    # The SGLang server (and its scheduler/detokenizer children) ends with this actor, including on SIGTERM.
    import atexit
    import signal

    def stop_sglang(*_):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
    atexit.register(stop_sglang)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    url = f'http://127.0.0.1:{port}'
    deadline = time.time() + 1800
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'SGLang server exited with {process.returncode} before becoming healthy')
        try:
            with urllib.request.urlopen(url + '/health_generate', timeout=60) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        if time.time() > deadline:
            process.kill()
            raise RuntimeError('SGLang server did not become healthy within 30 min')
        time.sleep(5)
    engine.bind_sglang(url, export_manifest=manifest, fork={key: fork[key] for key in ('repository', 'branch', 'commit')})
    return process


if __name__ == '__main__':
    main()
