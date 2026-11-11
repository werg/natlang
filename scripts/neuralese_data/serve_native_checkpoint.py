#!/usr/bin/env python3
"""Serve one exact native shared-state checkpoint as a rollout actor."""
from __future__ import annotations

import argparse
import json
import subprocess
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
    parser.add_argument('--transport-scope',choices=('independent-recovery','diagnostic-local-pin','rl-controller-pin'))
    parser.add_argument('--actor-export-format',help='bind this exact-weights export format in the receipt and every capture (an actor on another device: the learner guards the engine mismatch)')
    parser.add_argument('--coefficient-policy', type=Path)
    parser.add_argument('--coefficient-policy-sha256')
    parser.add_argument('--coefficient-rebase-out', type=Path)
    parser.add_argument('--coefficient-cancel-stale-reason')
    parser.add_argument('--prompt-rendering', choices=('full', 'program-header'), default='full')
    parser.add_argument('--prefill-tokens', type=int, default=None,
                        help='native prompt-forward chunk size (default: 2048 on GPUs <=12 GiB, otherwise 8192)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--prefix-cache-gb', type=float, default=2.0, help='native prompt-prefix cache budget; 0 disables')
    parser.add_argument('--prefix-block', type=int, default=None, help='native prefill cache grid (default: existing prefill chunk size)')
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
    parser.add_argument('--sglang-adapter-slots', type=int, default=4,
                        help='resident native adapter stacks on SGLang (0 opts out: adapted turns run on the native '
                             'engine); batches without an adapted request replay the base-only decode graph')
    parser.add_argument('--sglang-adapter-rank', type=int, default=32,
                        help='largest summed adapter rank per SGLang module and slot')
    parser.add_argument('--sglang-projection', choices=('native', 'vendor'), default='vendor',
                        help="LFM3 projections on SGLang: the learner's invariant native kernel, or one fused vendor "
                             'BF16 GEMM (fp32 accumulate; the learner corrects the gap with importance weights)')
    parser.add_argument('--sglang-overlap-schedule', action='store_true',
                        help="SGLang's overlapped scheduler. Off by default: on DGX GB10 with CUDA graphs it put "
                             'tokens of a wrong distribution into 2-6%% of real RL turns (up to 24 nats from the '
                             'learner); without it 0 of 64 (2026-11-03, plans/neuralese/LFM3_RL_ACTOR_ENGINE_20261028.md)')
    parser.add_argument('--sglang-arg', action='append', default=[], help='extra sglang.launch_server argument')
    parser.add_argument('--hf-export-dir', type=Path, help='HF export cache (default: <store-dir or checkpoint dir>/sglang-hf/<sha>)')
    args = parser.parse_args()
    if args.engine == 'sglang' and (args.coefficient_policy or args.transport_receipt):
        parser.error('the SGLang actor serves ordinary checkpoints only (no coefficient policy or transport view)')
    if bool(args.native_actor_view) != bool(args.native_actor_view_sha256):
        parser.error('native actor view requires its exact receipt SHA-256')
    transport_values=(args.transport_receipt,args.transport_receipt_sha256,args.transport_view_receipt,
        args.transport_view_receipt_sha256,args.training_source_sha256,args.transport_scope)
    recovery_values=(args.transport_recovery_receipt,args.transport_recovery_receipt_sha256)
    if any((*transport_values,*recovery_values)) and (not all(transport_values) or args.native_actor_view or args.native_participant
            or (args.transport_scope=='independent-recovery' and not all(recovery_values))
            or (args.transport_scope!='independent-recovery' and any(recovery_values))):
        parser.error('transported actor requires all exact receipt bindings and no full-source actor view or participant')
    if bool(args.coefficient_policy) != bool(args.coefficient_policy_sha256):
        parser.error('coefficient policy requires its exact SHA-256')

    import torch

    torch.set_num_threads(args.threads)
    from natlang_neuralese.common.attention_backend import configure_sdpa_backend
    configure_sdpa_backend(args.sdpa_backend)
    checkpoint = args.checkpoint.resolve(strict=True)
    bindings = [('checkpoint', checkpoint, args.checkpoint_sha256)]
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
        device=args.device,
        prefill_tokens=args.prefill_tokens,
        **view_options)
    from natlang_neuralese.serve import configure_native_prefix_cache
    configure_native_prefix_cache(engine, args.prefix_cache_gb, args.prefix_block)
    if args.engine == 'sglang':
        launch_sglang(engine, args, checkpoint)
        # The native model is loaded again only when a turn SGLang cannot serve arrives (serve/sglang_actor.py).
        engine.release_native({'checkpoint': str(checkpoint), 'base': str(args.base) if args.base else None,
                               'device': args.device, 'participant': args.native_participant,
                               **({'actor_view': view_options['actor_view']} if 'actor_view' in view_options else {})})
    if args.store_dir is not None:
        from natlang_neuralese.serve.store import TensorStore
        engine.store = TensorStore(args.store_dir)
    if args.actor_export_format:
        engine.set_actor_export({'kind': 'exact', 'format': args.actor_export_format, 'sha256': args.checkpoint_sha256})
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
        'serving_config': engine.serving_config,
        'prompt_rendering': args.prompt_rendering,
        'listening': f'http://{server.server_address[0]}:{server.server_address[1]}'}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        engine.stop()


ENGINE_DIED_EXIT = 70   # the SGLang engine exited under a live actor (2026-11-10: the wrapper kept answering /v1/models)


def supervise_sglang(process):
    """Tie the actor to its SGLang engine in both directions.

    The server (and its scheduler/detokenizer children) ends with this actor, including on SIGTERM. When the engine
    exits first (a scheduler crash takes the whole server down), this process must stop reporting healthy: it logs the
    reason and exits ``ENGINE_DIED_EXIT`` so the controller sees a dead actor and relaunches it (rl.managed_actor)."""
    import atexit
    import os
    import signal
    import threading
    import time
    state = {'stopping': False, 'died': None}

    def stop_sglang(*_):
        state['stopping'] = True
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)

    def terminate(signum, _frame):
        state['stopping'] = True
        sys.exit(ENGINE_DIED_EXIT if state['died'] is not None else 128 + signum)

    def watch():
        code = process.wait()
        if state['stopping']:
            return
        state['died'] = code
        print(json.dumps({'event': 'sglang_engine_exited', 'returncode': code, 'pid': process.pid,
                          'action': f'actor exits {ENGINE_DIED_EXIT}'}), file=sys.stderr, flush=True)
        os.kill(os.getpid(), signal.SIGTERM)   # the main thread unwinds (server_close, engine.stop) through terminate()
        time.sleep(30)                         # an unwind that hangs must not leave an unhealthy actor answering
        os._exit(ENGINE_DIED_EXIT)
    atexit.register(stop_sglang)
    signal.signal(signal.SIGTERM, terminate)
    threading.Thread(target=watch, name='sglang-engine-watch', daemon=True).start()


def attest_sglang_source(python, fork, pythonpath=''):
    """Record the interpreter's actual SGLang fork commit (and dirty flag) for the actor attestation.

    The declared commit in training/neuralese/sglang-fork.json is a recommendation, never a gate: a different or
    modified fork checkout is recorded and warned about, and the actor launches (owner, 2026-11-09)."""
    import os
    import subprocess
    env = {**os.environ, 'PYTHONPATH': pythonpath}
    probe = subprocess.run([python, '-c',
        'import importlib.util, pathlib; s=importlib.util.find_spec("sglang"); '
        'print(pathlib.Path(s.origin).resolve())'], env=env, check=True, text=True, capture_output=True)
    package = Path(probe.stdout.strip())
    if package.name != '__init__.py' or package.parent.name != 'sglang':
        raise RuntimeError('SGLang interpreter did not resolve a source package')
    repository = package.parent.parent.parent

    def git(*arguments):
        done = subprocess.run(['git', '-C', str(repository), *arguments], text=True, capture_output=True)
        return done.stdout.strip() if done.returncode == 0 else None
    actual = git('rev-parse', 'HEAD') if git('rev-parse', '--show-toplevel') is not None else None
    dirty = bool(git('status', '--porcelain', '--untracked-files=all', '--', 'python/sglang')) if actual else None
    declared = fork.get('commit')
    record = {'commit': actual, 'declared_commit': declared, 'matches_declared': actual == declared, 'dirty': dirty,
              'package': str(package), 'python': str(Path(python).resolve())}
    if actual != declared or dirty:
        print(json.dumps({'event': 'sglang_fork_differs_from_declared', **record}), file=sys.stderr, flush=True)
    return record


def launch_sglang(engine, args, checkpoint):
    """Export the checkpoint for SGLang (cached by SHA-256), start the fork's server and bind the front to it."""
    import os
    import time
    import urllib.request
    from natlang_neuralese.export.native_hf import export_loaded
    root = Path(__file__).resolve().parents[2]
    fork = json.loads((root / 'training/neuralese/sglang-fork.json').read_text())
    python = args.sglang_python or fork['python']
    if engine.model_config.get('native_linear_precision', 'ieee') != 'ieee':
        raise ValueError('SGLang native projection requires the ieee native linear policy')
    if engine.model_config.get('native_shared_execution', 'source-v1') not in ('source-v1', 'ffn-stable-v1'):
        raise ValueError('SGLang LFM3 FFN has not qualified this native shared execution policy')
    # The backend's native projection uses the same frozen serving runtime as
    # the front. Put it first so an editable or user-installed package cannot
    # silently substitute a different arithmetic implementation.
    backend_pythonpath = os.pathsep.join(filter(None, [str(root / 'training/neuralese'),
        os.environ.get('NATLANG_SGLANG_PYTHONPATH', '')]))
    source = attest_sglang_source(python, fork, backend_pythonpath)
    export_dir = args.hf_export_dir or ((args.store_dir or checkpoint.parent) / 'sglang-hf' / args.checkpoint_sha256[:16])
    manifest = export_loaded(engine.shared_model, engine.tokenizer, engine.role_tokens, engine.model_config,
                             checkpoint=checkpoint, checkpoint_sha256=args.checkpoint_sha256, out_dir=export_dir)
    port = args.sglang_port or args.port + 1000
    context = min(args.sglang_context, engine.context)
    # The front sizes each reply to the room left in the backend's context; with the trained maximum it asked for
    # 8192 tokens after a 24.6k prompt and SGLang refused the request (w5 round 2: 2 episodes lost, 5 held-eval).
    engine.context = context
    total = args.sglang_max_total_tokens or 4 * context
    if total < context:
        raise ValueError('--sglang-max-total-tokens must hold one full-context sequence')
    command = [python, '-m', 'sglang.launch_server', '--model-path', str(export_dir), '--host', '127.0.0.1',
               '--port', str(port), '--dtype', 'bfloat16', '--mem-fraction-static', str(args.sglang_mem_fraction),
               '--context-length', str(context), '--max-total-tokens', str(total),
               '--max-running-requests', str(args.sglang_max_running),
               '--max-mamba-cache-size', str(8 * args.sglang_max_running),
               '--cuda-graph-max-bs-decode', str(args.sglang_max_running),
               '--incremental-streaming-output', '--stream-interval', '8',
               '--skip-tokenizer-init', '--trust-remote-code',
               *([] if args.sglang_overlap_schedule else ['--disable-overlap-schedule']), *args.sglang_arg]
    print(json.dumps({'event': 'sglang_launch', 'command': command, 'export': manifest, 'source': source}), flush=True)
    # FlashInfer JIT-compiles kernels: the fork venv's ninja and the CUDA toolkit's nvcc on PATH.
    cuda = os.environ.get('CUDA_HOME', '/usr/local/cuda')
    path = os.pathsep.join([str(Path(python).parent), cuda + '/bin', os.environ.get('PATH', '')])
    # NATLANG_SGLANG_PYTHONPATH: a fork worktree under test instead of the pinned checkout's editable install.
    process = subprocess.Popen(command, env={**os.environ, 'PYTHONPATH': backend_pythonpath,
                                             'PATH': path, 'CUDA_HOME': cuda,
                                             'NATLANG_SGLANG_PROJECTION': args.sglang_projection,
                                             'NATLANG_ADAPTER_SLOTS': str(args.sglang_adapter_slots),
                                             'NATLANG_ADAPTER_RANK': str(args.sglang_adapter_rank)},
                               start_new_session=True)
    supervise_sglang(process)
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
    engine.bind_sglang(url, export_manifest=manifest,
                       fork={**{key: fork[key] for key in ('repository', 'branch')}, 'commit': source['commit'],
                             'declared_commit': source['declared_commit'], 'dirty': source['dirty']},
                       adapter_slots=args.sglang_adapter_slots, adapter_rank=args.sglang_adapter_rank,
                       projection=args.sglang_projection, overlap_schedule=args.sglang_overlap_schedule)
    return process


if __name__ == '__main__':
    main()
