"""A compiled site (``compiler/compiled_site.py``) served by the engines equals the training executor.

usage: check_compiled_site_parity.py CHECKPOINT BASE RELEASE WORK --phase reference|native|sglang-collect|llama|score
           [--tokens T] [--port P] [--kv-gb G] [--sglang-python PY] [--llama-server BIN] [--threads N]

RELEASE is ``run_t2``'s ``release/`` (``tune.build`` layout: ``result.json``, ``artifacts/`` with the xs basis and the
compiled cores, ``states/`` with the compiled header state, and ``site-row.json``, the real call it was compiled for).
The call is rendered with the header replaced by the compiled state (``place_states``, as served), and decoded greedily:

- ``reference``: the training executor itself: ``engine.shared_model`` over the graph that imports the state at its
  span (the same consumption ``native_graph_gradient`` trains), under ``run_native`` with the release's adapter stack,
  one argmax per step. Also the base arm (the header as text, no adapter) as the control the compiled site must change.
- ``native``: the Python native engine's ``generate`` (GenerationRequest with the native_state part and the adapter
  stack, T=0).
- ``sglang-collect``: the SGLang actor (``serve_native_checkpoint.py``'s own launch; the front hands the state to the
  fork as a store file and the stack to an adapter slot), T=0 with capture.
- ``llama``: the llama.cpp fork's llama-server (CPU): the checkpoint's natlang GGUF (bf16, the identity the state was
  written under), the release's cores as a GGUF LoRA (``tune.export``: the tuned-fit export), the state file as
  ``--native-state``; the prompt is the engine's rendered text with the state's marker where it stands (the fork adds
  the consume markers), checked token for token against natlang's tokenization of the same segments.
- ``score``: per engine, the greedy tokens against the reference (agreement up to the first difference, and the
  reference's top-2 margin there), and whether the compiled continuation differs from the base one.

Each phase writes ``WORK/<phase>.json``; run them as separate processes on a small GPU (one model resident at a time).
"""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))


def load(args, engine_class=None, device='cuda'):
    from natlang_neuralese.serve import load_native_engine
    from natlang_neuralese.serve.store import TensorStore
    from natlang_neuralese.tune.build import import_archive
    engine = load_native_engine(str(args.checkpoint), base=str(args.base), device=device, dtype=torch.bfloat16,
                                training_state=False, store=TensorStore(args.work / f'store-{args.phase}'),
                                **({'engine_class': engine_class} if engine_class else {}))
    import_archive(engine, [args.release / 'artifacts'], [args.release / 'states'])
    return engine


def release_call(args):
    from natlang_neuralese.compiler.compiled_site import place_header
    result = json.loads((args.release / 'result.json').read_text())
    row = json.loads((args.release / 'site-row.json').read_text())
    state = result['softStates']['soft:instructions']['state']
    stack = [{'id': item['artifact'], 'scale': 1.0} for item in result['artifacts']]
    compiled = place_header(row['messages'], row['header'], state)
    return row, state, stack, compiled


def prompt_graph(engine, messages, tools):
    """Prompt token ids and import spans as the engine renders the call (state parts become consume-marked spans)."""
    from natlang_neuralese.serve.targets import _items
    from natlang_neuralese.tune.soft import soft_items_to_node
    rendered = engine.render_request(messages, tools)
    items = _items(engine, rendered.segments, rendered.blocks, rendered.escape_nonce)
    names = {value: f'state:{index}' for index, (kind, value) in enumerate(items) if kind == 'block'}
    ids, _, imports, external = soft_items_to_node(engine, items, [], prompt_weight=0., state_names=names)
    return ids, imports, external


def reference_greedy(engine, messages, tools, stack, steps):
    from natlang_neuralese.model.native_graph_plan import Invocation, compile_graph
    ids, imports, external = prompt_graph(engine, messages, tools)
    blocks = {name: engine.store.get_native_state(binding['id']).block for name, binding in external.items()}
    identity = {name: {key: getattr(block, key) for key in ('checkpoint_sha256', 'tokenizer_fingerprint',
                                                             'embedding_weight_sha256', 'architecture')}
                for name, block in blocks.items()}
    sizes = {name: binding['token_count'] for name, binding in external.items()}
    out, margins = [], []
    for _ in range(steps):
        plan = compile_graph([Invocation('turn', tuple(ids + out), imports=tuple(imports))], external_sizes=sizes)
        with torch.no_grad():
            logits = engine._native_run(lambda: engine.shared_model.logits(engine.shared_model(
                plan, external_blocks=blocks, external_identity=identity)['h_final'][:, -1:])[0, -1].float(), stack)
        top = torch.topk(logits, 2)
        out.append(int(top.indices[0]))
        margins.append(float(top.values[0] - top.values[1]))
    return {'prompt_ids': ids, 'tokens': out, 'margins': margins}


def generate(engine, messages, tools, stack, steps):
    from natlang_neuralese.serve.engine import GenerationRequest
    request = GenerationRequest(messages=messages, tools=tools, temperature=0.0, max_tokens=steps, rl_capture=True,
                                native_recording=True, seed=7, adapters=stack)
    capture = engine.generate(request)['x_natlang_rl_capture']
    return {'tokens': [sample['token_id'] for sample in capture['samples']],
            'adapter_binding': capture.get('adapter_binding') is not None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('base', type=Path)
    parser.add_argument('release', type=Path)
    parser.add_argument('work', type=Path)
    parser.add_argument('--phase', required=True, choices=('reference', 'native', 'sglang-collect', 'llama', 'score'))
    parser.add_argument('--llama-server', help="the fork's llama-server binary (phase llama)")
    parser.add_argument('--threads', default='8')
    parser.add_argument('--tokens', type=int, default=32)
    parser.add_argument('--port', type=int, default=8771)
    parser.add_argument('--kv-gb', type=float, default=2.0)
    parser.add_argument('--sglang-python')
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    out = args.work / f'{args.phase}.json'
    if args.phase == 'score':
        return score(args)
    row, state, stack, compiled = release_call(args)
    tools = row['tools'] or None
    if args.phase == 'reference':
        engine = load(args)
        result = {'compiled': reference_greedy(engine, compiled, tools, stack, args.tokens),
                  'base': reference_greedy(engine, row['messages'], tools, [], args.tokens)}
    elif args.phase == 'llama':
        result = llama(args, row, state, stack, compiled, tools)
    elif args.phase == 'native':
        engine = load(args)
        engine.start()
        try:
            result = {'compiled': generate(engine, compiled, tools, stack, args.tokens),
                      'base': generate(engine, row['messages'], tools, [], args.tokens)}
        finally:
            engine.stop()
    else:
        import os
        import signal
        from natlang_neuralese.common.hashing import sha256_file_hex_cached
        from natlang_neuralese.serve.sglang_actor import SglangEngine
        from serve_native_checkpoint import build_parser, launch_sglang
        engine = load(args, SglangEngine, device='cpu')
        argv = ['--checkpoint', str(args.checkpoint), '--checkpoint-sha256', sha256_file_hex_cached(args.checkpoint),
                '--port', str(args.port), '--store-dir', str(args.work / f'store-{args.phase}'),
                '--hf-export-dir', str(args.work / 'export'), '--sglang-kv-gb', str(args.kv_gb),
                '--sglang-max-running', '4', '--sglang-context', '16384', '--sglang-adapter-slots', '2',
                '--adapter-release', str(args.release), *(['--sglang-python', args.sglang_python] if args.sglang_python else [])]
        process = launch_sglang(engine, build_parser().parse_args(argv), args.checkpoint,
                                release={'checkpoint': str(args.checkpoint), 'base': str(args.base), 'device': 'cuda',
                                         'participant': None})
        engine.start()
        try:
            result = {'compiled': generate(engine, compiled, tools, stack, args.tokens),
                      'base': generate(engine, row['messages'], tools, [], args.tokens),
                      'metrics': {k: v for k, v in engine.progress_metrics().items() if 'sglang' in k or 'fallback' in k}}
        finally:
            engine.stop()
            os.killpg(process.pid, signal.SIGTERM)
    result['release'] = str(args.release)
    result['state'], result['stack'] = state, stack
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items() if k in ('compiled', 'base')}, default=str)[:2000])
    return 0


def llama(args, row, state, stack, compiled, tools):
    import subprocess
    import time
    import urllib.request
    from natlang_neuralese.export.native_gguf import export as export_native_gguf
    from natlang_neuralese.tune.export import export_gguf, export_peft
    from natlang_neuralese.tune.stack import stack_factors
    from natlang_neuralese.tune.stack_export import load_artifacts, members_from_artifacts
    from natlang_neuralese.tune.targets import module_map
    base_dir = args.work / 'gguf'
    if not (base_dir / 'model-bf16.gguf').exists():
        export_native_gguf(args.checkpoint, base_dir, outtypes=('bf16',), base=str(args.base))
    result_json = json.loads((args.release / 'result.json').read_text())
    engine = load(args, device='cpu')
    identity = (result_json.get('receipt') or {}).get('base_native_identity') or dict(engine.native_identity)
    if not (args.work / 'lora.gguf').exists():
        artifacts = load_artifacts(args.release / 'artifacts')
        members, bases, shapes = members_from_artifacts(
            artifacts, [{'id': item['artifact'], 'scale': 1., 'node': item['node']} for item in result_json['artifacts']])
        targets, _ = module_map(sorted(shapes))
        factors = stack_factors(members, bases, shapes)
        export_peft(factors, {name: targets[name] for name in factors if factors[name].rank}, args.work / 'lora.peft',
                    base_identity=identity,
                    base_model_name_or_path=str(base_dir / 'hf-config'))
        export_gguf(args.work / 'lora.peft', base_dir / 'hf-config', args.work / 'lora.gguf')
    # the prompt as text: the engine's rendered segments with each state's marker; natlang's own tokenization of the
    # same segments is the check that llama reads the same tokens
    expected = {}
    def prompt_text(messages):
        rendered = engine.render_request(messages, tools)
        parts, ids = [], []
        for segment in rendered.segments:
            if isinstance(segment, str):
                parts.append(segment)
                ids.extend(engine._template_tokens(segment, rendered.escape_nonce))
            else:
                parts.append(f'<__natlang_native_state:{rendered.blocks[segment]}__>')
                ids.append(('state', rendered.blocks[segment]))
        return ''.join(parts), ids
    port = args.port + 1
    log = open(args.work / 'llama.log', 'w')
    server = subprocess.Popen([args.llama_server, '-m', str(base_dir / 'model-bf16.gguf'), '--port', str(port), '-c', '16384',
                               '-t', args.threads, '-ngl', '0', '--no-warmup', '-np', '1', '--lora-init-without-apply',
                               '--lora-per-sequence', '--lora', str(args.work / 'lora.gguf'),
                               '--native-state', str(args.release / 'states' / f'{state}.nzs')],
                              stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(1200):
            try:
                urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1)
                break
            except Exception:
                if server.poll() is not None:
                    raise SystemExit(f'llama-server exited {server.returncode}; see {log.name}')
                time.sleep(.5)
        def post(path, body):
            return json.load(urllib.request.urlopen(urllib.request.Request(
                f'http://127.0.0.1:{port}{path}', json.dumps(body).encode(), {'content-type': 'application/json'}), timeout=3600))
        out = {}
        for arm, messages, scale in (('compiled', compiled, 1.), ('base', row['messages'], 0.)):
            text, natlang_ids = prompt_text(messages)
            pieces = text.split('<__natlang_native_state:')
            llama_ids = []
            for index, piece in enumerate(pieces):
                if index:
                    ident, piece = piece.split('__>', 1)
                    llama_ids.append(('state', ident))
                if piece:
                    llama_ids.extend(post('/tokenize', {'content': piece, 'parse_special': True, 'add_special': False})['tokens'])
            reply = post('/completion', {'prompt': text, 'n_predict': args.tokens, 'temperature': 0, 'top_k': 1,
                                         'cache_prompt': False, 'return_tokens': True, 'lora': [{'id': 0, 'scale': scale}]})
            out[arm] = {'tokens': reply.get('tokens') or [], 'prompt_tokens_match': llama_ids == natlang_ids,
                        'tokens_evaluated': reply.get('tokens_evaluated'), 'text': reply.get('content', '')[:400]}
        return out
    finally:
        server.terminate()
        server.wait(60)


def score(args):
    reference = json.loads((args.work / 'reference.json').read_text())
    report = {'reference_compiled_differs_from_base': reference['compiled']['tokens'] != reference['base']['tokens']}
    for phase in ('native', 'sglang-collect', 'llama'):
        path = args.work / f'{phase}.json'
        if not path.exists():
            continue
        got = json.loads(path.read_text())
        report[phase] = {}
        for arm in ('compiled', 'base'):
            want, have = reference[arm]['tokens'], got[arm]['tokens']
            same = next((i for i, (a, b) in enumerate(zip(have, want)) if a != b), min(len(have), len(want)))
            report[phase][arm] = {'tokens': len(have), 'agree_prefix': same,
                                  'agreement': same / max(1, min(len(have), len(want))),
                                  'first_diff_margin': reference[arm]['margins'][same] if same < len(want) and same < len(have) else None}
    (args.work / 'report.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
