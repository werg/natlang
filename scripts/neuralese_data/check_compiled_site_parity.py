"""A compiled site (``compiler/compiled_site.py``) served by the engines equals the training executor.

usage: check_compiled_site_parity.py CHECKPOINT BASE RELEASE WORK --phase reference|native|sglang-collect|score
           [--tokens T] [--port P] [--kv-gb G] [--sglang-python PY]

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
    parser.add_argument('--phase', required=True, choices=('reference', 'native', 'sglang-collect', 'score'))
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
