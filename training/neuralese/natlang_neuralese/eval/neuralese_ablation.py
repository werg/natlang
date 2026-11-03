"""Is the Neuralese used? One shared ablation over native graphs (AGENTS.md 2026-11-03, learned operators).

For every graph with internal producer->consumer imports, score the consumers' supervised targets three ways:

* ``actual``: the graph as trained, one forward, consumers import their producers' state;
* ``shuffled``: every internal import is replaced by the producer state of a *different* row (a derangement over
  the scored rows, edge by edge; the donor's own length is imported, so slot counts and positions stay natural);
* ``zeroed``: every internal import keeps its length but all imported keys, values, hybrid payloads and embeddings
  are zero.

Producers still execute in the ablated graphs (their own tokens are unchanged); only what consumers import changes.
Consumers are the nodes with at least one internal import (``score='consumers'``) or every supervised node
(``score='all'``). Optional ``text-cut`` rows (the consumer reading the runtime's text cut of the same value, see
``data/operator_graph.py``) are scored as a fourth, separately supplied condition with the same target tokens.

The metric is weighted target NLL per supervised token and token accuracy. ``use`` = NLL(shuffled) - NLL(actual)
and NLL(zeroed) - NLL(actual): positive means consumers depend on the specific state they import. A learned operator
is promoted only when its consumers beat shuffled, zeroed and text-cut on held worlds.

    python -m natlang_neuralese.eval.neuralese_ablation --checkpoint CKPT --graphs G.jsonl[.zst] \
        [--text-cut T.jsonl[.zst]] [--limit N] [--device cuda] --out report.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import random
import sys
import time

import torch

from ..data.native_graph_schema import parse_graph
from ..model.native_graph_plan import ImportSpan, Invocation, compile_graph

CONDITIONS = ('actual', 'shuffled', 'zeroed')
SCHEMA = 'natlang.neuralese-ablation/1'
# A synthetic identity: ablation blocks are captured from and imported into the same model in one process.
ABLATION_IDENTITY = {'checkpoint_sha256': 'a' * 64, 'tokenizer_fingerprint': 'b' * 64,
                     'embedding_weight_sha256': 'c' * 64}


def internal_edges(plan):
    """Distinct (consumer index, source node, start, stop) of imports from nodes in the same graph."""
    names = {node.name for node in plan.invocations}
    out = []
    for index, node in enumerate(plan.invocations):
        for edge in node.imports:
            if edge.source in names and not edge.logical:
                out.append((index, edge.source, edge.start, edge.stop))
    return out


def consumer_weights(example, score='consumers'):
    """Target weights restricted to the scored nodes."""
    plan, weights = example.plan, list(example.target_weights)
    if score == 'all':
        return tuple(weights)
    consumers = {index for index, *_ in internal_edges(plan)}
    out = [0.] * len(weights)
    for index, (node, offset) in enumerate(zip(plan.invocations, plan.offsets)):
        if index in consumers:
            out[offset:offset + len(node.token_ids)] = weights[offset:offset + len(node.token_ids)]
    return tuple(out)


def _identity(model):
    return {**ABLATION_IDENTITY, 'architecture': str(getattr(model.port.config, 'model_type', type(model.port).__name__))}


def _zero_block(block):
    from ..model.state_block import NativeStateBlock
    layers = tuple({key: (value if key == 'kind' else torch.zeros_like(value)) for key, value in layer.items()}
                   for layer in block.layers)
    return NativeStateBlock(text='', token_ids=block.token_ids, tokenizer_fingerprint=block.tokenizer_fingerprint,
                            checkpoint_sha256=block.checkpoint_sha256,
                            embedding_weight_sha256=block.embedding_weight_sha256,
                            embeddings=torch.zeros_like(block.embeddings), positions=block.positions,
                            layers=layers, architecture=block.architecture)


def externalized(example, replacements):
    """The same graph with every internal import read from an external block: ``replacements[(source,start,stop)]``."""
    plan = example.plan
    names = {node.name for node in plan.invocations}
    external, sizes, invocations = {}, {}, []
    for node in plan.invocations:
        imports = []
        for edge in node.imports:
            key = (edge.source, edge.start, edge.stop)
            if edge.source in names and not edge.logical:
                block = replacements[key]
                name = 'ablation:%s:%d:%d' % key
                external[name] = block
                sizes[name] = len(block.token_ids)
                imports.append(ImportSpan(edge.at, name, 0, len(block.token_ids), stop_gradient=edge.stop_gradient))
            else:
                imports.append(edge)
        invocations.append(Invocation(node.name, node.token_ids, tuple(imports), node.producing,
                                      node.logical_producing, node.cache_prefix))
    for name, size in zip(plan.external_names, plan.external_sizes):
        sizes[name] = size
    return compile_graph(invocations, external_sizes=sizes), external


def _score(model, example, plan, weights, embeddings, external_blocks, external_identity, chunk_tokens):
    from ..data.native_graph import graph_loss_from_result
    scored = dataclasses.replace(example, plan=plan, target_weights=weights)
    result = model(plan, embeddings=embeddings, external_blocks=external_blocks or None,
                   external_identity=external_identity if external_blocks else None)
    loss, detail = graph_loss_from_result(model, scored, result, chunk_tokens=chunk_tokens)
    return {'nll_sum': float(loss) * detail['weight'], 'weight': detail['weight'],
            'correct_sum': detail['accuracy'] * detail['weight'], 'tokens': detail['supervised_tokens']}, result


def ablate(model, examples, *, rows=None, operator_spans=None, score='consumers', seed=0, chunk_tokens=128,
           text_cut=None):
    """Score ``examples`` (parsed NativeGraphExamples with internal imports) under every condition.

    ``rows``/``operator_spans`` (one tuple per example) substitute learned operator rows. ``text_cut`` maps an
    example id to its parsed text-cut consumer example. Returns the report dict (aggregate and per-row).
    """
    from ..data.native_graph import graph_loss_from_result
    from ..model.learned_operators import operator_embeddings
    from ..model.shared_state import capture_block
    identity = _identity(model)
    usable = [example for example in examples if internal_edges(example.plan)
              and any(weight > 0 for weight in consumer_weights(example, score))]
    if len(usable) < 2:
        raise ValueError('ablation needs at least two graphs with scored consumers (shuffling needs a donor)')
    spans = operator_spans or {}
    captured, embedded, per_row = [], [], []
    with torch.no_grad():
        for example in usable:
            embeddings = operator_embeddings(model, rows, example.plan, spans.get(example.id, ()))
            weights = consumer_weights(example, score)
            result = model(example.plan, embeddings=embeddings, external_blocks=None, external_identity=None)
            blocks = {}
            for _, source, start, stop in internal_edges(example.plan):
                if (source, start, stop) not in blocks:
                    blocks[(source, start, stop)] = capture_block(result, source, start, stop, text='', **identity)
            loss, detail = graph_loss_from_result(model, dataclasses.replace(example, target_weights=weights),
                                                  result, chunk_tokens=chunk_tokens)
            del result
            actual = {'nll_sum': float(loss) * detail['weight'], 'weight': detail['weight'],
                      'correct_sum': detail['accuracy'] * detail['weight'], 'tokens': detail['supervised_tokens']}
            captured.append(blocks)
            embedded.append(embeddings)
            per_row.append({'id': example.id, 'edges': len(blocks), 'actual': actual})
        order = list(range(len(usable)))
        random.Random(seed).shuffle(order)
        donor_of = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}  # a derangement
        for index, example in enumerate(usable):
            keys = list(captured[index])
            donor = list(captured[donor_of[index]].values())
            shuffled = {key: donor[position % len(donor)] for position, key in enumerate(keys)}
            zeroed = {key: _zero_block(block) for key, block in captured[index].items()}
            weights = consumer_weights(example, score)
            for condition, replacements in (('shuffled', shuffled), ('zeroed', zeroed)):
                plan, external = externalized(example, replacements)
                identities = {name: identity for name in external}
                for name in plan.external_names:
                    if name not in external:
                        raise ValueError('ablation does not supply stored external states of the source graph')
                scored, _ = _score(model, example, plan, weights, embedded[index], external, identities, chunk_tokens)
                per_row[index][condition] = scored
            per_row[index]['donor'] = usable[donor_of[index]].id
            if text_cut is not None and example.id in text_cut:
                cut = text_cut[example.id]
                cut_embeddings = operator_embeddings(model, None, cut.plan, ())
                scored, _ = _score(model, cut, cut.plan, cut.target_weights, cut_embeddings, None, None, chunk_tokens)
                per_row[index]['text-cut'] = scored
    return summarize(per_row, score=score, seed=seed)


def summarize(per_row, *, score, seed):
    conditions = [*CONDITIONS, *(['text-cut'] if any('text-cut' in row for row in per_row) else [])]
    aggregate = {}
    for condition in conditions:
        rows = [row[condition] for row in per_row if condition in row]
        weight = sum(item['weight'] for item in rows)
        aggregate[condition] = {'rows': len(rows), 'weight': weight, 'tokens': sum(item['tokens'] for item in rows),
                                'nll': sum(item['nll_sum'] for item in rows) / weight if weight else None,
                                'accuracy': sum(item['correct_sum'] for item in rows) / weight if weight else None}
    use = {f'{condition}-minus-actual': (aggregate[condition]['nll'] - aggregate['actual']['nll'])
           for condition in conditions if condition != 'actual' and aggregate[condition]['nll'] is not None}
    # Row-level sign test: the fraction of rows whose consumers lose more than they gain without their own state.
    wins = {condition: sum(1 for row in per_row if condition in row and
                           row[condition]['nll_sum'] / row[condition]['weight'] >
                           row['actual']['nll_sum'] / row['actual']['weight']) / max(1, sum(condition in row for row in per_row))
            for condition in conditions if condition != 'actual'}
    return {'schema': SCHEMA, 'score': score, 'seed': seed, 'rows': len(per_row), 'aggregate': aggregate,
            'use_nll_delta': use, 'actual_better_fraction': wins, 'per_row': per_row}


def _read_jsonl(path, limit=None):
    import io
    opener = open
    if str(path).endswith('.zst'):
        import zstandard
        def opener(p, mode):  # noqa: E306
            return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(p, 'rb')), encoding='utf-8')
    with opener(path, 'r') as stream:
        for count, line in enumerate(stream):
            if limit is not None and count >= limit:
                break
            if line.strip():
                yield json.loads(line)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--graphs', required=True)
    parser.add_argument('--text-cut')
    parser.add_argument('--operator-rows', help='a saved OperatorRows state (learned_operators.save_rows)')
    parser.add_argument('--text-init-operators', action='store_true',
                        help='operator rows text-initialized from this checkpoint (the untrained operator)')
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--max-tokens', type=int, default=24_000, help='skip graphs with more physical tokens')
    parser.add_argument('--score', choices=('consumers', 'all'), default='consumers')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    from ..model.native_model import load_native_checkpoint
    from ..model.learned_operators import ROW_FIELD, load_rows, resolve_spans
    started = time.time()
    model, tokenizer, roles, config, _ = load_native_checkpoint(args.checkpoint, device=args.device,
                                                                allow_weights_only=True)
    model.eval()
    fingerprint = config['tokenizer_fingerprint']
    rows = load_rows(args.operator_rows, device=args.device) if args.operator_rows else None
    if args.text_init_operators:
        from ..model.learned_operators import OperatorRows
        rows = OperatorRows.text_initialized(model.port.embedding_weight, tokenizer, tokenizer_fingerprint=fingerprint)
    examples, spans, raw_rows, skipped = [], {}, {}, {'no-internal-import': 0, 'over-max-tokens': 0}
    for row in _read_jsonl(args.graphs):
        if len(examples) >= args.limit:
            break
        example = parse_graph(row, tokenizer_fingerprint=fingerprint)
        if not internal_edges(example.plan):
            skipped['no-internal-import'] += 1
            continue
        if example.plan.token_count > args.max_tokens:
            skipped['over-max-tokens'] += 1
            continue
        if row.get(ROW_FIELD):
            if rows is None:
                raise SystemExit('graphs declare learned operator spans; pass --operator-rows or --text-init-operators')
            spans[example.id] = resolve_spans(row, example.plan, rows.bindings)
        examples.append(example)
    text_cut = None
    if args.text_cut:
        wanted = {example.id for example in examples}
        text_cut = {}
        for row in _read_jsonl(args.text_cut):
            source = row.get('text_cut_of')
            if source in wanted:
                text_cut[source] = parse_graph(row, tokenizer_fingerprint=fingerprint)
    report = ablate(model, examples, rows=rows, operator_spans=spans, score=args.score, seed=args.seed,
                    text_cut=text_cut)
    report.update({'checkpoint': args.checkpoint, 'graphs': args.graphs, 'text_cut': args.text_cut,
                   'operator_rows': args.operator_rows, 'skipped': skipped, 'seconds': time.time() - started})
    with open(args.out, 'w') as stream:
        json.dump(report, stream, indent=1)
    json.dump({key: report[key] for key in ('rows', 'aggregate', 'use_nll_delta', 'actual_better_fraction', 'skipped')},
              sys.stdout, indent=1)
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
