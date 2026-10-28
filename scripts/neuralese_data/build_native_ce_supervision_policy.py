#!/usr/bin/env python3
"""Declare balanced CE for exact currently admitted native source containers.

This creates an objective declaration, never corpus admission or row credits.
Use the same source manifests the two production recipes actually consume.
"""
from __future__ import annotations
import argparse
import copy
import json
import math
from pathlib import Path

from natlang_neuralese.common.hashing import sha256_file_hex_cached as sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.data.native_sources import _jsonl
from natlang_neuralese.train.native_supervision import SCHEMA, load_policy
from natlang_neuralese.train.recipe import native_source_dependencies


def complete_source_contract(prior, bindings, *, context_weight, inventory):
    """Declare row intent for every exact active leaf, without inventing a census.

    The caller supplies the authenticated active DAG's leaf bindings. This is an
    explicit objective contract, not source admission: normal source/index and
    per-row decoding checks remain mandatory at consumption.
    """
    if type(context_weight) not in (float, int) or not math.isfinite(context_weight) or not 0 <= context_weight < 1:
        raise ValueError('complete CE contract requires an explicit raw context coefficient')
    result = copy.deepcopy({key:value for key,value in prior.items() if not key.startswith('_')})
    previous = {entry['sha256']:entry for entry in result['sources']}
    held = {entry['sha256'] for entry in result.get('held_sources', []) + result.get('retained_held_sources', [])}
    seen = {}
    for binding in bindings:
        digest = binding['sha256']
        if (not isinstance(digest, str) or len(digest) != 64
                or any(char not in '0123456789abcdef' for char in digest)
                or not isinstance(binding.get('path'), str)):
            raise ValueError('complete CE contract requires exact source bindings')
        if digest in held:
            raise ValueError('complete TRAIN contract overlaps protected held declarations')
        if digest in seen:
            continue  # Identical immutable bytes can have multiple DAG aliases.
        old = previous.get(digest)
        coefficient = old['context_weight'] if old else context_weight
        seen[digest] = {'sha256':digest, 'encoding':'native-row-objectives-v1',
                        'context_weight':coefficient}
    if not seen:
        raise ValueError('complete CE contract requires a nonempty active source set')
    # Keep old declared sources too: objective continuation never silently
    # removes a prior declaration or any mask/held/source custody.
    previous.update(seen)
    result['sources'] = [previous[key] for key in sorted(previous)]
    result['require_eligible_sources'] = True
    result['source_contract'] = {'encoding':'native-row-objectives-v1',
        'inventory':copy.deepcopy(inventory), 'active_sources':copy.deepcopy(list(bindings)),
        'active_source_count':len(seen), 'default_raw_context_weight':context_weight,
        'scope':'Explicit per-row intent contract; not a full row census or new source admission. '
                'Existing source coefficients preserved; undeclared segmented/native masks retained; '
                'every consumed raw row validates exact coefficient, suffix and provenance.'}
    return result


def build(sources, *, legacy_pair_context_weight, held_sources=(), expected_split='train'):
    leaves, roots = {}, []
    for source in sources:
        path = Path(source).resolve(strict=True)
        roots.append({'path':str(path),'sha256':sha256_file_hex(path)})
        dependencies = native_source_dependencies(path)
        for binding in dependencies or [roots[-1]]:
            child = Path(binding['path'])
            if child.suffix != '.json':
                leaves[str(child)] = binding
    declarations, skipped = {}, []
    for binding in leaves.values():
        path = Path(binding['path'])
        if sha256_file_hex(path) != binding['sha256']:
            raise ValueError('CE declaration source changed after inventory validation')
        row = next(_jsonl(path), None)
        if row is None:
            raise ValueError('CE declaration source is empty: ' + str(path))
        if row.get('schema') == 'natlang.native-semantic-pair/1':
            graphs = [view['graph'] for view in row['models'].values()]
        elif row.get('schema') == 'natlang.neuralese-native-graph/1':
            graphs = [row]
        else:
            raise ValueError('CE declaration input is not an admitted native graph/pair source')
        if row.get('split') != expected_split or any(graph.get('split') != expected_split for graph in graphs):
            raise ValueError('CE declaration source has wrong explicit split: ' + str(path))
        encodings = set()
        for graph in graphs:
            conversion = graph.get('conversion') or {}
            if conversion.get('supervision_policy'):
                encodings.add(('already-declared', None))
            elif conversion.get('role_interpretation') == 'ordinary-text' and 'source_suffix_start' in conversion:
                encodings.add(('ordinary-suffix-raw-v1', conversion.get('context_weight')))
            elif graph.get('source_pairing'):
                encodings.add(('witnessed-pair-raw-v1', legacy_pair_context_weight))
            elif (conversion.get('role_interpretation') != 'ordinary-text'
                    and 'source_suffix_start' not in conversion and 'source_pairing' not in graph):
                # Direct admitted native objectives already carry explicit node
                # masks. Validate those fields with the shared schema law without
                # compiling KV positions, and retain their weights exactly.
                from natlang_neuralese.data.native_graph_schema import _graph_fields
                if not graph.get('nodes'):
                    raise ValueError('retained native CE source needs explicit node masks: ' + str(path))
                _graph_fields(graph, tokenizer_fingerprint=graph.get('tokenizer_fingerprint'))
                encodings.add(('retained-native-node-objective', None))
            else:
                raise ValueError('CE source lacks authenticated target intent: ' + str(path))
        if len(encodings) != 1:
            raise ValueError('CE source contains incompatible model-view supervision declarations')
        encoding, weight = encodings.pop()
        if encoding in {'already-declared', 'retained-native-node-objective'}:
            reason = ('existing trajectory supervision retained exactly' if encoding == 'already-declared'
                      else 'explicit native node target masks retained exactly; no raw legacy CE encoding')
            skipped.append({**binding, 'reason':reason})
            continue
        declaration = {'sha256':binding['sha256'],'encoding':encoding,'context_weight':weight}
        if declaration['sha256'] in declarations and declarations[declaration['sha256']] != declaration:
            raise ValueError('CE source digest has conflicting objective declarations')
        declarations[declaration['sha256']] = declaration
    value = {'schema':SCHEMA,'require_eligible_sources':True,
        'sources':[declarations[key] for key in sorted(declarations)],
        'inventory':roots,'retained_sources':[], 'first_row_observations':skipped,
        'scope':'First-row preliminary schema/intent checks only; complete source-bound row-objective contract follows.',
        'history':'Original graph bytes, IDs, row credits, node mass, optimizer/RNG and raw held CE retained.'}
    # In particular, full-session conversions intentionally put reasoning-turn
    # raw rows after segmented whole trajectories in the SAME container. A
    # first-row observation can never authorize source-wide retention.
    value = complete_source_contract(value, list(leaves.values()),
        context_weight=legacy_pair_context_weight, inventory=roots)
    if held_sources:
        held = build(held_sources, legacy_pair_context_weight=legacy_pair_context_weight, expected_split='test')
        if set(item['sha256'] for item in value['sources']) & set(item['sha256'] for item in held['sources']):
            raise ValueError('training and held CE source declarations must be disjoint')
        value.update(held_sources=held['sources'], held_inventory=held['inventory'],
                     retained_held_sources=held['retained_sources'])
    return value


def own_output_objective(args):
    """The declared own-outputs/1 objective with the tokenizer's exact chat-role token IDs."""
    from transformers import AutoTokenizer
    from natlang_neuralese.train.own_output_supervision import NAME, resolve, role_tokens_from_tokenizer
    if args.objective_tokenizer is None or args.objective_tokenizer_fingerprint is None:
        raise ValueError('own-output objective needs --objective-tokenizer and its exact fingerprint')
    tokenizer = AutoTokenizer.from_pretrained(str(args.objective_tokenizer))
    objective = resolve({'name': NAME, 'reasoning_share': args.objective_reasoning_share,
        'family_balance': args.objective_family_balance,
        'role_tokens': role_tokens_from_tokenizer(tokenizer,
            tokenizer_fingerprint=args.objective_tokenizer_fingerprint)})
    result = {'objective': objective}
    if args.objective_transition_from is not None:
        result['objective_transition'] = {'from': args.objective_transition_from}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', action='append', type=Path, default=[])
    parser.add_argument('--prior-policy',type=Path,
        help='append masks to this exact existing CE declaration without rescanning its source corpus')
    parser.add_argument('--complete-source-contract', action='store_true',
        help='explicitly cover every leaf in --source with mixed row objectives; requires prior policy')
    parser.add_argument('--expected-source-count', type=int,
        help='require this exact unique leaf count for the complete active DAG')
    parser.add_argument('--legacy-pair-context-weight', type=float,
        help='explicit original raw context coefficient for witnessed pairs and mixed retained sources, never estimated from weights')
    parser.add_argument('--held-source', action='append', type=Path, default=[],
        help='exact protected test source; evaluation declaration only, never training exposure')
    parser.add_argument('--failed-history-mask-manifest',type=Path,action='append',default=[],
        help='source-bound same-ID TRAIN mask continuation, with exact tokenizer snapshot')
    parser.add_argument('--objective-reasoning-share', type=float,
        help='declare the own-outputs/1 objective (owner 2026-10-28) with this reasoning share of assistant loss')
    parser.add_argument('--objective-tokenizer', type=Path,
        help='expanded model tokenizer whose exact chat-role token IDs the objective declares')
    parser.add_argument('--objective-tokenizer-fingerprint',
        help='exact expanded tokenizer fingerprint carried by the native graph rows')
    parser.add_argument('--objective-family-balance', type=json.loads,
        help='optional JSON {key: source-cohort|source-group, [pattern], max_scale}')
    parser.add_argument('--objective-transition-from',
        help='previous objective sha256 an existing lineage explicitly transitions from')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.prior_policy:
        if not args.complete_source_contract and (args.source or args.held_source or args.legacy_pair_context_weight is not None):
            raise ValueError('prior-policy mask continuation preserves all existing source declarations')
        from natlang_neuralese.train.native_failed_history import binding
        value=load_policy(binding(args.prior_policy.resolve(strict=True)))
        value={key:item for key,item in value.items()if not key.startswith('_')}
        if args.complete_source_contract:
            if not args.source or args.held_source or args.legacy_pair_context_weight is None:
                raise ValueError('complete source contract needs actual TRAIN sources and explicit raw coefficient; held stays unchanged')
            leaves, roots = {}, []
            for source in args.source:
                root = binding(source.resolve(strict=True)); roots.append(root)
                for child in native_source_dependencies(source) or [root]:
                    if Path(child['path']).suffix != '.json':
                        leaves[child['sha256']] = child
            if args.expected_source_count is not None and len(leaves) != args.expected_source_count:
                raise ValueError('complete CE active source count differs from the declared DAG')
            value = complete_source_contract(value, list(leaves.values()),
                context_weight=args.legacy_pair_context_weight, inventory=roots)
        elif not args.failed_history_mask_manifest:
            raise ValueError('prior-policy continuation requires explicit failed-history manifests')
    else:
        if args.complete_source_contract or args.expected_source_count is not None:
            raise ValueError('complete source contract requires an existing bound policy')
        if not args.source or args.legacy_pair_context_weight is None:
            raise ValueError('source policy preparation requires sources and explicit legacy pair coefficient')
        value = build(args.source, legacy_pair_context_weight=args.legacy_pair_context_weight, held_sources=args.held_source)
    if args.failed_history_mask_manifest:
        from natlang_neuralese.train.native_failed_history import binding
        value['failed_history_masks']=value.get('failed_history_masks',[])+[binding(path.resolve(strict=True))for path in args.failed_history_mask_manifest]
    if args.objective_reasoning_share is not None:
        value.update(own_output_objective(args))
    elif any(item is not None for item in (args.objective_tokenizer, args.objective_tokenizer_fingerprint,
                                           args.objective_family_balance, args.objective_transition_from)):
        raise ValueError('objective options require --objective-reasoning-share')
    if args.out.exists():
        if json.loads(args.out.read_text()) != value:
            raise ValueError('existing CE policy differs; write a new explicit continuation artifact')
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_durable(args.out, value)
    binding = {'path':str(args.out.resolve()),'sha256':sha256_file_hex(args.out)}
    load_policy(binding)
    print(json.dumps({'policy':binding,'declared_sources':len(value['sources']),
                      'retained_sources':len(value.get('retained_sources',[]))}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
