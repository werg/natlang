#!/usr/bin/env python3
"""Keep novel acquired graphs while retaining qualified predecessor versions.

Identity collisions are not equivalent examples: parse and record both actual
training bindings. The original source stays immutable for later correction
review. This neither grants source admission nor changes historical exposure.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'training/neuralese')]
from natlang_neuralese.common.custody import ProtectedWorlds
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from natlang_neuralese.common.jsonio import open_jsonl_text, write_json_durable
from scripts.neuralese_data.build_acquired_streaming_successor import bind, checked, read, receipt_protection_digest
from scripts.neuralese_data.build_native_graph_text_corpus import GraphWriter
from scripts.neuralese_data.parallel_native_graph_index import _init, _parse_line

SCHEMA = 'natlang.acquired-predecessor-collision-filter/1'


def filter_source(predecessor, screen_path, census_path, output):
    predecessor, screen_path, census_path = [Path(p).resolve(strict=True) for p in (predecessor, screen_path, census_path)]
    controls = [bind(p) for p in (predecessor, screen_path, census_path)]
    previous, screen, census = [read(p) for p in (predecessor, screen_path, census_path)]
    protection = ProtectedWorlds.load()
    if (receipt_protection_digest(previous) != protection.digest
            or screen.get('protected_worlds', {}).get('identity_set_sha256') != protection.digest
            or screen.get('protected_worlds', {}).get('matches_expected') is not True
            or census.get('screen') != controls[1]):
        raise ValueError('collision filter requires the exact screened census and current protection union')
    family_path = checked(previous['family_map'], 'predecessor family map')
    family = read(family_path)
    if family.get('source_sha256') != previous['source']['sha256']:
        raise ValueError('predecessor family map source differs')
    old = {row['id']: row['binding'] for row in family['rows']}
    if len(old) != previous['rows'] or len(old) != len(family['rows']):
        raise ValueError('predecessor repeats or omits graph identities')
    graph_path = checked(screen['output'], 'screened source')
    signals_path = checked(census['row_signals'], 'source census signals')
    geometry = read(checked(previous['geometry'], 'predecessor geometry'))
    _init(geometry['identity']['tokenizer'], geometry['identity']['roles'])
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    writer = GraphWriter(output / 'train/native-graphs.jsonl.zst')
    omitted, seen, kept, counts = [], set(), [], Counter()
    with ExitStack() as cleanup, open_jsonl_text(graph_path) as graphs, open_jsonl_text(signals_path) as signals:
        cleanup.callback(writer.abort)
        signal_iter = (json.loads(line) for line in signals if line.strip())
        for line in graphs:
            if not line.strip(): continue
            row = json.loads(line); signal = next(signal_iter, None)
            if (signal is None or row['id'] in seen or row.get('split') != 'train'
                    or signal['id'] != row['id'] or signal['graph_sha256'] != canonical_json_sha256_hex(row)):
                raise ValueError('graph/census order, content, split or identity differs')
            seen.add(row['id'])
            if row['id'] in old:
                entry, _, _ = _parse_line(line)
                omitted.append({'id': row['id'], 'previous_binding': old[row['id']],
                    'candidate_binding': entry['binding'], 'candidate_graph_sha256': signal['graph_sha256'],
                    'equivalent_training_binding': old[row['id']] == entry['binding'],
                    'disposition': 'retain-qualified-predecessor; candidate retained in original immutable source'})
                continue
            writer.write(row); kept.append(row['id'])
            counts['node_forward_tokens'] += signal['forward_tokens']
            counts['positive_target_tokens'] += signal['raw_positive_tokens']
            if len(seen) % 512 == 0:
                print(json.dumps({'kind': 'acquired_collision_filter', 'rows': len(seen), 'retained': len(kept), 'collisions': len(omitted)}), flush=True)
        if next(signal_iter, None) is not None or len(seen) != census['rows'] or len(seen) != screen['output']['rows']:
            raise ValueError('screen/census row counts differ')
        result = writer.finish()
    if not kept: raise ValueError('collision filtering leaves no novel graphs')
    if ([bind(p) for p in (predecessor, screen_path, census_path)] != controls
            or bind(graph_path)['sha256'] != screen['output']['sha256']
            or bind(signals_path) != census['row_signals']
            or bind(family_path) != previous['family_map']
            or ProtectedWorlds.load().digest != protection.digest):
        raise ValueError('collision filter inputs changed')
    graph = {**screen['output'], 'path': str(output / 'train/native-graphs.jsonl.zst'),
        **{key: result[key] for key in ('rows', 'bytes', 'sha256')},
        'graph_ids_sha256': canonical_json_sha256_hex(sorted(kept))}
    proof_path = output / 'collision-filter.json'
    write_json_durable(proof_path, {'schema': SCHEMA, 'predecessor': controls[0], 'original_screen': controls[1],
        'original_census': controls[2], 'family_map': previous['family_map'], 'input_graph': screen['output'],
        'output_graph': graph, 'excluded': omitted, 'protected_worlds_sha256': protection.digest,
        'builder': bind(Path(__file__)), 'training_admission': False,
        'scope': 'retain existing qualified versions; all changed candidates remain available for explicit source correction'})
    derived = copy.deepcopy(screen)
    derived['predecessor_collision_filter'] = bind(proof_path)
    derived['output'] = graph
    derived['accounting'].update(eligible_train_graphs=len(kept), predecessor_collisions=len(omitted), **counts)
    derived['status'] = 'Novel rows only; predecessor versions retained; no source admission or capability claim.'
    write_json_durable(output / 'receipt.json', derived)
    return {'receipt': bind(output / 'receipt.json'), 'retained': len(kept), 'collisions': len(omitted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ('predecessor', 'screen', 'census', 'out'): parser.add_argument('--' + flag, required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(filter_source(args.predecessor, args.screen, args.census, args.out)), flush=True)


if __name__ == '__main__': main()
