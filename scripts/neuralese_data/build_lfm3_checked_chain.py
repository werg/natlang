"""Build one complete, source-bound LFM3 training chain in a fresh namespace."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from transformers import AutoTokenizer
from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common import paths
from natlang_neuralese.data.native_graph_schema import parse_graph, communication_classification
from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
from scripts.neuralese_data.decision_holds import load_decision_holds
from scripts.neuralese_data.native_chain_render import render_chain
from scripts.neuralese_data.native_chain_materialize import materialize
from scripts.neuralese_data.native_chain_assembly import assemble


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--source-corpus-id', required=True)
    p.add_argument('--trajectory', required=True)
    p.add_argument('--tokenizer', type=Path, required=True)
    p.add_argument('--protection', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--current-source-reviews', type=Path,
                   default=paths.resolve('repo', 'training/source-reviews'),
                   help='live review authority; set explicitly when running a frozen or relocated checkout')
    args = p.parse_args()
    if args.out.exists(): raise ValueError('chain output must be a fresh namespace')
    root = Path(__file__).resolve().parents[2]
    code_paths = [Path(__file__).resolve(), *(root/path for path in (
        'scripts/neuralese_data/native_chain_sources.py', 'scripts/neuralese_data/native_chain_render.py',
        'scripts/neuralese_data/native_chain_materialize.py', 'scripts/neuralese_data/native_chain_assembly.py',
        'scripts/neuralese_data/build_native_semantic_pairs_e2e.py',
        'training/neuralese/natlang_neuralese/data/text_corpus.py',
        'training/neuralese/natlang_neuralese/data/trajectory_text.py',
        'training/neuralese/natlang_neuralese/data/prompt_rendering.py',
        'training/neuralese/natlang_neuralese/serve/chat.py',
        'training/neuralese/natlang_neuralese/data/native_graph_conversion.py',
        'training/neuralese/natlang_neuralese/data/native_graph_schema.py',
        'training/neuralese/natlang_neuralese/train/own_output_supervision.py',
        'training/neuralese/natlang_neuralese/model/native_graph_plan.py',
        'training/neuralese/natlang_neuralese/model/shared_state.py'))]
    code = {str(path): sha256_file_hex(path) for path in code_paths}
    reviews = {}
    for name in ('holds.jsonl', 'family-holds.json', 'datasets.json'):
        frozen = root/'training/source-reviews'/name
        current = args.current_source_reviews/name
        frozen_sha, current_sha = sha256_file_hex(frozen), sha256_file_hex(current)
        if frozen_sha != current_sha:
            raise ValueError('frozen source reviews differ from the live current review: '+name)
        reviews[name] = {'frozen': str(frozen), 'current': str(current), 'sha256': frozen_sha}
    receipt_path = args.source/'extract-receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if (receipt.get('parent') or {}).get('id') != args.source_corpus_id:
        raise ValueError('source corpus identity differs from the extraction receipt parent')
    refs = {}
    def stat_key(value):
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    for name in ('chains.jsonl', 'edges.jsonl', 'records.jsonl', 'pieces.jsonl'):
        path = args.source/name; expected = receipt['files'][name]
        initial = stat_key(path.stat())
        if initial[2] != expected['bytes'] or (name != 'records.jsonl' and sha256_file_hex(path) != expected['sha256']):
            raise ValueError('source differs from its exact extraction receipt: '+name)
        if stat_key(path.stat()) != initial:
            raise ValueError('source changed while checking extraction receipt: '+name)
        refs[name] = {'path': str(path.resolve()), **expected,
                      'stat': {'device': initial[0], 'inode': initial[1], 'size': initial[2],
                               'mtime_ns': initial[3], 'ctime_ns': initial[4]}}
    matches = [json.loads(line) for line in (args.source/'chains.jsonl').open()
               if json.loads(line).get('trajectory') == args.trajectory]
    if len(matches) != 1 or matches[0]['split'] != 'train':
        raise ValueError('select exactly one authenticated training chain')
    chain = matches[0]
    membership = [x for x in receipt['membership']['chains'] if x['trajectory'] == args.trajectory]
    if len(membership) != 1 or membership[0]['record_ids'] != chain['records'] or membership[0]['edge_count'] != chain['edges']:
        raise ValueError('chain differs from original extraction membership')
    wanted = set(chain['records']); wanted_bytes = {x.encode() for x in wanted}; records = {}
    record_hash = hashlib.sha256()
    with (args.source/'records.jsonl').open('rb') as stream:
        before = stat_key(os.fstat(stream.fileno()))
        if before != tuple(refs['records.jsonl']['stat'][key] for key in ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns')):
            raise ValueError('source records changed before the hash and extraction pass')
        for line in stream:
            record_hash.update(line)
            if not any(rid in line for rid in wanted_bytes): continue
            row = json.loads(line)
            if row.get('id') in wanted:
                if row['id'] in records: raise ValueError('duplicate source record identity')
                records[row['id']] = row
        if stat_key(os.fstat(stream.fileno())) != before:
            raise ValueError('source records changed during the hash and extraction pass')
    if stat_key((args.source/'records.jsonl').stat()) != before:
        raise ValueError('source records were replaced during the hash and extraction pass')
    if record_hash.hexdigest() != receipt['files']['records.jsonl']['sha256']:
        raise ValueError('source records differ from the exact extraction receipt')
    refs['records.jsonl']['stat'] = {'device': before[0], 'inode': before[1], 'size': before[2],
                                    'mtime_ns': before[3], 'ctime_ns': before[4]}
    edges = [json.loads(line) for line in (args.source/'edges.jsonl').open()
             if json.loads(line).get('trajectory') == args.trajectory]
    protection = current_protection(args.protection)
    blocked = protection['held'] | protection['quality'] | protection['pending']
    holds, hold_bindings = load_decision_holds(sorted((args.protection.parent/'decision-holds').glob('*.jsonl')))
    if wanted & set(holds) or any(set(row.get('source_groups') or []) & blocked for row in records.values()):
        raise ValueError('source trajectory intersects current protected or quality-held data')
    tok = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, trust_remote_code=False)
    role_ids = provision(tok); fingerprint = tokenizer_fingerprint(tok)
    if fingerprint != '133504d493a77e3121f62cc8c83699cb222ee993c46959d604c4b683a2ffdaf3' or list(role_ids) != [125033,125034,125035,125036]:
        raise ValueError('LFM3 tokenizer or role marker fingerprint differs')
    pieces = [json.loads(line) for line in (args.source/'pieces.jsonl').open()]
    rendered = render_chain(chain, records, edges, pieces, tokenizer=tok, tokenizer_fingerprint=fingerprint)
    nodes = materialize(rendered, records, edges, tokenizer=tok, role_ids=role_ids)
    binding = {'source_corpus_id': args.source_corpus_id,
               'extraction_receipt': {'path': str(receipt_path.resolve()), 'sha256': sha256_file_hex(receipt_path)},
               'files': refs, 'protection': protection['proof'], 'decision_holds': hold_bindings}
    graph = assemble(chain, records, edges, nodes, tokenizer=tok, tokenizer_fingerprint=fingerprint,
                     role_ids=role_ids, source_binding=binding)
    if any(sha256_file_hex(Path(path)) != sha for path, sha in code.items()):
        raise ValueError('chain renderer code changed during this source conversion')
    if stat_key((args.source/'records.jsonl').stat()) != before:
        raise ValueError('source records changed before chain publication')
    for item in refs.values():
        expected_stat = item['stat']
        if stat_key(Path(item['path']).stat()) != tuple(expected_stat[key] for key in
                ('device', 'inode', 'size', 'mtime_ns', 'ctime_ns')):
            raise ValueError('source dependency changed before chain publication')
    if any(sha256_file_hex(Path(item['current'])) != item['sha256'] for item in reviews.values()):
        raise ValueError('current source reviews changed during this source conversion')
    final_protection = current_protection(args.protection)
    final_holds, final_hold_bindings = load_decision_holds(sorted((args.protection.parent/'decision-holds').glob('*.jsonl')))
    final_blocked = final_protection['held'] | final_protection['quality'] | final_protection['pending']
    if (final_protection['proof'] != protection['proof'] or final_hold_bindings != hold_bindings
            or wanted & final_holds or
            any(set(row.get('source_groups') or []) & final_blocked for row in records.values())):
        raise ValueError('source chain became protected while conversion was running')
    parsed = parse_graph(graph, tokenizer_fingerprint=fingerprint)
    if len(parsed.plan.invocations) < 3 or not communication_classification(parsed)['producer_consumer']:
        raise ValueError('complete chain lacks a genuine multi-node state transfer')
    args.out.mkdir(parents=True)
    data = (json.dumps(graph, ensure_ascii=False, sort_keys=True, separators=(',', ':'))+'\n').encode()
    graph_path = args.out/'native-graphs.jsonl'; graph_path.write_bytes(data)
    result = {'schema': 'natlang.lfm3-checked-source-chain/1', 'source': binding, 'code': code, 'reviews': reviews, 'trajectory': args.trajectory,
              'split': chain['split'], 'prompt_rendering': 'program-header', 'nodes': len(graph['nodes']), 'edges': len(edges),
              'imports': sum(len(x['imports']) for x in graph['nodes']), 'fingerprint': fingerprint,
              'graph': {'path': str(graph_path.resolve()), 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)},
              'note': 'source-bound parse only; actual one-pass model and gradient qualification remains required'}
    (args.out/'receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__': main()
