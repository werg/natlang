#!/usr/bin/env python3
"""Learned-operator case files -> native graphs plus text-cut siblings (COMPACTION_OPERATOR_CONVERTERS.md §1).

    build_operator_graphs.py --cases CASES/<operator>/<source>.port-records.jsonl | CASES [--operator NAME]
        [--out-root DIR] [--corpus-id ID] [--workers 4] [--limit N]

Writes OUT/<operator>/<source>/{graphs,textcut}-{train,test}.jsonl.zst, omissions.jsonl and receipt.json. A finished
output (receipt over the same input bytes) is skipped, so the command can be rerun as converters land.

Custody: a train case whose split groups meet the current protection projection (held, quality-only or pending
source-review groups, `current_protection`) or whose world is related to a ProtectedWorlds identity is omitted with
its reason. Test cases become held (`split: test`) graphs: they score the ablation and are never trained. Only
gold/checked targets are supervised; prompts, sources and tool output carry zero weight.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'training' / 'neuralese'), str(ROOT)]
from natlang_neuralese.common.hashing import sha256_file_hex  # noqa: E402
from natlang_neuralese.common.paths import resolve as data_path  # noqa: E402

LFM3_FINGERPRINT = '133504d493a77e3121f62cc8c83699cb222ee993c46959d604c4b683a2ffdaf3'
LFM3_ROLES = [125033, 125034, 125035, 125036]
OUT_ROOT = data_path('data_hdd', 'natlang-compaction-operators-20261103', 'graphs')
CODE = ('scripts/neuralese_data/build_operator_graphs.py', 'training/neuralese/natlang_neuralese/data/operator_graph.py',
        'training/neuralese/natlang_neuralese/model/learned_operators.py',
        'training/neuralese/natlang_neuralese/data/port_record_graph.py',
        'training/neuralese/natlang_neuralese/serve/chat.py')
_RENDERER = None


def load_tokenizer():
    """The expanded LFM3 tokenizer (four role markers), checked against the production fingerprint."""
    from transformers import AutoTokenizer
    from scripts.neuralese_data.build_chain_graphs import LFM3
    from scripts.neuralese_data.build_native_semantic_pairs_e2e import provision
    from natlang_neuralese.data.text_corpus import tokenizer_fingerprint
    tokenizer = AutoTokenizer.from_pretrained(str(LFM3), trust_remote_code=False, local_files_only=True)
    roles = provision(tokenizer)
    fingerprint = tokenizer_fingerprint(tokenizer)
    if fingerprint != LFM3_FINGERPRINT or list(roles) != LFM3_ROLES:
        raise SystemExit('expanded LFM3 tokenizer differs from the production fingerprint/roles')
    return tokenizer, roles, fingerprint


def make_renderer(max_tokens):
    from natlang_neuralese.data.operator_graph import OperatorGraphRenderer
    from natlang_neuralese.model.learned_operators import OPERATORS, bind_tokens
    tokenizer, roles, fingerprint = load_tokenizer()
    bindings = {name: bind_tokens(tokenizer, name, tokenizer_fingerprint=fingerprint) for name in OPERATORS}
    return OperatorGraphRenderer(tokenizer, role_ids=roles, tokenizer_fingerprint=fingerprint, bindings=bindings,
                                 max_tokens=max_tokens)


def _init(max_tokens):
    global _RENDERER
    _RENDERER = make_renderer(max_tokens)


def _render_batch(task):
    from natlang_neuralese.data.operator_graph import OperatorCaseError
    operator, corpus_id, path, items = task
    out = []
    for line_number, offset, line in items:
        sha = hashlib.sha256(line.encode()).hexdigest()
        record = json.loads(line)
        try:
            graph, cut = _RENDERER.render(record, operator, corpus_id=corpus_id, source_row_sha256=sha,
                                          source_ref={'path': path, 'line': line_number, 'byte_offset': offset})
            out.append(('ok', record['id'], graph, cut))
        except (OperatorCaseError, ValueError, KeyError, TypeError) as error:
            out.append(('omit', record.get('id'), {'reason': str(error)[:300], 'line': line_number}, None))
    return out


class ZstLines:
    def __init__(self, path):
        import zstandard
        self.raw = open(path, 'wb')
        self.stream = zstandard.ZstdCompressor(level=6).stream_writer(self.raw)
        self.count = 0

    def write(self, row):
        self.stream.write((json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n').encode())
        self.count += 1

    def close(self):
        self.stream.close()


def batches(path, size, limit):
    batch, offset = [], 0
    with open(path, 'rb') as stream:
        for line_number, raw in enumerate(stream, 1):
            start, offset = offset, offset + len(raw)
            if limit is not None and line_number > limit:
                break
            if raw.strip():
                batch.append((line_number, start, raw.decode('utf-8').rstrip('\n')))
            if len(batch) == size:
                yield batch
                batch = []
    if batch:
        yield batch


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--cases', type=Path, required=True)
    parser.add_argument('--operator')
    parser.add_argument('--out-root', type=Path, default=OUT_ROOT)
    parser.add_argument('--corpus-id', default='natlang-compaction-operators-20261103')
    parser.add_argument('--protection', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--max-tokens', type=int, default=32768)
    args = parser.parse_args(argv)
    if args.cases.is_dir():
        # Every finished converter output under a cases root (its manifest is written last), each rendered once.
        status = 0
        for manifest in sorted(args.cases.glob('*/*.manifest.json')):
            cases = manifest.with_name(manifest.name.removesuffix('.manifest.json') + '.port-records.jsonl')
            forwarded = [item for item in (argv if argv is not None else sys.argv[1:])]
            index = forwarded.index('--cases')
            forwarded[index + 1] = str(cases)
            status |= main(forwarded)
        return status
    from natlang_neuralese.common.custody import ProtectedWorlds
    from scripts.neuralese_data.build_chain_graphs import PROTECTION
    from scripts.neuralese_data.build_verified_teacher_trajectory_graphs import current_protection
    operator = args.operator or args.cases.parent.name
    source = args.cases.name.removesuffix('.port-records.jsonl')
    out = args.out_root / operator / source
    input_sha = sha256_file_hex(args.cases)
    receipt_path = out / 'receipt.json'
    if receipt_path.exists() and json.loads(receipt_path.read_text())['input']['sha256'] == input_sha \
            and json.loads(receipt_path.read_text()).get('limit') == args.limit:
        print('already complete:', out)
        return 0
    out.mkdir(parents=True, exist_ok=True)
    protection = current_protection(args.protection or PROTECTION)
    blocked = {'held': protection['held'], 'quality': protection['quality'], 'pending': protection['pending']}
    worlds = ProtectedWorlds.load()
    writers = {(kind, split): ZstLines(out / f'{kind}-{split}.jsonl.zst.tmp')
               for kind in ('graphs', 'textcut') for split in ('train', 'test')}
    omissions = open(out / 'omissions.jsonl', 'w')
    reasons, started = {}, time.time()
    tokens = {'train': 0, 'test': 0}

    def omit(record_id, reason, detail=None):
        key = reason.split(':')[0]
        reasons[key] = reasons.get(key, 0) + 1
        omissions.write(json.dumps({'id': record_id, 'reason': reason, **(detail or {})}) + '\n')

    tasks = ((operator, args.corpus_id, str(args.cases), batch) for batch in batches(args.cases, args.batch, args.limit))
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(args.max_tokens,)) as pool:
        for results in pool.map(_render_batch, tasks):
            for status, record_id, graph, cut in results:
                if status == 'omit':
                    omit(record_id, graph['reason'], {'line': graph['line']})
                    continue
                if graph['split'] == 'train':
                    groups = set(graph['source_groups'])
                    hit = next((name for name, members in blocked.items() if groups & members), None)
                    if hit:
                        omit(record_id, f'protection-{hit}-group')
                        continue
                    kinds = worlds.related({'id': record_id, 'split': 'train', 'source_groups': graph['source_groups']})
                    if kinds:
                        omit(record_id, 'protected-world-related', {'kinds': kinds})
                        continue
                writers[('graphs', graph['split'])].write(graph)
                writers[('textcut', graph['split'])].write(cut)
                tokens[graph['split']] += sum(len(node['token_ids']) for node in graph['nodes'])
    omissions.close()
    counts = {}
    for (kind, split), writer in writers.items():
        writer.close()
        final = out / f'{kind}-{split}.jsonl.zst'
        os.replace(out / f'{kind}-{split}.jsonl.zst.tmp', final)
        counts[f'{kind}-{split}'] = {'rows': writer.count, 'sha256': sha256_file_hex(final)}
    from natlang_neuralese.model.learned_operators import registry_sha256
    receipt = {'schema': 'natlang.operator-graphs-build/1', 'operator': operator, 'source': source,
               'input': {'path': str(args.cases), 'sha256': input_sha, 'bytes': args.cases.stat().st_size},
               'limit': args.limit, 'corpus_id': args.corpus_id, 'registry_sha256': registry_sha256([operator]),
               'tokenizer': {'fingerprint': LFM3_FINGERPRINT, 'role_ids': LFM3_ROLES},
               'protection': protection['proof'], 'protected_worlds_sha256': worlds.digest,
               'code': {path: sha256_file_hex(ROOT / path) for path in CODE},
               'outputs': counts, 'graph_tokens': tokens, 'omissions': reasons,
               'max_tokens': args.max_tokens, 'seconds': round(time.time() - started, 1),
               'admission': 'rendered operator graphs; training admission is a separate recorded stage decision'}
    receipt_path.write_text(json.dumps(receipt, indent=1) + '\n')
    print(json.dumps({key: receipt[key] for key in ('operator', 'source', 'outputs', 'omissions', 'graph_tokens')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
