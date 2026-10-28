#!/usr/bin/env python3
"""Exact-oracle ``decide`` readouts -> native decision graph rows (data/decision_graph.py) for the student tokenizers.

Input: ``natlang.decision-prompt/1`` rows (the runtime's exact decision prompts: ``messages``, ``options``, ``gold``,
``split``), e.g. ``refine-judge-exact`` (crisp-checker labels, scripts/refine_data/exact.py, regenerable at any
scale with ``scripts/refine_data_cli.py exact --per-predicate N``). Only rows whose label comes from an exact oracle
are converted (``--label-source-prefix``, default ``crisp:``); train rows only, held rows are counted and left out.

Each row becomes one graph: K option nodes (the served prompt + each option's closed continuation, rendered by the
serving engine's own renderer) and a ``decisions`` entry with the exact label, supervised over the option set by the
own-output objective's decision readout (train/decision_supervision.py). Replaces the earlier text conversion that
trained the gold option as a generated reply (convert_refine_rows.py judge rows).

Usage (CPU, under the DGX memory ledger):
  build_decision_graphs.py --source judge-rows.jsonl [...] --student lfm3 [--student lfm25] --out-dir OUT
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT))

DECISION_PROMPT = 'natlang.decision-prompt/1'


def decision_record(row, *, label_source_prefix):
    """(record, omission reason)."""
    if row.get('schema') != DECISION_PROMPT:
        return None, 'not a decision-prompt row'
    if row.get('split') != 'train':
        return None, f"held split {row.get('split')}"
    if not str(row.get('label_source', '')).startswith(label_source_prefix):
        return None, 'label is not an exact oracle'
    gold, options = row.get('gold'), row.get('options')
    if not isinstance(options, list) or len(options) < 2 or not isinstance(gold, list) or len(gold) != len(options):
        return None, 'malformed options or gold'
    record = {'id': row['id'] + ':decision', 'split': 'train',
              'source_groups': [f"decision-prompt-group:{row.get('group') or row['id']}"],
              'messages': row['messages'], 'options': [str(option) for option in options],
              'label_source': 'exact-oracle'}
    if sorted(gold) == [0.] * (len(gold) - 1) + [1.]:
        record['label'] = gold.index(1.)
    else:
        record['probabilities'] = [float(p) for p in gold]
    return record, None


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', type=Path, nargs='+', required=True)
    parser.add_argument('--student', action='append', required=True, help='student tokenizer key (lfm3, lfm25, ...)')
    parser.add_argument('--out-dir', type=Path, required=True)
    parser.add_argument('--label-source-prefix', default='crisp:')
    args = parser.parse_args()
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.common.jsonio import write_jsonl_text
    from natlang_neuralese.data.decision_graph import decision_graph_row, tokenizer_renderer
    from natlang_neuralese.data.native_graph import parse_graph
    from scripts.neuralese_data.teacher_graph_student_twin import student_tokenizer
    from natlang_neuralese.common import line_backbones
    records, omissions = [], collections.Counter()
    for source in args.source:
        for line in source.read_text().splitlines():
            if not line.strip():
                continue
            record, reason = decision_record(json.loads(line), label_source_prefix=args.label_source_prefix)
            if reason:
                omissions[reason] += 1
            else:
                records.append(record)
    receipt = {'schema': 'natlang.decision-graph-conversion/1',
               'sources': [{'path': str(path.resolve()), 'sha256': sha256_file_hex(path)} for path in args.source],
               'records': len(records), 'omissions': dict(omissions), 'students': {}}
    for key in args.student:
        tokenizer, identity = student_tokenizer(key)
        # The backbone's declared model_type selects its tool-call format and history-reasoning declaration.
        model_type = line_backbones.backbone(key)['model_id_architecture']
        renderer = tokenizer_renderer(tokenizer, model_type=model_type,
                                      history_reasoning_policy=identity['history_reasoning']['policy'])
        rows, tokens = [], 0
        for record in records:
            row = decision_graph_row(renderer, record, tokenizer_fingerprint=identity['expanded_fingerprint'])
            parse_graph(row, tokenizer_fingerprint=identity['expanded_fingerprint'])
            tokens += sum(len(node['token_ids']) for node in row['nodes'])
            rows.append(row)
        out = args.out_dir / key / 'train' / 'native-graphs.jsonl'
        out.parent.mkdir(parents=True, exist_ok=True)
        with write_jsonl_text(out) as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + '\n')
        receipt['students'][key] = {'tokenizer': {k: identity[k] for k in ('source_fingerprint', 'expanded_fingerprint',
                                                                            'model_id')},
                                    'rows': len(rows), 'decisions': len(rows), 'tokens': tokens,
                                    'path': str(out), 'sha256': sha256_file_hex(out)}
    (args.out_dir / 'receipt.json').write_text(json.dumps(receipt, indent=1, sort_keys=True) + '\n')
    print(json.dumps({key: value['rows'] for key, value in receipt['students'].items()} | {'omissions': dict(omissions)}))


if __name__ == '__main__':
    main()
