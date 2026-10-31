#!/usr/bin/env python3
"""Re-encode the admitted acquired agent sessions for another student tokenizer (LFM3, 2026-10-31).

The admitted acquired-session corpora (registry source_class
``source-curated-complete-acquired-agent-native-session-graphs``) were converted for LFM2.5 and Mellum2.1 only. Their
normalized complete-session exports are immutable; this runs the same converter
(build_acquired_full_session_graphs.py: eligibility, current protection and decision holds, held-group closure, exact
receipts) over every export those registry entries bind, with the student's tokenizer prepared once per process.

  convert_acquired_sessions_student.py exports --out EXPORTS.json
      every export receipt bound by an admitted entry's admission audit (sorted, de-duplicated)
  convert_acquired_sessions_student.py convert --exports EXPORTS.json --out-root DIR --protection MANIFEST
      [--student lfm3] [--shard K --shards N]
      resumable: an export whose output has a receipt is skipped
  convert_acquired_sessions_student.py summary --exports EXPORTS.json --out-root DIR --out SUMMARY.json

Output: DIR/<source>/<export id>/{train,test}/native-graphs.jsonl.zst + receipt.json, one per export. Admission is
inherited from the registry entries; a mixture still applies the complete current protection union.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
sys.path.insert(0, str(ROOT))
REGISTRY = ROOT / 'training' / 'neuralese_corpora.json'
SOURCE_CLASS = 'source-curated-complete-acquired-agent-native-session-graphs'


def exports(args):
    corpora = json.loads(REGISTRY.read_text())['corpora']
    found = set()
    for entry in corpora:
        if entry.get('source_class') != SOURCE_CLASS or not entry.get('training_admission'):
            continue
        audit = (entry.get('build') or {}).get('audit')
        if not audit:
            continue
        text = (ROOT / entry['path'] / audit).read_text()
        found.update(re.findall(r'"(/[^"]*?/export\.json)"', text))
        # Older audits bind the converted graph receipts, which name their export receipt.
        for receipt in set(re.findall(r'"(/[^"]*?/lfm25/receipt\.json)"', text)):
            if Path(receipt).exists():
                found.add(json.loads(Path(receipt).read_text())['export_receipt']['path'])
    listed = sorted(p for p in found if Path(p).exists())
    Path(args.out).write_text(json.dumps({'schema': 'natlang.acquired-session-exports/1', 'exports': listed,
                                          'missing': sorted(found - set(listed))}, indent=1) + '\n')
    print(json.dumps({'exports': len(listed), 'missing': len(found) - len(listed)}))


def output_dir(root: Path, export: str) -> Path:
    """DIR/<source>/<export id>: the dataset label (D12, D32, ...) and the export's own directory chain."""
    parts = Path(export).parent.parts
    label = next((p for p in reversed(parts) if re.fullmatch(r'D\d+', p)), 'other')
    tail = parts[parts.index(label) + 1:] if label in parts else parts[-1:]
    return root / label / '/'.join(tail)


def convert(args):
    from scripts.neuralese_data import build_acquired_full_session_graphs as builder
    sys.path.insert(0, str(ROOT / 'scripts' / 'neuralese_data'))
    from student_tokenizers import resolve
    listed = json.loads(Path(args.exports).read_text())['exports']
    mine = [e for i, e in enumerate(listed) if i % args.shards == args.shard]
    tokenizer = Path(resolve(args.student)['path'])
    prepared = builder.prepare_tokenizer(tokenizer)
    # The protection projection is the same for every export: verify and load it once per process (the converter
    # reloads it per call; 14k small exports would spend hours re-hashing it). The cached value is never mutated.
    loaded = builder.current_protection(Path(args.protection))

    def cached_protection(path, **_):
        if Path(path) != Path(args.protection):
            raise ValueError('one protection manifest per conversion process')
        return loaded
    builder.current_protection = cached_protection
    holds = sorted((ROOT / 'training' / 'decision-holds').glob('*.jsonl'))
    done = skipped = 0
    for export in mine:
        out = output_dir(Path(args.out_root), export)
        if (out / 'receipt.json').exists():
            skipped += 1
            continue
        argv = ['--export-receipt', export, '--base-tokenizer', str(tokenizer), '--protection', args.protection,
                '--output', str(out)]
        for hold in holds:
            argv += ['--decision-hold', str(hold)]
        out.parent.mkdir(parents=True, exist_ok=True)
        builder.main(argv, prepared=prepared)
        done += 1
        print(json.dumps({'converted': export, 'out': str(out)}), flush=True)
    print(json.dumps({'shard': args.shard, 'converted': done, 'skipped_complete': skipped}))


def summary(args):
    listed = json.loads(Path(args.exports).read_text())['exports']
    totals, by_source, missing = {}, {}, []
    for export in listed:
        out = output_dir(Path(args.out_root), export)
        receipt = out / 'receipt.json'
        if not receipt.exists():
            missing.append(export)
            continue
        value = json.loads(receipt.read_text())
        source = out.relative_to(args.out_root).parts[0]
        bucket = by_source.setdefault(source, {'exports': 0, 'counts': {}, 'outputs': []})
        bucket['exports'] += 1
        for key, count in (value.get('counts') or {}).items():
            bucket['counts'][key] = bucket['counts'].get(key, 0) + count
            totals[key] = totals.get(key, 0) + count
        bucket['outputs'] += [{k: o[k] for k in ('path', 'rows', 'bytes', 'sha256')} for o in value['outputs'] if o['rows']]
    Path(args.out).write_text(json.dumps({'schema': 'natlang.acquired-session-student-conversion/1',
                                          'totals': totals, 'by_source': by_source, 'incomplete': missing},
                                         indent=1) + '\n')
    print(json.dumps({'totals': totals, 'incomplete': len(missing)}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('exports')
    p.add_argument('--out', required=True)
    p = sub.add_parser('convert')
    p.add_argument('--exports', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--protection', required=True)
    p.add_argument('--student', default='lfm3')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p = sub.add_parser('summary')
    p.add_argument('--exports', required=True)
    p.add_argument('--out-root', required=True)
    p.add_argument('--out', required=True)
    args = ap.parse_args()
    {'exports': exports, 'convert': convert, 'summary': summary}[args.cmd](args)


if __name__ == '__main__':
    main()
