#!/usr/bin/env python3
"""Verify and compact reviewed CLINC label-derived static references.

The semantic decision is supplied in --review. This script only checks exact
source/label/replay/turn bindings, records the approved static scope, and
compresses the large replay and turn streams. It never claims teacher traces.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def compact(path):
    compressed = path.with_name(path.name + '.zst')
    if compressed.exists():
        raise ValueError(f'already compacted; review this batch before rerun: {compressed}')
    original_sha = digest(path)
    size = path.stat().st_size
    subprocess.run(['/usr/bin/zstd', '-q', '-3', '-T1', '--rm', '--', str(path)], check=True)
    return {'path': str(compressed), 'sha256': digest(compressed),
            'decoded_sha256': original_sha, 'decoded_bytes': size,
            'compressed_bytes': compressed.stat().st_size}


def batch_review(root, number, all_matching, all_groups):
    source = root / f'gemini-clinc-fresh-batch{number:02d}'
    adapted = root / f'gemini-clinc-batch{number:02d}-static-adapter-v2-full-replay'
    source_cases = source / 'cases.jsonl'
    labels = source / 'labels.jsonl'
    quality = json.loads((source / 'quality-summary.json').read_text())
    summary = json.loads((adapted / 'summary.json').read_text())
    source_manifest = json.loads((source / 'source-manifest.json').read_text())
    label_manifest = json.loads((source / 'labels.jsonl.manifest.json').read_text())
    if (source_manifest['cases_sha256'] != digest(source_cases) or
            quality['cases_sha256'] != digest(source_cases) or
            quality['labels_sha256'] != digest(labels) or
            label_manifest['cases_sha256'] != digest(source_cases) or
            summary['schema'] != 'natlang.held-static-decision-adapter-summary/1' or
            summary['execution_component_verification']['verified'] is not True or
            summary['training_admission'] is not False):
        raise ValueError(f'batch {number}: source/adapter binding failed')
    cases = {row['id']: row for row in rows(source_cases)}
    labels_by_id = {row['id']: row for row in rows(labels)}
    if len(cases) != len(labels_by_id) or len(cases) != summary['source_cases']:
        raise ValueError(f'batch {number}: missing or duplicate labels')
    matching = {key for key, case in cases.items()
                if isinstance(labels_by_id[key].get('answer'), dict) and
                labels_by_id[key]['answer'].get('choice') == case['answer']}
    if matching != set(quality['source_matching_ids']) or len(matching) != summary['source_labels_accepted']:
        raise ValueError(f'batch {number}: source-matching label set changed')
    programs = {}
    used = set()
    for program in rows(adapted / 'cases.jsonl'):
        pid = program['id']
        if pid in programs:
            raise ValueError(f'batch {number}: duplicate program')
        ids = {receipt['id'] for receipt in program['generation']['label_receipts']}
        if not ids or used.intersection(ids) or not ids.issubset(matching):
            raise ValueError(f'batch {number}: program labels are duplicate or unreviewed')
        used.update(ids)
        programs[pid] = ids
    if used != matching or len(programs) != summary['cases']:
        raise ValueError(f'batch {number}: eligible labels not fully converted')
    replays = {}
    for result in rows(adapted / 'replay.results.jsonl'):
        program = result['row']['task']['program_ir']['id']
        if (program in replays or program not in programs or
                result['admission']['admitted'] is not True or
                result['row']['outcome']['accepted'] is not True or
                result['capture_visibility']['visible'] is not True):
            raise ValueError(f'batch {number}: incomplete authored reference replay')
        replays[program] = True
    if (set(replays) != set(programs) or summary['replay_attempted'] != len(programs) or
            summary['replay_failures'] or (adapted / 'replay.failures.jsonl').stat().st_size):
        raise ValueError(f'batch {number}: replay coverage or failure mismatch')
    turns = 0
    for row in rows(adapted / 'native.held.jsonl'):
        if (row['program_id'] not in programs or row['split'] != 'train' or
                row['training_admission']['kind'] != 'authored-static-reference-pending-review' or
                row['training_admission']['approved'] is not False or
                row['trace_admission']['admitted'] is not False or
                row['decision']['training_approved'] is not False):
            raise ValueError(f'batch {number}: native turn admission/provenance mismatch')
        turns += 1
    if turns != summary['native_turns']:
        raise ValueError(f'batch {number}: native turn count mismatch')
    for key in labels_by_id:
        if key not in matching:
            continue
        group = cases[key]['group']
        if key in all_matching or group in all_groups:
            raise ValueError(f'batch {number}: duplicate source id/group across batches')
        all_matching[key] = {'batch': number, 'program': next(p for p, ids in programs.items() if key in ids),
                             'state': cases[key]['state'], 'answer': cases[key]['answer']}
        all_groups.add(group)
    return {'batch': number, 'source_rows': len(cases), 'source_matching': len(matching),
            'programs': len(programs), 'native_turns': turns,
            'files': {name: digest(path) for name, path in {
                'cases': source_cases, 'labels': labels,
                'source_manifest': source / 'source-manifest.json',
                'label_manifest': source / 'labels.jsonl.manifest.json',
                'quality_summary': source / 'quality-summary.json',
                'adapted_cases': adapted / 'cases.jsonl',
                'adapted_summary': adapted / 'summary.json',
                'adapted_source_manifest': adapted / 'source-manifest.json',
                'adapted_label_manifest': adapted / 'label-manifest.json',
                'adapted_scorer_manifest': adapted / 'scorer-manifest.json',
                'replays': adapted / 'replay.results.jsonl',
                'native_turns': adapted / 'native.held.jsonl',
            }.items()}, 'adapted_dir': str(adapted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--review', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--batches', type=int, default=44)
    args = parser.parse_args()
    review = json.loads(args.review.read_text())
    if (review.get('schema') != 'natlang.gemini-clinc-static-semantic-review/1' or
            review.get('root_approved') is not True or not review.get('rationale') or
            review.get('sample_seeds') != [44, 529, 771] or
            review.get('sample_sizes') != [25, 70, 100] or
            review.get('sample_scopes') != ['batch44', 'all', 'all']):
        raise ValueError('semantic review decision missing or incomplete')
    root, out = args.root.resolve(), args.out.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'refusing to overwrite review: {out}')
    all_matching, all_groups = {}, set()
    batches = [batch_review(root, n, all_matching, all_groups) for n in range(1, args.batches + 1)]
    items = [(key, value) for key, value in all_matching.items()]
    sampled = []
    for seed, size, scope in zip(review['sample_seeds'], review['sample_sizes'], review['sample_scopes']):
        chosen = [item for item in items if scope == 'all' or item[1]['batch'] == 44]
        random.Random(seed).shuffle(chosen)
        sampled.extend(key for key, _ in chosen[:size])
    held = set(review['held_source_ids'])
    if not held.issubset(set(sampled)) or any(key not in all_matching for key in held):
        raise ValueError('semantic holds must be inspected source-matching rows')
    excluded_programs = {all_matching[key]['program'] for key in held}
    out.mkdir(parents=True, exist_ok=True)
    compressed = []
    for batch in batches:
        adapted = Path(batch['adapted_dir'])
        compressed.append({'batch': batch['batch'],
                           'native': compact(adapted / 'native.held.jsonl'),
                           'replays': compact(adapted / 'replay.results.jsonl')})
    exclusions = out / 'excluded-programs.jsonl'
    with exclusions.open('w') as stream:
        for pid in sorted(excluded_programs):
            stream.write(json.dumps({'id': pid, 'reason_code': 'semantic_label_ambiguity_in_reviewed_item'},
                                    sort_keys=True) + '\n')
    receipt = {'schema': 'natlang.gemini-clinc-static-freeze/1',
               'review_path': str(args.review.resolve()), 'review_sha256': digest(args.review),
               'training_admission': True, 'trace_admission': False,
               'training_scope': 'authored static reference decisions from source-matching typed labels; two ambiguous programs excluded at render',
               'batches': batches, 'compressed': compressed,
               'totals': {'source_cases': sum(x['source_rows'] for x in batches),
                          'source_matching_labels': len(all_matching),
                          'programs': sum(x['programs'] for x in batches),
                          'native_turns_before_exclusions': sum(x['native_turns'] for x in batches),
                          'sampled_unique_items': len(set(sampled)),
                          'held_source_items': sorted(held),
                          'excluded_programs': len(excluded_programs)},
               'sampled_source_ids': sorted(set(sampled)),
               'excluded_programs_path': str(exclusions),
               'excluded_programs_sha256': digest(exclusions)}
    (out / 'freeze-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt['totals']))


if __name__ == '__main__':
    main()
