#!/usr/bin/env python3
"""Program-header rendering of an e2e cohort (records.jsonl + pieces.jsonl + edges.jsonl), from the same source records.

Each record goes through the shared data/prompt_rendering.program_header_native_record: the leading system message is
reduced to the program header (generic runtime prompt dropped, owner 2026-10-11) and the runtime tool schemas are
dropped; messages after it, the target and provenance are unchanged. The producer->consumer edges are re-derived with
train/trajectories.authenticated_recurrence_context_view on the rendered records and must equal the source cohort's
edges.jsonl exactly, otherwise nothing is written. Output: records.jsonl (program-header), pieces.jsonl and edges.jsonl
(byte copies), receipt.json. Run under the memory ledger (CPU, ~4 GB for the 1389-record cohort).

  render_e2e_program_header.py SOURCE_COHORT_DIR OUT_DIR --id CORPUS_ID
"""
import argparse, hashlib, json, os, shutil, sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))
from natlang_neuralese.data.prompt_rendering import program_header_native_record, prompt_rendering_identity  # noqa: E402
from natlang_neuralese.train.trajectories import authenticated_recurrence_context_view  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def edge_set(records, pieces):
    _, review = authenticated_recurrence_context_view(records, pieces)
    return {(ref.get('writer_record_id'), d['id'], ref.get('name'), ref.get('body_sha256'))
            for d in review['decisions'] for ref in d.get('references') or [] if ref.get('gradient_edge')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source'); ap.add_argument('out'); ap.add_argument('--id', required=True)
    a = ap.parse_args()
    src, out = Path(a.source), Path(a.out)
    pieces = {}
    for line in open(src / 'pieces.jsonl'):
        row = json.loads(line); pieces[row['name']] = row['text']
    records = [json.loads(line) for line in open(src / 'records.jsonl') if line.strip()]
    rendered = [program_header_native_record(r, pieces) for r in records]
    expected = {(e['writer_record_id'], e['reader_record_id'], e['write_name'], e['body_sha256'])
                for e in map(json.loads, open(src / 'edges.jsonl'))}
    got = edge_set(rendered, pieces)
    kept = {e for e in got if e in expected}
    if kept != expected:
        raise SystemExit(f'edges differ after program-header rendering: {len(expected - got)} lost, '
                         f'{len(got - expected)} extra among source endpoints')
    staged = out.with_name(out.name + '.building')
    shutil.rmtree(staged, ignore_errors=True); staged.mkdir(parents=True)
    with open(staged / 'records.jsonl', 'w') as f:
        for r in rendered: f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for name in ('pieces.jsonl', 'edges.jsonl'): shutil.copyfile(src / name, staged / name)
    receipt = {
        'schema': 'natlang.neuralese-e2e-cohort/1', 'id': a.id, 'prompt_rendering': 'program-header',
        'rendering_identity': prompt_rendering_identity('program-header'),
        'derived_from': {'cohort': str(src), 'receipt_sha256': sha(src / 'receipt.json') if (src / 'receipt.json').exists() else None,
                         'records_sha256': sha(src / 'records.jsonl')},
        'transformation': 'data/prompt_rendering.program_header_native_record per record: system soft pieces resolved to '
                          'text and reduced to the program header (system turn dropped when none), tools []; every other '
                          'field unchanged. The full rendering is the source cohort itself.',
        'records': len(rendered), 'system_prompt_mode': Counter(r['system_prompt_mode'] for r in rendered),
        'edges': len(expected), 'edges_reauthenticated_equal': True,
        'files': {n: {'sha256': sha(staged / n), 'bytes': os.path.getsize(staged / n)}
                  for n in ('records.jsonl', 'pieces.jsonl', 'edges.jsonl')},
    }
    json.dump(receipt, open(staged / 'receipt.json', 'w'), indent=1, sort_keys=True, default=dict)
    if out.exists(): shutil.rmtree(out)
    staged.rename(out)
    print(json.dumps({k: receipt[k] for k in ('id', 'records', 'system_prompt_mode', 'edges', 'files')}, indent=1, default=dict))


if __name__ == '__main__':
    main()
