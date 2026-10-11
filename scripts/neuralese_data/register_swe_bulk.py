#!/usr/bin/env python3
"""Register the bulk harness-bench SWE-rebench corpus (D13) and its two text twins, publish their manifests, commit
only the registry and those manifests, and brief Pop. Run by swe_bulk_followup.sh when the twins are assembled.

  register_swe_bulk.py --bulk BULK_OUT --twin-root TWIN_ROOT [--no-commit]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECORDS_ID = 'harness-bench-swe-rebench-openhands-pi-records-bulk-20261011-v1'
TWINS = {'lfm25': ('harness-bench-swe-rebench-openhands-pi-text-lfm25-350m-bulk-20261011-v1', 'last_turn_only', 'LFM'),
         'mellum21': ('harness-bench-swe-rebench-openhands-pi-text-mellum21-12b-bulk-20261011-v1', 'keep', 'Mellum')}
ADMISSION_REASON = ("owner 2026-10-11 maximize data: admit unless a concrete defect (broken format, leak into held/eval, "
                    "protected eval, verified wrong labels); 'pending review' is recorded as review status, not a reason "
                    "to exclude; splits and source-disjointness kept (plans/neuralese/DATA_INVENTORY_20261011.md)")


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def link(name: str, target: Path) -> str:
    rel = f'data/neuralese/corpora/{name}'
    path = ROOT / rel
    if not path.is_symlink():
        os.symlink(target, path)
    return rel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bulk', type=Path, required=True)
    parser.add_argument('--twin-root', type=Path, required=True)
    parser.add_argument('--no-commit', action='store_true')
    args = parser.parse_args()
    shards = sorted(p for p in (args.bulk / 'shards').iterdir() if p.is_dir())
    summaries = [json.loads((s / 'summary.json').read_text()) for s in shards]
    total = {'trajectories': sum(s['trajectories'] for s in summaries), 'records': sum(s['records'] for s in summaries),
             'by_split': {k: sum(s['by_split'].get(k, 0) for s in summaries) for k in ('train', 'validation', 'test')},
             'view_parts': sum(s.get('view_parts', 0) for s in summaries),
             'repositories': sum(s['placement']['repositories'] for s in summaries),
             'repositories_by_split': {k: sum(s['placement']['by_split'].get(k, 0) for s in summaries) for k in ('train', 'validation', 'test')},
             'unplaceable_trajectories': sum(s.get('unplaceable_trajectories', 0) for s in summaries)}
    replay = {'fullyVerified': 0, 'diverged': 0, 'failed': 0, 'trajectories': 0}
    for s in shards:
        lines = [l for l in (s / 'replay.log').read_text().splitlines() if l.startswith('{"trajectories"')]
        if lines:
            last = json.loads(lines[-1])
            for k in replay:
                replay[k] += last.get(k, 0)
    prepared = [json.loads(p.read_text()) for p in sorted((args.bulk / 'prepared').glob('*.summary.json'))]
    prep = {k: sum(p.get(k, 0) for p in prepared) for k in ('seen', 'written', 'no_task', 'unresolved_skipped')}
    commit = git('rev-parse', 'HEAD').strip()
    registry_path = ROOT / 'training/neuralese_corpora.json'
    # Another session's uncommitted registry edits must not ride along in this commit.
    foreign = bool(git('status', '--porcelain', '--', 'training/neuralese_corpora.json').strip())
    registry = json.loads(registry_path.read_text())
    known = {c['id'] for c in registry['corpora']}
    entries = []
    rel = link(RECORDS_ID, args.bulk)
    entries.append({
        'id': RECORDS_ID, 'owner': 'dgx', 'path': rel,
        'include': ['surface.json', 'prepared/*.summary.json', 'shards/*/records.jsonl', 'shards/*/pieces.jsonl',
                    'shards/*/summary.json', 'shards/*/prepared.jsonl', 'shards/*/replayed.jsonl', 'shards/*/replay.log'],
        'split': ("by repository, as harness-bench-swe-rebench-openhands-pi-records-20261010-v4: a repository S1 places keeps "
                  "S1's split (cross-corpus-index-s1-full-final-20261003-v1), others split_of(repo, 5); source groups "
                  "repository and instance; shards are disjoint sets of whole repositories, so per-shard placement equals "
                  "whole-corpus placement, and repositories shared with v4 get v4's split"),
        'source_class': 'teacher-agent-trajectories-normalized-to-pi-and-replayed-in-pi-tools',
        'admission': 'admitted (maximize-data rule); records are read directly only by the recurrence stage, which stays held as for v4; the text twins are the training input',
        'training_admission': True, 'training_admission_reason': ADMISSION_REASON,
        'review_status': f"replay: {replay['fullyVerified']} fully verified, {replay['diverged']} diverged (records up to the divergence), {replay['failed']} failed (no records)",
        'corpus_publication_for_training': False,
        'derived_from': ['nebius-swe-rebench-openhands-trajectories-20261009-v1'],
        'license': "CC-BY-4.0 (source dataset cards); repositories' own licences per instance",
        'build': {'script': 'training/neuralese/scripts/harness_bench_bulk.sh OUT 24 6 1000', 'code_commit': commit,
                  'builder': 'natlang.harness_bench.records/1', 'dialect': 'openhands-v0.54', 'mapping': 'pi',
                  'surface_file_sha256': 'bf007f819f00fb9107b687c38dcd82be391ab2a6261b4e0a06814ff2d22fa050 (identical to v4)',
                  'prepare': prep, 'replay': replay, 'shards': len(shards), 'offset': 1000},
        'counts': total,
        'notes': ("D13 (plans/neuralese/DATA_INVENTORY_20261011.md): every resolved SWE-rebench OpenHands trajectory after "
                  "parquet row 1000 (v4 is rows 0-999), normalized to pi transcripts, replayed in pi's tools and turned "
                  "into native records (8 sampled targets per trajectory, the v4 default). Additive to v4: no trajectory in "
                  "common. Built on the external HDD; text twins on NVMe."),
    })
    for name, (twin_id, policy, line) in TWINS.items():
        receipt = json.loads((args.twin_root / name / 'receipt.json').read_text())
        rel_t = link(twin_id, args.twin_root / name)
        entries.append({
            'id': twin_id, 'owner': 'dgx', 'path': rel_t,
            'include': ['text.jsonl', 'receipt.json', 'shards/*/receipt.json', 'shards/*/provenance.jsonl', 'shards/*/omissions.jsonl'],
            'split': 'inherited from the source records (by repository; source groups repository and instance)',
            'source_class': 'derived-native-chat-gold-text-twin-of-harness-bench-records',
            'admission': f'admitted-harness_bench-cohort-text-stages, as the v4 twin ({line} line text stages beside the native text); test documents are a reported stratum, never qualification evidence',
            'training_admission': True, 'training_admission_reason': ADMISSION_REASON,
            'corpus_publication_for_training': False, 'derived_from': [RECORDS_ID],
            'license': "CC-BY-4.0 (source dataset cards); repositories' own licences per instance",
            'build': {'script': 'scripts/neuralese_data/swe_bulk_twins.py (scripts/build_neuralese_text_packet.py per shard, concatenated)',
                      'code_commit': commit, 'renderer': 'natlang_neuralese.data.text_corpus.gold_text_rows',
                      'renderer_code': receipt['renderer_code'], 'tokenizer': receipt['tokenizer'],
                      'history_reasoning_policy': policy, 'text_jsonl_sha256': receipt['text_jsonl_sha256']},
            'tokens': receipt['tokens'],
            'notes': f"Text twin of {RECORDS_ID}, same renderer and history-reasoning policy as the v4 twin; per-shard receipts under shards/.",
        })
    added = [e for e in entries if e['id'] not in known]
    registry['corpora'].extend(added)
    registry_path.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + '\n')
    for e in entries:
        subprocess.run([sys.executable, str(ROOT / 'scripts/sync_training_corpora.py'), 'publish', '--machine', 'dgx', '--id', e['id']],
                       cwd=ROOT, check=True)
    print(json.dumps({'registered': [e['id'] for e in added], 'counts': total,
                      'tokens': {e['id']: e.get('tokens') for e in entries[1:]}}, indent=1))
    if args.no_commit:
        return 0
    paths = ['training/neuralese_corpora.json'] + [f'training/corpus-manifests/{e["id"]}.json' for e in entries]
    if foreign or git('diff', '--cached', '--name-only').strip():
        print('registry or index holds another session\'s changes; registered but left uncommitted', file=sys.stderr)
        return 2
    git('add', '--', *paths)
    git('commit', '-q', '-m', f'Register bulk SWE-rebench harness-bench records and text twins (D13): {total["trajectories"]} trajectories, {total["records"]} records\n\n'
        'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n'
        'Claude-Session: https://claude.ai/code/session_01Q53kXkxDK2onzJmxGfyHnP', '--', *paths)
    return 0


if __name__ == '__main__':
    sys.exit(main())
