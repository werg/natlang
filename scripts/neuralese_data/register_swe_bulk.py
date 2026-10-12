#!/usr/bin/env python3
"""Register the bulk harness-bench SWE-rebench corpus (D13) and its four text twins: entries and manifests are committed
and pushed by register_corpora.py from a private worktree off origin/main (never the shared checkout's registry).
Run by swe_bulk_followup.sh when the twins are assembled.

  register_swe_bulk.py --bulk BULK_OUT --twin-root TWIN_ROOT
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/neuralese_data'))
from register_corpora import link as data_link, register  # noqa: E402
from swe_bulk_twins import file_sha  # noqa: E402
RECORDS_ID = 'harness-bench-swe-rebench-openhands-pi-records-bulk-20261011-v1'
# Whole-trajectory twins (coordinator 2026-10-11): one document per trajectory chain, not one per decision.
TWINS = {
    ('whole-trajectory', 'lfm25'): ('harness-bench-swe-rebench-openhands-pi-wholetraj-lfm25-350m-bulk-20261011-v1', 'last_turn_only', 'LFM'),
    ('whole-trajectory', 'mellum21'): ('harness-bench-swe-rebench-openhands-pi-wholetraj-mellum21-12b-bulk-20261011-v1', 'keep', 'Mellum'),
    ('program-header', 'lfm25'): ('harness-bench-swe-rebench-openhands-pi-program-header-lfm25-350m-bulk-20261012-v1', 'last_turn_only', 'LFM'),
    ('program-header', 'mellum21'): ('harness-bench-swe-rebench-openhands-pi-program-header-mellum21-12b-bulk-20261012-v1', 'keep', 'Mellum'),
}
ADMISSION_REASON = ("owner 2026-10-11 maximize data: admit unless a concrete defect (broken format, leak into held/eval, "
                    "protected eval, verified wrong labels); 'pending review' is recorded as review status, not a reason "
                    "to exclude; splits and source-disjointness kept (plans/neuralese/DATA_INVENTORY_20261011.md)")


def git(*args: str) -> str:
    return subprocess.run(['git', *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def link(name: str, target: Path) -> str:
    rel = f'data/neuralese/corpora/{name}'
    data_link(rel, target)
    return rel


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--bulk', type=Path, required=True)
    parser.add_argument('--twin-root', type=Path, required=True)
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
                  "common. Built and stored on NVMe."),
    })
    for (rendering, name), (twin_id, policy, line) in TWINS.items():
        twin_path = args.twin_root / name if rendering == 'whole-trajectory' else args.twin_root / rendering / name
        receipt = json.loads((twin_path / 'receipt.json').read_text())
        # The twin is one zstd file (swe_bulk_twins.py): the receipt binds its file hash and its plain-text hash.
        if (receipt.get('rendering') != rendering or receipt.get('text_file') != 'text.jsonl.zst'
                or receipt.get('text_zst_sha256') != file_sha(twin_path / 'text.jsonl.zst')):
            raise ValueError(f'{twin_path}: rendering receipt mismatch')
        rel_t = link(twin_id, twin_path)
        entries.append({
            'id': twin_id, 'owner': 'dgx', 'path': rel_t,
            'include': ['text.jsonl.zst', 'receipt.json', 'shards/*/receipt.json', 'shards/*/omissions.jsonl',
                        'shards/*/render-omissions.jsonl'],
            'split': 'inherited from the source records (by repository; source groups repository and instance)',
            'source_class': f'rendered-{rendering}-text-of-harness-bench-records',
            'admission': f'admitted-harness_bench-cohort-text-stages ({line} line); {rendering} documents, tool output at the shared feedback weight; test documents are a reported stratum, never qualification evidence',
            'training_admission': True, 'training_admission_reason': ADMISSION_REASON,
            'corpus_publication_for_training': False, 'derived_from': [RECORDS_ID],
            'license': "CC-BY-4.0 (source dataset cards); repositories' own licences per instance",
            'build': {'script': 'scripts/neuralese_data/swe_bulk_twins.py (render_native_turn_corpus.py --renderings '
                                'program-header whole-trajectory --program-header-as-full per shard, concatenated)',
                      'code_commit': commit, 'renderer': f'scripts/neuralese_data/render_native_turn_corpus.py {rendering}',
                      'renderer_code': receipt['renderer_code'], 'tokenizer': receipt['tokenizer'], 'rendering': rendering,
                      'history_reasoning_policy': policy, 'text_file': 'text.jsonl.zst',
                      'text_jsonl_sha256': receipt['text_jsonl_sha256'], 'text_zst_sha256': receipt['text_zst_sha256']},
            'tokens': receipt['tokens'],
            'notes': (f"{rendering} text of {RECORDS_ID} (the system prompt is the pi harness's own, kept whole with its "
                      "tool schemas). Chosen over v4's one document per decision, which repeats each trajectory's context "
                      "per target: on 120 v4 trajectories (940 decisions) all 940 decisions are covered, and Mellum train "
                      "text is 4.06M tokens with 817k supervised assistant tokens against 17.1M / 145k per decision. "
                      "Per-shard receipts under shards/."),
        })
        if rendering == 'program-header':
            entries[-1]['notes'] = (f"Program-header decisions of {RECORDS_ID}, using the source pi harness's own "
                                    "system prompt and tool schemas. Suffix targets and balanced context supervision "
                                    "use the shared renderer; whole-trajectory twins separately cover complete clean "
                                    "assistant chains. Repository/source splits are inherited, not resampled.")
    added = register(entries, f'Register bulk SWE-rebench harness-bench records and text twins (D13): '
                              f'{total["trajectories"]} trajectories, {total["records"]} records')
    print(json.dumps({'registered': added, 'counts': total,
                      'tokens': {e['id']: e.get('tokens') for e in entries[1:]}}, indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
