"""Export the exact D12 source-reported feature-status view as full sessions.

DECOMPOSITION
1. Bind each selected D12 archive to its immutable acquisition receipt.
2. Reuse d12_feature_status_diagnoses for source-derived status and full context.
3. Preserve each accepted Trajectory unchanged with its archive and evaluator
   provenance; write every adapter rejection to a separate omission ledger.
4. Commit each archive export atomically with source, live-measurement and code
   identities. This is normalization plumbing, not a source-quality decision.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'training' / 'neuralese'))

from natlang_neuralese.common.hashing import sha256_file_hex
from natlang_neuralese.common.jsonio import write_json_durable, write_jsonl_text
from neuralese_data import acquired_agents, agents
from neuralese_data.acquired_diagnosis import d12_feature_status_diagnoses
from neuralese_data.common import Reject


def export(root: Path, output: Path, live48: Path) -> dict:
    root = root.resolve(strict=True)
    live48 = live48.resolve(strict=True)
    measurement = json.loads(live48.read_text())
    if (measurement.get('schema') != 'natlang.d12-live48-measurement-summary/1'
            or measurement.get('rows') != 48 or measurement.get('exact_matches') != 48
            or measurement.get('extra_json_claims') != 0):
        raise ValueError('D12 view requires the completed exact-match live JSON measurement')
    packet = measurement['packet_sha256']
    summary_sha = sha256_file_hex(live48)
    source_root = root / 'D12'
    acquisition_path = source_root / 'acquisition.json'
    acquisition = json.loads(acquisition_path.read_text())
    if acquisition.get('status') != 'acquired' or not acquisition.get('revision'):
        raise ValueError('D12 source acquisition is not complete and pinned')
    receipt = acquired_agents.validate_acquisition(
        'D12', acquisition, acquisition_path, root=root / 'D12')
    files = [entry for entry in receipt['files'] if entry['path'].endswith('.tar.gz')]
    if not files:
        raise ValueError('D12 pinned source has no selected complete archives')
    output.mkdir(parents=True, exist_ok=True)
    agents.EXPANSION_ROOT = root
    completed = []
    code_sha = sha256_file_hex(Path(__file__))
    for entry in files:
        key = hashlib.sha256((entry['path'] + '\0' + entry['sha256'] + '\0' + packet).encode()).hexdigest()
        target = output / key
        target.mkdir(parents=True, exist_ok=True)
        final = target / 'full-trajectories.jsonl.zst'
        omission_final = target / 'omissions.jsonl'
        done = target / 'export.json'
        contract = {'source': entry, 'revision': receipt['revision'], 'exporter_sha256': code_sha,
            'view': 'd12-source-reported-feature-status/1', 'live48_summary_sha256': summary_sha,
            'live48_packet_sha256': packet, 'live48_exact_matches': 48,
            'instruction': 'Return only one JSON object preserving the same schema, keys and values; do not infer a cause or repair.',
            'source_policy': 'exact reported feature statuses only; original failed actions remain context'}
        if done.exists():
            previous = json.loads(done.read_text())
            if previous.get('contract') != contract:
                raise ValueError(f'Existing D12 export is bound to another contract: {done}')
            for name, digest in previous['outputs'].items():
                if sha256_file_hex(target / name) != digest:
                    raise ValueError(f'Existing D12 export output changed: {target / name}')
            completed.append({'status': 'reused', 'receipt': str(done), 'sessions': previous['complete_sessions']})
            continue
        acquired_agents.ACQUIRED_FILES = frozenset({entry['path']})
        temp = target / 'full-trajectories.pending.zst'
        omissions_temp = target / 'omissions.pending.jsonl'
        candidates = rejected = 0
        seen = set()
        with write_jsonl_text(temp) as stream, omissions_temp.open('w', encoding='utf-8') as omissions:
            for source_id, trajectory in d12_feature_status_diagnoses():
                if not source_id.startswith(entry['path'] + ':'):
                    raise ValueError(f'D12 adapter escaped selected archive {entry["path"]}: {source_id}')
                if source_id in seen:
                    raise ValueError(f'D12 adapter duplicated source view identity: {source_id}')
                seen.add(source_id)
                if isinstance(trajectory, Reject):
                    omissions.write(json.dumps({'source_row': source_id, 'reason': str(trajectory)},
                        ensure_ascii=False, sort_keys=True, separators=(',', ':')) + '\n')
                    rejected += 1
                    continue
                provenance = trajectory.lineage.get('notes', {}).get('source_reported_feature_status', {})
                original = trajectory.lineage.get('notes', {}).get('original_cooperation_context', {})
                archive = original.get('archive_context') or {}
                if (provenance.get('review_kind') != 'source-curated-reported-status'
                        or provenance.get('failed_actions_positive_targets') is not False
                        or provenance.get('causal_claims') is not False
                        or provenance.get('native_execution_verified') is not False
                        or not archive.get('member', {}).get('sha256')):
                    raise ValueError(f'D12 derived view lacks exact source status/member provenance: {source_id}')
                row = {'schema': 'natlang.normalized-agent-trajectory/1', 'source_row': source_id,
                    'source_file': entry, 'source_revision': receipt['revision'],
                    'trajectory': asdict(trajectory), 'positive_candidate': True,
                    'training_admission': False, 'whole_trajectory': True, 'windowed': False,
                    'view': contract['view'], 'live48_summary_sha256': summary_sha}
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
                candidates += 1
            omissions.flush()
            os.fsync(omissions.fileno())
        os.replace(temp, final)
        os.replace(omissions_temp, omission_final)
        record = {'schema': 'natlang.normalized-agent-source-file/1', 'contract': contract,
            'outputs': {path.name: sha256_file_hex(path) for path in (final, omission_final)},
            'complete_sessions': candidates, 'positive_candidates': candidates,
            'non_positive_sessions': 0, 'explicit_adapter_omissions': rejected,
            'adapter_source_rows_accounted': len(seen), 'training_admission': False,
            'created_at': time.time()}
        write_json_durable(done, record)
        completed.append({'status': 'exported', 'receipt': str(done), 'sessions': candidates,
                          'adapter_omissions': rejected, 'accounted_adapter_rows': len(seen)})
        print(json.dumps(completed[-1]), flush=True)
    summary = {'schema': 'natlang.acquired-d12-status-source-exports/1', 'ref': 'D12',
        'source_revision': receipt['revision'], 'live48_summary': {'path': str(live48),
            'sha256': summary_sha, 'packet_sha256': packet, 'exact_matches': 48},
        'files': completed, 'training_admission': False}
    write_json_durable(output / 'complete.json', summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live48-summary', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(export(args.source_root, args.output, args.live48_summary), sort_keys=True))


if __name__ == '__main__':
    main()
