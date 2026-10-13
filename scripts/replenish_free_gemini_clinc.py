#!/usr/bin/env python3
"""Keep source-disjoint, held CLINC decision batches flowing through the free Gemini pool.

This is queue plumbing. The pool chooses among verified free model quota groups;
semantic review and training admission remain separate.
"""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261012-v1/'
              'clinc-plus-full-train-expanded-20261012-v1/decision-cases.jsonl')
CAMPAIGN = ROOT / 'runs/free-provider-generation-20261013-v1'
STATE = ROOT / 'runs/free-provider-generation-20261009-v1/google-project-quota-state-v1.json'
SALT = 'gemini-clinc-20261013:'
BATCH_SIZE = 256


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def lines(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def unit_active(unit):
    status = subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit], check=False)
    return status.returncode == 0


def launch(batch, number, retry=0):
    unit = f'natlang-pop-gemini-clinc-batch{number:02d}' + (f'-r{retry}' if retry else '')
    key_file = Path.home() / '.config/natlang/gemini.env'
    command = (f'set -a; source {key_file}; set +a; '
               'exec python3 scripts/run_gemini_decision_pool.py '
               f'--cases {batch.relative_to(ROOT)}/cases.jsonl '
               f'--out {batch.relative_to(ROOT)}/labels.jsonl '
               f'--state {STATE.relative_to(ROOT)} --workers 4 '
               '--max-inflight-per-quota-group 1 --max-output-tokens 512 '
               '--choice-contract label-confidence --response-format json_schema '
               '--choice-enum-policy omit --wait')
    subprocess.run(['python3', 'scripts/memory_ledger.py', 'run', '--unit', unit,
                    '--budget-gb', '1.2', '--class', 'collection', '--priority', 'low',
                    '--workdir', str(ROOT), '--', 'bash', '-lc', command],
                   cwd=ROOT, check=True)
    print(json.dumps({'event': 'launched', 'batch': number, 'unit': unit}), flush=True)
    return unit + '.service'


def resume_incomplete(batch, number, completed):
    state_path = batch / 'relaunch-state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    unit = state.get('unit', f'natlang-pop-gemini-clinc-batch{number:02d}.service')
    if unit_active(unit):
        print(json.dumps({'event': 'active', 'batch': number, 'completed': completed}), flush=True)
        return
    now = time.time()
    if now < state.get('not_before', 0):
        print(json.dumps({'event': 'cooldown', 'batch': number,
                          'seconds': round(state['not_before'] - now)}), flush=True)
        return
    failures = 0 if completed > state.get('last_completed', -1) else state.get('failures', 0) + 1
    if failures >= 4:
        print(json.dumps({'event': 'needs_investigation', 'batch': number,
                          'completed': completed, 'no_progress_failures': failures}), flush=True)
        return
    retry = state.get('retry', 0) + 1
    try:
        next_unit = launch(batch, number, retry)
    except subprocess.CalledProcessError:
        next_unit = unit
        failures += 1
    delay = min(7200, 600 * 2 ** failures)
    state_path.write_text(json.dumps({'unit': next_unit, 'retry': retry,
                                      'last_completed': completed, 'failures': failures,
                                      'not_before': now + delay}, indent=2) + '\n')


def quality_summary(batch, cases, labels):
    path = batch / 'quality-summary.json'
    case_hash = digest(batch / 'cases.jsonl')
    label_hash = digest(batch / 'labels.jsonl')
    if path.exists():
        summary = json.loads(path.read_text())
        if summary['cases_sha256'] != case_hash or summary['labels_sha256'] != label_hash:
            raise ValueError(f'closed batch changed: {batch}')
        return summary
    gold = {row['id']: row['answer'] for row in cases}
    exact_ids = sorted(row['id'] for row in labels
                       if isinstance(row.get('answer'), dict) and
                       row['answer'].get('choice') == gold[row['id']])
    valid = sum(isinstance(row.get('answer'), dict) and
                'choice' in row['answer'] for row in labels)
    by_model = {}
    for model in sorted({row['teacher'] for row in labels}):
        group = [row for row in labels if row['teacher'] == model]
        by_model[model] = {'total': len(group), 'exact': sum(row['id'] in exact_ids for row in group)}
    summary = {'schema': 'natlang.free-api-held-label-quality/1', 'training_admission': False,
               'cases_sha256': case_hash, 'labels_sha256': label_hash,
               'source_rows': len(cases), 'typed_valid': valid, 'source_label_match': len(exact_ids),
               'source_matching_ids': exact_ids, 'by_model': by_model,
               'decision': 'Only source-matching rows are candidates; independent semantic review remains required.'}
    path.write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--campaign', type=Path, default=CAMPAIGN)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    campaign.mkdir(parents=True, exist_ok=True)
    with (campaign / '.replenisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        batches = sorted(campaign.glob('gemini-clinc-fresh-batch[0-9][0-9]'))
        source = args.source.resolve()
        if batches and digest(source) != json.loads((batches[0] / 'source-manifest.json').read_text())['source_sha256']:
            raise ValueError('pinned CLINC source changed; review before replenishing')
        for batch in batches:
            number = int(batch.name[-2:])
            cases = lines(batch / 'cases.jsonl')
            labels = lines(batch / 'labels.jsonl')
            case_ids = {row['id'] for row in cases}
            label_ids = {row['id'] for row in labels}
            if len(labels) < len(cases) or case_ids != label_ids or len(labels) != len(label_ids):
                if len(labels) > len(cases) or not label_ids.issubset(case_ids) or len(labels) != len(label_ids):
                    print(json.dumps({'event': 'needs_investigation', 'batch': number,
                                      'completed': len(labels), 'target': len(cases),
                                      'reason': 'duplicate_or_foreign_label_ids'}))
                    return
                resume_incomplete(batch, number, len(labels))
                return
            if cases:
                summary = quality_summary(batch, cases, labels)
                valid, exact = summary['typed_valid'], summary['source_label_match']
                if valid / len(cases) < 0.6 or exact / len(cases) < 0.6:
                    print(json.dumps({'event': 'quality_pause', 'batch': number,
                                      'valid': valid, 'exact': exact, 'target': len(cases)}))
                    return
        if len(batches) >= 99:
            print('{"event":"campaign_limit"}')
            return
        used = {row['id'] for batch in batches for row in lines(batch / 'cases.jsonl')}
        rows = [row for row in lines(source) if row.get('role') == 'train'
                and row['id'] not in used]
        rows.sort(key=lambda row: hashlib.sha256((SALT + row['group']).encode()).hexdigest())
        selected = rows[:BATCH_SIZE]
        if not selected:
            print('{"event":"source_exhausted"}')
            return
        number = len(batches) + 1
        batch = campaign / f'gemini-clinc-fresh-batch{number:02d}'
        batch.mkdir(exist_ok=False)
        cases_path = batch / 'cases.jsonl'
        with cases_path.open('w') as stream:
            for row in selected:
                stream.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
        manifest = {'schema': 'natlang.free-api-source-selection/1', 'source': str(source),
                    'source_sha256': digest(source), 'cases_sha256': digest(cases_path),
                    'count': len(selected), 'selection': 'sha256(gemini-clinc-20261013:group) order, excluding prior campaign IDs',
                    'role': 'train', 'training_admission': False,
                    'quality': 'held pending semantic review'}
        (batch / 'source-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
        try:
            launch(batch, number)
        except subprocess.CalledProcessError:
            (batch / 'relaunch-state.json').write_text(json.dumps({
                'unit': f'natlang-pop-gemini-clinc-batch{number:02d}.service',
                'retry': 0, 'last_completed': 0, 'failures': 1,
                'not_before': time.time() + 600}, indent=2) + '\n')
            print(json.dumps({'event': 'launch_deferred', 'batch': number}), flush=True)


if __name__ == '__main__':
    main()
