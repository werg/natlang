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


def launch(batch, number):
    unit = f'natlang-pop-gemini-clinc-batch{number:02d}'
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
        for batch in batches:
            number = int(batch.name[-2:])
            cases = lines(batch / 'cases.jsonl')
            labels = lines(batch / 'labels.jsonl')
            case_ids = {row['id'] for row in cases}
            label_ids = {row['id'] for row in labels}
            if len(labels) < len(cases) or case_ids != label_ids or len(labels) != len(label_ids):
                unit = f'natlang-pop-gemini-clinc-batch{number:02d}.service'
                print(json.dumps({'event': 'active' if unit_active(unit) else 'needs_investigation',
                                  'batch': number, 'completed': len(labels), 'target': len(cases)}))
                return
            if cases:
                valid = sum(isinstance(row.get('answer'), dict) and
                            'error' not in row['answer'] for row in labels)
                gold = {row['id']: row['answer'] for row in cases}
                exact = sum(row.get('answer', {}).get('choice') == gold[row['id']]
                            for row in labels if isinstance(row.get('answer'), dict))
                if valid / len(cases) < 0.6 or exact / len(cases) < 0.75:
                    print(json.dumps({'event': 'quality_pause', 'batch': number,
                                      'valid': valid, 'exact': exact, 'target': len(cases)}))
                    return
        if len(batches) >= 99:
            print('{"event":"campaign_limit"}')
            return
        used = {row['id'] for batch in batches for row in lines(batch / 'cases.jsonl')}
        source = args.source.resolve()
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
        launch(batch, number)


if __name__ == '__main__':
    main()
