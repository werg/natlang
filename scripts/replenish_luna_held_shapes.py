#!/usr/bin/env python3
"""Supply bounded, source-disjoint Luna curriculum campaigns on Pop.

This schedules generation only. Successful outputs still require a separate
quality/decontamination review and explicit corpus admission.
"""
import collections
import fcntl
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex

BASE = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261013-v1')
HELD = ROOT / 'runs/natlang-held-task-packet-v3-20261012/cases.jsonl'
RUNTIME = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261012-v1/runtime-held-task-empty-eos-v1')
FAMILIES = ('contract_diagnosis', 'workbench_tickets', 'logic_proof_verifier',
            'actor_route_replan', 'folder_criteria_reducer',
            'iterate_schedule_repair', 'relational_late_argmax')
START_SEED = 52007
WORKERS = 6
PER_FAMILY = 20


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def world(row):
    semantics = row.get('semantics', {})
    return json.dumps([row['family'], {key: semantics.get(key)
                                      for key in ('inputs', 'files', 'services')}],
                      sort_keys=True, separators=(',', ':'))


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit],
                          check=False).returncode == 0


def complete(campaign):
    manifest = json.loads((campaign / 'campaign-manifest.json').read_text())
    pending = False
    for number in range(1, WORKERS + 1):
        worker = campaign / f'worker-{number:02d}'
        queue = rows(worker / 'queue.jsonl')
        unit = f'natlang-pop-luna-s{manifest["seed"]}-w{number:02d}.service'
        if active(unit):
            pending = True
            continue
        journal = worker / 'journal.jsonl'
        finished = {event['key']: event for event in (rows(journal) if journal.exists() else [])
                    if event.get('event') == 'finish'}
        missing = [job['key'] for job in queue if job['key'] not in finished]
        if missing:
            retry_file = worker / 'relaunch-state.json'
            previous = json.loads(retry_file.read_text()) if retry_file.exists() else {}
            failures = previous.get('failures', 0) + (len(finished) <= previous.get('finished', -1))
            if failures >= 4:
                print(json.dumps({'event': 'needs_review', 'unit': unit,
                                  'reason': 'four_no_progress_relaunches', 'missing': len(missing)}), flush=True)
                return False
            subprocess.run(['systemctl', '--user', 'reset-failed', unit], check=False)
            launch_worker(campaign, manifest['seed'], number)
            retry_file.write_text(json.dumps({'failures': failures, 'finished': len(finished)}) + '\n')
            pending = True
            continue
        for job in queue:
            result = finished.get(job['key'], {})
            if result.get('status') not in ('complete', 'complete_with_skips') or not result.get('output_accounting', {}).get('complete'):
                print(json.dumps({'event': 'needs_review', 'campaign': str(campaign),
                                  'key': job['key'], 'finish': result.get('status')}), flush=True)
                return False
    return not pending


def launch_worker(campaign, seed, number):
    worker = campaign / f'worker-{number:02d}'
    unit = f'natlang-pop-luna-s{seed}-w{number:02d}'
    command = ['python3', 'scripts/run_bonsai_queue.py', str(worker / 'queue.jsonl'),
               str(worker / 'journal.jsonl'), '--runtime', str(RUNTIME),
               '--case-seconds', '1800', '--provider', 'openai-codex',
               '--model-id', 'gpt-6-luna', '--model-concurrency', '1',
               '--execution-plans', '--reasoning-effort', 'low', '--min-free-mib', '1024']
    subprocess.run(['python3', 'scripts/memory_ledger.py', 'run', '--unit', unit,
                    '--budget-gb', '0.8', '--class', 'collection', '--priority', 'low',
                    '--workdir', str(ROOT), '--', *command], cwd=ROOT, check=True)
    print(json.dumps({'event': 'launched', 'unit': unit, 'cases': len(rows(worker / 'queue.jsonl'))}), flush=True)


def build_source(campaign, seed):
    source = campaign / 'source' / f'held-shapes-s{seed}-n8.ir.jsonl'
    if source.exists():
        return source
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['node', 'ts-host/scripts/inline-curriculum/build.mjs',
                    '--seed', str(seed), '--shapes', '8', '--split', 'train',
                    '--families', ','.join(FAMILIES), '--out', str(source)],
                   cwd=ROOT, check=True)
    return source


def prior_worlds(current):
    worlds = {world(row) for row in rows(HELD)}
    # These are prior *sources*, not only successful outputs: never resample
    # a world just because a previous model attempt failed.
    old = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261012-v1')
    for source in old.glob('provider-luna-held-shape-remediation*/source/*.ir.jsonl'):
        worlds.update(world(row) for row in rows(source))
    for source in BASE.glob('provider-luna-held-shapes-s*-v1/source/*.ir.jsonl'):
        if current in source.parents:
            continue
        worlds.update(world(row) for row in rows(source))
    return worlds


def prepare(campaign, seed, source, excluded):
    source_rows = rows(source)
    report = json.loads(Path(str(source) + '.report.json').read_text())
    if report['cases'] != len(source_rows) or report['verified'] != len(source_rows):
        raise ValueError('Source replay verification incomplete')
    selected = []
    seen = set()
    used_groups = set()
    counts = collections.Counter()
    # Choose distinct groups before sibling variants, avoiding exact held or
    # prior-generation input worlds. Do not force a quota if the source lacks it.
    eligible = [row for row in source_rows if row['split'] == 'train' and world(row) not in excluded]
    for distinct in (True, False):
        for row in eligible:
            family = row['curriculum']['family']
            if counts[family] >= PER_FAMILY or row['id'] in seen:
                continue
            if distinct and used_groups.intersection(row['source_groups']):
                continue
            selected.append(row)
            seen.add(row['id'])
            used_groups.update(row['source_groups'])
            counts[family] += 1
    if len(selected) < 100:
        raise ValueError(f'Too few decontaminated source cases: {len(selected)}')
    for number in range(1, WORKERS + 1):
        worker = campaign / f'worker-{number:02d}'
        worker.mkdir(parents=True, exist_ok=False)
        with (worker / 'queue.jsonl').open('w') as stream:
            for ordinal, row in enumerate(selected[number - 1::WORKERS]):
                index = source_rows.index(row)
                case = worker / f'case-{ordinal:03d}'
                job = dict(key=f'luna-held-shape-s{seed}-w{number:02d}-{ordinal:03d}',
                           source=str(source), index=index, count=1,
                           output=str(case / 'results.jsonl'), jobs=str(case / 'jobs'),
                           log=str(case / 'collector.log'), seed=seed * 1000 + (number - 1) * 100 + ordinal,
                           context_tokens=32768, action_max_tokens=4096,
                           max_turns=60, max_model_requests=384, case_seconds=1800,
                           request_retries=3, transport_retries=0,
                           text_neuralese_emulation=True)
                stream.write(json.dumps(job, sort_keys=True) + '\n')
    manifest = dict(schema='natlang.generation-campaign/1', seed=seed,
                    source_path=str(source), source_sha256=sha256_file_hex(source),
                    source_verified_cases=len(source_rows), selected_count=len(selected),
                    selected_by_family=dict(counts), selected_ids=[row['id'] for row in selected],
                    held_path=str(HELD), held_sha256=sha256_file_hex(HELD),
                    world_exclusion='same family and exact semantics.inputs/files/services against held and earlier sources',
                    training_admission=False, status='generation_only_pending_quality_review')
    (campaign / 'campaign-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for number in range(1, WORKERS + 1):
        launch_worker(campaign, seed, number)


def main():
    BASE.mkdir(parents=True, exist_ok=True)
    if not RUNTIME.exists():
        raise FileNotFoundError(RUNTIME)
    with (BASE / '.luna-replenisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        campaigns = sorted(BASE.glob('provider-luna-held-shapes-s*-v1'))
        for campaign in campaigns:
            manifest_path = campaign / 'campaign-manifest.json'
            if not manifest_path.exists():
                source = campaign / 'source' / f'held-shapes-s{int(campaign.name.split("-s")[-1].split("-")[0])}-n8.ir.jsonl'
                if source.exists():
                    if active(f'natlang-pop-build-held-s{int(campaign.name.split("-s")[-1].split("-")[0])}.service'):
                        print(json.dumps({'event': 'source_build_wait', 'campaign': str(campaign)}), flush=True)
                        return
                    prepare(campaign, int(campaign.name.split('-s')[-1].split('-')[0]),
                            source, prior_worlds(campaign))
                    return
                raise ValueError(f'Incomplete campaign preparation: {campaign}')
            if not complete(campaign):
                return
        seed = START_SEED if not campaigns else max(int(p.name.split('-s')[-1].split('-')[0]) for p in campaigns) + 1
        campaign = BASE / f'provider-luna-held-shapes-s{seed}-v1'
        campaign.mkdir(exist_ok=True)
        source = build_source(campaign, seed)
        excluded = prior_worlds(campaign)
        prepare(campaign, seed, source, excluded)


if __name__ == '__main__':
    main()
