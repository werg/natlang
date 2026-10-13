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
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex

BASE = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261013-v1')
HELD = ROOT / 'runs/natlang-held-task-packet-v3-20261012/cases.jsonl'
RUNTIME = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261012-v1/runtime-held-task-empty-eos-v1')
FAMILIES = ('contract_diagnosis', 'workbench_tickets', 'logic_proof_verifier',
            'actor_route_replan', 'folder_criteria_reducer',
            'iterate_schedule_repair', 'relational_late_argmax',
            'actor_greenhouse', 'edit_stream', 'claims_desk',
            'dataset_workbench', 'inline_review_each',
            'counterexample_revision', 'policy_after_measure',
            'recurrence_inline_graphs', 'composed_process',
            'kqapro_question', 'scienceworld_task', 'alfworld_task',
            'musique_hops', 'knowledge_research', 'folder_edit')
START_SEED = 52007
WORKERS = 6
PER_FAMILY = {family: 6 for family in FAMILIES}
# inline_review_each also emits its paired no-crisp family. Preserve both
# behaviours without letting this one builder take two full family slots.
PER_FAMILY.update(inline_review_each=3, inline_avoid_crisp=3)
TARGET_GROUP_INDEX = BASE / 'target-r3-source-groups.jsonl'
TARGET_GROUP_INPUT_HASHES = {
    'program-header': '0974200754e27da114f429de3142fb3be8500431bcb0837cbea35868ebf30ba9',
    'whole-trajectory': 'c30b6bbeb2887bc9a533a5aaa18111216caf6dc4252511a7d0985e32f7044501',
}
BACKFILL = BASE / 'provider-luna-folder-backfill-s52007-s52010-v1'
BACKFILL_UNIT_SEED = 52070


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def world(row):
    semantics = row.get('semantics', {})
    # Folder reducers expose their decisive contents through folder_files;
    # operations and other future visible fields also belong to the world.
    # Gold expectations and oracles are the only semantics fields omitted.
    visible = {key: value for key, value in semantics.items()
               if key not in ('expected', 'expected_files', 'oracle', 'files_oracle')}
    return json.dumps([row['family'], visible],
                      sort_keys=True, separators=(',', ':'))


def active(unit):
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', unit],
                          check=False).returncode == 0


def complete(campaign):
    manifest = json.loads((campaign / 'campaign-manifest.json').read_text())
    exclusions_path = campaign / 'reviewed-exclusions.json'
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    if exclusions and exclusions.get('schema') != 'natlang.generation-reviewed-exclusions/1':
        raise ValueError(f'Unrecognized reviewed exclusions: {exclusions_path}')
    excluded = exclusions.get('cases', {})
    pending = False
    for number in range(1, WORKERS + 1):
        worker = campaign / f'worker-{number:02d}'
        queue = rows(worker / 'queue.jsonl')
        unit = f'natlang-pop-luna-s{manifest["seed"]}-w{number:02d}.service'
        retry_dirs = sorted(worker.glob('retry-*'))
        if active(unit) or any(active(f'natlang-pop-luna-s{manifest["seed"]}-w{number:02d}-r{path.name.removeprefix("retry-")}.service')
                               for path in retry_dirs):
            pending = True
            continue
        journals = [worker / 'journal.jsonl', *(path / 'journal.jsonl' for path in retry_dirs)]
        finished = {event['key']: event for journal in journals if journal.exists()
                    for event in rows(journal) if event.get('event') == 'finish'}
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
            if not launch_worker(campaign, manifest['seed'], number):
                return False  # memory pressure is a deferred launch, not a no-progress attempt
            retry_file.write_text(json.dumps({'failures': failures, 'finished': len(finished)}) + '\n')
            pending = True
            continue
        for job in queue:
            result = finished.get(job['key'], {})
            if result.get('status') not in ('complete', 'complete_with_skips') or not result.get('output_accounting', {}).get('complete'):
                review = excluded.get(job['key'])
                if review is not None:
                    accounting = result.get('output_accounting', {})
                    if (result.get('status') != review.get('finish_status') or
                            accounting.get('exact_result_rows') != 0 or
                            accounting.get('output_sha256') != review.get('output_sha256') or
                            not review.get('reason')):
                        raise ValueError(f'Reviewed exclusion no longer matches terminal output: {job["key"]}')
                    continue
                if retry_transient(campaign, worker, manifest['seed'], number, job, result, retry_dirs):
                    pending = True
                    continue
                print(json.dumps({'event': 'needs_review', 'campaign': str(campaign),
                                  'key': job['key'], 'finish': result.get('status')}), flush=True)
                return False
    return not pending


def launch_queue(queue_dir, unit):
    command = ['python3', 'scripts/run_bonsai_queue.py', str(queue_dir / 'queue.jsonl'),
               str(queue_dir / 'journal.jsonl'), '--runtime', str(RUNTIME),
               '--case-seconds', '1800', '--provider', 'openai-codex',
               '--model-id', 'gpt-6-luna', '--model-concurrency', '1',
               '--execution-plans', '--reasoning-effort', 'low', '--min-free-mib', '1024']
    run = subprocess.run(['python3', 'scripts/memory_ledger.py', 'run', '--unit', unit,
                          '--budget-gb', '0.5', '--reserve-gb', '0.2', '--class', 'collection',
                          '--priority', 'low', '--workdir', str(ROOT), '--', *command],
                         cwd=ROOT, capture_output=True, text=True)
    if run.returncode:
        if 'not admitted:' in run.stderr:
            print(json.dumps({'event': 'deferred_memory', 'unit': unit,
                              'reason': run.stderr.strip()}), flush=True)
            return False
        raise subprocess.CalledProcessError(run.returncode, run.args, run.stdout, run.stderr)
    print(json.dumps({'event': 'launched', 'unit': unit,
                      'cases': len(rows(queue_dir / 'queue.jsonl'))}), flush=True)
    return True


def launch_worker(campaign, seed, number):
    return launch_queue(campaign / f'worker-{number:02d}', f'natlang-pop-luna-s{seed}-w{number:02d}')


def retry_transient(campaign, worker, seed, number, job, result, retry_dirs):
    """Retry a resultless provider error without reclassifying it as a model rejection."""
    if result.get('status') != 'incomplete' or result.get('exit_code') != 2:
        return False
    if result.get('output_accounting', {}).get('exact_result_rows') != 0:
        return False
    own_retries = [path for path in retry_dirs
                   if (path / 'review.json').exists() and
                   json.loads((path / 'review.json').read_text()).get('original_key') == job['key']]
    if own_retries and not (own_retries[-1] / 'journal.jsonl').exists():
        last_retry = own_retries[-1]
        launch_queue(last_retry, f'natlang-pop-luna-s{seed}-w{number:02d}-r{last_retry.name.removeprefix("retry-")}')
        return True
    last = own_retries[-1] if own_retries else worker / f'case-{queue_ordinal(worker, job["key"]):03d}'
    errors = sorted((last / 'jobs').glob('*.error.json'))
    if not errors:
        return False
    message = json.loads(errors[-1].read_text()).get('error', '')
    if not any(phrase in message for phrase in ('Codex error: An error occurred while processing your request',
                                                'rate limit', '429 Too Many Requests', 'Error: terminated')):
        return False
    attempt = len(own_retries) + 1
    if attempt > 3:
        print(json.dumps({'event': 'needs_review', 'key': job['key'],
                          'reason': 'transient_retry_limit'}), flush=True)
        return False
    not_before = max(result.get('next_allowed_at', 0),
                     (own_retries[-1] / 'review.json').stat().st_mtime + min(900, 30 * 2 ** (attempt - 1))
                     if own_retries else 0)
    if time.time() < not_before:
        print(json.dumps({'event': 'retry_cooldown', 'key': job['key'],
                          'seconds': round(not_before - time.time())}), flush=True)
        return True
    ordinal = queue_ordinal(worker, job['key'])
    retry = worker / f'retry-{ordinal:03d}-{attempt:02d}'
    retry.mkdir(exist_ok=False)
    again = dict(job, output=str(retry / 'results.jsonl'), jobs=str(retry / 'jobs'),
                 log=str(retry / 'collector.log'), seed=job['seed'] + attempt * 100000)
    (retry / 'queue.jsonl').write_text(json.dumps(again, sort_keys=True) + '\n')
    (retry / 'review.json').write_text(json.dumps({
        'schema': 'natlang.teacher-transient-retry/1', 'original_key': job['key'],
        'attempt': attempt, 'error': message, 'training_admission': False}, indent=2) + '\n')
    launch_queue(retry, f'natlang-pop-luna-s{seed}-w{number:02d}-r{ordinal:03d}-{attempt:02d}')
    return True


def queue_ordinal(worker, key):
    return next(index for index, job in enumerate(rows(worker / 'queue.jsonl')) if job['key'] == key)


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


def prior_groups(current):
    index_manifest = Path(str(TARGET_GROUP_INDEX) + '.manifest.json')
    if not TARGET_GROUP_INDEX.exists() and not index_manifest.exists():
        from neuralese_data.source_group_index import build
        union = ROOT / 'runs/natlang-target-union-20261011-r3'
        build([union / rendering / 'lfm25/text-data.jsonl'
               for rendering in TARGET_GROUP_INPUT_HASHES], TARGET_GROUP_INDEX)
    manifest = json.loads(index_manifest.read_text())
    if (manifest.get('schema') != 'natlang.training-source-group-index/1' or
            manifest.get('index_sha256') != sha256_file_hex(TARGET_GROUP_INDEX) or
            {Path(item['path']).parts[-3]: item['sha256'] for item in manifest['inputs']}
            != TARGET_GROUP_INPUT_HASHES):
        raise ValueError('target r3 source-group exclusion index is stale')
    groups = {json.loads(line) for line in TARGET_GROUP_INDEX.read_text().splitlines()}
    old = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261012-v1')
    sources = list(old.glob('provider-luna-held-shape-remediation*/source/*.ir.jsonl'))
    sources += list(BASE.glob('provider-luna-held-shapes-s*-v1/source/*.ir.jsonl'))
    for source in sources:
        if current in source.parents:
            continue
        for row in rows(source):
            groups.update(row.get('source_groups') or ())
    return groups


def write_queues(campaign, seed, selected):
    """Write six source-pinned one-program queues and launch their workers."""
    for number in range(1, WORKERS + 1):
        worker = campaign / f'worker-{number:02d}'
        worker.mkdir(parents=True, exist_ok=False)
        with (worker / 'queue.jsonl').open('w') as stream:
            for ordinal, (source, index, row) in enumerate(selected[number - 1::WORKERS]):
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
    for number in range(1, WORKERS + 1):
        launch_worker(campaign, seed, number)


def prepare(campaign, seed, source, excluded, excluded_groups):
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
    excluded_world_cases = sum(row['split'] == 'train' and world(row) in excluded for row in source_rows)
    excluded_group_cases = sum(row['split'] == 'train' and
                               bool(excluded_groups.intersection(row.get('source_groups') or ()))
                               for row in source_rows)
    eligible = [row for row in source_rows if row['split'] == 'train' and world(row) not in excluded
                and not excluded_groups.intersection(row.get('source_groups') or ())]
    for distinct in (True, False):
        for row in eligible:
            family = row['curriculum']['family']
            if counts[family] >= PER_FAMILY[family.removeprefix('curriculum_')] or row['id'] in seen:
                continue
            if distinct and used_groups.intersection(row['source_groups']):
                continue
            selected.append(row)
            seen.add(row['id'])
            used_groups.update(row['source_groups'])
            counts[family] += 1
    if len(selected) < 100:
        raise ValueError(f'Too few decontaminated source cases: {len(selected)}')
    manifest = dict(schema='natlang.generation-campaign/1', seed=seed,
                    source_path=str(source), source_sha256=sha256_file_hex(source),
                    source_verified_cases=len(source_rows), selected_count=len(selected),
                    selected_by_family=dict(counts), selected_ids=[row['id'] for row in selected],
                    held_path=str(HELD), held_sha256=sha256_file_hex(HELD),
                    target_group_index=str(TARGET_GROUP_INDEX),
                    target_group_index_sha256=sha256_file_hex(TARGET_GROUP_INDEX),
                    excluded_existing_source_groups=len(excluded_groups),
                    excluded_world_cases=excluded_world_cases,
                    excluded_group_cases=excluded_group_cases,
                    world_exclusion='same family and all visible semantics (including folder_files and operation; excluding gold expected/oracle) against held and earlier sources',
                    training_admission=False, status='generation_only_pending_quality_review')
    (campaign / 'campaign-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_queues(campaign, seed, [(source, source_rows.index(row), row) for row in selected])


def prepare_backfill():
    """Recover folder worlds falsely omitted by the old input fingerprint."""
    old = Path('/srv/storage/natlang-artifacts/free-provider-generation-20261012-v1')
    seen = {world(row) for row in rows(HELD)}
    for source in old.glob('provider-luna-held-shape-remediation*/source/*.ir.jsonl'):
        seen.update(world(row) for row in rows(source))
    selected = []
    sources = {}
    for seed in range(52007, 52011):
        source = BASE / f'provider-luna-held-shapes-s{seed}-v1/source/held-shapes-s{seed}-n8.ir.jsonl'
        source_rows = rows(source)
        report = json.loads(Path(str(source) + '.report.json').read_text())
        if report['verified'] != len(source_rows):
            raise ValueError(f'backfill source unverified: {source}')
        sources[str(source)] = sha256_file_hex(source)
        for index, row in enumerate(source_rows):
            if row['family'] != 'curriculum_folder_criteria_reducer':
                continue
            key = world(row)
            if key in seen:
                continue
            seen.add(key)
            selected.append((source, index, row))
    if len(selected) != 128:
        raise ValueError(f'expected 128 source-disjoint folder backfill cases, got {len(selected)}')
    BACKFILL.mkdir(exist_ok=False)
    manifest = dict(schema='natlang.generation-campaign/1', seed=BACKFILL_UNIT_SEED,
                    source_paths=sources, selected_count=len(selected),
                    selected_by_family={'folder_criteria_reducer': len(selected)},
                    selected_ids=[row['id'] for _, _, row in selected],
                    held_path=str(HELD), held_sha256=sha256_file_hex(HELD),
                    world_exclusion='all visible semantics including folder_files; old s52005/s52006 and held sources excluded',
                    correction='restore folder cases lost by old fingerprint omission',
                    training_admission=False, status='generation_only_pending_quality_review')
    (BACKFILL / 'campaign-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    write_queues(BACKFILL, BACKFILL_UNIT_SEED, selected)


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
                    source, prior_worlds(campaign), prior_groups(campaign))
                    return
                raise ValueError(f'Incomplete campaign preparation: {campaign}')
            if not complete(campaign):
                return
        if not BACKFILL.exists():
            prepare_backfill()
            return
        if not complete(BACKFILL):
            return
        seed = START_SEED if not campaigns else max(int(p.name.split('-s')[-1].split('-')[0]) for p in campaigns) + 1
        campaign = BASE / f'provider-luna-held-shapes-s{seed}-v1'
        campaign.mkdir(exist_ok=True)
        source = build_source(campaign, seed)
        excluded = prior_worlds(campaign)
        prepare(campaign, seed, source, excluded, prior_groups(campaign))


if __name__ == '__main__':
    main()
