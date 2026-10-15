#!/usr/bin/env python3
"""Supply bounded, source-disjoint Luna curriculum campaigns on Pop.

This schedules generation only. Successful outputs still require a separate
quality/decontamination review and explicit corpus admission.
"""
import collections
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training/neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex
from neuralese_data.prepare_recovery_teacher_queue import prepare as prepare_recovery_queue
from neuralese_data.select_repair_handoffs import select as select_repair_handoffs

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
            'musique_hops', 'knowledge_research')
START_SEED = 52007
WORKERS = 6
PER_FAMILY = {family: 6 for family in FAMILIES}
# Both MuSiQue hop variants are under a source-contract family hold. Keep the
# key for already prepared campaigns, but do not spend new teacher calls on it.
PER_FAMILY['musique_hops'] = 0
# Large CoEdIT folder edits require file-by-file semantic review even when the
# numeric return matches. The exact reference replay has a separate admitted
# static route. CUAD folder_extract presently takes hundreds of requests per
# 20-35-contract case; keep the six pinned s52029 trials, but do not repeat
# this shape in later campaigns before a smaller variant is measured.
# Keep the zero cap for an already-built source encountered on resume.
PER_FAMILY['folder_edit'] = 0
PER_FAMILY['folder_extract'] = 0
PER_FAMILY['folder_criteria_reducer'] = 8
# inline_review_each also emits its paired no-crisp family. Preserve both
# behaviours without letting this one builder take two full family slots.
PER_FAMILY.update(inline_review_each=3, inline_avoid_crisp=3)
# Source-disjoint candidates remain plentiful in these deeper semantic and
# reducer families after the MuSiQue hold. Use them rather than requiring an
# arbitrary total that the old six-per-family caps cannot reach.
PER_FAMILY.update(actor_greenhouse=10, dataset_workbench=10,
                  workbench_tickets=8, composed_process=8,
                  recurrence_inline_graphs=8, folder_criteria_reducer=10,
                  knowledge_research=8, relational_late_argmax=8)
# Exact final answers are insufficient for these shapes: the backlog-v5 audit
# found that their successful trajectories systematically bypassed the intended
# per-item NL decision or measured iterateOn process. Fresh s52065/s52066
# successes show the same pattern. The wider scan also found greenhouse and
# schedule repair almost never used iterateOn. Hold *new* teacher allocation until the
# visible task and source oracle enforce the process, then measure the changed
# task live. Folder criteria version 2 passed its 48-case live shape trial.
# Greenhouse version 3 gave actual iteration steps in all 38 exact outcomes of
# 48 measured cases; broken-heater failures stay rejected by the status oracle.
# New allocation for those two families resumes behind exact and process review.
# Existing campaigns still finish and remain available for review.
PER_FAMILY.update(actor_greenhouse=10, composed_process=0,
                  edit_stream=0, folder_criteria_reducer=2,
                  counterexample_revision=0, iterate_schedule_repair=0)
PER_FAMILY['inline_review_each'] = 4
# The v2 folder task is still useful, but seed52096 admitted only one of its
# ten attempts after exact and process review. Keep a small live sample while
# the 48-case direct-judgment trial runs; spend the freed calls on families
# with stronger recent exact/process yield. Restore the folder share if its
# revised source qualifies on the live executor.
PER_FAMILY.update(recurrence_inline_graphs=16, knowledge_research=10,
                  actor_greenhouse=12, relational_late_argmax=10)
SOURCE_FAMILIES = tuple(family for family in FAMILIES if PER_FAMILY[family] > 0)
TARGET_GROUP_INDEX = BASE / 'target-r3-source-groups.jsonl'
TARGET_GROUP_INPUT_HASHES = {
    'program-header': '0974200754e27da114f429de3142fb3be8500431bcb0837cbea35868ebf30ba9',
    'whole-trajectory': 'c30b6bbeb2887bc9a533a5aaa18111216caf6dc4252511a7d0985e32f7044501',
}
BACKFILL = BASE / 'provider-luna-folder-backfill-s52007-s52010-v1'
BACKFILL_UNIT_SEED = 52070
REPAIR = BASE / 'repair-handoffs'
FIRST_REPAIR_SEED = 52026
REPAIR_EVERY_SEEDS = 8
# Student-root handoffs are already selected from exact replay. Give their
# bounded correction queues first claim on the existing six Luna slots.
STUDENT_REPAIR_CAMPAIGNS = (REPAIR / 'qad-r8-student-root-v4/luna',)


def rows(path):
    # JSONL uses LF as its record boundary. str.splitlines() also splits on
    # U+2028 inside a valid JSON string (present in the source datasets).
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


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


def active_luna_workers():
    result = subprocess.run(['systemctl', '--user', 'list-units', '--type=service',
                             '--state=running', '--plain', '--no-legend'],
                            check=True, capture_output=True, text=True)
    units = [line.split()[0] for line in result.stdout.splitlines() if line.split()]
    return sum(unit.startswith('natlang-pop-luna-') and '-w' in unit for unit in units)


def next_source_can_prebuild(campaign):
    """Prepare the next seed before a draining campaign releases teacher slots."""
    manifest = json.loads((campaign / 'campaign-manifest.json').read_text())
    selected = manifest['selected_count']
    finished = set()
    for worker in campaign.glob('worker-[0-9][0-9]'):
        for journal in [worker / 'journal.jsonl', *(path / 'journal.jsonl' for path in worker.glob('retry-*'))]:
            if journal.exists():
                finished.update(event['key'] for event in rows(journal) if event.get('event') == 'finish')
    available_kib = next(int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))
    return len(finished) >= max(0, selected - 30) and available_kib >= 4 * 1024 * 1024


def luna_unit_prefix(campaign, seed, *, repair=False):
    if repair:
        return f'natlang-pop-luna-repair-s{seed}'
    if campaign == BACKFILL:
        return f'natlang-pop-luna-backfill-s{seed}'
    return f'natlang-pop-luna-s{seed}'


def durable_progress(journals, missing, finished):
    """Count completed jobs and the largest saved turn of an unfinished job.

    A resumed collector may advance hundreds of durable turns before it can
    emit one terminal result. Treat those turns as progress across relaunches.
    """
    saved_turns = max((event.get('saved_turns', 0)
                       for journal in journals if journal.exists()
                       for event in rows(journal)
                       if event.get('key') in missing and event.get('event') == 'activity'), default=0)
    return len(finished), saved_turns


def launch_budget_gb(queue_dir):
    """Allow a growing resumable partial enough RAM for its retained context."""
    # Ordinary workers store one job directory per case; a retry queue writes
    # directly into retry-NNN-NN/jobs. Inspect both before admitting the unit.
    partials = list(queue_dir.glob('case-*/jobs/*.partial.json'))
    partials.extend(queue_dir.glob('jobs/*.partial.json'))
    if any(path.stat().st_size >= 8_000_000 for path in partials):
        return '2.0'
    return '1.5' if queue_dir.name.startswith('retry-') else '0.9'


def complete(campaign):
    manifest = json.loads((campaign / 'campaign-manifest.json').read_text())
    repair = manifest.get('repair_cycle', False)
    exclusions_path = campaign / 'reviewed-exclusions.json'
    exclusions = json.loads(exclusions_path.read_text()) if exclusions_path.exists() else {}
    if exclusions and exclusions.get('schema') != 'natlang.generation-reviewed-exclusions/1':
        raise ValueError(f'Unrecognized reviewed exclusions: {exclusions_path}')
    excluded = exclusions.get('cases', {})
    pending = False
    for number in range(1, manifest.get('workers', WORKERS) + 1):
        worker = campaign / f'worker-{number:02d}'
        queue = rows(worker / 'queue.jsonl')
        prefix = luna_unit_prefix(campaign, manifest['seed'], repair=repair)
        unit = f'{prefix}-w{number:02d}.service'
        retry_dirs = sorted(worker.glob('retry-*'))
        if active(unit) or any(active(f'{prefix}-w{number:02d}-r{path.name.removeprefix("retry-")}.service')
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
            progress = durable_progress(journals, set(missing), finished)
            old_progress = (previous.get('finished', -1), previous.get('saved_turns', -1))
            failures = 0 if progress > old_progress else previous.get('failures', 0) + 1
            if failures >= 4:
                print(json.dumps({'event': 'needs_review', 'unit': unit,
                                  'reason': 'four_no_progress_relaunches', 'missing': len(missing),
                                  'durable_progress': progress}), flush=True)
                return False
            subprocess.run(['systemctl', '--user', 'reset-failed', unit], check=False)
            if not launch_worker(campaign, manifest['seed'], number, repair=repair):
                return False  # memory pressure is a deferred launch, not a no-progress attempt
            retry_file.write_text(json.dumps({'failures': failures, 'finished': progress[0],
                                               'saved_turns': progress[1]}) + '\n')
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
                if retry_transient(campaign, worker, manifest['seed'], number, job, result, retry_dirs,
                                   repair=repair):
                    pending = True
                    continue
                print(json.dumps({'event': 'needs_review', 'campaign': str(campaign),
                                  'key': job['key'], 'finish': result.get('status')}), flush=True)
                return False
    return not pending


def has_launchable_waiting_worker(campaign):
    """Prefer unfinished workers of an older prepared campaign over a new seed."""
    manifest = json.loads((campaign / 'campaign-manifest.json').read_text())
    prefix = luna_unit_prefix(campaign, manifest['seed'], repair=manifest.get('repair_cycle', False))
    for number in range(1, manifest.get('workers', WORKERS) + 1):
        worker = campaign / f'worker-{number:02d}'
        if active(f'{prefix}-w{number:02d}.service') or any(
                active(f'{prefix}-w{number:02d}-r{path.name.removeprefix("retry-")}.service')
                for path in worker.glob('retry-*')):
            continue
        state_path = worker / 'relaunch-state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if state.get('failures', 0) >= 4:
            continue  # this worker needs investigation; other campaigns may proceed
        journals = [worker / 'journal.jsonl', *(path / 'journal.jsonl' for path in worker.glob('retry-*'))]
        finished = {event['key'] for journal in journals if journal.exists()
                    for event in rows(journal) if event.get('event') == 'finish'}
        if any(job['key'] not in finished for job in rows(worker / 'queue.jsonl')):
            return True
    return False


def launch_queue(queue_dir, unit):
    # The worker cap is global across overlapping campaigns and retries. The
    # memory ledger constrains RAM independently; it is not a concurrency cap.
    if active_luna_workers() >= WORKERS:
        print(json.dumps({'event': 'deferred_parallelism', 'unit': unit,
                          'active_luna_workers': WORKERS}), flush=True)
        return False
    command = ['python3', 'scripts/run_bonsai_queue.py', str(queue_dir / 'queue.jsonl'),
               str(queue_dir / 'journal.jsonl'), '--runtime', str(RUNTIME),
               '--case-seconds', '1800', '--provider', 'openai-codex',
               '--model-id', 'gpt-6-luna', '--model-concurrency', '1',
               '--execution-plans', '--reasoning-effort', 'low', '--min-free-mib', '1024']
    # A retry may process a much larger partially completed program than its
    # original one-case queue. Give its collector enough cgroup RAM to resume.
    budget = launch_budget_gb(queue_dir)
    run = subprocess.run(['python3', 'scripts/memory_ledger.py', 'run', '--unit', unit,
                          '--budget-gb', budget, '--reserve-gb', '0.2', '--class', 'collection',
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


def launch_worker(campaign, seed, number, *, repair=False):
    prefix = luna_unit_prefix(campaign, seed, repair=repair)
    return launch_queue(campaign / f'worker-{number:02d}', f'{prefix}-w{number:02d}')


def copy_partial_for_resume(previous_jobs, retry_jobs, root_seed):
    """Preserve the collector's exact request journal for a larger budget run.

    The collector replays saved responses by request hash only when the partial
    provenance still matches, including the root seed. Evidence snapshots are
    copied with their content hashes; no mutable file is hardlinked.
    """
    partials = list(previous_jobs.glob('*.partial.json'))
    if len(partials) != 1:
        raise ValueError(f'Expected one resumable partial in {previous_jobs}, got {len(partials)}')
    partial = partials[0]
    state = json.loads(partial.read_text())
    if state.get('version') != 'natlang.teacher_partial.native/1' or not isinstance(state.get('turns'), list):
        raise ValueError(f'Invalid collector partial: {partial}')
    if state.get('provenance', {}).get('seed_policy', {}).get('root') != root_seed:
        raise ValueError(f'Collector partial seed differs from retry seed: {partial}')
    retry_jobs.mkdir(parents=True, exist_ok=False)
    for snapshot in state.get('evidence_snapshots') or []:
        name = snapshot['path']
        if Path(name).name != name:
            raise ValueError(f'Unsafe collector evidence path: {name}')
        source = previous_jobs / name
        if (source.stat().st_size != snapshot['bytes'] or
                sha256_file_hex(source) != snapshot['sha256']):
            raise ValueError(f'Collector evidence changed after partial: {source}')
        shutil.copy2(source, retry_jobs / name)
    shutil.copy2(partial, retry_jobs / partial.name)
    return sha256_file_hex(partial)


def retry_transient(campaign, worker, seed, number, job, result, retry_dirs, *, repair=False):
    """Retry a resultless provider error or measured collection budget exhaustion."""
    stream_overflow = (result.get('status') == 'provider_stream_overflow' and
                       result.get('resource_limit_reason') == 'provider_tool_call_stream_overflow')
    if not stream_overflow and (result.get('status') != 'incomplete' or result.get('exit_code') != 2):
        return False
    if result.get('output_accounting', {}).get('exact_result_rows') != 0:
        return False
    own_retries = [path for path in retry_dirs
                   if (path / 'review.json').exists() and
                   json.loads((path / 'review.json').read_text()).get('original_key') == job['key']]
    retry_journal = own_retries[-1] / 'journal.jsonl' if own_retries else None
    if own_retries and (not retry_journal.exists() or
                        not any(event.get('event') == 'finish' for event in rows(retry_journal))):
        last_retry = own_retries[-1]
        prefix = luna_unit_prefix(campaign, seed, repair=repair)
        unit = f'{prefix}-w{number:02d}-r{last_retry.name.removeprefix("retry-")}'
        state_file = last_retry / 'relaunch-state.json'
        state = json.loads(state_file.read_text()) if state_file.exists() else {'failures': 0}
        if state['failures'] >= 4:
            print(json.dumps({'event': 'needs_review', 'unit': unit,
                              'reason': 'four_interrupted_retry_relaunches'}), flush=True)
            return False
        subprocess.run(['systemctl', '--user', 'reset-failed', unit], check=False)
        if launch_queue(last_retry, unit):
            state_file.write_text(json.dumps({'failures': state['failures'] + 1}) + '\n')
        return True
    last = own_retries[-1] if own_retries else worker / f'case-{queue_ordinal(worker, job["key"]):03d}'
    errors = sorted((last / 'jobs').glob('*.error.json'))
    if not errors and not stream_overflow:
        return False
    message = json.loads(errors[-1].read_text()).get('error', '') if errors else ''
    resource_budget = (bool(errors) and result.get('resource_limit_reason') == 'model_request_budget' and
                       'NATLANG_MODEL_REQUEST_BUDGET' in errors[-1].read_text())
    provider_error = any(phrase in message for phrase in
                         ('Codex error: An error occurred while processing your request',
                          'rate limit', '429 Too Many Requests', 'Error: terminated',
                          'upstream connect error or disconnect/reset before headers',
                          'ECONNRESET', '502 Bad Gateway', '503 Service Unavailable',
                          '504 Gateway Timeout'))
    if not (resource_budget or provider_error or stream_overflow):
        return False
    attempt = len(own_retries) + 1
    if attempt > (1 if resource_budget else 3):
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
                 log=str(retry / 'collector.log'),
                 seed=job['seed'] if resource_budget else job['seed'] + attempt * 100000)
    if resource_budget:
        again['max_model_requests'] = max(192 if repair else 768,
                                          job['max_model_requests'] * 2)
        again['case_seconds'] = max(1800 if repair else 3600,
                                    job['case_seconds'] * (1 if repair else 2))
    if resource_budget:
        try:
            partial_sha256 = copy_partial_for_resume(last / 'jobs', retry / 'jobs', job['seed'])
        except Exception:
            shutil.rmtree(retry)
            raise
    (retry / 'queue.jsonl').write_text(json.dumps(again, sort_keys=True) + '\n')
    (retry / 'review.json').write_text(json.dumps({
        'schema': 'natlang.teacher-transient-retry/1', 'original_key': job['key'],
        'attempt': attempt, 'error': message,
        'reason': 'observed_model_request_budget' if resource_budget else 'provider_transport',
        **({'resumed_partial_sha256': partial_sha256, 'seed_preserved': True} if resource_budget else {}),
        'training_admission': False}, indent=2) + '\n')
    prefix = luna_unit_prefix(campaign, seed, repair=repair)
    launch_queue(retry, f'{prefix}-w{number:02d}-r{ordinal:03d}-{attempt:02d}')
    return True


def prepare_repair_cycle(campaigns, seed):
    """Replay failed train runs and pin a balanced correction queue once per eight broad batches."""
    cycle = REPAIR / f'luna-s{seed - REPAIR_EVERY_SEEDS}-s{seed - 1}-v1'
    if cycle.exists():
        if not ((cycle / 'teacher/campaign-manifest.json').exists() or
                (cycle / 'empty-selection.json').exists()):
            raise ValueError(f'incomplete repair cycle preparation: {cycle}')
        return cycle
    cycle.mkdir(parents=True)
    recent = [campaign for campaign in campaigns if
              seed - REPAIR_EVERY_SEEDS <= int(campaign.name.split('-s')[-1].split('-')[0]) < seed]
    if len(recent) != REPAIR_EVERY_SEEDS:
        raise ValueError(f'repair cycle needs eight closed source campaigns, found {len(recent)}')
    results = sorted(path for source in recent for path in source.glob('worker-*/case-*/results.jsonl'))
    raw = cycle / 'handoffs.raw.ir.jsonl'
    subprocess.run(['node', str(RUNTIME / 'scripts/build-handoffs.mjs'), str(raw),
                    *map(str, results), '--workers', '6'], cwd=ROOT, check=True)
    selected = cycle / 'handoffs.selected.ir.jsonl'
    selection = select_repair_handoffs(raw, HELD, selected)
    if not selection['selected']:
        (cycle / 'empty-selection.json').write_text(json.dumps(selection, indent=2) + '\n')
        return cycle
    prepare_recovery_queue(selected, cycle / 'teacher', line='lfm25', workers=WORKERS,
                           case_seconds=1800)
    manifest_path = cycle / 'teacher/campaign-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(seed=seed, repair_cycle=True, runtime=str(RUNTIME),
                    runtime_sha256=sha256_file_hex(RUNTIME / 'frozen-runtime.json'),
                    source_campaigns={str(source): sha256_file_hex(source / 'campaign-manifest.json')
                                      for source in recent})
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return cycle


def finalize_repair_cycle(cycle):
    """Prepare exact-success turns and causal pairs for review, never training admission."""
    package = cycle / 'review-candidates.json'
    if package.exists():
        return
    teacher = cycle / 'teacher'
    parents = cycle / 'accepted-parents.jsonl'
    parent_review = parents.with_suffix('.review.json')
    if not parents.exists() and not parent_review.exists():
        subprocess.run(['python3', 'scripts/collect_luna_accepted_parents.py',
                        '--campaign', str(teacher), '--held', str(HELD),
                        '--out', str(parents)], cwd=ROOT, check=True)
    if not parents.is_file() or not parent_review.is_file():
        raise ValueError(f'incomplete repair parent collection: {cycle}')
    native = cycle / 'native-turns.review-candidates.jsonl'
    if not native.exists() and parents.stat().st_size:
        subprocess.run(['node', str(RUNTIME / 'scripts/materialize-native-teacher.mjs'),
                        str(parents), str(native)], cwd=ROOT, check=True)
    pairs = cycle / 'preference-pairs.review-candidates.jsonl'
    if not pairs.exists() and parents.stat().st_size:
        subprocess.run(['node', str(RUNTIME / 'scripts/build-preference-pairs.mjs'),
                        str(pairs), '--handoffs', str(parents), '--workers', '6'], cwd=ROOT, check=True)
    outputs = {name: {'path': str(path), 'sha256': sha256_file_hex(path),
                      **({'rows': sum(1 for line in path.open() if line.strip())}
                         if path.suffix == '.jsonl' else {})}
               for name, path in (('accepted_parents', parents), ('parent_review', parent_review),
                                  ('native_turns', native), ('preference_pairs', pairs)) if path.exists()}
    package.write_text(json.dumps({
        'schema': 'natlang.repair-review-candidates/1',
        'source_manifest_sha256': sha256_file_hex(teacher / 'campaign-manifest.json'),
        'runtime_sha256': sha256_file_hex(RUNTIME / 'frozen-runtime.json'),
        'outputs': outputs, 'training_admission': False,
        'required_review': 'native decision quality, split/source decontamination, causal pair audit, explicit corpus registration',
    }, indent=2) + '\n')


def run_repair_cycle(campaigns):
    by_seed = {int(campaign.name.split('-s')[-1].split('-')[0]): campaign for campaign in campaigns}
    # Overlapping source campaigns can be prepared before the last worker in
    # the preceding eight finishes. Discover every closed boundary instead of
    # testing only the newest directory name, or that repair window is lost.
    for seed in range(FIRST_REPAIR_SEED, max(by_seed) + 1, REPAIR_EVERY_SEEDS):
        recent = [by_seed.get(previous) for previous in range(seed - REPAIR_EVERY_SEEDS, seed)]
        if any(campaign is None or not complete(campaign) for campaign in recent):
            continue
        cycle = REPAIR / f'luna-s{seed - REPAIR_EVERY_SEEDS}-s{seed - 1}-v1'
        if (cycle / 'empty-selection.json').exists() or (cycle / 'review-candidates.json').exists():
            continue
        cycle = prepare_repair_cycle(campaigns, seed)
        if (cycle / 'empty-selection.json').exists():
            continue
        teacher = cycle / 'teacher'
        manifest = json.loads((teacher / 'campaign-manifest.json').read_text())
        for number in range(1, manifest['workers'] + 1):
            worker = teacher / f'worker-{number:02d}'
            if not rows(worker / 'queue.jsonl'):
                continue
            if not (worker / 'journal.jsonl').exists() and not active(f'natlang-pop-luna-repair-s{seed}-w{number:02d}.service'):
                launch_worker(teacher, seed, number, repair=True)
        if not complete(teacher):
            # An old repair's slow final case must not strand five teacher
            # slots. Offer them to the next closed repair window; if there is
            # no such window, the caller may prepare a new broad campaign.
            if active_luna_workers() >= WORKERS:
                return True
            continue
        finalize_repair_cycle(cycle)
        if active_luna_workers() >= WORKERS:
            return True
    return False


def queue_ordinal(worker, key):
    return next(index for index, job in enumerate(rows(worker / 'queue.jsonl')) if job['key'] == key)


def build_source(campaign, seed):
    source = campaign / 'source' / f'held-shapes-s{seed}-n8.ir.jsonl'
    if source.exists():
        verified_source_report(source)
        return source
    source.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['node', 'ts-host/scripts/inline-curriculum/build.mjs',
                    '--seed', str(seed), '--shapes', '8', '--split', 'train',
                    '--families', ','.join(SOURCE_FAMILIES), '--allow-failures',
                    '--out', str(source)],
                   cwd=ROOT, check=True)
    try:
        report = verified_source_report(source)
    except ValueError:
        source.unlink(missing_ok=True)
        raise
    if report['failures']:
        print(json.dumps({'event': 'source_review_excluded', 'seed': seed,
                          'cases': len(report['failures']),
                          'ids': [failure['id'] for failure in report['failures']]}), flush=True)
    return source


def verified_source_report(source):
    report = json.loads(Path(str(source) + '.report.json').read_text())
    source_rows = rows(source)
    failures = report['failures']
    unexpected = [failure for failure in failures
                  if failure['problems'] != ['reference not admitted: source_review_pending']]
    source_ids = {row['id'] for row in source_rows}
    if (unexpected or report['verified'] != len(source_rows) or
            report['cases'] != len(source_rows) + len(failures) or
            any(failure['id'] in source_ids for failure in failures)):
        raise ValueError(f'Source replay verification incomplete: {source}')
    return report


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
    groups = {row for row in rows(TARGET_GROUP_INDEX)}
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
                expensive = row['family'] == 'curriculum_counterexample_revision'
                job = dict(key=f'luna-held-shape-s{seed}-w{number:02d}-{ordinal:03d}',
                           source=str(source), index=index, count=1,
                           output=str(case / 'results.jsonl'), jobs=str(case / 'jobs'),
                           log=str(case / 'collector.log'), seed=seed * 1000 + (number - 1) * 100 + ordinal,
                           context_tokens=32768, action_max_tokens=4096,
                           max_turns=60, max_model_requests=768 if expensive else 384,
                           case_seconds=3600 if expensive else 1800,
                           request_retries=3, transport_retries=0,
                           text_neuralese_emulation=True)
                stream.write(json.dumps(job, sort_keys=True) + '\n')
    for number in range(1, WORKERS + 1):
        launch_worker(campaign, seed, number)


def prepare(campaign, seed, source, excluded, excluded_groups):
    source_rows = rows(source)
    report = verified_source_report(source)
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
    if len(selected) < WORKERS:
        raise ValueError(f'Fewer decontaminated source cases than workers: {len(selected)} < {WORKERS}')
    manifest = dict(schema='natlang.generation-campaign/1', seed=seed,
                    source_path=str(source), source_sha256=sha256_file_hex(source),
                    source_verified_cases=len(source_rows), selected_count=len(selected),
                    selection_target=100, selection_shortfall=max(0, 100 - len(selected)),
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
    # This script is intentionally optionless; still parse the CLI so --help
    # remains read-only instead of unexpectedly starting a generation cycle.
    import argparse
    argparse.ArgumentParser(description='Replenish the reviewed Luna held-shapes generation campaigns.').parse_args()
    BASE.mkdir(parents=True, exist_ok=True)
    if not RUNTIME.exists():
        raise FileNotFoundError(RUNTIME)
    with (BASE / '.luna-replenisher.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        student_repair_pending = False
        for campaign in STUDENT_REPAIR_CAMPAIGNS:
            if campaign.exists() and not complete(campaign):
                print(json.dumps({'event': 'student_repair_pending', 'campaign': str(campaign)}), flush=True)
                student_repair_pending = True
        if student_repair_pending:
            return
        campaigns = sorted(BASE.glob('provider-luna-held-shapes-s*-v1'))
        pending_campaigns = []
        for campaign in campaigns:
            manifest_path = campaign / 'campaign-manifest.json'
            if not manifest_path.exists():
                seed = int(campaign.name.split('-s')[-1].split('-')[0])
                if active(f'natlang-pop-build-held-s{seed}.service'):
                    print(json.dumps({'event': 'source_build_wait', 'campaign': str(campaign)}), flush=True)
                    return
                # A failed source build leaves the campaign directory and its
                # verification report, but no shard. Rebuild that exact seed
                # and keep valid cases after reviewed source holds are removed.
                source = build_source(campaign, seed)
                prepare(campaign, seed, source, prior_worlds(campaign), prior_groups(campaign))
                return
            if not complete(campaign):
                pending_campaigns.append(campaign)
        if not BACKFILL.exists():
            prepare_backfill()
            return
        if not complete(BACKFILL):
            return
        # One slow source shape must not leave the other teacher slots idle.
        # Every prepared campaign's source worlds/groups enter prior_worlds and
        # prior_groups, so overlapping campaigns stay decontaminated.
        # Bound the overlap to three campaigns so a stalled pilot plus a
        # draining ordinary campaign can still replenish free worker slots.
        # Repair cycles still require their preceding campaigns to close.
        if campaigns and run_repair_cycle(campaigns):
            return
        # A full teacher pool must not postpone CPU-only finalization of a
        # closed repair window. The repair cycle itself applies the worker cap
        # before launching any new provider attempt.
        full_pool = active_luna_workers() >= WORKERS
        prebuild = (full_pool and len(pending_campaigns) == 1 and
                    next_source_can_prebuild(pending_campaigns[0]))
        if (len(pending_campaigns) >= 3 or (full_pool and not prebuild) or
                any(has_launchable_waiting_worker(campaign) for campaign in pending_campaigns)):
            return
        seed = START_SEED if not campaigns else max(int(p.name.split('-s')[-1].split('-')[0]) for p in campaigns) + 1
        campaign = BASE / f'provider-luna-held-shapes-s{seed}-v1'
        campaign.mkdir(exist_ok=True)
        source = build_source(campaign, seed)
        excluded = prior_worlds(campaign)
        prepare(campaign, seed, source, excluded, prior_groups(campaign))


if __name__ == '__main__':
    main()
