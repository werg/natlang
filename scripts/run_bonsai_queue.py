#!/usr/bin/env python3
"""Run a frozen, interleaved teacher queue with a hard per-case wall-clock budget.

Timeouts and failures remain explicit in the append-only journal and are not training results.
Restart skips attempted entries; use a new journal for an explicitly reviewed retry pass.
"""
import argparse
import hashlib
import json
import subprocess
import signal
import time
import math
import random
import shutil
import urllib.request
from pathlib import Path


OUTPUT_ACCOUNTING_VERSION = 'natlang.supervisor_output_accounting/2'


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return None


def canonical_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def frozen_record_digests(records, runtime=None):
    runtime = Path(runtime) if runtime else Path(__file__).resolve().parents[1] / 'ts-host'
    code = """
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const [collector] = process.argv.slice(1);
const { recordDigest } = await import(pathToFileURL(collector).href);
const records = JSON.parse(readFileSync(0, 'utf8'));
process.stdout.write(JSON.stringify(records.map(record => recordDigest(record))));
"""
    output = subprocess.check_output(['node', '--input-type=module', '-e', code,
        str(runtime / 'dist/teacher/collector.js')], input=json.dumps(records), text=True)
    return json.loads(output)


def output_accounting(entry, runtime=None):
    """Validate exact collector rows and batch manifest before a queue key can be called complete."""
    jobs = resolve_entry_jobs(entry, runtime)
    root = Path(entry['jobs'])
    states, result_ids, result_digests, saved_result_rows = [], [], [], []
    candidate_result_rows = []
    for job in jobs:
        result_path = root / f"{job['key']}.result.json"
        partial_path = root / f"{job['key']}.partial.json"
        error_path = root / f"{job['index']:06d}.error.json"
        row = read_json(result_path)
        if result_path.is_file() and row is None:
            state = 'invalid_result'
        elif row is not None:
            program = row.get('task', {}).get('program_ir', {})
            provenance = row.get('provenance', {})
            terminal = row.get('outcome', {}).get('status') in {'done', 'quiesced', 'failed'} and \
                isinstance(row.get('outcome', {}).get('accepted'), bool)
            exact = program.get('id') == job['program_id'] and \
                provenance.get('program_ir_sha256') == job['digest'] and terminal
            state = 'result' if exact else 'invalid_result'
            if exact:
                candidate_result_rows.append((job, row, result_path))
        else:
            error = read_json(error_path)
            if error and error.get('index') == job['index'] and error.get('program_id') == job['program_id'] and \
                    error.get('generation_hold') and error.get('generation_hold') == job.get('generation_hold'):
                state = 'generation_held'
            elif error and error.get('index') == job['index'] and error.get('program_id') == job['program_id'] and error.get('generation_hold'):
                state = 'invalid_generation_hold'
            elif error and error.get('index') == job['index'] and error.get('program_id') == job['program_id']:
                message = str(error.get('error', '')).lower()
                state = 'transport_failed' if any(term in message for term in
                    ('transport', 'rate limit', 'timed out', 'timeout', 'socket', 'connection', 'fetch failed')) else 'failed'
            elif partial_path.is_file():
                partial = read_json(partial_path)
                state = 'partial_without_terminal_result' if partial and \
                    partial.get('program_id') == job['program_id'] and \
                    partial.get('provenance', {}).get('program_ir_sha256') == job['digest'] else 'invalid_partial'
            else:
                state = 'missing_terminal_result'
        states.append({'index': job['index'], 'program_id': job['program_id'],
                       'digest': job['digest'], 'state': state})

    embedded_records = [row.get('task', {}).get('program_ir') for _, row, _ in candidate_result_rows]
    try:
        embedded_digests = frozen_record_digests(embedded_records, runtime)
    except Exception as error:
        embedded_digests = []
        embedded_digest_error = f'{type(error).__name__}: {error}'
    else:
        embedded_digest_error = None
    states_by_job = {(state['index'], state['digest']): state for state in states}
    for position, (job, row, result_path) in enumerate(candidate_result_rows):
        state = states_by_job[(job['index'], job['digest'])]
        if position >= len(embedded_digests) or embedded_digests[position] != job['digest']:
            state['state'] = 'invalid_result_ir_digest'
            continue
        state['state'] = 'result'
        result_ids.append(job['program_id'])
        result_digests.append(job['digest'])
        saved_result_rows.append(row)

    output = Path(entry['output'])
    manifest = read_json(str(output) + '.manifest.json')
    output_exists = output.is_file()
    output_hash = sha256_file(output) if output_exists else None
    output_rows, output_error = [], None
    if output_exists:
        try:
            with output.open() as source:
                for line_number, line in enumerate(source, 1):
                    if line.strip():
                        output_rows.append(json.loads(line))
        except (OSError, ValueError) as error:
            output_error = f'{type(error).__name__}: {error}'
    output_row_errors = []
    if output_rows != saved_result_rows:
        output_row_errors.append('merged output rows do not structurally equal the ordered exact saved job rows')
    output_row_hashes = [canonical_json_sha256(row) for row in output_rows]
    saved_row_hashes = [canonical_json_sha256(row) for row in saved_result_rows]
    manifest_ok = bool(manifest and manifest.get('version') == 'natlang.teacher_batch.native/1' and
        manifest.get('range') == {'start': entry['index'], 'count': len(jobs)} and
        manifest.get('completed') == len(result_ids) and
        manifest.get('missing') == [state['index'] for state in states if state['state'] != 'result'] and
        manifest.get('output_sha256') == output_hash and manifest.get('source') == entry['source'] and
        manifest.get('source_sha256') == sha256_file(entry['source']) and not output_error and
        output_row_hashes == saved_row_hashes and not output_row_errors)
    all_accounted = all(state['state'] in {'result', 'generation_held'} for state in states)
    any_held = any(state['state'] == 'generation_held' for state in states)
    unresolved = [state['state'] for state in states if state['state'] not in {'result', 'generation_held'}]
    if not manifest_ok or not all_accounted:
        disposition = next((value for value in ('transport_failed', 'partial_without_terminal_result',
            'invalid_result_ir_digest', 'invalid_generation_hold', 'invalid_result', 'invalid_partial',
            'failed', 'missing_terminal_result') if value in unresolved),
            'export_validation_failed')
        complete = False
    else:
        disposition = 'explicit_generation_hold' if any_held else 'all_exact_results_exported'
        complete = True
    transport_retry_observed = False
    if not complete and disposition == 'partial_without_terminal_result':
        collector_log = Path(entry.get('log', ''))
        if collector_log.is_file():
            try:
                text = collector_log.read_text(errors='replace')
                transport_retry_observed = any(job['program_id'] in line and 'retry:' in line and
                    ('transport_failure' in line or 'rate_limit' in line) for line in text.splitlines())
            except OSError:
                pass
    if not complete and transport_retry_observed and disposition == 'partial_without_terminal_result':
        disposition = 'transport_failure_with_partial_checkpoint'
    return {'version': OUTPUT_ACCOUNTING_VERSION, 'complete': complete, 'disposition': disposition,
            'expected_jobs': len(jobs), 'exact_result_rows': len(result_ids),
            'explicitly_generation_held': sum(state['state'] == 'generation_held' for state in states),
            'job_states': states, 'output_exists': output_exists, 'output_sha256': output_hash,
            'output_rows': len(output_rows), 'output_error': output_error, 'output_row_errors': output_row_errors,
            'saved_row_hashes': saved_row_hashes, 'output_row_hashes': output_row_hashes,
            'embedded_digest_error': embedded_digest_error,
            'manifest_present': manifest is not None, 'manifest_valid': manifest_ok,
            'transport_retry_observed': transport_retry_observed}


def resolve_entry_jobs(entry, runtime=None):
    """Use the collector's own digest: indices can overlap between source campaigns."""
    if '_resolved_jobs' not in entry:
        runtime = Path(runtime) if runtime else Path(__file__).resolve().parents[1] / 'ts-host'
        code = """
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const [collector, source, start, count] = process.argv.slice(1);
const collectorUrl = pathToFileURL(collector);
const { recordDigest } = await import(collectorUrl.href);
const { generationHoldReason } = await import(new URL('./curriculum-policy.js', collectorUrl).href);
const lines = readFileSync(source, 'utf8').split(/\\r?\\n/);
const jobs = [];
for (let index = Number(start); index < lines.length && jobs.length < Number(count); index++) {
  if (!lines[index]?.trim()) continue;
  const record = JSON.parse(lines[index]);
  const digest = recordDigest(record);
  jobs.push({index, program_id: record.id, digest, generation_hold: generationHoldReason(record) ?? null,
    key: `${String(index).padStart(6, '0')}-${digest.slice(0,16)}`});
}
if (jobs.length !== Number(count)) throw new Error('source range exceeds frozen batch');
process.stdout.write(JSON.stringify(jobs));
"""
        output = subprocess.check_output(['node', '--input-type=module', '-e', code,
            str(runtime / 'dist/teacher/collector.js'), entry['source'],
            str(entry['index']), str(entry.get('count', 1))], text=True)
        entry['_resolved_jobs'] = json.loads(output)
    return entry['_resolved_jobs']


def local_decode_progress():
    """A long batched reply can be decoding before its next durable checkpoint exists."""
    try:
        with urllib.request.urlopen('http://127.0.0.1:8081/metrics', timeout=2) as response:
            for line in response.read().decode().splitlines():
                if line.startswith('llamacpp:n_decode_total '):
                    value = float(line.split()[1])
                    return value if math.isfinite(value) and value >= 0 else None
    except (OSError, ValueError):
        pass
    return None


def partial_metrics(entry):
    """Saved replies show activity. Repeated action shapes across distinct child inputs do not imply a loop."""
    metrics = dict(saved_turns=0, fresh_model_replies=0, completion_tokens=0, repeated_action_sets=0, repeated_request_hashes=0)
    seen = set()
    requests = set()
    def note_request(turn):
        digest = turn.get("request_sha256")
        if digest:
            if digest in requests:
                metrics["repeated_request_hashes"] += 1
            requests.add(digest)
    # Resolve each root independently, taking its latest exact-identity artifact.
    # A fresh partial can supersede a result from an earlier attempt of the same IR.
    for job in resolve_entry_jobs(entry):
        result = Path(entry['jobs']) / f"{job['key']}.result.json"
        partial = Path(entry['jobs']) / f"{job['key']}.partial.json"
        paths = sorted((path for path in (result, partial) if path.exists()),
                       key=lambda path: path.stat().st_mtime_ns, reverse=True)
        for path in paths:
            completed = path.name.endswith('.result.json')
            try:
                row = json.loads(path.read_text())
                program_id = row.get('task', {}).get('program_ir', {}).get('id') if completed else row.get('program_id')
                if program_id != job['program_id'] or row.get('provenance', {}).get('program_ir_sha256') != job['digest']:
                    continue
                turns = row['trajectory' if completed else 'turns']
            except (OSError, ValueError, KeyError):
                continue
            metrics['saved_turns'] += len(turns)
            for turn in turns:
                response = turn.get('model_response' if completed else 'response', {})
                if not (turn.get('raw_response_sha256') if completed else response.get('raw_response')):
                    continue
                note_request(turn)
                metrics['fresh_model_replies'] += 1
                metrics['completion_tokens'] += response.get('completion_tokens', 0) or 0
                action = json.dumps(response.get('calls', []), sort_keys=True)
                if action in seen:
                    metrics['repeated_action_sets'] += 1
                seen.add(action)
            break  # Count this root once, never both its partial and finished result.
    metrics['unique_action_sets'] = len(seen)
    metrics['unique_request_hashes'] = len(requests)
    metrics['repetition_scope'] = 'action shapes across all child calls; not a semantic stall detector'
    return metrics


def retry_deadline(entry):
    """Provider minimum delays survive exhausted retries, timeouts, and supervisor restarts."""
    deadline = 0
    for job in resolve_entry_jobs(entry):
        paths = [Path(entry['jobs']) / f"{job['key']}.retry.json",
                 Path(entry['jobs']) / f"{job['index']:06d}.error.json"]
        for path in paths:
            try:
                row = json.loads(path.read_text())
                if row.get('program_id') != job['program_id']:
                    continue
                value = float(row.get('until', row.get('retry_not_before', 0))) / 1000
                if math.isfinite(value):
                    deadline = max(deadline, value)
            except (OSError, ValueError, KeyError, TypeError):
                continue
    return deadline


def retry_waiting(entry):
    return retry_deadline(entry) > time.time()


def failure_cooldown(streak):
    return min(300, 30 * 2 ** min(max(0, streak - 1), 20) * (0.5 + random.random()))


def run_queue(queue, journal, runtime, seconds=600, model_id='Ternary-Bonsai-2-27B',
              provider=None, model_concurrency=None, execution_plans=False, reasoning_effort='low', min_free_mib=0):
    if model_concurrency is None:
        model_concurrency = 1 if provider else 4
    queue, journal, runtime = map(Path, (queue, journal, runtime))
    entries = [json.loads(line) for line in queue.read_text().splitlines() if line.strip()]
    attempted = set()
    failure_streak, next_allowed_at = 0, 0
    if journal.exists():
        attempted = {json.loads(line)['key'] for line in journal.read_text().splitlines()
                     if line.strip() and json.loads(line).get('event') == 'finish'}
        for line in journal.read_text().splitlines():
            row = json.loads(line)
            if row.get('event') == 'finish' and not row.get('batch_key'):
                failure_streak = 0 if row['status'] in {'complete', 'complete_with_skips', 'skipped'} else failure_streak + 1
                next_allowed_at = row.get('next_allowed_at', 0)
    journal.parent.mkdir(parents=True, exist_ok=True)
    def record(value):
        with journal.open('a') as output:
            output.write(json.dumps(value) + '\n')
            output.flush()
        print(json.dumps(value), flush=True)
    for entry in entries:
        if entry['key'] in attempted:
            continue
        resolve_entry_jobs(entry, runtime)
        if min_free_mib:
            free_mib = shutil.disk_usage(journal.parent).free // (1024 * 1024)
            if free_mib < min_free_mib:
                record(dict(event='storage_pause', key=entry['key'], time=time.time(),
                            free_mib=free_mib, minimum_free_mib=min_free_mib,
                            disposition='stopped_before_case; restart same queue/journal after freeing space'))
                return
        if not provider:
            # An offline server must not consume every remaining queue key.
            # Waiting happens before the attempt starts or its case budget begins.
            while True:
                try:
                    with urllib.request.urlopen('http://127.0.0.1:8081/health', timeout=5) as response:
                        if response.status != 200:
                            raise RuntimeError(f'server health status {response.status}')
                    break
                except Exception as error:
                    record(dict(event='service_wait', key=entry['key'], time=time.time(),
                                retry_seconds=30, error=f'{type(error).__name__}: {error}',
                                disposition='unstarted; waiting for local teacher health'))
                    time.sleep(30)
        if next_allowed_at > time.time():
            record(dict(event='cooldown', key=entry['key'], time=time.time(),
                        failure_streak=failure_streak, until=next_allowed_at))
            while next_allowed_at > time.time():
                time.sleep(min(30, next_allowed_at - time.time()))
        max_turns = entry.get('max_turns', 20)
        count = entry.get('count', 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 5:
            raise ValueError('entry count must be between one and five')
        if count > 1 and provider:
            raise ValueError('provider queues remain single-case; batching is only enabled for local Bonsai')
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError('entry max_turns must be a positive integer')
        command = ['node', '--max-old-space-size=3000', str(runtime / 'dist/teacher/cli.js'),
                   entry['source'], entry['jobs'], entry['output'], '--start', str(entry['index']), '--limit', str(count),
                   '--model-id', model_id,
                   '--root-seed', str(entry['seed']), '--workers', str(count), '--max-turns', str(max_turns),
                   '--model-concurrency', str(model_concurrency), '--max-model-requests', str(entry.get('max_model_requests', 128)),
                   '--transport-retries', '1', '--file-tools', entry.get('surface', 'all')]
        command += ['--provider', provider, '--context-tokens', '16384', '--reasoning-effort', reasoning_effort] if provider else [
            '--server', 'http://127.0.0.1:8081', '--kv-tokens', '40000']
        if provider:
            command += ['--retry-delay-ms', '15000']
        if execution_plans:
            command.append('--execution-plans')
        case_seconds = entry.get('case_seconds', seconds)
        if not isinstance(case_seconds, int) or case_seconds < 1:
            raise ValueError('entry case_seconds must be a positive integer')
        start = time.monotonic()
        last_activity, previous = start, partial_metrics(entry)
        previous_decode = None
        record({'event': 'start', 'key': entry['key'], 'time': time.time(), 'budget_seconds': case_seconds,
                'max_turns': max_turns})
        with Path(entry['log']).open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                for tick in range(math.ceil(case_seconds / 30)):
                    remaining = case_seconds - (time.monotonic() - start)
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(command, case_seconds)
                    try:
                        code = child.wait(timeout=min(30, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        current = partial_metrics(entry)
                        decode = local_decode_progress() if not provider else None
                        decoding = decode is not None and previous_decode is not None and decode > previous_decode
                        previous_decode = decode
                        if current['saved_turns'] != previous['saved_turns']:
                            last_activity, previous = time.monotonic(), current
                        elif decoding or (provider and retry_waiting(entry)):
                            last_activity = time.monotonic()
                        elif time.monotonic() - last_activity >= 300:
                            raise TimeoutError('no checkpoint or local decode progress for 300 seconds')
                        record({'event': 'activity', 'key': entry['key'], 'time': time.time(),
                                'elapsed_seconds': round(time.monotonic() - start, 1),
                                'local_decode_total': decode, **partial_metrics(entry)})
                else:
                    raise subprocess.TimeoutExpired(command, case_seconds)
                status = 'complete' if code == 0 else 'incomplete'
            except (subprocess.TimeoutExpired, TimeoutError) as error:
                status, code = ('inactivity_timeout' if isinstance(error, TimeoutError) else 'timeout'), None
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        try:
            accounting = output_accounting(entry, runtime)
        except Exception as error:
            accounting = {'version': OUTPUT_ACCOUNTING_VERSION, 'complete': False,
                          'disposition': 'accounting_error', 'error': f'{type(error).__name__}: {error}'}
        normal_collector_exit = status in {'complete', 'incomplete'} and code in {0, 2}
        if accounting['complete'] and normal_collector_exit:
            if accounting['explicitly_generation_held'] == accounting['expected_jobs']:
                status = 'skipped'
            elif accounting['explicitly_generation_held'] and code in {0, 2}:
                status = 'complete_with_skips'
            elif code == 0:
                status = 'complete'
        elif status == 'complete' and not accounting['complete']:
            # A successful process exit is not proof that the exact job rows reached the merged output.
            status = 'incomplete_export'
        failure_streak = 0 if status in {'complete', 'complete_with_skips', 'skipped'} else failure_streak + 1
        next_allowed_at = max(time.time() + failure_cooldown(failure_streak), retry_deadline(entry)) if failure_streak else 0
        record({'event': 'finish', 'key': entry['key'], 'status': status, 'exit_code': code,
                'failure_streak': failure_streak, 'next_allowed_at': next_allowed_at,
                'elapsed_seconds': round(time.monotonic() - start, 1), 'time': time.time(),
                'output_accounting': accounting, **partial_metrics(entry)})
        # Preserve original attempt keys when roots share one collector batch.
        # A terminal batch event covers both attempts, even when only one produced a result.
        for member in entry.get('members', []):
            record({'event': 'finish', 'key': member['key'], 'status': status, 'exit_code': code,
                    'batch_key': entry['key'], 'time': time.time()})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue')
    parser.add_argument('journal')
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--case-seconds', type=int, default=600)
    parser.add_argument('--model-id', default='Ternary-Bonsai-2-27B')
    parser.add_argument('--provider')
    parser.add_argument('--model-concurrency', type=int, help='global request cap, including children (local: 4; provider: 1)')
    parser.add_argument('--min-free-mib', type=int, default=0, help='stop before a case if filesystem free space falls below this floor; zero disables')
    parser.add_argument('--execution-plans', action='store_true')
    parser.add_argument('--reasoning-effort', default='low')
    args = parser.parse_args()
    if args.case_seconds < 1:
        parser.error('--case-seconds must be positive')
    if args.min_free_mib < 0:
        parser.error('--min-free-mib must not be negative')
    if args.model_concurrency is not None and args.model_concurrency < 1:
        parser.error('--model-concurrency must be positive')
    def stop(signum, frame):
        raise KeyboardInterrupt('queue stopped')
    signal.signal(signal.SIGTERM, stop)
    run_queue(args.queue, args.journal, args.runtime, args.case_seconds, args.model_id, args.provider,
              args.model_concurrency, args.execution_plans, args.reasoning_effort, args.min_free_mib)
