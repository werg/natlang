#!/usr/bin/env python3
"""Run a frozen, interleaved teacher queue with a hard per-case wall-clock budget.

Timeouts and failures remain explicit in the append-only journal and are not training results.
Restart skips attempted entries; use a new journal for an explicitly reviewed retry pass.
"""
import argparse
import hashlib
import json
import os
import subprocess
import signal
import time
import math
import random
import shutil
import tempfile
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256_file  # noqa: E402


OUTPUT_ACCOUNTING_VERSION = 'natlang.supervisor_output_accounting/2'
OBSERVATION_POLICY_VERSION = 'natlang.collection_observations/2'
REPETITION_GUARD_POLL_SECONDS = 1.0
ACTIVITY_HEARTBEAT_SECONDS = 30.0


class ProviderStreamOverflow(RuntimeError):
    """A tool-call stream outgrew the action's output ceiling without a reply."""


def jsonl_lines(path):
    # Only LF delimits JSONL. str.splitlines() also splits valid JSON strings
    # containing U+2028, which occurs in several curriculum source datasets.
    with Path(path).open(encoding='utf-8') as stream:
        yield from stream


class ProviderObservations:
    """Read only this child's appended, content-free provider events. Log writes are not progress."""

    def __init__(self, path):
        self.path = Path(path)
        info = self.path.stat() if self.path.exists() else None
        self.offset = info.st_size if info else 0
        self.identity = (info.st_dev, info.st_ino) if info else None
        self.pending = b''
        self.counts = {}
        self.requests = {}
        self.latest_delta_at = None
        self.max_toolcall_delta_bytes = 0
        self.invalid_events = 0

    def poll(self):
        delta = False
        try:
            with self.path.open('rb') as source:
                info = os.fstat(source.fileno())
                identity = (info.st_dev, info.st_ino)
                if ((self.identity is not None and identity != self.identity)
                        or info.st_size < self.offset):
                    # Rotation/truncation cannot turn old lines into current-child evidence.
                    self.offset = info.st_size
                    self.pending = b''
                    self.invalid_events += 1
                self.identity = identity
                source.seek(self.offset)
                payload = source.read(512 * 1024)
                self.offset = source.tell()
        except OSError:
            return False, {'read_error': True, 'pending_requests': list(self.requests.values())}
        lines = (self.pending + payload).split(b'\n')
        self.pending = lines.pop()
        if len(self.pending) > 65536:
            self.pending = b''
            self.invalid_events += 1
        for line in lines:
            if len(line) > 65536:
                continue
            try:
                event = json.loads(line)
            except (ValueError, UnicodeError):
                continue
            try:
                if not isinstance(event, dict) or event.get('event') not in {
                        'provider_request_phase', 'provider_stream_progress'}:
                    continue
                if event['event'] == 'provider_request_phase':
                    phase = event.get('phase')
                    # Provider cleanup is intentionally outside request-progress tracking and its
                    # records omit `role`. Ignore this one known phase before identity validation.
                    if phase == 'provider_close':
                        continue
                    if phase not in {'provider_prepare', 'provider_turn', 'provider_action_cycle'}:
                        raise ValueError('invalid provider observation phase')
                role, provider = event.get('role'), event.get('provider')
                ordinal = event.get('request_ordinal')
                if (role not in {'teacher', 'judge'} or not isinstance(provider, str)
                        or len(provider) > 80 or (ordinal is not None and
                        (type(ordinal) is not int or ordinal < 1))):
                    raise ValueError('invalid provider observation identity')
                if event['event'] == 'provider_request_phase':
                    if phase == 'provider_action_cycle':
                        continue
                    key = (role, provider, ordinal, phase)
                    if event.get('status') == 'started':
                        if len(self.requests) >= 1024 and key not in self.requests:
                            raise ValueError('too many pending requests')
                        self.requests[key] = {'role': role, 'provider': provider,
                                              'request_ordinal': ordinal, 'phase': phase}
                    elif event.get('status') in {'completed', 'failed', 'deadline_exceeded',
                                                'action_cycle_deadline_exceeded', 'collection_aborted'}:
                        self.requests.pop(key, None)
                    continue
                events, size = event.get('deltaEvents'), event.get('deltaBytes')
                toolcall_bytes = event.get('toolCallDeltaBytes', 0)
                if (event.get('status') not in {'progress', 'completed', 'failed'}
                        or type(events) is not int or type(size) is not int or events < 0 or size < 0
                        or type(toolcall_bytes) is not int or not 0 <= toolcall_bytes <= size):
                    raise ValueError('invalid delta counters')
                started = event.get('startedAt')
                observed = datetime.fromisoformat(event['observedAt'].replace('Z', '+00:00'))
                began = datetime.fromisoformat(started.replace('Z', '+00:00'))
                if (observed.utcoffset() is None or began.utcoffset() is None
                        or observed < began or observed.timestamp() > time.time() + 5):
                    raise ValueError('invalid observation timestamp')
                key = (role, provider, ordinal, started)
                previous_events, previous_size = self.counts.get(key, (0, 0))
                if events < previous_events or size < previous_size:
                    raise ValueError('delta counters regressed')
                if len(self.counts) >= 4096 and key not in self.counts:
                    raise ValueError('too many observed requests')
                self.counts[key] = (events, size)
                self.max_toolcall_delta_bytes = max(self.max_toolcall_delta_bytes, toolcall_bytes)
                if events > previous_events and size > previous_size:
                    self.latest_delta_at = max(self.latest_delta_at or 0, observed.timestamp())
                    delta = True
            except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
                self.invalid_events += 1
        return delta, {'pending_requests': list(self.requests.values()),
                       'latest_delta_at': self.latest_delta_at,
                       'max_toolcall_delta_bytes': self.max_toolcall_delta_bytes,
                       'invalid_events': self.invalid_events}


def read_json(path, *, with_sha=False):
    try:
        payload = Path(path).read_bytes()
        value = json.loads(payload)
        return (value,hashlib.sha256(payload).hexdigest()) if with_sha else value
    except (OSError, ValueError):
        return (None,None) if with_sha else None


def canonical_json_sha256(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def same_resolved_path(left, right):
    """Compare path identity after resolution without relaxing any content hash checks."""
    try:
        return Path(left).expanduser().resolve(strict=True) == Path(right).expanduser().resolve(strict=True)
    except (OSError, RuntimeError, TypeError, ValueError):
        return False


def frozen_record_digests(records, runtime=None):
    runtime = Path(runtime) if runtime else Path(__file__).resolve().parents[1] / 'ts-host'
    code = """
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const [collector, inputPath] = process.argv.slice(1);
const { recordDigest } = await import(pathToFileURL(collector).href);
const records = JSON.parse(readFileSync(inputPath, 'utf8'));
process.stdout.write(JSON.stringify(records.map(record => recordDigest(record))));
"""
    # Imported modules can make stdin nonblocking; large piped JSON then fails EAGAIN.
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as payload:
        json.dump(records, payload)
        payload.flush()
        output = subprocess.check_output(['node', '--input-type=module', '-e', code,
            str(runtime / 'dist/teacher/collector.js'), payload.name], text=True)
    return json.loads(output)


def output_accounting(entry, runtime=None):
    """Validate exact collector rows and batch manifest before a queue key can be called complete."""
    jobs = resolve_entry_jobs(entry, runtime)
    root = Path(entry['jobs'])
    source_sha256 = sha256_file(entry['source'])
    states, result_ids, result_digests, saved_result_rows = [], [], [], []
    candidate_result_rows = []
    for job in jobs:
        resource_limit_reason = None
        terminal_error = None
        unverified_error = None
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
            error,error_sha256 = read_json(error_path,with_sha=True)
            if error and error.get('index') == job['index'] and error.get('program_id') == job['program_id'] and \
                    error.get('generation_hold') and error.get('generation_hold') == job.get('generation_hold'):
                state = 'generation_held'
            elif error and error.get('index') == job['index'] and error.get('program_id') == job['program_id'] and error.get('generation_hold'):
                state = 'invalid_generation_hold'
            elif error and error.get('index') == job['index'] and error.get('program_id') == job['program_id'] and (
                    error.get('program_ir_sha256') != job['digest'] or error.get('source_sha256') != source_sha256):
                state = 'unverified_terminal_error'
                unverified_error = {'path':str(error_path),'sha256':error_sha256,
                    'reason':'terminal error lacks matching immutable record/source digests'}
            elif error and error.get('index') == job['index'] and error.get('program_id') == job['program_id']:
                if error.get('code') == 'NATLANG_MODEL_REQUEST_BUDGET':
                    resource_limit_reason = 'model_request_budget'
                terminal_error = {'error': str(error.get('error', '')), 'path': str(error_path),
                                  'sha256': error_sha256, 'program_ir_sha256':error['program_ir_sha256'],
                                  'source_sha256':error['source_sha256'], 'record_binding_verified':False}
                message = terminal_error['error'].lower()
                state = 'transport_failed' if message.strip() == 'typeerror: terminated' or any(term in message for term in
                    ('transport', 'rate limit', 'timed out', 'timeout', 'socket', 'connection', 'fetch failed')) else 'failed'
            elif partial_path.is_file():
                partial = read_json(partial_path)
                state = 'partial_without_terminal_result' if partial and \
                    partial.get('program_id') == job['program_id'] and \
                    partial.get('provenance', {}).get('program_ir_sha256') == job['digest'] else 'invalid_partial'
            else:
                state = 'missing_terminal_result'
        states.append({'index': job['index'], 'program_id': job['program_id'],
                       'digest': job['digest'], 'state': state,
                       **({'resource_limit_reason': resource_limit_reason} if resource_limit_reason else {}),
                       **({'terminal_error': terminal_error} if terminal_error else {}),
                       **({'unverified_error':unverified_error} if unverified_error else {})})

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
        if embedded_digest_error is not None:
            state['state'] = 'accounting_error'
            continue
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
    manifest_errors = []
    if not manifest:
        manifest_errors.append('manifest_missing_or_unreadable')
    else:
        checks = [
            ('manifest_version_mismatch', manifest.get('version') == 'natlang.teacher_batch.native/1'),
            ('manifest_range_mismatch', manifest.get('range') == {'start': entry['index'], 'count': len(jobs)}),
            ('manifest_completed_mismatch', manifest.get('completed') == len(result_ids)),
            ('manifest_missing_indices_mismatch', manifest.get('missing') ==
             [state['index'] for state in states if state['state'] != 'result']),
            ('manifest_output_sha256_mismatch', manifest.get('output_sha256') == output_hash),
            ('manifest_source_path_mismatch', same_resolved_path(manifest.get('source'), entry['source'])),
            ('manifest_source_sha256_mismatch', manifest.get('source_sha256') == source_sha256),
            ('output_parse_error', not output_error),
            ('manifest_output_rows_mismatch', output_row_hashes == saved_row_hashes and not output_row_errors),
        ]
        manifest_errors.extend(reason for reason, passed in checks if not passed)
    manifest_ok = not manifest_errors
    for state in states:
        if state.get('terminal_error'):
            state['terminal_error']['record_binding_verified'] = manifest_ok
    all_accounted = all(state['state'] in {'result', 'generation_held'} for state in states)
    any_held = any(state['state'] == 'generation_held' for state in states)
    unresolved = [state['state'] for state in states if state['state'] not in {'result', 'generation_held'}]
    if not manifest_ok or not all_accounted:
        disposition = next((value for value in ('transport_failed', 'partial_without_terminal_result',
            'accounting_error', 'invalid_result_ir_digest', 'invalid_generation_hold', 'invalid_result', 'invalid_partial',
            'unverified_terminal_error', 'failed', 'missing_terminal_result') if value in unresolved),
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
            'resource_limit_reason': next((state['resource_limit_reason'] for state in states
                                          if state.get('resource_limit_reason')), None),
            'expected_jobs': len(jobs), 'exact_result_rows': len(result_ids),
            'explicitly_generation_held': sum(state['state'] == 'generation_held' for state in states),
            'job_states': states, 'output_exists': output_exists, 'output_sha256': output_hash,
            'output_rows': len(output_rows), 'output_error': output_error, 'output_row_errors': output_row_errors,
            'saved_row_hashes': saved_row_hashes, 'output_row_hashes': output_row_hashes,
            'embedded_digest_error': embedded_digest_error,
            'manifest_present': manifest is not None, 'manifest_valid': manifest_ok, 'source_sha256':source_sha256,
            'manifest_errors': manifest_errors,
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


def local_decode_progress(server=None):
    """A long batched reply can be decoding before its next durable checkpoint exists."""
    endpoint = urllib.parse.urlsplit(server or 'http://127.0.0.1:8081')
    if endpoint.scheme not in {'http', 'https'} or endpoint.hostname not in {'localhost', '127.0.0.1', '::1'}:
        return None
    metrics_url = urllib.parse.urlunsplit((endpoint.scheme, endpoint.netloc, '/metrics', '', ''))
    try:
        with urllib.request.urlopen(metrics_url, timeout=2) as response:
            for line in response.read().decode().splitlines():
                if line.startswith(('llamacpp:n_decode_total ', 'natlang_native_decode_tokens_total ')):
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
                provenance = (turn.get('action_provenance') or
                              (response.get('raw_response') or {}).get('natlang_action_provenance') or {})
                if provenance.get('sampled') is False or provenance.get('kind') == 'authored_reference_root_eval':
                    continue
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


class IdenticalTurnLimit(Exception):
    """The same action saw the same observation repeatedly in one invocation."""


def _completed_no_action_truncation(turn):
    response = turn.get('response') or {}
    raw = response.get('raw_response') or {}
    provenance = raw.get('natlang_action_provenance') or {}
    choices = raw.get('choices') or []
    tokens = response.get('completion_tokens')
    return (response.get('truncated') is True and not response.get('calls')
            and type(tokens) is int and tokens > 0 and isinstance(choices, list) and bool(choices)
            and choices[0].get('finish_reason') == 'length'
            and provenance.get('sampled') is not False
            and provenance.get('kind') != 'authored_reference_root_eval'
            and isinstance(turn.get('request_sha256'), str) and len(turn['request_sha256']) == 64
            and all(char in '0123456789abcdef' for char in turn['request_sha256']))


def identical_turn_streak(entry, limit):
    """Verify configured no-progress limits from authentic completed partials.

    The original exact action/observation law stays intact. Completed truncated
    replies with no executable action also consume the declared bound. Real
    successful execution receipts reset that count; emitted proposals cannot.
    Legacy partials prove only a contiguous same-invocation no-action suffix.
    """
    if not limit or entry.get('count', 1) != 1:
        return None
    for job in resolve_entry_jobs(entry):
        path = Path(entry['jobs']) / f"{job['key']}.partial.json"
        try:
            partial = json.loads(path.read_text())
            turns = partial.get('turns', [])
        except (OSError, ValueError):
            continue
        if (partial.get('program_id') != job['program_id']
                or partial.get('provenance', {}).get('program_ir_sha256') != job['digest']
                or len(turns) < limit):
            continue
        latest = turns[-1].get('invocation_id')
        if latest and _completed_no_action_truncation(turns[-1]):
            scoped = [turn for turn in turns if turn.get('invocation_id') == latest]
            count, successful, sequence = 0, 0, None
            requests = []
            if all(isinstance(turn.get('execution_progress'), dict) for turn in scoped):
                for turn in scoped:
                    progress = turn['execution_progress']
                    current, seq = progress.get('successful_actions'), progress.get('last_success_seq')
                    if (progress.get('schema') != 'natlang.completed-action-progress/1'
                            or progress.get('invocation_id') != latest or type(current) is not int
                            or current < successful or (current == 0 and seq is not None)
                            or (current > 0 and (type(seq) is not int or seq < 0))
                            or (current == successful and seq != sequence)
                            or (current > successful and sequence is not None and seq <= sequence)):
                        raise ValueError('collector completed-action progress receipt is invalid')
                    if current > successful:
                        count, requests = 0, []
                    successful, sequence = current, seq
                    if _completed_no_action_truncation(turn):
                        count += 1; requests.append(turn['request_sha256'])
            else:
                # Unknown execution outcomes cannot be relabeled successful.
                # A contiguous all-no-call suffix itself proves no action ran.
                for turn in reversed(turns):
                    if turn.get('invocation_id') != latest or not _completed_no_action_truncation(turn):
                        break
                    count += 1; requests.append(turn['request_sha256'])
                requests.reverse()
            if count >= limit:
                return {'saved_turns': len(turns), 'invocation_id': latest,
                        'reason': 'consecutive_truncated_no_action', 'limit': limit,
                        'completed_no_action_replies': count, 'successful_actions': successful,
                        'request_sha256': requests}
        signatures = []
        for turn in turns[-limit:]:
            calls = (turn.get('response') or {}).get('calls')
            observed = (turn.get('last_tool_observation') or {}).get('content_sha256')
            if not calls or not observed or not turn.get('invocation_id'):
                break
            signatures.append((turn['invocation_id'], json.dumps(calls, sort_keys=True), observed))
        if len(signatures) == limit and len(set(signatures)) == 1:
            return {'saved_turns': len(turns), 'invocation_id': signatures[0][0],
                    'action_sha256': hashlib.sha256(signatures[0][1].encode()).hexdigest(),
                    'observation_sha256': signatures[0][2]}
    return None


def partial_progress_signature(entry, jobs=None):
    """Snapshot durable per-job partial identities without reading their contents."""
    jobs = resolve_entry_jobs(entry) if jobs is None else jobs
    root = Path(entry['jobs'])
    rows = []
    for job in jobs:
        path = root / f"{job['key']}.partial.json"
        try:
            stat = path.stat()
        except FileNotFoundError:
            rows.append((job['key'], None))
        else:
            rows.append((job['key'], stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    return tuple(rows)


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
              provider=None, model_concurrency=None, execution_plans=False, reasoning_effort='low', min_free_mib=0,
              no_observation_seconds=0, max_batch_cases=5, provider_request_config=None,
              server=None, chat_completions_url=None, api_key_env=None, chat_request_config=None,
              identical_turn_limit=0, neuralese_server=False, drain_file=None, teacher_identity=None):
    if provider and (server or chat_completions_url or api_key_env):
        raise ValueError('Pi provider and external HTTP controls are mutually exclusive')
    if (chat_completions_url or api_key_env) and not server:
        raise ValueError('external HTTP controls require an explicit server')
    if neuralese_server and (not server or provider):
        raise ValueError('Neuralese server collection requires an explicit external server')
    external = bool(server)
    remote = bool(provider or external)
    if chat_request_config is not None:
        if not external:
            raise ValueError('chat request config requires an explicit external server')
        chat_request_config = Path(chat_request_config).resolve()
        if not isinstance(json.loads(chat_request_config.read_text()), dict):
            raise ValueError('chat request config must be a JSON object')
    if model_concurrency is None:
        model_concurrency = 1 if remote else 4
    queue, journal, runtime = map(Path, (queue, journal, runtime))
    if provider_request_config is not None:
        if not provider:
            raise ValueError('provider request config requires an explicit provider')
        provider_request_config = Path(provider_request_config).resolve()
        controls = json.loads(provider_request_config.read_text())
        if not isinstance(controls, dict):
            raise ValueError('provider request config must be a JSON object')
    entries = [json.loads(line) for line in jsonl_lines(queue) if line.strip()]
    # Validate every batch before starting a collector, including later queue entries.
    for entry in entries:
        count = entry.get('count', 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= max_batch_cases:
            raise ValueError(f'entry count exceeds reviewed batch ceiling: {entry.get("key")}')
        if count > 1 and remote:
            raise ValueError('provider queues remain single-case')
        if 'request_retries' in entry:
            retries = entry['request_retries']
            if type(retries) is not int or retries < 0:
                raise ValueError(f'entry request_retries must be a nonnegative integer: {entry.get("key")}')
        if 'system_file' in entry and (not isinstance(entry['system_file'], str) or not Path(entry['system_file']).is_file()):
            raise ValueError(f'entry system_file must name an existing file: {entry.get("key")}')
        if 'temperature' in entry:
            temperature = entry['temperature']
            if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
                raise ValueError(f'entry temperature must be a number in [0, 2]: {entry.get("key")}')
        if 'action_max_tokens' in entry:
            action_max_tokens = entry['action_max_tokens']
            if type(action_max_tokens) is not int or action_max_tokens < 1:
                raise ValueError(f'entry action_max_tokens must be a positive integer: {entry.get("key")}')
        if 'context_tokens' in entry:
            context_tokens = entry['context_tokens']
            if type(context_tokens) is not int or context_tokens < 1:
                raise ValueError(f'entry context_tokens must be a positive integer: {entry.get("key")}')
            if not remote:
                raise ValueError('entry context_tokens is only supported by provider queues')
    attempted = set()
    failure_streak, next_allowed_at = 0, 0
    if journal.exists():
        attempted = {json.loads(line)['key'] for line in jsonl_lines(journal)
                     if line.strip() and json.loads(line).get('event') == 'finish'}
        for line in jsonl_lines(journal):
            row = json.loads(line)
            if row.get('event') == 'finish' and not row.get('batch_key'):
                failure_streak = 0 if (row['status'] in {'complete', 'complete_with_skips', 'skipped', 'repetition', 'model_incomplete'} or
                                       row.get('resource_limit_reason') == 'model_request_budget') else failure_streak + 1
                next_allowed_at = row.get('next_allowed_at', 0)
    journal.parent.mkdir(parents=True, exist_ok=True)
    def record(value, *, durable=False):
        with journal.open('a') as output:
            output.write(json.dumps(value) + '\n')
            output.flush()
            if durable: os.fsync(output.fileno())
        print(json.dumps(value), flush=True)
    for entry in entries:
        if entry['key'] in attempted:
            continue
        if drain_file is not None and Path(drain_file).exists():
            record({'event':'operator_drain','time':time.time(),'next_key':entry['key'],
                'queue':str(queue),'queue_sha256':hashlib.sha256(queue.read_bytes()).hexdigest(),
                'completed_keys':sorted(attempted),
                'unstarted_keys':[row['key'] for row in entries if row['key'] not in attempted],
                'disposition':'active case finished; no next request; remove drain file to resume exact remaining keys'}, durable=True)
            return 75
        entry_jobs = resolve_entry_jobs(entry, runtime)
        if min_free_mib:
            free_mib = shutil.disk_usage(journal.parent).free // (1024 * 1024)
            if free_mib < min_free_mib:
                record(dict(event='storage_pause', key=entry['key'], time=time.time(),
                            free_mib=free_mib, minimum_free_mib=min_free_mib,
                            disposition='stopped_before_case; restart same queue/journal after freeing space'))
                return 75
        if not remote:
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
        transport_retries = entry.get('transport_retries', 1)
        if type(transport_retries) is not int or transport_retries < 0:
            raise ValueError(f'entry transport_retries must be a nonnegative integer: {entry.get("key")}')
        count = entry.get('count', 1)
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= max_batch_cases:
            raise ValueError(f'entry count must be between one and {max_batch_cases}')
        if count > 1 and remote:
            raise ValueError('provider queues remain single-case; batching is only enabled for local Bonsai')
        if isinstance(max_turns, bool) or not isinstance(max_turns, int) or max_turns < 1:
            raise ValueError('entry max_turns must be a positive integer')
        command = ['node', '--max-old-space-size=3000', str(runtime / 'dist/teacher/cli.js'),
                   entry['source'], entry['jobs'], entry['output'], '--start', str(entry['index']), '--limit', str(count),
                   '--model-id', model_id,
                   '--root-seed', str(entry['seed']), '--workers', str(count), '--max-turns', str(max_turns),
                   '--model-concurrency', str(model_concurrency), '--max-model-requests', str(entry.get('max_model_requests', 128)),
                   '--transport-retries', str(transport_retries), '--file-tools', entry.get('surface', 'all')]
        if 'prompt_rendering' in entry:
            if entry['prompt_rendering'] not in ('full', 'program-header'):
                raise ValueError('entry prompt_rendering must be full or program-header')
            command += ['--prompt-rendering', entry['prompt_rendering']]
        if 'request_retries' in entry:
            command += ['--request-retries', str(entry['request_retries'])]
        if 'action_max_tokens' in entry:
            command += ['--action-max-tokens', str(entry['action_max_tokens'])]
        if 'system_file' in entry:
            # The run's system prompt (the collector's --system-file), e.g. a memetic individual's guidance.
            command += ['--system-file', entry['system_file']]
        if 'temperature' in entry:
            # Sampled rollouts (S7 GRPO groups); absent means the collector's greedy default.
            command += ['--temperature', str(entry['temperature'])]
        command += ['--provider', provider, '--context-tokens', str(entry.get('context_tokens', 16384)), '--reasoning-effort', reasoning_effort] if provider else [
            '--server', server or 'http://127.0.0.1:8081']
        if external:
            if neuralese_server:
                command += ['--neuralese-server']
            command += ['--context-tokens', str(entry.get('context_tokens', 16384)), '--kv-tokens', '0']
            if chat_completions_url:
                command += ['--chat-completions-url', chat_completions_url]
            if api_key_env:
                command += ['--api-key-env', api_key_env]
            if chat_request_config is not None:
                command += ['--chat-request-config', str(chat_request_config)]
        elif not provider:
            command += ['--kv-tokens', '40000']
        if remote:
            command += ['--retry-delay-ms', '15000']
            if provider_request_config is not None:
                command += ['--provider-request-config', str(provider_request_config)]
        if execution_plans:
            command.append('--execution-plans')
        text_neuralese_emulation = entry.get('text_neuralese_emulation', False)
        if not isinstance(text_neuralese_emulation, bool):
            raise ValueError('entry text_neuralese_emulation must be a boolean')
        if text_neuralese_emulation:
            command.append('--text-neuralese-emulation')
        case_seconds = entry.get('case_seconds', seconds)
        if not isinstance(case_seconds, int) or case_seconds < 1:
            raise ValueError('entry case_seconds must be a positive integer')
        # Fresh reviewed queues need no manually precreated collector directories.
        # Prepare them before recording an attempt or opening its log.
        Path(entry['jobs']).mkdir(parents=True, exist_ok=True)
        Path(entry['output']).parent.mkdir(parents=True, exist_ok=True)
        Path(entry['log']).parent.mkdir(parents=True, exist_ok=True)
        start = time.monotonic()
        last_activity, previous = start, partial_metrics(entry)
        previous_poll, unobserved_seconds, was_waiting = start, 0, False
        last_activity_heartbeat = start
        repetition_poll_seconds = (REPETITION_GUARD_POLL_SECONDS if identical_turn_limit
                                   else ACTIVITY_HEARTBEAT_SECONDS)
        # Force one initial read at the first supervision tick. A partial may
        # already contain the repeated window when the child starts (or before
        # this supervisor attaches), and baseline-only stat comparison would
        # otherwise suppress that evidence indefinitely.
        last_partial_signature = None
        observer = ProviderObservations(entry['log']) if provider else None
        observation = {'pending_requests': []}
        observations = []
        limit_reason = None
        limit_evidence = None
        previous_decode = None
        teacher_proof = None
        if teacher_identity is not None:
            try:
                from scripts.coreweave.teacher_generation_guardian import verify_teacher_identity
            except ModuleNotFoundError:
                from teacher_generation_guardian import verify_teacher_identity
            teacher_proof = verify_teacher_identity(teacher_identity,server=server,model_id=model_id)
        record({'event': 'start', 'key': entry['key'], 'time': time.time(), 'budget_seconds': case_seconds,
                'max_turns': max_turns, 'observation_policy': OBSERVATION_POLICY_VERSION,
                'no_observation_seconds': no_observation_seconds,
                'budget_semantics': 'collection_resource_limit; not a language deadline or model-negative label',
                **({'teacher_identity':teacher_proof} if teacher_proof is not None else {})})
        with Path(entry['log']).open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                for tick in range(math.ceil(case_seconds / repetition_poll_seconds)):
                    remaining = case_seconds - (time.monotonic() - start)
                    if remaining <= 0:
                        limit_reason = 'case_wall_budget'
                        raise subprocess.TimeoutExpired(command, case_seconds)
                    try:
                        code = child.wait(timeout=min(repetition_poll_seconds, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        now = time.monotonic()
                        partial_changed = False
                        if identical_turn_limit:
                            current_partial_signature = partial_progress_signature(entry, entry_jobs)
                            partial_changed = (last_partial_signature is None or
                                               current_partial_signature != last_partial_signature)
                            last_partial_signature = current_partial_signature
                            current = partial_metrics(entry) if partial_changed else previous
                        else:
                            current = partial_metrics(entry)
                        heartbeat = (not identical_turn_limit or
                                     now - last_activity_heartbeat >= ACTIVITY_HEARTBEAT_SECONDS)
                        repeated = (identical_turn_streak(entry, identical_turn_limit)
                                    if identical_turn_limit and partial_changed else None)
                        if repeated:
                            limit_reason = repeated.get('reason', 'identical_turn_limit')
                            limit_evidence = repeated
                            record({'event': 'repetition_detected', 'key': entry['key'], 'time': time.time(),
                                    'limit': identical_turn_limit, **repeated})
                            raise IdenticalTurnLimit('same action and observation repeated without progress')
                        if not partial_changed and not heartbeat:
                            continue
                        if heartbeat:
                            last_activity_heartbeat = now
                        decode = (local_decode_progress(server) if neuralese_server else
                                  local_decode_progress() if not remote else None)
                        decoding = decode is not None and previous_decode is not None and decode > previous_decode
                        previous_decode = decode
                        stream_delta, observation = observer.poll() if observer else (False, {'pending_requests': []})
                        # The provider's token ceiling should bound an action's streamed
                        # arguments. A growing tool-call stream far beyond that ceiling,
                        # without any completed turn, is a measured collection failure.
                        # Ordinary long reasoning and silent rate-limit waits do not trip it.
                        action_tokens = int(entry.get('action_max_tokens') or 4096)
                        toolcall_stream_limit = max(131_072, 32 * action_tokens)
                        if observation.get('max_toolcall_delta_bytes', 0) > toolcall_stream_limit:
                            limit_reason = 'provider_tool_call_stream_overflow'
                            raise ProviderStreamOverflow(
                                f'tool-call stream exceeded {toolcall_stream_limit} bytes '
                                f'for {action_tokens} requested output tokens')
                        waiting = bool(remote and retry_waiting(entry))
                        observations = []
                        if current['saved_turns'] > previous['saved_turns']:
                            observations.append('durable_checkpoint')
                        if decoding:
                            observations.append('local_decode_delta')
                        if stream_delta:
                            observations.append('provider_stream_delta')
                        previous = current
                        if observations:
                            # A delayed log read must not present an old delta as current output.
                            age = max(0, time.time() - observation['latest_delta_at']) if observations == ['provider_stream_delta'] else 0
                            last_activity = max(last_activity, now - age)
                            unobserved_seconds = max(0, now - last_activity)
                        elif not waiting and not was_waiting:
                            unobserved_seconds += now - previous_poll
                        previous_poll, was_waiting = now, waiting
                        state = 'known_retry_wait' if waiting else (
                            'observed_output' if observations else 'pending_no_observation')
                        if (no_observation_seconds and not waiting
                                and unobserved_seconds >= no_observation_seconds):
                            limit_reason = 'no_observation_limit'
                            raise TimeoutError('configured collection no-observation limit elapsed; stuckness unknown')
                        record({'event': 'activity', 'key': entry['key'], 'time': time.time(),
                                'elapsed_seconds': round(time.monotonic() - start, 1),
                                'observation_policy': OBSERVATION_POLICY_VERSION,
                                'observation_state': state, 'observations': observations,
                                'seconds_since_observed_output': round(now - last_activity, 1),
                                'no_observation_budget_used_seconds': round(unobserved_seconds, 1),
                                'provider_observation': observation,
                                'local_decode_total': decode, **current})
                else:
                    limit_reason = 'case_wall_budget'
                    raise subprocess.TimeoutExpired(command, case_seconds)
                status = 'complete' if code == 0 else 'incomplete'
            except (subprocess.TimeoutExpired, TimeoutError, IdenticalTurnLimit, ProviderStreamOverflow) as error:
                status, code = (('model_incomplete' if limit_reason == 'consecutive_truncated_no_action' else 'repetition') if isinstance(error, IdenticalTurnLimit) else
                                'provider_stream_overflow' if isinstance(error, ProviderStreamOverflow) else
                                'no_observation_limit' if isinstance(error, TimeoutError) else 'timeout'), None
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
            if observer:
                _, observation = observer.poll()
            accounting = output_accounting(entry, runtime)
        except Exception as error:
            accounting = {'version': OUTPUT_ACCOUNTING_VERSION, 'complete': False,
                          'disposition': 'accounting_error', 'error': f'{type(error).__name__}: {error}'}
        normal_collector_exit = status in {'complete', 'incomplete'} and code in {0, 2}
        limit_reason = limit_reason or accounting.get('resource_limit_reason')
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
        # Deterministic task limits are not upstream-provider failures; a
        # cooldown before an unrelated next item only wastes the queue slot.
        failure_streak = 0 if (status in {'complete', 'complete_with_skips', 'skipped', 'repetition', 'model_incomplete'} or
                               limit_reason == 'model_request_budget') else failure_streak + 1
        next_allowed_at = max(time.time() + failure_cooldown(failure_streak), retry_deadline(entry)) if failure_streak else 0
        record({'event': 'finish', 'key': entry['key'], 'status': status, 'exit_code': code,
                'observation_policy': OBSERVATION_POLICY_VERSION,
                **({'model_incomplete_evidence': limit_evidence} if limit_reason == 'consecutive_truncated_no_action' else {}),
                'resource_limit_reason': limit_reason, 'stuckness': 'unknown' if limit_reason else None,
                'provider_observation': observation,
                'failure_streak': failure_streak, 'next_allowed_at': next_allowed_at,
                'elapsed_seconds': round(time.monotonic() - start, 1), 'time': time.time(),
                'output_accounting': accounting, **partial_metrics(entry)})
        attempted.add(entry['key'])
        # Preserve original attempt keys when roots share one collector batch.
        # A terminal batch event covers both attempts, even when only one produced a result.
        for member in entry.get('members', []):
            record({'event': 'finish', 'key': member['key'], 'status': status, 'exit_code': code,
                    'batch_key': entry['key'], 'time': time.time()})
    # Exhausting the attempt list is not successful completion. Preserve failed
    # attempts and their cooldowns, but let callers distinguish them from a
    # queue whose exact outputs (or explicit source-review skips) are closed.
    latest = {}
    if journal.exists():
        for line in jsonl_lines(journal):
            if line.strip():
                event = json.loads(line)
                if event.get('event') == 'finish':
                    latest[event['key']] = event
    closed = {'complete', 'complete_with_skips', 'skipped'}
    return 0 if all(latest.get(entry['key'], {}).get('status') in closed
                    for entry in entries) else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue')
    parser.add_argument('journal')
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--case-seconds', type=int, default=600)
    parser.add_argument('--model-id', default='Ternary-Bonsai-2-27B')
    parser.add_argument('--provider')
    parser.add_argument('--server', help='explicit external HTTP server; skips local Bonsai health polling')
    parser.add_argument('--chat-completions-url', help='exact external completion URL, including its API path')
    parser.add_argument('--api-key-env', help='environment variable name only; never pass a key value')
    parser.add_argument('--neuralese-server', action='store_true', help='use native value-channel server transport')
    parser.add_argument('--chat-request-config', type=Path, help='external HTTP sampling controls passed unchanged to the collector')
    parser.add_argument('--provider-request-config', type=Path, help='explicit provider request controls passed unchanged to the collector')
    parser.add_argument('--model-concurrency', type=int, help='global request cap, including children (local: 4; provider: 1)')
    parser.add_argument('--min-free-mib', type=int, default=0, help='stop before a case if filesystem free space falls below this floor; zero disables')
    parser.add_argument('--no-observation-seconds', type=int, default=0,
                        help='optional collection silence ceiling; zero (default) disables; never diagnoses stuckness; excludes explicit retry waits')
    parser.add_argument('--identical-turn-limit', type=int, default=0,
                        help='stop after this many exact action/observation repeats or completed truncated no-action continuations in one invocation; zero disables')
    parser.add_argument('--max-batch-cases', type=int, default=5, help='explicit local queue batch ceiling; request concurrency remains separate')
    parser.add_argument('--teacher-identity', type=Path, help='verify immutable actual teacher actor and engine process identity before each case')
    parser.add_argument('--drain-file', type=Path, help='finish the active case, then stop before another case with a durable unstarted frontier')
    parser.add_argument('--execution-plans', action='store_true')
    parser.add_argument('--reasoning-effort', default='low')
    args = parser.parse_args()
    if not 1 <= args.max_batch_cases <= 64:
        parser.error('--max-batch-cases must be between one and 64')
    if args.case_seconds < 1:
        parser.error('--case-seconds must be positive')
    if args.min_free_mib < 0:
        parser.error('--min-free-mib must not be negative')
    if args.no_observation_seconds < 0:
        parser.error('--no-observation-seconds must not be negative')
    if args.identical_turn_limit < 0:
        parser.error('--identical-turn-limit must not be negative')
    if args.model_concurrency is not None and args.model_concurrency < 1:
        parser.error('--model-concurrency must be positive')
    def stop(signum, frame):
        raise KeyboardInterrupt('queue stopped')
    signal.signal(signal.SIGTERM, stop)
    raise SystemExit(run_queue(args.queue, args.journal, args.runtime, args.case_seconds, args.model_id, args.provider,
              args.model_concurrency, args.execution_plans, args.reasoning_effort, args.min_free_mib,
              args.no_observation_seconds, args.max_batch_cases, args.provider_request_config,
              args.server, args.chat_completions_url, args.api_key_env, args.chat_request_config,
              args.identical_turn_limit, args.neuralese_server, args.drain_file, args.teacher_identity))
