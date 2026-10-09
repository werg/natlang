#!/usr/bin/env python3
"""Run one reviewed native-collector pool over one immutable contiguous IR range.

This opt-in campaign runner is separate from run_bonsai_queue.py. It never rewrites
the source or changes case identity. SIGTERM requests a drain: the collector stops
admitting cases and lets active cases checkpoint. An explicit hard-abort file sends
SIGTERM to the collector and leaves its per-case partials for exact resume.
"""
import argparse
import ctypes
import fcntl
import hashlib
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = str(Path(__file__).resolve().parent)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from reviewed_pool_paths import ReviewedPoolPaths
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
from natlang_neuralese.common.hashing import sha256_file_hex as sha256  # noqa: E402
from natlang_neuralese.common.jsonio import utc_now_iso as utc_now  # noqa: E402

ACTIVE_COLLECTOR_PID = None
ACTIVE_COLLECTOR = None
OWNED_POOL_LOCK = None


def atomic_json(path, value):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + f'.tmp-{os.getpid()}-{time.time_ns()}')
    with temporary.open('x') as output:
        output.write(json.dumps(value, indent=2, sort_keys=True) + '\n')
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, target)
    fd = os.open(target.parent, os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)


def verify_runtime(runtime, manifest_path, expected_manifest_sha):
    runtime = Path(runtime).resolve()
    manifest_path = Path(manifest_path).resolve()
    actual_manifest_sha = sha256(manifest_path)
    if actual_manifest_sha != expected_manifest_sha:
        raise ValueError(f'runtime manifest SHA mismatch: expected {expected_manifest_sha}, got {actual_manifest_sha}')
    manifest = json.loads(manifest_path.read_text())
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('frozen runtime manifest has no file inventory')
    expected = {}
    for relative, digest in files.items():
        target = (runtime / relative).resolve()
        if not target.is_relative_to(runtime) or sha256(target) != digest:
            raise ValueError(f'frozen runtime file mismatch: {relative}')
        expected[str(target)] = digest
    actual = set()
    for root, _dirs, names in os.walk(runtime):
        for name in names:
            actual.add(str((Path(root) / name).resolve()))
    if str(manifest_path) in actual:
        actual.remove(str(manifest_path))
    if actual != set(expected):
        raise ValueError('runtime file tree does not exactly match its frozen manifest')
    dependency_root = runtime / 'node_modules'
    if not dependency_root.is_dir() or not any(name.startswith('node_modules/') for name in files):
        raise ValueError('runtime must contain its pinned Node dependency closure; parent-workspace resolution is not sealed')
    return manifest, expected


def preflight(runtime, ir, start, count):
    """Use the pinned TS runtime for record parsing, curriculum validation, and live holds."""
    code = r"""
import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
const [runtime, irPath, startText, countText] = process.argv.slice(1);
const base = pathToFileURL(runtime + '/dist/teacher/');
const { recordDigest, loadRecords } = await import(new URL('collector.js', base).href);
const { validateCurriculum } = await import(new URL('curriculum.js', base).href);
const { quarantineReason, generationHoldReason, retiredFamily } = await import(new URL('curriculum-policy.js', base).href);
const start = Number(startText), count = Number(countText);
const rows = await loadRecords(irPath, start, count);
const holds = [];
for (const {index, record} of rows) {
  validateCurriculum(record);
  const reason = generationHoldReason(record) ?? retiredFamily(record) ?? quarantineReason(record);
  if (reason) holds.push({index, id: record.id, reason});
}
const source = readFileSync(irPath);
process.stdout.write(JSON.stringify({
  source_sha256: await (async()=>{const {createHash}=await import('node:crypto');return createHash('sha256').update(source).digest('hex')})(),
  selected: rows.map(({index,record})=>({index,id:record.id,digest:recordDigest(record)})), holds
}));
"""
    return json.loads(subprocess.check_output(['node', '--input-type=module', '-e', code,
        str(Path(runtime).resolve()), str(Path(ir).resolve()), str(start), str(count)], text=True))


def validate_collector_capabilities(runtime):
    """Fail closed when an older pinned CLI silently ignores runner controls."""
    teacher = Path(runtime) / 'dist/teacher'
    cli = (teacher / 'cli.js').read_text()
    collector = (teacher / 'collector.js').read_text()
    required = {
        '--case-events-file': "flags.has('--case-events-file')" in cli and
            'caseEventsFile:' in cli and 'config.caseEventsFile' in collector,
        '--final-export-only': "flags.has('--final-export-only')" in cli and
            'finalExportOnly:' in cli and 'config.finalExportOnly' in collector,
        '--drain-file': "flags.has('--drain-file')" in cli and 'drainFile' in cli and
            'admissionSignal?.aborted' in collector,
        '--context-tokens': "integer(flags, '--context-tokens'" in cli and 'contextTokens:' in cli,
        '--max-model-requests': "integer(flags, '--max-model-requests'" in cli and 'maxModelRequests:' in cli and
            'config.maxModelRequests' in collector,
        '--transport-retries': "integer(flags, '--transport-retries'" in cli and 'transportRetries:' in cli and
            'config.transportRetries' in collector,
        '--retry-delay-ms': "flags.get('--retry-delay-ms')" in cli and 'retryDelayMs:' in cli and
            'config.retryDelayMs' in collector,
        '--execution-plans': "flags.has('--execution-plans')" in cli and 'executionPlans:' in cli and
            'config.executionPlans' in collector,
    }
    missing = [flag for flag, supported in required.items() if not supported]
    if missing:
        raise ValueError('pinned collector runtime does not implement runner-required controls: ' + ', '.join(missing))
    return {'checked_flags': list(required), 'unsupported_flags': []}


def imported_output_accounting(runtime):
    helper_path = Path(__file__).resolve().with_name('run_bonsai_queue.py')
    spec = importlib.util.spec_from_file_location('run_bonsai_queue_reviewed_pool_helpers', helper_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.output_accounting, module.resolve_entry_jobs


def write_event(path, value):
    path = Path(path)
    with path.open('a') as output:
        output.write(json.dumps(value, sort_keys=True) + '\n')
        output.flush()
        os.fsync(output.fileno())


def run(args):
    global ACTIVE_COLLECTOR_PID, ACTIVE_COLLECTOR, OWNED_POOL_LOCK
    ir, runtime, manifest_path = map(lambda x: Path(x).resolve(), (args.ir, args.runtime, args.runtime_manifest))
    jobs, output, control = map(lambda x: Path(x).resolve(), (args.jobs, args.output, args.control_dir))
    if args.start < 0 or not 1 <= args.count <= 2048:
        raise ValueError('first reviewed pool supports at most 2,048 selected cases; source bounds are checked by the pinned loader')
    if args.workers < 1 or args.model_concurrency < 1 or args.max_model_requests < 1 or args.max_turns < 1:
        raise ValueError('workers, request cap, per-case request budget, and turns must be positive')
    if args.wall_seconds < 0 or args.min_free_mib < 12_288:
        raise ValueError('resource wall must be nonnegative; reviewed pool requires at least 12 GiB free before launch')
    if args.min_available_memory_mib < 12_288:
        raise ValueError('reviewed pool requires at least 12 GiB MemAvailable before launch')
    if args.workers > 512 or args.model_concurrency > 256:
        raise ValueError('pool worker/request settings exceed reviewed hard ceilings (512/256)')
    if args.max_model_requests != 384 or args.max_turns != 40:
        raise ValueError('this reviewed pool contract pins 384 requests and 40 turns per case')
    if args.kv_tokens != 0:
        raise ValueError('this pool contract disables local KvBudget; the model server is the shared KV authority')
    if args.provider:
        raise ValueError('reviewed single-pool mode is only for the local OpenAI-compatible server')
    if not args.authorization:
        raise ValueError('an approved exact-scope authorization receipt is required')
    if not args.native_review:
        raise ValueError('a root-approved native reference review receipt is required')
    if not args.chat_request_config:
        raise ValueError('pin the reviewed chat-request configuration explicitly')

    paths = ReviewedPoolPaths.from_runner_args(args)
    jobs, output, control = paths.jobs, paths.output, paths.control
    status_path, journal_path = paths.worker_status, paths.supervisor_journal
    case_events_path, hard_abort_file = paths.case_events, paths.hard_abort
    runtime_manifest, runtime_files = verify_runtime(runtime, manifest_path, args.runtime_manifest_sha256)
    collector_capabilities = validate_collector_capabilities(runtime)
    actual_script_sha = sha256(Path(__file__))
    pool_paths_helper_sha256 = sha256(Path(__file__).resolve().with_name('reviewed_pool_paths.py'))
    helper_path = Path(__file__).resolve().with_name('run_bonsai_queue.py')
    output_accounting_helper_sha256 = sha256(helper_path)
    input_proof = preflight(runtime, ir, args.start, args.count)
    if input_proof['source_sha256'] != args.ir_sha256:
        raise ValueError('source IR SHA-256 differs from the reviewed pin')
    if input_proof['holds']:
        raise ValueError(f'current generation/quarantine policy holds selected cases: {input_proof["holds"][:10]}')
    if len(input_proof['selected']) != args.count:
        raise ValueError('preflight selected case count differs from requested range')

    chat_config = Path(args.chat_request_config).resolve()
    if sha256(chat_config) != args.chat_request_config_sha256:
        raise ValueError('chat request config SHA-256 differs from the reviewed pin')
    authorization = json.loads(Path(args.authorization).read_text())
    auth_status = authorization.get('status')
    allowed_auth_statuses = {'approved'}
    if args.preflight_only:
        allowed_auth_statuses.add('prepared_not_approved')
    if auth_status not in allowed_auth_statuses:
        raise ValueError(f'authorization status must be one of {sorted(allowed_auth_statuses)}')
    native_review_path = Path(args.native_review).resolve()
    native_review_sha256 = sha256(native_review_path)
    native_review = json.loads(native_review_path.read_text())
    if (native_review.get('version') != 'natlang.reviewed_single_pool_native_review/1' or
            native_review.get('status') != 'approved' or
            native_review.get('ir_sha256') != args.ir_sha256 or
            native_review.get('runtime_manifest_sha256') != args.runtime_manifest_sha256 or
            native_review.get('range') != {'start': args.start, 'count': args.count} or
            native_review.get('native_reference_rows') != args.count or
            native_review.get('current_admitted') != args.count or
            native_review.get('materializer_accepted') != args.count):
        raise ValueError('native review receipt is not approved for this exact source/runtime/range')
    native_rows_path = (native_review_path.parent / native_review.get('native_reference_rows_path', '')).resolve()
    if not native_rows_path.is_file() or sha256(native_rows_path) != native_review.get('native_reference_rows_sha256'):
        raise ValueError('native reference row file does not match its approved review receipt')
    native_row_count = 0
    with native_rows_path.open() as native_rows:
        for position, line in enumerate(native_rows):
            if position >= args.count:
                raise ValueError('native reference proof has extra rows')
            row = json.loads(line)
            expected = input_proof['selected'][position]
            record = row.get('task', {}).get('program_ir', {})
            if (record.get('id') != expected['id'] or row.get('outcome', {}).get('accepted') is not True or
                    row.get('provenance', {}).get('program_ir_sha256') != expected['digest']):
                raise ValueError(f'native reference row identity/admission mismatch at selected position {position}')
            native_row_count += 1
    if native_row_count != args.count:
        raise ValueError('native reference proof row count differs from selected range')
    required_authorization = {
        'version': 'natlang.reviewed_single_pool_authorization/1', 'status': auth_status,
        'ir_sha256': args.ir_sha256, 'runtime_manifest_sha256': args.runtime_manifest_sha256,
        'start': args.start, 'count': args.count, 'root_seed': args.root_seed,
        'model_id': args.model_id, 'workers': args.workers,
        'model_concurrency': args.model_concurrency, 'max_model_requests': args.max_model_requests,
        'max_turns': args.max_turns, 'chat_request_config_sha256': args.chat_request_config_sha256,
        'transport_retries': args.transport_retries, 'retry_delay_ms': args.retry_delay_ms,
        'pool_runner_sha256': actual_script_sha, 'final_export_only': True,
        'pool_paths_helper_sha256': pool_paths_helper_sha256,
        'native_review_sha256': native_review_sha256,
        'output_accounting_helper_sha256': output_accounting_helper_sha256,
        'jobs_path': str(jobs), 'output_path': str(output), 'control_dir': str(control),
        'worker_status_path': str(status_path), 'supervisor_journal_path': str(journal_path),
        'case_events_path': str(case_events_path), 'hard_abort_file': str(hard_abort_file),
        'server': args.server, 'reasoning_effort': args.reasoning_effort, 'file_tools': args.file_tools,
        'execution_plans': args.execution_plans, 'heap_mib': args.heap_mib,
        'kv_tokens': args.kv_tokens, 'context_tokens': args.context_tokens,
        'wall_seconds': args.wall_seconds, 'min_free_mib': args.min_free_mib,
        'min_available_memory_mib': args.min_available_memory_mib,
        'status_interval': args.status_interval,
        'authorization_path': str(Path(args.authorization).resolve()),
        'native_review_path': str(native_review_path),
        'runtime_path': str(runtime), 'runtime_manifest_path': str(manifest_path),
        'chat_request_config_path': str(chat_config),
        'pool_status_path': str(control / 'pool-status.json'),
    }
    if any(authorization.get(key) != value for key, value in required_authorization.items()):
        raise ValueError('authorization receipt does not exactly approve this source/runtime/range/config/pool')
    current_review_sources = {
        'source_review_source_sha256': sha256(runtime / 'src/teacher/source-review.ts'),
        'source_review_compiled_sha256': sha256(runtime / 'dist/teacher/source-review.js'),
    }
    if any(authorization.get(key) != value for key, value in current_review_sources.items()):
        raise ValueError('runtime source-review implementation differs from the authorized current hold policy')

    space_path = control
    while not space_path.exists() and space_path != space_path.parent: space_path = space_path.parent
    free_mib = shutil.disk_usage(space_path).free // (1024 * 1024)
    if args.min_free_mib and free_mib < args.min_free_mib:
        raise RuntimeError(f'storage pause before launch: {free_mib} MiB < {args.min_free_mib} MiB')
    meminfo = Path('/proc/meminfo').read_text()
    available_line = next((line for line in meminfo.splitlines() if line.startswith('MemAvailable:')), None)
    if available_line is None:
        raise RuntimeError('resource pause before launch: /proc/meminfo has no MemAvailable value')
    available_mib = int(available_line.split()[1]) // 1024
    if available_mib < args.min_available_memory_mib:
        raise RuntimeError(f'memory pause before launch: {available_mib} MiB MemAvailable < {args.min_available_memory_mib} MiB')

    if args.preflight_only:
        plan_path = control / 'pool-plan.json'
        if plan_path.exists():
            raise ValueError('preflight-only requires a fresh control directory without pool-plan.json')
        if jobs.exists() and any(jobs.iterdir()):
            raise ValueError('preflight-only refuses a nonempty jobs directory')
        if output.exists() or Path(str(output) + '.manifest.json').exists():
            raise ValueError('preflight-only refuses existing output artifacts')
        for existing in (status_path, journal_path, case_events_path, hard_abort_file):
            if existing.exists():
                raise ValueError(f'preflight-only refuses an existing run artifact: {existing}')
        lock_path = control / 'pool.lock'
        if lock_path.exists():
            with lock_path.open('r+') as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise ValueError('preflight-only refuses a live pool lock') from error
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        report = {
            'version': 'natlang.reviewed_single_pool_preflight/1',
            'status': 'preflight_passed_no_pool_started',
            'source_ir_sha256': input_proof['source_sha256'],
            'selected_cases': len(input_proof['selected']),
            'selected_case_identity_sha256': hashlib.sha256(
                json.dumps(input_proof['selected'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
            'runtime_manifest_sha256': args.runtime_manifest_sha256,
            'native_review_sha256': native_review_sha256,
            'pool_runner_sha256': actual_script_sha,
            'pool_paths_helper_sha256': pool_paths_helper_sha256,
            'collector_capabilities': collector_capabilities,
            'output_accounting_helper_sha256': output_accounting_helper_sha256,
            'source_review': current_review_sources,
            'authorization_status': auth_status,
            'server_health_checked': False,
            'pool_plan_written': False,
            'jobs_started': False,
        }
        print(json.dumps(report, sort_keys=True))
        return 0

    control.mkdir(parents=True, exist_ok=True)
    plan_path = control / 'pool-plan.json'
    prior_plan_exists = plan_path.exists()
    if prior_plan_exists and not args.resume:
        raise ValueError('pool plan already exists; pass --resume after checking no process remains')
    if not prior_plan_exists and (any(jobs.iterdir()) if jobs.exists() else False):
        raise ValueError('jobs directory contains prior artifacts without this pool plan')
    if not prior_plan_exists and (output.exists() or Path(str(output) + '.manifest.json').exists()):
        raise ValueError('output artifacts exist without this pool plan; refusing to adopt prior results')
    jobs.mkdir(parents=True, exist_ok=True)
    lock = (control / 'pool.lock').open('a+')
    try: fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error: raise RuntimeError('another pool supervisor owns this control directory') from error
    OWNED_POOL_LOCK = lock
    plan = {
        'version': 'natlang.reviewed_single_pool/1', 'ir': str(ir), 'ir_sha256': args.ir_sha256,
        'range': {'start': args.start, 'count': args.count}, 'selected_case_identity_sha256': hashlib.sha256(
            json.dumps(input_proof['selected'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'root_seed': args.root_seed, 'model_id': args.model_id, 'runtime': str(runtime),
        'runtime_manifest': str(manifest_path), 'runtime_manifest_sha256': args.runtime_manifest_sha256,
        'runtime_file_count': len(runtime_files), 'runner_sha256': actual_script_sha,
        'pool_paths_helper_sha256': pool_paths_helper_sha256,
        'output_accounting_helper_sha256': output_accounting_helper_sha256,
        'collector_capabilities': collector_capabilities,
        **current_review_sources,
        'chat_request_config': str(chat_config), 'chat_request_config_sha256': args.chat_request_config_sha256,
        'workers': args.workers, 'model_concurrency': args.model_concurrency,
        'per_case_max_model_requests': args.max_model_requests, 'per_case_max_turns': args.max_turns,
        'transport_retries': args.transport_retries, 'retry_delay_ms': args.retry_delay_ms,
        'export_policy': 'live_per_case_jobs_final_ordered_merge_only',
        'heap_mib': args.heap_mib, 'kv_tokens': args.kv_tokens, 'context_tokens': args.context_tokens,
        'resource_wall_seconds': args.wall_seconds, 'min_free_mib': args.min_free_mib,
        'min_available_memory_mib': args.min_available_memory_mib,
        'server': args.server, 'reasoning_effort': args.reasoning_effort, 'file_tools': args.file_tools,
        'execution_plans': args.execution_plans, 'jobs': str(jobs), 'output': str(output),
        'control_dir': str(control), 'worker_status': str(status_path), 'supervisor_journal': str(journal_path),
        'case_events_file': str(case_events_path), 'hard_abort_file': str(hard_abort_file),
        'status_interval': args.status_interval,
        'authorization': str(Path(args.authorization).resolve()),
        'native_review': str(native_review_path), 'native_review_sha256': native_review_sha256,
        'authorization_sha256': sha256(args.authorization),
    }
    if plan_path.exists():
        old_plan = json.loads(plan_path.read_text())
        if {key: value for key, value in old_plan.items() if key != 'created_at'} != plan:
            raise ValueError('existing pool plan differs; create a new isolated control directory')
    else:
        atomic_json(plan_path, {**plan, 'created_at': utc_now()})
    if output.exists() and not journal_path.exists():
        raise ValueError('output exists without this pool’s journal; refusing to adopt prior artifacts')
    if (control / 'drain.request').exists():
        if not args.resume:
            raise ValueError('drain request already exists; review prior terminal state before explicit resume')
        (control / 'drain.request').unlink()

    output.parent.mkdir(parents=True, exist_ok=True)
    if args.server.rstrip('/'):
        with urllib.request.urlopen(args.server.rstrip('/') + '/health', timeout=5) as response:
            if response.status != 200: raise RuntimeError(f'model server health status {response.status}')
    drain_file = control / 'drain.request'
    events_file = case_events_path
    log_path = control / 'collector.log'
    entry = {'source': str(ir), 'index': args.start, 'count': args.count, 'seed': args.root_seed,
        'jobs': str(jobs), 'output': str(output), 'max_turns': args.max_turns,
        'max_model_requests': args.max_model_requests, 'log': str(log_path)}

    command = ['node', f'--max-old-space-size={args.heap_mib}', str(runtime / 'dist/teacher/cli.js'),
        str(ir), str(jobs), str(output), '--start', str(args.start), '--limit', str(args.count),
        '--model-id', args.model_id, '--root-seed', str(args.root_seed), '--workers', str(args.workers),
        '--max-turns', str(args.max_turns), '--model-concurrency', str(args.model_concurrency),
        '--max-model-requests', str(args.max_model_requests), '--transport-retries', str(args.transport_retries),
        '--retry-delay-ms', str(args.retry_delay_ms), '--kv-tokens', '0', '--context-tokens', str(args.context_tokens),
        '--server', args.server, '--reasoning-effort', args.reasoning_effort, '--file-tools', args.file_tools,
        '--chat-request-config', str(chat_config), '--drain-file', str(drain_file),
        '--case-events-file', str(events_file)]
    command.append('--final-export-only')
    if args.execution_plans: command.append('--execution-plans')
    start_time = time.monotonic()
    stop_reason = None
    previous_sig = {}
    def request_drain(signum, _frame):
        nonlocal stop_reason
        stop_reason = 'signal_drain'
        drain_file.touch(exist_ok=True)
        write_event(journal_path, {'event': 'drain_requested', 'signal': signum, 'at': utc_now()})
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_sig[signum] = signal.signal(signum, request_drain)
    write_event(journal_path, {'event': 'pool_start', 'at': utc_now(), 'plan_sha256': sha256(plan_path),
        'pid': os.getpid(), 'selected_cases': args.count})
    atomic_json(control / 'pool-status.json', {'state': 'running', 'started_at': utc_now(),
        'pid': os.getpid(), 'progress': {'completed_results': 0, 'partials': 0, 'errors': 0}})
    atomic_json(status_path, {'state': 'running', 'updated_at': utc_now(), 'worker_pid': os.getpid(),
        'control_dir': str(control), 'status': 'running'})
    accounting = None
    code = None
    with log_path.open('ab') as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        ACTIVE_COLLECTOR = child
        ACTIVE_COLLECTOR_PID = child.pid
        atomic_json(control / 'pool-status.json', {'state': 'running', 'started_at': utc_now(),
            'pid': os.getpid(), 'collector_pid': child.pid, 'progress': {'completed_results': 0, 'partials': 0, 'errors': 0},
            'merged_export_status': 'final_merge_after_drain_or_completion'})
        last_scan = 0.0
        event_offset = events_file.stat().st_size if args.resume and events_file.exists() else 0
        event_buffer = b''
        failure_streak = 0
        if args.resume and events_file.exists():
            for line in events_file.read_text().splitlines():
                if not line.strip(): continue
                prior = json.loads(line)
                if prior.get('event') == 'case_finish':
                    failure_streak = failure_streak + 1 if prior.get('status') == 'error' else 0
                elif prior.get('event') == 'case_checkpoint': failure_streak = 0
        while child.poll() is None:
            now = time.monotonic()
            if args.wall_seconds and now - start_time >= args.wall_seconds and stop_reason is None:
                stop_reason = 'resource_wall_drain'
                drain_file.touch(exist_ok=True)
                write_event(journal_path, {'event': 'resource_wall_drain', 'at': utc_now(),
                    'wall_seconds': args.wall_seconds})
            if hard_abort_file.exists() and child.poll() is None:
                stop_reason = 'explicit_hard_abort'
                write_event(journal_path, {'event': 'hard_abort_requested', 'at': utc_now(),
                    'request_file': str(hard_abort_file)})
                os.killpg(child.pid, signal.SIGTERM)
                hard_abort_file.unlink()
            if events_file.exists():
                with events_file.open('rb') as events:
                    events.seek(event_offset)
                    payload = events.read()
                    event_offset = events.tell()
                lines = (event_buffer + payload).split(b'\n')
                event_buffer = lines.pop()
                for line in lines:
                    if not line: continue
                    item = json.loads(line)
                    write_event(journal_path, {'event': 'collector_case_event', **item})
                    if item.get('event') == 'case_finish':
                        failure_streak = failure_streak + 1 if item.get('status') == 'error' else 0
                    elif item.get('event') == 'case_checkpoint': failure_streak = 0
                    if item.get('event') in {'case_finish', 'case_checkpoint'} and failure_streak >= 3 and stop_reason is None:
                        stop_reason = 'collection_error_pause'
                        drain_file.touch(exist_ok=True)
                        write_event(journal_path, {'event': 'failure_pause_drain', 'at': utc_now(),
                            'index': item.get('index'), 'program_id': item.get('program_id'),
                            'consecutive_error_streak': failure_streak,
                            'failure_policy': 'three consecutive terminal collector-error completions in event order; successful wrong_return outcomes are result events'})
            if now - last_scan >= args.status_interval:
                names = list(jobs.iterdir())
                result_count = sum(path.name.endswith('.result.json') for path in names)
                partial_count = sum(path.name.endswith('.partial.json') for path in names)
                retry_count = sum(path.name.endswith('.retry.json') for path in names)
                error_count = sum(path.name.endswith('.error.json') for path in names)
                status = {'state': 'running', 'updated_at': utc_now(), 'pid': os.getpid(), 'collector_pid': child.pid,
                    'elapsed_seconds': round(now - start_time, 1), 'stop_reason': stop_reason,
                    'progress': {'completed_result_files': result_count, 'partial_files': partial_count,
                        'retry_files': retry_count, 'error_files': error_count, 'expected_cases': args.count},
                    'merged_export_status': 'final_merge_after_drain_or_completion',
                    'aggregate_may_be_stale_until_final_merge': bool(output.exists()),
                    'last_event_offset': event_offset}
                atomic_json(control / 'pool-status.json', status)
                atomic_json(status_path, {'state': 'running', 'updated_at': utc_now(), 'worker_pid': os.getpid(),
                    'collector_pid': child.pid, 'control_dir': str(control), 'status': 'running',
                    'progress': status['progress'], 'stop_reason': stop_reason,
                    'aggregate_may_be_stale_until_final_merge': bool(output.exists())})
                write_event(journal_path, {'event': 'pool_progress', 'at': utc_now(), **status['progress'],
                    'elapsed_seconds': status['elapsed_seconds'], 'stop_reason': stop_reason})
                last_scan = now
            time.sleep(1)
        code = child.wait()
        ACTIVE_COLLECTOR = None
        ACTIVE_COLLECTOR_PID = None
        # The final fsynced event can land just before the collector exits;
        # consume its tail before deciding whether an infrastructure streak
        # requires a paused terminal status.
        if events_file.exists():
            with events_file.open('rb') as events:
                events.seek(event_offset)
                tail = event_buffer + events.read()
            for line in tail.split(b'\n'):
                if not line: continue
                item = json.loads(line)
                write_event(journal_path, {'event': 'collector_case_event', **item})
                if item.get('event') == 'case_finish':
                    failure_streak = failure_streak + 1 if item.get('status') == 'error' else 0
                elif item.get('event') == 'case_checkpoint': failure_streak = 0
            if failure_streak >= 3 and stop_reason is None:
                stop_reason = 'collection_error_pause'
                write_event(journal_path, {'event': 'failure_pause_drain', 'at': utc_now(),
                    'consecutive_error_streak': failure_streak,
                    'failure_policy': 'three consecutive terminal collector-error completions in event order; successful wrong_return outcomes are result events'})

    output_accounting, _resolve = imported_output_accounting(runtime)
    try: accounting = output_accounting(entry, runtime)
    except Exception as error:
        accounting = {'version': 'natlang.supervisor_output_accounting/2', 'complete': False,
            'disposition': 'accounting_error', 'error': f'{type(error).__name__}: {error}'}
    if code == 0 and accounting.get('complete') and accounting.get('disposition') == 'all_exact_results_exported':
        status, worker_state, disposition = 'complete', 'finished', 'complete'
    elif stop_reason in {'signal_drain', 'resource_wall_drain', 'collection_error_pause'}:
        if stop_reason == 'collection_error_pause':
            status, worker_state, disposition = 'paused_collection_error', 'paused', 'paused_collection_error'
        else: status, worker_state, disposition = 'drained_partial', 'stopped', 'explicit_drain_partial'
    elif stop_reason == 'explicit_hard_abort': status, worker_state, disposition = 'hard_aborted_partial', 'stopped', 'explicit_hard_abort'
    else: status, worker_state, disposition = 'incomplete', 'paused', 'incomplete_or_infrastructure_failure'
    finish = {'event': 'pool_finish', 'at': utc_now(), 'status': status, 'exit_code': code,
        'stop_reason': stop_reason, 'elapsed_seconds': round(time.monotonic() - start_time, 1),
        'output_accounting': accounting}
    write_event(journal_path, finish)
    atomic_json(control / 'pool-status.json', {**finish, 'state': status})
    atomic_json(status_path, {**finish, 'state': worker_state, 'disposition': disposition,
        'updated_at': utc_now(), 'worker_pid': os.getpid(), 'control_dir': str(control)})
    for signum, handler in previous_sig.items(): signal.signal(signum, handler)
    lock.close()
    OWNED_POOL_LOCK = None
    return 0 if status == 'complete' else 2


def record_setup_failure(args, error):
    """Record failure only while this process owns, or has acquired, the pool lock."""
    global ACTIVE_COLLECTOR_PID, ACTIVE_COLLECTOR, OWNED_POOL_LOCK
    lock = OWNED_POOL_LOCK
    opened_here = False
    try:
        control = Path(args.control_dir).resolve()
        control.mkdir(parents=True, exist_ok=True)
        if lock is None:
            lock = (control / 'pool.lock').open('a+')
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
                return
            opened_here = True
        if ACTIVE_COLLECTOR is not None:
            child = ACTIVE_COLLECTOR
            try:
                os.killpg(child.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # This is infrastructure cleanup after a supervisor exception,
                # not a per-case deadline. Do not release the pool lock while
                # the collector can still write jobs or status.
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    child.wait()
                except Exception:
                    # Fall back to the direct child handle and reap it before
                    # allowing the pool lock to be released.
                    child.kill()
                    child.wait()
            ACTIVE_COLLECTOR = None
            ACTIVE_COLLECTOR_PID = None
        now = utc_now()
        detail = f'{type(error).__name__}: {error}'
        write_event(Path(args.journal).resolve(), {'event': 'pool_setup_failure', 'at': now, 'error': detail})
        payload = {'state': 'paused', 'status': 'paused_setup', 'disposition': 'paused_setup',
            'updated_at': now, 'worker_pid': os.getpid(), 'control_dir': str(control), 'error': detail}
        atomic_json(Path(args.status_file).resolve(), payload)
        atomic_json(control / 'pool-status.json', payload)
    except Exception:
        # Paths may themselves be invalid. Preserve the original failure and
        # never claim a status without a successfully acquired lock.
        # If status publication itself fails, still stop and reap the direct
        # collector child before this process can exit and release its lock.
        if ACTIVE_COLLECTOR is not None:
            child = ACTIVE_COLLECTOR
            try:
                child.kill()
            except ProcessLookupError:
                pass
            child.wait()
            ACTIVE_COLLECTOR = None
            ACTIVE_COLLECTOR_PID = None
        return
    finally:
        if ACTIVE_COLLECTOR is None and lock is not None and (opened_here or lock is OWNED_POOL_LOCK):
            lock.close()
            if lock is OWNED_POOL_LOCK: OWNED_POOL_LOCK = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ir', required=True)
    parser.add_argument('--ir-sha256', required=True)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=2048)
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--runtime-manifest', required=True)
    parser.add_argument('--runtime-manifest-sha256', required=True)
    parser.add_argument('--authorization', required=True, help='root-approved exact-range/pool authorization JSON')
    parser.add_argument('--native-review', required=True, help='root-approved native/admission/materialization proof JSON')
    parser.add_argument('--jobs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--control-dir', required=True)
    parser.add_argument('--model-id', default='nvidia/Qwen3.6-35B-A3B-NVFP4')
    parser.add_argument('--server', default='http://127.0.0.1:8082')
    parser.add_argument('--chat-request-config', required=True)
    parser.add_argument('--chat-request-config-sha256', required=True)
    parser.add_argument('--root-seed', required=True, type=int)
    parser.add_argument('--workers', type=int, default=320)
    parser.add_argument('--model-concurrency', type=int, default=256)
    parser.add_argument('--max-model-requests', type=int, default=384)
    parser.add_argument('--max-turns', type=int, default=40)
    parser.add_argument('--heap-mib', type=int, default=8192)
    parser.add_argument('--kv-tokens', type=int, default=0)
    parser.add_argument('--context-tokens', type=int, default=16384)
    parser.add_argument('--transport-retries', type=int, default=1)
    parser.add_argument('--retry-delay-ms', type=int, default=5000)
    parser.add_argument('--reasoning-effort', default='high')
    parser.add_argument('--file-tools', default='all')
    parser.add_argument('--execution-plans', action='store_true')
    parser.add_argument('--wall-seconds', type=int, default=0, help='0 disables the optional graceful resource wall')
    parser.add_argument('--min-free-mib', type=int, default=12_288, help='reviewed pool requires at least 12 GiB free before launch')
    parser.add_argument('--min-available-memory-mib', type=int, default=12_288,
        help='reviewed pool requires at least 12 GiB MemAvailable before launch')
    parser.add_argument('--status-interval', type=int, default=10)
    parser.add_argument('--hard-abort-file', help='touch to cancel active requests; partials remain resumable')
    parser.add_argument('--provider', help='not supported by this local shared-pool runner')
    parser.add_argument('--status-file', required=True, help='shared worker-status.json consumed by sync_remote_teacher.py')
    parser.add_argument('--journal', required=True, help='shared journal-pool.jsonl append-only supervisor journal')
    parser.add_argument('--case-events-file', required=True, help='shared journal-cases.jsonl append-only collector case events')
    parser.add_argument('--resume', action='store_true', help='resume exact pool plan after verifying no prior process remains')
    parser.add_argument('--preflight-only', action='store_true',
        help='validate source/runtime/native/auth/resource/path gates and exit before writes, server health, or child launch')
    args = parser.parse_args()
    if args.heap_mib < 1 or args.kv_tokens < 0 or args.context_tokens < 1 or args.status_interval < 1:
        parser.error('heap/context/status interval must be positive and KV tokens nonnegative')
    try:
        result = run(args)
    except Exception as error:
        if not args.preflight_only:
            record_setup_failure(args, error)
        raise
    raise SystemExit(result)


if __name__ == '__main__': main()
