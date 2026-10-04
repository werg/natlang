/** Correctness-first evaluation for small Python functions on host-held cases. */
import { canonical as canonicalJSON } from '../adaptation/identity.js';
import { randomBytes } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { chmodSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { PYTHON_SANDBOX_IMAGE } from './graded.js';

export type CodeObjectiveCase = { id: string; group: string; input: unknown; expected: unknown };
export type CodeObjective = {
  schema: 'natlang.skill-code-objective/1';
  id: string;
  revision: string;
  description: string;
  functionName: 'solve';
  cases: readonly CodeObjectiveCase[];
  /** A correct reference implementation and a separately supplied best-known byte lower bound. */
  sizeObjective?: { referenceSource: string; bestKnownLowerBoundBytes: number };
};
export type CodeSandboxRequest = { source: string; inputs: readonly unknown[] };
export type CodeSandboxResult =
  | { kind: 'ok'; value: unknown }
  | { kind: 'candidate-error'; error: { category: string; message: string } }
  | { kind: 'infrastructure-error'; category: string; message: string };
export type CodeSandboxBatchResult =
  | { kind: 'results'; results: readonly CodeSandboxResult[] }
  | { kind: 'infrastructure-error'; category: string; message: string };
export type CodeSandboxExecutor = (request: CodeSandboxRequest) => CodeSandboxBatchResult;
export type CodeObjectiveOptions = { execute?: CodeSandboxExecutor; maxSourceBytes?: number };
export type CodeObjectiveResult = {
  status: 'scored' | 'candidate-error' | 'infrastructure-error' | 'invalid-task';
  quality?: number;
  gates: Record<string, boolean>;
  detail: Record<string, unknown>;
};

const MAX_CASES = 64;
const MAX_INPUT_BYTES = 64 * 1024;
const MAX_OUTPUT_BYTES = 64 * 1024;
const DEFAULT_MAX_SOURCE_BYTES = 32 * 1024;
const CONTAINER_WALL_TIMEOUT_MS = 8000;
const CPU_SECONDS = 3;
const ADDRESS_SPACE_BYTES = 224 * 1024 * 1024;
const checkedImages = new Set<string>();

const CASE_RUNNER = String.raw`
import contextlib, importlib.util, json, os, resource, sys

MAX_INPUT = 65536
MAX_OUTPUT = 65536
try:
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (224 * 1024 * 1024, 224 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    raw = sys.stdin.buffer.readline(MAX_INPUT + 1)
    if len(raw) > MAX_INPUT or not raw.endswith(b'\n'):
        raise ValueError('input exceeds sandbox limit')
    request = json.loads(raw)
    if not isinstance(request, dict) or set(request) != {'input', 'token'} or not isinstance(request['token'], str):
        raise ValueError('invalid runner request')
    token = request['token']
    protocol = os.fdopen(os.dup(1), 'w', encoding='utf-8', closefd=True)
    try:
        with contextlib.redirect_stdout(sys.stderr):
            spec = importlib.util.spec_from_file_location('candidate_solution', '/work/solution.py')
            if spec is None or spec.loader is None:
                raise ImportError('solution module unavailable')
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            solve = getattr(module, 'solve', None)
            if not callable(solve):
                raise TypeError('solution.py must define callable solve(x)')
            result = solve(request['input'])
        encoded = json.dumps(result, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        if len(encoded.encode('utf-8')) > MAX_OUTPUT:
            raise ValueError('result exceeds sandbox output limit')
        response = {'token': token, 'kind': 'ok', 'value': json.loads(encoded)}
    except BaseException as error:
        if isinstance(error, MemoryError):
            response = {'token': token, 'kind': 'infrastructure-error', 'category': 'memory-limit', 'message': 'case worker exhausted its memory limit'}
        else:
            response = {'token': token, 'kind': 'candidate-error', 'error': {
                'category': type(error).__name__, 'message': str(error)[:300]}}
    protocol.write(json.dumps(response, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    protocol.flush()
except BaseException as error:
    sys.stderr.write(type(error).__name__ + ': ' + str(error)[:300] + '\n')
    sys.exit(70)
`;

const RUNNER = String.raw`
import ctypes, json, os, resource, subprocess, sys

MAX_INPUT = 65536
MAX_OUTPUT = 65536
try:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(4, 0, 0, 0, 0) != 0:  # Linux PR_SET_DUMPABLE
        raise OSError(ctypes.get_errno(), 'could not disable process dumps')
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (224 * 1024 * 1024, 224 * 1024 * 1024))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))
    raw = sys.stdin.buffer.readline(MAX_INPUT + 1)
    if len(raw) > MAX_INPUT or not raw.endswith(b'\n'):
        raise ValueError('input exceeds sandbox limit')
    request = json.loads(raw)
    if not isinstance(request, dict) or set(request) != {'inputs', 'token'} or not isinstance(request['token'], str) or not isinstance(request['inputs'], list) or not (1 <= len(request['inputs']) <= 64):
        raise ValueError('invalid runner request')
    token = request['token']
    results = []
    for item in request['inputs']:
        case_token = os.urandom(24).hex()
        case_input = json.dumps({'input': item, 'token': case_token}, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n'
        try:
            completed = subprocess.run([sys.executable, '-B', '-I', '/work/case_runner.py'], input=case_input,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=4, check=False)
        except subprocess.TimeoutExpired:
            protocol = os.fdopen(os.dup(1), 'w', encoding='utf-8', closefd=True)
            protocol.write(json.dumps({'token': token, 'kind': 'infrastructure-error', 'category': 'case-time-limit', 'message': 'a case worker exceeded the wall-time limit; batch is incomplete'}, separators=(',', ':')) + '\n')
            protocol.flush()
            sys.exit(0)
        if completed.returncode == 137 or completed.returncode == -9:
            protocol = os.fdopen(os.dup(1), 'w', encoding='utf-8', closefd=True)
            protocol.write(json.dumps({'token': token, 'kind': 'infrastructure-error', 'category': 'case-resource-limit', 'message': 'a case worker was killed at a memory or CPU limit; batch is incomplete'}, separators=(',', ':')) + '\n')
            protocol.flush()
            sys.exit(0)
        lines = completed.stdout.strip().split('\n')
        try:
            case_result = json.loads(lines[-1] if lines else '')
        except Exception:
            diagnostic = completed.stderr[-240:].replace('\\n', ' ')
            results.append({'kind': 'candidate-error', 'error': {'category': 'candidate-process', 'message': ('case worker exited without a result: ' + diagnostic)[:300]}})
            continue
        if not isinstance(case_result, dict) or case_result.get('token') != case_token:
            results.append({'kind': 'candidate-error', 'error': {'category': 'candidate-process', 'message': 'case worker returned no authenticated result'}})
        elif case_result.get('kind') == 'ok' and 'value' in case_result:
            results.append({'kind': 'ok', 'value': case_result['value']})
        elif case_result.get('kind') == 'candidate-error' and isinstance(case_result.get('error'), dict):
            results.append({'kind': 'candidate-error', 'error': case_result['error']})
        elif case_result.get('kind') == 'infrastructure-error' and isinstance(case_result.get('category'), str) and isinstance(case_result.get('message'), str):
            protocol = os.fdopen(os.dup(1), 'w', encoding='utf-8', closefd=True)
            protocol.write(json.dumps({'token': token, 'kind': 'infrastructure-error', 'category': case_result['category'], 'message': case_result['message']}, separators=(',', ':')) + '\n')
            protocol.flush()
            sys.exit(0)
        else:
            results.append({'kind': 'candidate-error', 'error': {'category': 'candidate-process', 'message': 'case worker returned an invalid result'}})
    protocol = os.fdopen(os.dup(1), 'w', encoding='utf-8', closefd=True)
    protocol.write(json.dumps({'token': token, 'kind': 'results', 'results': results}, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    protocol.flush()
except BaseException as error:
    sys.stderr.write(type(error).__name__ + ': ' + str(error)[:300] + '\n')
    sys.exit(70)
`;

function safeJson(value: unknown, label: string): string {
  try { return canonicalJSON(value); } catch { throw Error(label + ' must contain finite JSON values'); }
}
const canonical = canonicalJSON;

function taskError(task: CodeObjective): string | undefined {
  if (!task || task.schema !== 'natlang.skill-code-objective/1' || !task.id || !task.revision ||
      !task.description || task.functionName !== 'solve' || !Array.isArray(task.cases) ||
      task.cases.length < 1 || task.cases.length > MAX_CASES) return 'invalid task identity, schema, or case count';
  if (new Set(task.cases.map(item => item?.id)).size !== task.cases.length ||
      task.cases.some(item => !item || !item.id || !item.group)) return 'cases require unique IDs and source groups';
  try {
    if (Buffer.byteLength(safeJson({ inputs: task.cases.map(row => row.input) }, 'case inputs')) > MAX_INPUT_BYTES)
      return 'case batch exceeds sandbox input limit';
    for (const row of task.cases) {
      if (Buffer.byteLength(safeJson({ input: row.input }, 'case input')) > MAX_INPUT_BYTES) return `case input too large: ${row.id}`;
      if (Buffer.byteLength(safeJson(row.expected, 'expected output')) > MAX_OUTPUT_BYTES) return `expected output too large: ${row.id}`;
    }
  } catch (error) { return String((error as Error).message); }
  if (task.sizeObjective) {
    const { referenceSource, bestKnownLowerBoundBytes } = task.sizeObjective;
    if (typeof referenceSource !== 'string' || !referenceSource.trim() ||
        !Number.isSafeInteger(bestKnownLowerBoundBytes) || bestKnownLowerBoundBytes < 1) return 'invalid optional size objective';
  }
  return undefined;
}

function validSandboxResult(value: unknown): CodeSandboxResult {
  if (!value || typeof value !== 'object') return { kind: 'infrastructure-error', category: 'runner-protocol', message: 'sandbox returned no result object' };
  const result = value as Record<string, any>;
  if (result.kind === 'ok' && Object.hasOwn(result, 'value')) {
    try { canonicalJSON(result.value); } catch { return {kind:'infrastructure-error',category:'runner-protocol',message:'sandbox returned non-JSON output'}; }
    return { kind: 'ok', value: result.value };
  }
  if (result.kind === 'candidate-error' && result.error && typeof result.error.category === 'string' && typeof result.error.message === 'string')
    return { kind: 'candidate-error', error: { category: result.error.category, message: result.error.message.slice(0, 300) } };
  if (result.kind === 'infrastructure-error' && typeof result.category === 'string' && typeof result.message === 'string')
    return { kind: 'infrastructure-error', category: result.category, message: result.message.slice(0, 300) };
  return { kind: 'infrastructure-error', category: 'runner-protocol', message: 'sandbox returned an invalid result envelope' };
}

function validBatchResult(value: unknown, expectedCount: number): CodeSandboxBatchResult {
  if (!value || typeof value !== 'object') return { kind: 'infrastructure-error', category: 'runner-protocol', message: 'sandbox returned no batch result object' };
  const result = value as Record<string, unknown>;
  if (result.kind !== 'results' || !Array.isArray(result.results) || result.results.length !== expectedCount)
    return { kind: 'infrastructure-error', category: 'runner-protocol', message: 'sandbox returned an invalid batch envelope or result count' };
  return { kind: 'results', results: result.results.map(validSandboxResult) };
}

function ensurePinnedImage(): CodeSandboxResult | undefined {
  if (checkedImages.has(PYTHON_SANDBOX_IMAGE)) return undefined;
  const inspect = spawnSync('docker', ['image', 'inspect', PYTHON_SANDBOX_IMAGE, '--format', '{{.Id}}'], {
    encoding: 'utf8', timeout: 5000, maxBuffer: 16 * 1024,
  });
  if (inspect.error) return { kind: 'infrastructure-error', category: 'docker-unavailable', message: inspect.error.message.slice(0, 300) };
  if (inspect.status !== 0 || !inspect.stdout.trim()) return { kind: 'infrastructure-error', category: 'missing-pinned-image', message: 'pinned Python sandbox image is not present locally; no image pull was attempted' };
  checkedImages.add(PYTHON_SANDBOX_IMAGE);
  return undefined;
}

/** Execute a candidate once for the whole case batch; expected outputs are never sent to the sandbox. */
export function runPythonSolveBatch(request: CodeSandboxRequest): CodeSandboxBatchResult {
  const source = request.source;
  if (typeof source !== 'string' || !Array.isArray(request.inputs) || request.inputs.length < 1 || request.inputs.length > MAX_CASES)
    return { kind: 'infrastructure-error', category: 'request-shape', message: 'sandbox requires candidate source and 1 to 64 inputs' };
  const token = randomBytes(24).toString('hex');
  const input = safeJson({ inputs: request.inputs, token }, 'sandbox input');
  if (Buffer.byteLength(input) > MAX_INPUT_BYTES) return { kind: 'infrastructure-error', category: 'input-limit', message: 'case batch exceeds sandbox input limit' };
  const imageError = ensurePinnedImage();
  if (imageError?.kind === 'infrastructure-error') return { kind: 'infrastructure-error', category: imageError.category, message: imageError.message };
  const containerName = 'natlang-code-objective-' + token;
  const work = mkdtempSync(join(tmpdir(), 'natlang-code-objective-'));
  try {
    writeFileSync(join(work, 'solution.py'), source, { mode: 0o444 });
    writeFileSync(join(work, 'runner.py'), RUNNER, { mode: 0o444 });
    writeFileSync(join(work, 'case_runner.py'), CASE_RUNNER, { mode: 0o444 });
    chmodSync(work, 0o555);
    const result = spawnSync('docker', [
      'run', '--rm', '--name', containerName, '-i', '--network', 'none', '--memory', '256m', '--memory-swap', '256m',
      '--cpus', '1', '--pids-limit', '16', '--read-only', '--tmpfs', '/tmp:rw,noexec,nosuid,size=16m',
      '--cap-drop', 'ALL', '--security-opt', 'no-new-privileges', '--user', '65534:65534',
      '--workdir', '/work', '-v', `${join(work, 'solution.py')}:/work/solution.py:ro`,
      '-v', `${join(work, 'runner.py')}:/work/runner.py:ro`,
      '-v', `${join(work, 'case_runner.py')}:/work/case_runner.py:ro`, PYTHON_SANDBOX_IMAGE,
      'python', '-B', '-I', '/work/runner.py',
    ], {
      input: `${input}\n`, encoding: 'utf8', timeout: CONTAINER_WALL_TIMEOUT_MS,
      maxBuffer: 256 * 1024,
    });
    if (result.error) {
      if ((result.error as NodeJS.ErrnoException).code === 'ETIMEDOUT')
        return {kind:'infrastructure-error',category:'allocation-exhausted',message:'sandbox wall-time allocation expired; no correctness verdict'};
      if ((result.error as NodeJS.ErrnoException).code === 'ENOBUFS')
        return {kind:'infrastructure-error',category:'output-allocation-exhausted',message:'sandbox output allocation expired; no correctness verdict'};
      return { kind: 'infrastructure-error', category: 'sandbox-launch', message: result.error.message.slice(0, 300) };
    }
    if (result.status === 137) return {kind:'infrastructure-error',category:'unattributed-sigkill',message:'sandbox was killed; resource or operator cause is unproven'};
    if (result.status !== 0) return { kind: 'infrastructure-error', category: 'sandbox-exit', message: `sandbox exited ${result.status}: ${result.stderr.slice(-300)}` };
    const lines = result.stdout.trimEnd().split('\n');
    let parsed: unknown;
    try { parsed = JSON.parse(lines.at(-1) ?? ''); }
    catch { return { kind: 'infrastructure-error', category: 'runner-protocol', message: `sandbox result is not valid JSON: ${result.stderr.slice(-200)}` }; }
    if (Buffer.byteLength(lines.at(-1) ?? '') > MAX_OUTPUT_BYTES) return { kind: 'infrastructure-error', category: 'runner-protocol', message: 'batch response exceeds sandbox output limit' };
    if (!parsed || typeof parsed !== 'object' || (parsed as Record<string, unknown>).token !== token)
      return { kind: 'infrastructure-error', category: 'runner-authentication', message: 'sandbox result did not match this request token' };
    const { token: _token, ...resultEnvelope } = parsed as Record<string, unknown>;
    return validBatchResult(resultEnvelope, request.inputs.length);
  } finally {
    // A killed Docker client can leave its container running. Always reclaim this named invocation.
    spawnSync('docker', ['rm', '-f', containerName], {encoding:'utf8',timeout:3000,maxBuffer:16*1024});
    chmodSync(work, 0o755);
    rmSync(work, { recursive: true, force: true });
  }
}

/** Compatibility convenience for a single public input. */
export function runPythonSolve(request: { source: string; input: unknown }): CodeSandboxResult {
  const batch = runPythonSolveBatch({ source: request.source, inputs: [request.input] });
  return batch.kind === 'results' ? batch.results[0] ?? { kind: 'infrastructure-error', category: 'runner-protocol', message: 'empty batch response' } : batch;
}

function byteLength(source: string): number { return Buffer.byteLength(source, 'utf8'); }

/**
 * Run every exact case before applying any optional code-size reward. Expected values remain on the host;
 * executor receives candidate source and all case inputs in one launch. Infrastructure failures are unscored.
 */
export function evaluateCodeObjective(source: string, task: CodeObjective, options: CodeObjectiveOptions = {}): CodeObjectiveResult {
  const invalidTask = taskError(task);
  if (invalidTask) return { status: 'invalid-task', gates: { task_valid: false }, detail: { reason: invalidTask } };
  if (typeof source !== 'string') return { status: 'candidate-error', quality: 0, gates: { source_string: false }, detail: {} };
  const sourceBytes = byteLength(source), maxSourceBytes = options.maxSourceBytes ?? DEFAULT_MAX_SOURCE_BYTES;
  if (!Number.isSafeInteger(maxSourceBytes) || maxSourceBytes < 1) throw new Error('maxSourceBytes must be a positive integer');
  if (sourceBytes > maxSourceBytes) return { status: 'candidate-error', quality: 0, gates: { source_size: false, all_correct: false }, detail: { source_bytes: sourceBytes, max_source_bytes: maxSourceBytes } };

  const execute = options.execute ?? runPythonSolveBatch;
  const evaluateSource = (candidate: string): CodeSandboxBatchResult => {
    try { return execute({ source: candidate, inputs: task.cases.map(row => row.input) }); }
    catch (error) { return { kind: 'infrastructure-error', category: 'sandbox-executor', message: String((error as Error)?.message ?? error).slice(0, 300) }; }
  };
  const candidateBatch = validBatchResult(evaluateSource(source), task.cases.length);
  if (candidateBatch.kind === 'infrastructure-error') return {
    status: 'infrastructure-error', gates: { infrastructure: false },
    detail: { category: candidateBatch.category, message: candidateBatch.message, cases_completed: 0 },
  };
  const failures: { caseId: string; category: string; message?: string }[] = [];
  let passed = 0;
  for (const [index, row] of task.cases.entries()) {
    const result = candidateBatch.results[index]!;
    if (result.kind === 'infrastructure-error') return {
      status: 'infrastructure-error', gates: { infrastructure: false },
      detail: { category: result.category, message: result.message, cases_completed: passed + failures.length },
    };
    if (result.kind === 'candidate-error') {
      failures.push({ caseId: row.id, category: result.error.category, message: result.error.message });
      continue;
    }
    if (canonical(result.value) !== canonical(row.expected)) {
      failures.push({ caseId: row.id, category: 'wrong-output' });
      continue;
    }
    passed++;
  }
  const correctness = passed === task.cases.length && failures.length === 0;
  if (!correctness) return {
    status: 'candidate-error', quality: 0,
    gates: { source_size: true, all_correct: false },
    detail: { passed, total: task.cases.length, failed_case_ids: failures.map(item => item.caseId), failures: failures.slice(0, 12) },
  };

  const detail: Record<string, unknown> = { passed, total: task.cases.length, source_bytes: sourceBytes };
  if (!task.sizeObjective) return { status: 'scored', quality: 1, gates: { source_size: true, all_correct: true }, detail };

  const referenceBytes = byteLength(task.sizeObjective.referenceSource);
  const lowerBound = task.sizeObjective.bestKnownLowerBoundBytes;
  if (lowerBound > referenceBytes) return { status: 'invalid-task', gates: { task_valid: false }, detail: { reason: 'best-known lower bound exceeds reference implementation bytes' } };
  const referenceFailures: string[] = [];
  const referenceBatch = validBatchResult(evaluateSource(task.sizeObjective.referenceSource), task.cases.length);
  if (referenceBatch.kind === 'infrastructure-error') return {
    status: 'infrastructure-error', gates: { infrastructure: false },
    detail: { category: referenceBatch.category, message: referenceBatch.message, phase: 'reference-validation' },
  };
  for (const [index, row] of task.cases.entries()) {
    const result = referenceBatch.results[index]!;
    if (result.kind === 'infrastructure-error') return {
      status: 'infrastructure-error', gates: { infrastructure: false },
      detail: { category: result.category, message: result.message, phase: 'reference-validation' },
    };
    if (result.kind !== 'ok' || canonical(result.value) !== canonical(row.expected)) referenceFailures.push(row.id);
  }
  if (referenceFailures.length) return {
    status: 'invalid-task', gates: { task_valid: false },
    detail: { reason: 'reference implementation failed exact host-held cases', reference_failed_case_ids: referenceFailures },
  };
  const sizeScore = referenceBytes === lowerBound ? (sourceBytes <= referenceBytes ? 1 : 0)
    : Math.max(0, Math.min(1, (referenceBytes - sourceBytes) / (referenceBytes - lowerBound)));
  Object.assign(detail, {
    reference_bytes: referenceBytes,
    best_known_lower_bound_bytes: lowerBound,
    lower_bound_is_proven_minimum: false,
    bytes_below_best_known_bound: sourceBytes < lowerBound,
  });
  return { status: 'scored', quality: sizeScore, gates: { source_size: true, all_correct: true, reference_validated: true }, detail };
}
