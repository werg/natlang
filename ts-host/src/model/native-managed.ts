/**
 * A natlang-owned native reference server (docs/NATIVE_MODELS.md): a profile's `native` block names a pinned native
 * checkpoint, and the first model turn starts `python -m natlang_neuralese.serve` for it, or reuses a server already
 * serving that exact checkpoint at the profile's port. The server stops with the command that started it.
 * `natlang setup --native SOURCE` pins a checkpoint (a content-addressed copy, so a learner rewriting its own file
 * never changes what is served) and writes the profile.
 */
import { spawn, spawnSync, type ChildProcess } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, renameSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultNatlangDataDirectory } from '../package/store.js';

export type NativeModelSettings = {
  /** The pinned complete native checkpoint the server loads. */
  checkpoint: string;
  /** Its SHA-256; the running server must report exactly this checkpoint. */
  sha256: string;
  /** Where the pin came from (a checkpoint pointer, a file or a `nativeSources` name) and its training step. */
  source?: string; step?: number;
  /** The SHA-256 of the training checkpoint the pin was made from (an exact copy: equal to `sha256`). */
  sourceSha256?: string;
  /** Interpreter with natlang_neuralese importable, and extra PYTHONPATH entries. */
  python?: string; pythonPath?: string[];
  device?: string; dtype?: 'float32' | 'bfloat16'; memoryGb?: number; contextTokens?: number;
  /** Fixed local port; default derived from the checkpoint, so every natlang process finds the same server. */
  port?: number;
  /** Keep native blocks on disk (`--store-dir`): true for the server default, or a directory. */
  storeDir?: string | true;
  /** Memory for reusing prompt-prefix state across calls (`--prefix-cache-gb`; default 2, 0 turns it off). Calls of
   * one function share their long instructions, so later calls prefill only what differs. */
  prefixCacheGb?: number;
  /** Further server arguments. */
  args?: string[];
  /** Prefix that launches the server command, e.g. a memory ledger `run ... --`, and the command that stops what it
   * launched. `{port}` and `{unit}` are substituted. Without `stop`, the server is this process's child. */
  launcher?: string[]; stop?: string[];
  /** How long the first start may take (default 15 minutes: loading and a launcher's admission queue). */
  startTimeoutMs?: number;
};

const SHA256 = /^[0-9a-f]{64}$/;

export function validateNativeSettings(native: NativeModelSettings): void {
  if (!native || typeof native !== 'object' || Array.isArray(native)) throw new TypeError('native must be an object');
  if (typeof native.checkpoint !== 'string' || !native.checkpoint) throw new Error('native.checkpoint must name the pinned checkpoint; run natlang setup --native SOURCE');
  if (typeof native.sha256 !== 'string' || !SHA256.test(native.sha256)) throw new Error('native.sha256 must be the checkpoint SHA-256');
  if (native.port !== undefined && (!Number.isSafeInteger(native.port) || native.port < 1 || native.port > 65535))
    throw new Error('native.port must be a port number');
  if (native.dtype !== undefined && !['float32', 'bfloat16'].includes(native.dtype)) throw new Error('native.dtype must be float32 or bfloat16');
  if (native.prefixCacheGb !== undefined && !(Number.isFinite(native.prefixCacheGb) && native.prefixCacheGb >= 0))
    throw new Error('native.prefixCacheGb must be a number >= 0');
  for (const key of ['memoryGb', 'contextTokens', 'startTimeoutMs'] as const)
    if (native[key] !== undefined && !(Number.isFinite(native[key]) && native[key]! > 0)) throw new Error(`native.${key} must be positive`);
  for (const key of ['args', 'launcher', 'stop', 'pythonPath'] as const)
    if (native[key] !== undefined && (!Array.isArray(native[key]) || native[key]!.some(value => typeof value !== 'string')))
      throw new Error(`native.${key} must be an array of strings`);
  if (native.stop && !native.launcher) throw new Error('native.stop requires native.launcher');
}

export function nativePort(native: NativeModelSettings): number {
  return native.port ?? 20000 + parseInt(native.sha256.slice(0, 8), 16) % 10000;
}
export const nativeEndpoint = (native: NativeModelSettings) => `http://127.0.0.1:${nativePort(native)}`;
const unitName = (native: NativeModelSettings) => `natlang-native-${native.sha256.slice(0, 12)}-${nativePort(native)}`;
const substitute = (values: string[], native: NativeModelSettings) =>
  values.map(value => value.replaceAll('{port}', String(nativePort(native))).replaceAll('{unit}', unitName(native)));

/** The checkout this CLI was built from, when it is one (its Python training environment serves native models). */
function checkoutRoot(): string | undefined {
  const root = resolve(dirname(fileURLToPath(import.meta.url)), '../../..');
  return existsSync(join(root, 'training', 'neuralese', 'natlang_neuralese')) ? root : undefined;
}

/** Interpreter and PYTHONPATH defaults: the checkout's `.venv-neuralese` and training packages when present. */
export function defaultNativePython(environment: NodeJS.ProcessEnv = process.env): Pick<NativeModelSettings, 'python' | 'pythonPath'> {
  if (environment.NATLANG_NATIVE_PYTHON) return { python: environment.NATLANG_NATIVE_PYTHON };
  const root = checkoutRoot();
  if (!root) return {};
  const venv = join(root, '.venv-neuralese', 'bin', 'python');
  return { ...(existsSync(venv) ? { python: venv } : {}), pythonPath: [join(root, 'training', 'neuralese'), root] };
}

export function nativeServerCommand(native: NativeModelSettings, environment: NodeJS.ProcessEnv = process.env): { command: string[]; env: NodeJS.ProcessEnv } {
  const python = native.python ?? defaultNativePython(environment).python ?? 'python3';
  const pythonPath = native.pythonPath ?? (native.python ? [] : defaultNativePython(environment).pythonPath ?? []);
  const server = [python, '-u', '-m', 'natlang_neuralese.serve', '--native-checkpoint', native.checkpoint,
    '--host', '127.0.0.1', '--port', String(nativePort(native)), '--device', native.device ?? 'auto',
    ...(native.dtype ? ['--dtype', native.dtype] : []), ...(native.memoryGb ? ['--memory-gb', String(native.memoryGb)] : []),
    ...(native.contextTokens ? ['--ctx-size', String(native.contextTokens)] : []),
    ...((native.prefixCacheGb ?? 2) > 0 ? ['--prefix-cache-gb', String(native.prefixCacheGb ?? 2)] : []),
    ...(native.storeDir === true ? ['--store-dir'] : native.storeDir ? ['--store-dir', native.storeDir] : []), ...(native.args ?? [])];
  const joined = [...pythonPath, ...(environment.PYTHONPATH ? [environment.PYTHONPATH] : [])].join(':');
  const env = { ...environment, ...(joined ? { PYTHONPATH: joined } : {}) };
  // A launcher (a ledger unit) does not inherit this environment: PYTHONPATH goes through `env`.
  const command = native.launcher ? [...substitute(native.launcher, native), ...(joined ? ['env', `PYTHONPATH=${joined}`] : []), ...server] : server;
  return { command, env };
}

type Served = { checkpointSha256?: string; reachable: boolean };
async function servedCheckpoint(endpoint: string): Promise<Served> {
  try {
    const response = await fetch(endpoint + '/v1/neuralese/info', { signal: AbortSignal.timeout(5000) });
    if (!response.ok) return { reachable: true };
    const info = await response.json() as { native_state?: { checkpoint_sha256?: string } };
    return { reachable: true, checkpointSha256: info.native_state?.checkpoint_sha256 };
  } catch { return { reachable: false }; }
}

type Owned = { child: ChildProcess | null; native: NativeModelSettings; stderr: string };
// Servers this process started, by endpoint; one start per endpoint even when several sessions ask at once.
const owned = new Map<string, Owned>();
const starting = new Map<string, Promise<void>>();

/** Ensure the profile's native server is serving its exact checkpoint: reuse one that is, else start it. */
export function ensureNativeServer(native: NativeModelSettings, environment: NodeJS.ProcessEnv = process.env,
  error: NodeJS.WritableStream = process.stderr): Promise<void> {
  const endpoint = nativeEndpoint(native);
  let pending = starting.get(endpoint);
  if (!pending) {
    pending = start(native, endpoint, environment, error).finally(() => starting.delete(endpoint));
    starting.set(endpoint, pending);
  }
  return pending;
}

function mismatch(native: NativeModelSettings, endpoint: string, served: Served): Error {
  return new Error(`native server at ${endpoint} serves ${served.checkpointSha256 ? `checkpoint ${served.checkpointSha256.slice(0, 12)}` : 'no native checkpoint'}, ` +
    `not ${native.sha256.slice(0, 12)}; stop it or set native.port in the profile`);
}

async function start(native: NativeModelSettings, endpoint: string, environment: NodeJS.ProcessEnv, error: NodeJS.WritableStream): Promise<void> {
  const current = await servedCheckpoint(endpoint);
  if (current.checkpointSha256 === native.sha256) return;
  if (current.reachable) throw mismatch(native, endpoint, current);
  if (!existsSync(native.checkpoint)) throw new Error(`native checkpoint ${native.checkpoint} is missing; run natlang setup --native ${native.source ?? 'SOURCE'}`);
  const { command, env } = nativeServerCommand(native, environment);
  error.write(`natlang: starting native model ${native.sha256.slice(0, 12)}${native.step !== undefined ? ` (step ${native.step})` : ''} at ${endpoint}; ` +
    'loading takes a minute (natlang serve keeps it running between commands)\n');
  const record: Owned = { child: null, native, stderr: '' };
  const child = spawn(command[0]!, command.slice(1), { env, stdio: ['ignore', 'ignore', 'pipe'] });
  record.child = child;
  owned.set(endpoint, record);
  child.stderr?.on('data', chunk => { record.stderr = (record.stderr + String(chunk)).slice(-16000); });
  let spawnError: Error | undefined;
  child.once('error', failure => { spawnError = failure; });
  const deadline = Date.now() + (native.startTimeoutMs ?? 15 * 60_000);
  try {
    while (Date.now() < deadline) {
      if (spawnError) throw new Error(`could not start the native server (${command[0]}): ${spawnError.message}`);
      const served = await servedCheckpoint(endpoint);
      if (served.checkpointSha256 === native.sha256) return;
      if (served.reachable && served.checkpointSha256) throw mismatch(native, endpoint, served);
      // A launcher that hands the server to a service manager exits 0 while the server keeps loading.
      if (child.exitCode !== null && (child.exitCode !== 0 || !native.launcher)) {
        if (served.reachable) continue;
        throw new Error(`native server exited during startup (status ${child.exitCode})\n${record.stderr.trim()}`);
      }
      await new Promise(resolveWait => setTimeout(resolveWait, 500));
    }
    throw new Error(`native server did not become ready within ${Math.round((native.startTimeoutMs ?? 15 * 60_000) / 1000)} s\n${record.stderr.trim()}`);
  } catch (failure) { await stopNativeServer(endpoint); throw failure; }
}

/** Stop the server at `endpoint` if this process started it; a reused server keeps running. */
export async function stopNativeServer(endpoint: string): Promise<void> {
  const record = owned.get(endpoint);
  if (!record) return;
  owned.delete(endpoint);
  if (record.native.stop) spawnSync(record.native.stop[0]!, substitute(record.native.stop, record.native).slice(1), { stdio: 'ignore', timeout: 60_000 });
  const child = record.child;
  if (!child || child.exitCode !== null || child.signalCode !== null) return;
  const exited = new Promise<void>(resolveExit => child.once('exit', () => resolveExit()));
  child.kill('SIGTERM');
  let timer: NodeJS.Timeout | undefined;
  await Promise.race([exited, new Promise<void>(resolveTimeout => { timer = setTimeout(resolveTimeout, 15_000); })]);
  if (timer) clearTimeout(timer);
  if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
}
export const ownsNativeServer = (endpoint: string) => owned.has(endpoint);
process.once('exit', () => { for (const record of owned.values()) {
  if (record.native.stop) spawnSync(record.native.stop[0]!, substitute(record.native.stop, record.native).slice(1), { stdio: 'ignore', timeout: 30_000 });
  else if (record.child && record.child.exitCode === null) record.child.kill('SIGTERM');
} });

// ---- Pinning checkpoints ----

export type CheckpointPointer = { path: string; sha256?: string; bytes?: number; step?: number };

/**
 * A checkpoint source: a `.pt` file, or a JSON pointer to one. Pointers name the file as `local_full_state`,
 * `checkpoint` or top-level `path`, with optional `sha256`, `bytes` and `global_step`/`step`.
 */
export function readCheckpointPointer(source: string): CheckpointPointer {
  const path = resolve(source);
  if (!existsSync(path)) throw new Error(`native source ${source} does not exist`);
  if (!path.endsWith('.json')) return { path };
  const value = JSON.parse(readFileSync(path, 'utf8')) as Record<string, unknown>;
  const reference = (value.local_full_state ?? value.checkpoint ?? value) as Record<string, unknown>;
  if (typeof reference.path !== 'string') throw new Error(`${source} names no checkpoint (local_full_state, checkpoint or path)`);
  const step = value.global_step ?? value.step ?? reference.step;
  return { path: reference.path, ...(typeof reference.sha256 === 'string' ? { sha256: reference.sha256 } : {}),
    ...(typeof reference.bytes === 'number' ? { bytes: reference.bytes } : {}), ...(typeof step === 'number' ? { step } : {}) };
}

export const nativeModelsDirectory = (environment: NodeJS.ProcessEnv = process.env) =>
  environment.NATLANG_NATIVE_MODELS ? resolve(environment.NATLANG_NATIVE_MODELS) : join(defaultNatlangDataDirectory(environment), 'native-models');

type Pin = { checkpoint: string; sha256: string; bytes: number; sourceSha256: string; step?: number; reused: boolean };

// An exact byte copy through the trainers' verified copy (train/checkpoint_offload.py: the opened file's identity, size
// and SHA-256 are checked, written through a partial name and fsynced). The pin keeps the training checkpoint's own
// identity, which is what tuned releases, RL receipts and the native server's checkpoint_sha256 bind.
const PIN_SCRIPT = `import json, sys
from natlang_neuralese.train.checkpoint_offload import verified_copy
from natlang_neuralese.common.hashing import sha256_file_hex_cached
source, dest, expected = sys.argv[1], sys.argv[2], sys.argv[3]
source_sha = sha256_file_hex_cached(source)
if expected and source_sha != expected: sys.exit(f'{source} hashes to {source_sha}, its pointer says {expected}')
receipt = verified_copy(source, dest)
if receipt['sha256'] != source_sha: sys.exit(f'{source} changed while pinning')
print(json.dumps({'sha256': receipt['sha256'], 'bytes': receipt['bytes'], 'source_sha256': source_sha}))`;

/** Pin the source checkpoint as `<native-models>/<sha256>.pt`, an exact copy; reuse an earlier pin of the same source. */
export async function pinNativeCheckpoint(pointer: CheckpointPointer, python: Pick<NativeModelSettings, 'python' | 'pythonPath'> = {},
  environment: NodeJS.ProcessEnv = process.env, error: NodeJS.WritableStream = process.stderr): Promise<Pin> {
  const directory = nativeModelsDirectory(environment), index = join(directory, 'pins.json');
  mkdirSync(directory, { recursive: true });
  const pins = (existsSync(index) ? JSON.parse(readFileSync(index, 'utf8')) : {}) as Record<string, Omit<Pin, 'reused'>>;
  const known = pointer.sha256 ? pins[pointer.sha256] : undefined;
  if (known && existsSync(known.checkpoint) && statSync(known.checkpoint).size === known.bytes) return { ...known, reused: true };
  const size = statSync(pointer.path).size;
  if (pointer.bytes !== undefined && size !== pointer.bytes) throw new Error(`${pointer.path} has ${size} bytes, its pointer says ${pointer.bytes}`);
  error.write(`natlang: pinning ${pointer.path} (${(size / 2 ** 30).toFixed(1)} GiB)\n`);
  const temporary = join(directory, `.pin-${process.pid}-${Date.now()}.pt`);
  const { command, env } = nativeServerCommand({ checkpoint: '', sha256: '0'.repeat(64), ...python }, environment);
  try {
    const result = await new Promise<{ status: number | null; stdout: string; stderr: string }>((resolveRun, reject) => {
      const child = spawn(command[0]!, ['-c', PIN_SCRIPT, pointer.path, temporary, pointer.sha256 ?? ''], { env, stdio: ['ignore', 'pipe', 'pipe'] });
      let stdout = '', stderr = '';
      child.stdout.on('data', chunk => { stdout += chunk; }); child.stderr.on('data', chunk => { stderr = (stderr + chunk).slice(-8000); });
      child.once('error', reject); child.once('close', status => resolveRun({ status, stdout, stderr }));
    });
    if (result.status !== 0) throw new Error(`pinning ${pointer.path} failed\n${result.stderr.trim()}`);
    const receipt = JSON.parse(result.stdout.trim().split('\n').pop()!) as { sha256: string; bytes: number; source_sha256: string };
    const pinned = join(directory, `${receipt.sha256}.pt`);
    renameSync(temporary, pinned);
    const pin = { checkpoint: pinned, sha256: receipt.sha256, bytes: receipt.bytes, sourceSha256: receipt.source_sha256,
      ...(pointer.step !== undefined ? { step: pointer.step } : {}) };
    pins[receipt.source_sha256] = pin;
    const indexTemporary = `${index}.${process.pid}.tmp`;
    writeFileSync(indexTemporary, JSON.stringify(pins, null, 1) + '\n'); renameSync(indexTemporary, index);
    return { ...pin, reused: false };
  } finally { rmSync(temporary, { force: true }); }
}

/** Forget a pin no profile uses any more: its file and its index entry. */
export function removeNativePin(checkpoint: string, environment: NodeJS.ProcessEnv = process.env): boolean {
  const directory = nativeModelsDirectory(environment), index = join(directory, 'pins.json');
  if (dirname(resolve(checkpoint)) !== directory || !existsSync(checkpoint)) return false;
  rmSync(checkpoint);
  rmSync(join(directory, `gguf-${basename(checkpoint).slice(0, 12)}`), { recursive: true, force: true });
  if (existsSync(index)) {
    const pins = JSON.parse(readFileSync(index, 'utf8')) as Record<string, { checkpoint: string }>;
    for (const [key, pin] of Object.entries(pins)) if (pin.checkpoint === checkpoint) delete pins[key];
    writeFileSync(`${index}.${process.pid}.tmp`, JSON.stringify(pins, null, 1) + '\n'); renameSync(`${index}.${process.pid}.tmp`, index);
  }
  return true;
}

/** A newer checkpoint behind the pin's source pointer, for a one-line notice; undefined when current or unknown. */
export function newerNativeCheckpoint(native: NativeModelSettings, sources: Record<string, string> = {}): CheckpointPointer | undefined {
  if (!native.source) return undefined;
  try {
    const path = sources[native.source] ?? native.source;
    if (!path.endsWith('.json')) return undefined;
    const pointer = readCheckpointPointer(path);
    if (!pointer.sha256 || pointer.sha256 === (native.sourceSha256 ?? native.sha256)) return undefined;
    if (pointer.step !== undefined && native.step !== undefined && pointer.step <= native.step) return undefined;
    return pointer;
  } catch { return undefined; }
}

// ---- GGUF exports for the managed llama.cpp backend ----

export type NativeGguf = { modelPath: string; sha256: string; template: string; templateKwargs: Record<string, unknown>;
  logitBias: Record<string, number>; reused: boolean };

/** Export (or reuse) a pinned checkpoint as GGUF beside its pin (export/native_gguf.py: learned rows merged, the
 * native chat template, the producer marker banned). Ordinary-token execution only; see docs/NATIVE_MODELS.md. */
export async function exportNativeGguf(native: Pick<NativeModelSettings, 'checkpoint' | 'sha256' | 'python' | 'pythonPath'>, outtype: string,
  environment: NodeJS.ProcessEnv = process.env, error: NodeJS.WritableStream = process.stderr): Promise<NativeGguf> {
  if (!['f32', 'bf16', 'f16', 'q8_0'].includes(outtype)) throw new Error('--gguf must be f32, bf16, f16 or q8_0');
  const directory = join(dirname(native.checkpoint), `gguf-${native.sha256.slice(0, 12)}`), manifestPath = join(directory, 'manifest.json');
  type Manifest = { checkpoint_sha256: string; files: { outtype: string; file: string; sha256: string }[];
    serving: { chat_template: string; chat_template_kwargs: Record<string, unknown>; logit_bias: Record<string, number> } };
  const read = (): Manifest | undefined => { try { return JSON.parse(readFileSync(manifestPath, 'utf8')) as Manifest; } catch { return undefined; } };
  const found = (manifest: Manifest | undefined) => manifest?.checkpoint_sha256 === native.sha256 ?
    manifest.files.find(item => item.outtype === outtype && existsSync(join(directory, item.file))) : undefined;
  let manifest = read(), file = found(manifest), reused = true;
  if (!file) {
    reused = false;
    error.write(`natlang: exporting ${native.sha256.slice(0, 12)} as ${outtype} GGUF\n`);
    // The exporter runs here, with the profile's interpreter: never through the server's launcher.
    const { command, env } = nativeServerCommand({ checkpoint: native.checkpoint, sha256: native.sha256,
      ...(native.python ? { python: native.python } : {}), ...(native.pythonPath ? { pythonPath: native.pythonPath } : {}) }, environment);
    const result = spawnSync(command[0]!, ['-m', 'natlang_neuralese.export.native_gguf', native.checkpoint, directory, '--outtype', outtype],
      { env: { ...env, CUDA_VISIBLE_DEVICES: env.CUDA_VISIBLE_DEVICES ?? '' }, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], maxBuffer: 64 << 20 });
    if (result.status !== 0) throw new Error(`GGUF export failed\n${String(result.stderr).slice(-4000).trim()}`);
    manifest = read(); file = found(manifest);
    if (!file) throw new Error(`GGUF export wrote no ${outtype} file to ${directory}`);
  }
  return { modelPath: join(directory, file.file), sha256: file.sha256, template: join(directory, manifest!.serving.chat_template),
    templateKwargs: manifest!.serving.chat_template_kwargs, logitBias: manifest!.serving.logit_bias, reused };
}

// ---- Tuned programs serve the base they were tuned on ----

/**
 * Where the checkpoint with SHA-256 `sha256` can be read: an existing pin, a tune job of the program that trained on it
 * (`.natlang/tune/jobs/<key>/plan.json` records the student checkpoint and its exact base), or a `nativeSources`
 * pointer to it. A tune plan names a file without a hash of its own, so pinning verifies it against `sha256`.
 */
export function findNativeCheckpoint(sha256: string, options: { programRoot?: string; sources?: Record<string, string> } = {},
  environment: NodeJS.ProcessEnv = process.env): { pinned?: string; pointer?: CheckpointPointer; step?: number } | undefined {
  const directory = nativeModelsDirectory(environment);
  const pinned = join(directory, `${sha256}.pt`);
  let step: number | undefined;
  try { step = (JSON.parse(readFileSync(join(directory, 'pins.json'), 'utf8')) as Record<string, { step?: number }>)[sha256]?.step; } catch { /* no index */ }
  if (existsSync(pinned)) return { pinned, ...(step !== undefined ? { step } : {}) };
  if (options.programRoot) {
    const jobs = join(options.programRoot, '.natlang', 'tune', 'jobs');
    for (const job of existsSync(jobs) ? readdirSync(jobs).sort() : []) {
      try {
        const plan = JSON.parse(readFileSync(join(jobs, job, 'plan.json'), 'utf8')) as { base?: { checkpoint_sha256?: string }; student?: { checkpoint?: string } };
        if (plan.base?.checkpoint_sha256 === sha256 && plan.student?.checkpoint && existsSync(plan.student.checkpoint))
          return { pointer: { path: plan.student.checkpoint, sha256 } };
      } catch { /* not a readable plan */ }
    }
  }
  for (const source of Object.values(options.sources ?? {})) {
    try { const pointer = readCheckpointPointer(source); if (pointer.sha256 === sha256 && existsSync(pointer.path)) return { pointer, ...(pointer.step !== undefined ? { step: pointer.step } : {}) }; }
    catch { /* an unreadable pointer names nothing */ }
  }
  return undefined;
}
