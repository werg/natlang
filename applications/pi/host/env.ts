/**
 * The `env` service: one tool call's execution environment (pi's ExecutionEnv: file system and shell), for the coding
 * tools' natural-language functions and their crisp helpers. Results are `{ value }` or `{ error: { code, message } }`
 * instead of pi's Result objects, and no method takes a Context. Besides the file system it carries the host
 * mechanics the tools share: pi's path normalization (`toolPath`, `readPath`), the per-file mutation lock, shell
 * execution streamed into the call's output (`runShell`), the call's diagnostics, and the diff renderer for UIs.
 */
import type { Context } from '@earendil-works/chord';
import * as Diff from 'diff';
import type { BinaryReader, ExecutionEnv, FileInfo, LineScan } from '../vendor/durable/src/env/index.ts';
import type { ToolDiagnostic, ToolExecutionApi } from '../vendor/durable/src/harness/types.ts';

const DEFAULT_MAX_BYTES = 50 * 1024;
const DEFAULT_MAX_LINES = 2000;
const UNICODE_SPACES = /[  -   　]/g;
const NARROW_NO_BREAK_SPACE = ' ';

export type EnvError = { code: string; message: string; spillPath?: string };
export type EnvResult<T> = { value: T; error?: undefined } | { value?: undefined; error: EnvError };

const NO_ENV: EnvError = { code: 'no_env', message: 'No execution environment is configured' };

function failure(error: unknown): EnvError {
  const record = error as { code?: unknown; message?: unknown; spillPath?: unknown };
  return { code: typeof record?.code === 'string' ? record.code : 'unknown',
    message: error instanceof Error ? error.message : String(error),
    ...(typeof record?.spillPath === 'string' ? { spillPath: record.spillPath } : {}) };
}

type PiResult<T> = { ok: true; value: T } | { ok: false; error: unknown };
const wrap = <T>(result: PiResult<T>): EnvResult<T> => result.ok ? { value: result.value } : { error: failure(result.error) };

/** Tail of the mutation chain of each file, keyed by file system id and canonical path (pi's file-mutation-queue). */
const queues = new Map<string, Promise<void>>();

async function mutationKey(env: ExecutionEnv, path: string, context: Context): Promise<string> {
  const absolute = await env.absolutePath(path, context);
  if (!absolute.ok) throw absolute.error;
  return `${env.id}\0${await canonical(env, absolute.value, context)}`;
}

/** The canonical path; for a file that does not exist yet, its canonical parent joined with its name. */
async function canonical(env: ExecutionEnv, absolutePath: string, context: Context): Promise<string> {
  const result = await env.canonicalPath(absolutePath, context);
  if (result.ok) return result.value;
  if (result.error.code === 'not_supported') return absolutePath;
  if (result.error.code !== 'not_found') throw result.error;
  const parent = await env.joinPath([absolutePath, '..'], context);
  if (!parent.ok) throw parent.error;
  if (parent.value === absolutePath || !absolutePath.startsWith(parent.value)) return absolutePath;
  const name = absolutePath.slice(parent.value.length + (/[/\\]$/.test(parent.value) ? 0 : 1));
  const joined = await env.joinPath([await canonical(env, parent.value, context), name], context);
  if (!joined.ok) throw joined.error;
  return joined.value;
}

/** pi's display diff with line numbers and four lines of context, and the first changed line of the new text. */
function generateDiffString(oldContent: string, newContent: string, contextLines = 4): { diff: string; firstChangedLine: number | undefined } {
  const parts = Diff.diffLines(oldContent, newContent);
  const output: string[] = [];
  const lineNumWidth = String(Math.max(oldContent.split('\n').length, newContent.split('\n').length)).length;
  let oldLineNum = 1, newLineNum = 1, lastWasChange = false;
  let firstChangedLine: number | undefined;
  const context = (line: string) => { output.push(` ${String(oldLineNum).padStart(lineNumWidth, ' ')} ${line}`); oldLineNum++; newLineNum++; };
  const skip = (count: number) => { output.push(` ${''.padStart(lineNumWidth, ' ')} ...`); oldLineNum += count; newLineNum += count; };
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]!;
    const raw = part.value.split('\n');
    if (raw[raw.length - 1] === '') raw.pop();
    if (part.added || part.removed) {
      if (firstChangedLine === undefined) firstChangedLine = newLineNum;
      for (const line of raw) {
        if (part.added) { output.push(`+${String(newLineNum).padStart(lineNumWidth, ' ')} ${line}`); newLineNum++; }
        else { output.push(`-${String(oldLineNum).padStart(lineNumWidth, ' ')} ${line}`); oldLineNum++; }
      }
      lastWasChange = true;
      continue;
    }
    const trailing = i < parts.length - 1 && (parts[i + 1]!.added || parts[i + 1]!.removed);
    if (lastWasChange && trailing) {
      if (raw.length <= contextLines * 2) raw.forEach(context);
      else {
        const leading = raw.slice(0, contextLines), tail = raw.slice(raw.length - contextLines);
        leading.forEach(context);
        skip(raw.length - leading.length - tail.length);
        tail.forEach(context);
      }
    } else if (lastWasChange) {
      const shown = raw.slice(0, contextLines);
      shown.forEach(context);
      if (raw.length - shown.length > 0) skip(raw.length - shown.length);
    } else if (trailing) {
      const skipped = Math.max(0, raw.length - contextLines);
      if (skipped > 0) skip(skipped);
      raw.slice(skipped).forEach(context);
    } else { oldLineNum += raw.length; newLineNum += raw.length; }
    lastWasChange = false;
  }
  return { diff: output.join('\n'), firstChangedLine };
}

/**
 * The env service of one tool call. Locks taken through it are released by `release()`, which the tool's host
 * wrapper calls when the call's natural-language function ends, however it ends.
 */
export function envService(api: ToolExecutionApi, context: Context) {
  const env = api.env;
  const readers = new Map<number, BinaryReader>();
  const held = new Map<number, () => void>();
  let nextHandle = 1;
  const guarded = async <T>(body: (env: ExecutionEnv) => Promise<EnvResult<T>>): Promise<EnvResult<T>> => {
    if (!env) return { error: NO_ENV };
    try { return await body(env); } catch (error) { return { error: failure(error) }; }
  };
  const reader = (handle: number) => {
    const found = readers.get(handle);
    if (!found) throw new Error(`no open reader ${handle}`);
    return found;
  };
  const resolvePath = async (target: ExecutionEnv, path: string): Promise<EnvResult<string>> => {
    const normalized = path.replace(UNICODE_SPACES, ' ');
    return wrap(await target.absolutePath(normalized.startsWith('@') ? normalized.slice(1) : normalized, context));
  };
  const service = {
    /** The working directory commands run in. */
    cwd: (): EnvResult<string> => env ? { value: env.cwd } : { error: NO_ENV },
    toolPath: (path: string) => guarded(target => resolvePath(target, path)),
    readPath: (path: string) => guarded(async target => {
      const resolved = await resolvePath(target, path);
      if (resolved.error) return resolved;
      const absolute = resolved.value;
      const variants = [absolute, absolute.replace(/ (AM|PM)\./gi, `${NARROW_NO_BREAK_SPACE}$1.`), absolute.normalize('NFD'),
        absolute.replace(/'/g, '’'), absolute.normalize('NFD').replace(/'/g, '’')];
      for (const variant of new Set(variants)) {
        const exists = await target.exists(variant, context);
        if (!exists.ok) return { error: failure(exists.error) };
        if (exists.value) return { value: variant };
      }
      return { value: absolute };
    }),
    exists: (path: string) => guarded(async target => wrap(await target.exists(path, context))),
    fileInfo: (path: string): Promise<EnvResult<FileInfo>> => guarded(async target => wrap(await target.fileInfo(path, context))),
    readTextFile: (path: string) => guarded(async target => wrap(await target.readTextFile(path, context))),
    writeFile: (path: string, content: string) => guarded(async target => wrap(await target.writeFile(path, content, context))),
    openReader: (path: string) => guarded(async target => {
      const opened = await target.openBinaryReader(path, undefined, context);
      if (!opened.ok) return { error: failure(opened.error) };
      const handle = nextHandle++;
      readers.set(handle, opened.value);
      return { value: handle };
    }),
    readerInfo: (handle: number) => guarded(async () => wrap(await reader(handle).info(context))),
    readBytes: (handle: number, offset: number, length: number): Promise<EnvResult<Uint8Array>> =>
      guarded(async () => wrap(await reader(handle).read(offset, length, context))),
    scanLines: (handle: number, startLine: number, endLine?: number): Promise<EnvResult<LineScan>> =>
      guarded(async () => wrap(await reader(handle).scanLines({ startLine, ...(endLine === undefined ? {} : { endLine }) }, context))),
    closeReader: async (handle: number): Promise<void> => {
      const found = readers.get(handle);
      readers.delete(handle);
      await found?.close(context);
    },
    lock: (path: string) => guarded(async target => {
      const key = await mutationKey(target, path, context);
      const previous = queues.get(key) ?? Promise.resolve();
      let release = (): void => {};
      const done = new Promise<void>(resolve => { release = resolve; });
      const tail = previous.then(() => done);
      queues.set(key, tail);
      await previous;
      const handle = nextHandle++;
      held.set(handle, () => { release(); if (queues.get(key) === tail) queues.delete(key); });
      return { value: handle };
    }),
    unlock: (handle: number): void => { held.get(handle)?.(); held.delete(handle); },
    runShell: (command: string, timeout?: number): Promise<EnvResult<{ exitCode: number; spillPath?: string }>> => guarded<{ exitCode: number; spillPath?: string }>(async target => {
      const result = await target.exec(command, {
        cwd: target.cwd, env: {}, inheritEnv: true,
        ...(timeout === undefined || timeout === null ? {} : { timeout }),
        onOutput: (text, _context, info) => api.output(text, info.skipped),
        spill: { afterBytes: DEFAULT_MAX_BYTES, afterLines: DEFAULT_MAX_LINES },
        ...(api.outputWindow === undefined ? {} : { window: api.outputWindow }),
      }, context);
      if (result.ok) return { value: { exitCode: result.value.exitCode, ...(result.value.spillPath === undefined ? {} : { spillPath: result.value.spillPath }) } };
      if (result.error.code === 'aborted' && context.abortSignal?.aborted) throw result.error;
      return { error: failure(result.error) };
    }),
    diagnostic: (diagnostic: ToolDiagnostic): void => { api.diagnostic(diagnostic); },
    renderDiff: (path: string, before: string, after: string): { diff: string; patch: string; firstChangedLine?: number } => {
      const { diff, firstChangedLine } = generateDiffString(before, after);
      const patch = Diff.createTwoFilesPatch(path, path, before, after, undefined, undefined, { context: 4, headerOptions: Diff.FILE_HEADERS_ONLY });
      return { diff, patch, ...(firstChangedLine === undefined ? {} : { firstChangedLine }) };
    },
    /** Release every lock and reader this call still holds (host only; not declared to the model). */
    async release(): Promise<void> {
      for (const unlock of held.values()) unlock();
      held.clear();
      for (const handle of [...readers.keys()]) await service.closeReader(handle);
    },
  };
  return service;
}

export type EnvService = ReturnType<typeof envService>;

/** What the model reads of the env service. */
export const ENV_DECLARATION = `/**
 * The execution environment of this tool call: its file system and shell. Results are { value } or { error: { code,
 * message } }; error.code is one of not_found, permission_denied, not_directory, is_directory, invalid, not_supported,
 * aborted, timeout, shell_unavailable, spawn_error, unknown, or no_env (no environment is configured). A tool that meets
 * an error it has no rule for fails with error.message exactly.
 */
/** The directory commands run in. */
export function cwd(): { value?: string; error?: { code: string; message: string } };
/** pi's tool path: special Unicode spaces become spaces, a leading @ is dropped, and the path is made absolute against cwd. */
export function toolPath(path: string): Promise<{ value?: string; error?: { code: string; message: string } }>;
/** toolPath, then the first existing spelling variant (narrow no-break space before AM/PM, NFD, curly apostrophe), else toolPath. */
export function readPath(path: string): Promise<{ value?: string; error?: { code: string; message: string } }>;
export function exists(path: string): Promise<{ value?: boolean; error?: { code: string; message: string } }>;
/** kind is "file", "directory" or "symlink"; size in bytes; mtimeMs. */
export function fileInfo(path: string): Promise<{ value?: { name: string; path: string; kind: "file" | "directory" | "symlink"; size: number; mtimeMs: number }; error?: { code: string; message: string } }>;
/** The whole file decoded as UTF-8 (a byte-order mark stays in the text). */
export function readTextFile(path: string): Promise<{ value?: string; error?: { code: string; message: string } }>;
/** Create or overwrite the file, creating parent directories. */
export function writeFile(path: string, content: string): Promise<{ value?: null; error?: { code: string; message: string } }>;
/** Open the file for positional reads (used with readerInfo and the read tool's selection helper); close it with closeReader. */
export function openReader(path: string): Promise<{ value?: number; error?: { code: string; message: string } }>;
/** Size and mtimeMs of the opened file. */
export function readerInfo(reader: number): Promise<{ value?: { name: string; path: string; kind: string; size: number; mtimeMs: number }; error?: { code: string; message: string } }>;
export function closeReader(reader: number): Promise<void>;
/**
 * Wait for, then hold, this process's mutation lock of path's file (write and edit take it around read-modify-write).
 * Returns a lock number for unlock. Locks still held when the tool's function ends are released.
 */
export function lock(path: string): Promise<{ value?: number; error?: { code: string; message: string } }>;
export function unlock(lock: number): void;
/**
 * Run command through the environment's shell in cwd, with an optional timeout in seconds. Its output streams into
 * this call's output, which becomes the result content when the result has no content of its own. value.exitCode is
 * the exit code; spillPath (on value, or on error for a timeout or abort) names a file with the complete output when it
 * was longer than 50KB or 2000 lines. error.code "timeout": the timeout passed; "aborted": the command was aborted.
 */
export function runShell(command: string, timeout?: number): Promise<{ value?: { exitCode: number; spillPath?: string }; error?: { code: string; message: string; spillPath?: string } }>;
/** Record a model-visible remark about this call, before its result. */
export function diagnostic(diagnostic: { severity: "info" | "warn" | "error"; message: string; code?: string }): void;
/** pi's renderings of a change for UIs: a line-numbered diff, a unified patch, and the first changed line of after. */
export function renderDiff(path: string, before: string, after: string): { diff: string; patch: string; firstChangedLine?: number };`;
