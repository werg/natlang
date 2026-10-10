/**
 * pi-durable's ExecutionEnv over a natlang Folder: the browser host's workspace (browser.ts), where there is no file
 * system and no process to spawn. Platform-neutral, so the Node host and tests run it too.
 *
 * The folder is mounted at `/workspace`, as natlang's folder shell sees it (`runFolderBash`, just-bash over the same
 * folder): the read, write and edit tools change the folder through the file system methods, and bash commands run in
 * the shell over it, so both see the same files. `/tmp` is a second folder of the environment's own, for temporary
 * files (the complete output of a long command); the shell does not see it. Every other path does not exist, and
 * writing there is refused. Errors carry Node's codes and messages, as the coding tools show them to the model.
 *
 * What a folder does not have: directories exist only while something is in them, so an empty one made by
 * `createDir` lives in the environment (the shell does not see it, nor one made by `mkdir` after its command);
 * modification times are the environment's clock, advanced when it or one of its commands changes a file (a change
 * made to the folder outside the environment keeps the time); symbolic links do not exist; `watch` is not supported.
 * Environments over one folder share its namespace (`id`), its `/tmp` and its directories and times.
 */
import type { Context } from '@earendil-works/chord';
import { Folder, runFolderBash } from 'natlang:runtime';
import {
  type BinaryReader, type DirReader, type ExecutionEnv, ExecutionError, err, FileError, type FileInfo, type FileKind,
  type FileWatcher, type LineScan, ok, type Result, type ShellExecOptions, type ShellExecResult, type ShellOutputInfo,
  StreamDecoder, type TextLine, type TextLineReader,
} from '../vendor/durable/src/env/index.ts';
import { LineScanner } from '../vendor/durable/src/env/line-scan.ts';
import { basenamePath, dirnamePath, joinPath, resolvePath } from './paths.ts';

/** Where the folder is mounted: natlang's folder shell runs its commands there. */
export const WORKSPACE = '/workspace';
/** The environment's temporary files. */
export const TEMP = '/tmp';
const MAX_TIMEOUT_MS = 2_147_483_647;

export type FolderEnvOptions = {
  /** The workspace, mounted at /workspace. */
  folder: Folder;
  /** The working directory, under /workspace (default /workspace). */
  cwd?: string;
  /** Network access for commands (curl and the like through fetch); default off. */
  network?: boolean;
};

/** What every environment over one folder shares. */
type Mounts = { id: string; temp: Folder; dirs: Set<string>; times: Map<string, number>; clock: number };
const mounts = new WeakMap<Folder, Mounts>();
let folders = 0;

const decodeText = (bytes: Uint8Array) => new TextDecoder('utf-8', { ignoreBOM: true }).decode(bytes);
const bytesOf = (content: string | Uint8Array) => typeof content === 'string' ? new TextEncoder().encode(content) : content;
const randomName = () => crypto.randomUUID();
const quote = (word: string) => `'${word.replace(/'/g, `'\\''`)}'`;
const error = (code: FileError['code'], errno: string, text: string, syscall: string, path: string) =>
  new FileError(code, `${errno}: ${text}, ${syscall} '${path}'`, path);
const notFound = (syscall: string, path: string) => error('not_found', 'ENOENT', 'no such file or directory', syscall, path);
const isDirectory = (syscall: string, path: string) => error('is_directory', 'EISDIR', 'illegal operation on a directory', syscall, path);
const notDirectory = (syscall: string, path: string) => error('not_directory', 'ENOTDIR', 'not a directory', syscall, path);
const denied = (syscall: string, path: string) => error('permission_denied', 'EACCES', 'permission denied', syscall, path);
const abortedFile = (path?: string) => new FileError('aborted', 'aborted', path);

/** A file's bytes as they were when it was opened, for positional reads (`BinaryReader`). */
class SnapshotReader implements BinaryReader {
  #closed = false;
  private readonly bytes: Uint8Array;
  private readonly meta: FileInfo;
  constructor(bytes: Uint8Array, meta: FileInfo) { this.bytes = bytes; this.meta = meta; }
  #check<T>(context: Context): Result<T, FileError> | undefined {
    if (context.abortSignal?.aborted) return err(abortedFile(this.meta.path));
    if (this.#closed) return err(new FileError('invalid', 'Binary reader is closed', this.meta.path));
    return undefined;
  }
  async info(context: Context): Promise<Result<FileInfo, FileError>> { return this.#check<FileInfo>(context) ?? ok({ ...this.meta }); }
  async read(offset: number, length: number, context: Context): Promise<Result<Uint8Array, FileError>> {
    const failed = this.#check<Uint8Array>(context);
    if (failed) return failed;
    if (!Number.isSafeInteger(offset) || offset < 0 || !Number.isSafeInteger(length) || length < 0)
      return err(new FileError('invalid', 'Offset and length must be non-negative safe integers', this.meta.path));
    return ok(this.bytes.slice(offset, offset + length));
  }
  async scanLines(options: { startLine: number; endLine?: number }, context: Context): Promise<Result<LineScan, FileError>> {
    const failed = this.#check<LineScan>(context);
    if (failed) return failed;
    let scanner: LineScanner;
    try { scanner = new LineScanner(options.startLine, options.endLine); }
    catch { return err(new FileError('invalid', 'Invalid line range', this.meta.path)); }
    scanner.push(this.bytes);
    return ok(scanner.finish());
  }
  async close(): Promise<void> { this.#closed = true; }
}

/** Lines of a file's text as it was when opened, split at LF as pi's strict line reader splits them. */
class SnapshotLineReader implements TextLineReader {
  #at = 0;
  #closed = false;
  readonly #text: string;
  private readonly path: string;
  constructor(bytes: Uint8Array, path: string) {
    this.path = path;
    const decoder = new StreamDecoder();
    this.#text = decoder.decode(bytes) + decoder.decode();
  }
  async readLine(context: Context): Promise<Result<TextLine | undefined, FileError>> {
    if (context.abortSignal?.aborted) return err(abortedFile(this.path));
    if (this.#closed) return err(new FileError('invalid', 'Text line reader is closed', this.path));
    if (this.#at >= this.#text.length) return ok(undefined);
    const newline = this.#text.indexOf('\n', this.#at);
    const line = newline === -1 ? { text: this.#text.slice(this.#at), terminated: false } : { text: this.#text.slice(this.#at, newline), terminated: true };
    this.#at = newline === -1 ? this.#text.length : newline + 1;
    return ok(line);
  }
  async close(): Promise<void> { this.#closed = true; }
}

class ListReader implements DirReader {
  #at = 0;
  private readonly entries: FileInfo[];
  private readonly path: string;
  constructor(entries: FileInfo[], path: string) { this.entries = entries; this.path = path; }
  async next(maxEntries: number, context: Context): Promise<Result<{ entries: FileInfo[]; done: boolean }, FileError>> {
    if (context.abortSignal?.aborted) return err(abortedFile(this.path));
    if (!Number.isSafeInteger(maxEntries) || maxEntries <= 0) return err(new FileError('invalid', 'maxEntries must be a positive safe integer', this.path));
    const entries = this.entries.slice(this.#at, this.#at + maxEntries);
    this.#at += entries.length;
    return ok({ entries, done: this.#at >= this.entries.length });
  }
  async close(): Promise<void> { this.#at = this.entries.length; }
}

export class FolderExecutionEnv implements ExecutionEnv {
  readonly id: string;
  cwd: string;
  readonly folder: Folder;
  readonly #mounts: Mounts;
  readonly #network: boolean;
  readonly #running = new Set<AbortController>();

  constructor(options: FolderEnvOptions) {
    this.folder = options.folder;
    let shared = mounts.get(options.folder);
    if (!shared) {
      shared = { id: `folder:${++folders}`, temp: new Folder({}), dirs: new Set(), times: new Map(), clock: 0 };
      mounts.set(options.folder, shared);
    }
    this.#mounts = shared;
    this.id = shared.id;
    this.cwd = resolvePath(WORKSPACE, options.cwd ?? WORKSPACE);
    this.#network = options.network ?? false;
  }

  // --- paths ------------------------------------------------------------------------------------------------------

  #resolve(path: string): string {
    let normalized = path;
    if (normalized === '~' || normalized.startsWith('~/')) normalized = WORKSPACE + normalized.slice(1);
    else if (normalized.startsWith('file://')) { try { normalized = decodeURIComponent(new URL(normalized).pathname); } catch { /* an ordinary path */ } }
    return resolvePath(this.cwd, normalized);
  }
  /** The folder holding an absolute path and the path within it, or undefined outside both mounts. */
  #locate(absolute: string): { folder: Folder; path: string } | undefined {
    for (const [mount, folder] of [[WORKSPACE, this.folder], [TEMP, this.#mounts.temp]] as const) {
      if (absolute === mount) return { folder, path: '' };
      if (absolute.startsWith(`${mount}/`)) return { folder, path: absolute.slice(mount.length + 1) };
    }
    return undefined;
  }
  #kind(absolute: string): FileKind | undefined {
    if (absolute === '/') return 'directory';
    const at = this.#locate(absolute);
    if (!at) return undefined;
    if (!at.path) return 'directory';
    try {
      if (at.folder.isFile(at.path)) return 'file';
      if (at.folder.isFolder(at.path) || this.#mounts.dirs.has(absolute)) return 'directory';
    } catch { /* a name the folder does not accept: nothing is there */ }
    return undefined;
  }
  #info(absolute: string, kind: FileKind): FileInfo {
    const at = this.#locate(absolute);
    return { name: basenamePath(absolute), path: absolute, kind, size: kind === 'file' && at ? at.folder.size(at.path) : 0,
      mtimeMs: this.#mounts.times.get(absolute) ?? 0 };
  }
  /** Mark files as changed now: their modification times move past every earlier one. */
  #touch(...absolutes: string[]): void {
    const now = Math.max(Date.now(), this.#mounts.clock + 1);
    this.#mounts.clock = now;
    for (const path of absolutes) this.#mounts.times.set(path, now);
  }
  /** The bytes of an existing regular file, or the error the syscall would give. */
  #file(absolute: string, syscall: string): Result<Uint8Array, FileError> {
    const kind = this.#kind(absolute);
    if (kind === 'directory') return err(isDirectory(syscall === 'open' ? 'read' : syscall, absolute));
    if (kind !== 'file') return err(notFound(syscall, absolute));
    const at = this.#locate(absolute)!;
    return ok(at.folder.readBytesSync(at.path));
  }
  /** Where a write to `absolute` lands, or why it cannot. */
  #writable(absolute: string, syscall: string): Result<{ folder: Folder; path: string }, FileError> {
    const at = this.#locate(absolute);
    if (!at || !at.path) return err(at ? isDirectory(syscall, absolute) : denied(syscall, absolute));
    if (this.#kind(absolute) === 'directory') return err(isDirectory(syscall, absolute));
    for (let parent = dirnamePath(absolute); parent !== '/' && parent !== absolute; parent = dirnamePath(parent))
      if (this.#kind(parent) === 'file') return err(notDirectory(syscall, absolute));
    if (at.folder.access === 'read') return err(denied(syscall, absolute));
    return ok(at);
  }
  #put(absolute: string, bytes: Uint8Array, syscall: string): Result<void, FileError> {
    const target = this.#writable(absolute, syscall);
    if (!target.ok) return target;
    try { target.value.folder.writeBytes(target.value.path, bytes); }
    catch (caught) { return err(new FileError('permission_denied', caught instanceof Error ? caught.message : String(caught), absolute)); }
    this.#touch(absolute);
    return ok(undefined);
  }

  async absolutePath(path: string): Promise<Result<string, FileError>> { return ok(this.#resolve(path)); }
  async joinPath(parts: string[]): Promise<Result<string, FileError>> { return ok(joinPath(...parts)); }

  // --- reading ----------------------------------------------------------------------------------------------------

  async readTextFile(path: string, context: Context): Promise<Result<string, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    const bytes = this.#file(absolute, 'open');
    return bytes.ok ? ok(decodeText(bytes.value)) : bytes;
  }
  async readBinaryFile(path: string, context: Context): Promise<Result<Uint8Array, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    return this.#file(absolute, 'open');
  }
  async openTextLineReader(path: string, context: Context): Promise<Result<TextLineReader, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    const bytes = this.#file(absolute, 'open');
    return bytes.ok ? ok(new SnapshotLineReader(bytes.value, absolute)) : bytes;
  }
  async readTextLines(path: string, options: { maxLines?: number } | undefined, context: Context): Promise<Result<string[], FileError>> {
    if (options?.maxLines !== undefined && options.maxLines <= 0) return ok([]);
    const opened = await this.openTextLineReader(path, context);
    if (!opened.ok) return opened;
    const lines: string[] = [];
    while (options?.maxLines === undefined || lines.length < options.maxLines) {
      const line = await opened.value.readLine(context);
      if (!line.ok) return line;
      if (line.value === undefined) break;
      lines.push(line.value.text);
    }
    return ok(lines);
  }
  async openBinaryReader(path: string, _options: { noFollow?: boolean } | undefined, context: Context): Promise<Result<BinaryReader, FileError>> {
    // A folder has no symbolic links, so noFollow changes nothing.
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    const bytes = this.#file(absolute, 'open');
    return bytes.ok ? ok(new SnapshotReader(bytes.value, this.#info(absolute, 'file'))) : bytes;
  }
  async fileInfo(path: string, context: Context): Promise<Result<FileInfo, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    const kind = this.#kind(absolute);
    return kind ? ok(this.#info(absolute, kind)) : err(notFound('lstat', absolute));
  }
  async listDir(path: string, context: Context): Promise<Result<FileInfo[], FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    const kind = this.#kind(absolute);
    if (kind === 'file') return err(notDirectory('scandir', absolute));
    if (!kind) return err(notFound('scandir', absolute));
    const names = new Set<string>();
    if (absolute === '/') { names.add(WORKSPACE.slice(1)); names.add(TEMP.slice(1)); }
    const at = this.#locate(absolute);
    if (at) for (const name of at.folder.childNames(at.path)) names.add(name);
    for (const dir of this.#mounts.dirs) if (dirnamePath(dir) === absolute) names.add(basenamePath(dir));
    return ok([...names].map(name => joinPath(absolute, name)).flatMap(child => {
      const childKind = this.#kind(child);
      return childKind ? [this.#info(child, childKind)] : [];
    }));
  }
  async openDirReader(path: string, context: Context): Promise<Result<DirReader, FileError>> {
    const listed = await this.listDir(path, context);
    return listed.ok ? ok(new ListReader(listed.value, this.#resolve(path))) : listed;
  }
  async canonicalPath(path: string, context: Context): Promise<Result<string, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    return this.#kind(absolute) ? ok(absolute) : err(notFound('realpath', absolute));
  }
  async exists(path: string, context: Context): Promise<Result<boolean, FileError>> {
    const info = await this.fileInfo(path, context);
    if (info.ok) return ok(true);
    return info.error.code === 'not_found' ? ok(false) : info;
  }
  async watch(): Promise<Result<FileWatcher, FileError>> {
    return err(new FileError('not_supported', 'A folder environment cannot watch files'));
  }

  // --- writing ----------------------------------------------------------------------------------------------------

  async writeFile(path: string, content: string | Uint8Array, context: Context): Promise<Result<void, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    return this.#put(absolute, bytesOf(content), 'open');
  }
  async appendFile(path: string, content: string | Uint8Array, context: Context): Promise<Result<void, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    const before = this.#kind(absolute) === 'file' ? this.#file(absolute, 'open') : ok<Uint8Array, FileError>(new Uint8Array());
    if (!before.ok) return before;
    const added = bytesOf(content), joined = new Uint8Array(before.value.length + added.length);
    joined.set(before.value);
    joined.set(added, before.value.length);
    return this.#put(absolute, joined, 'open');
  }
  async truncateFile(path: string, size: number, context: Context): Promise<Result<void, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    if (!Number.isSafeInteger(size) || size < 0) return err(new FileError('invalid', 'File size must be a non-negative safe integer', absolute));
    const before = this.#file(absolute, 'open');
    if (!before.ok) return before;
    const resized = new Uint8Array(size);
    resized.set(before.value.subarray(0, size));
    return this.#put(absolute, resized, 'open');
  }
  async flushFile(path: string, context: Context): Promise<Result<void, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    if (this.#kind(absolute) === 'directory') return err(new FileError('is_directory', 'Is a directory', absolute));
    const file = this.#file(absolute, 'open');
    return file.ok ? ok(undefined) : file;
  }
  async renameFile(sourcePath: string, destinationPath: string, context: Context): Promise<Result<void, FileError>> {
    const source = this.#resolve(sourcePath), destination = this.#resolve(destinationPath);
    if (context.abortSignal?.aborted) return err(abortedFile(destination));
    const kind = this.#kind(source);
    if (!kind) return err(notFound('rename', source));
    if (kind === 'file') {
      const bytes = this.#file(source, 'rename');
      if (!bytes.ok) return bytes;
      const written = this.#put(destination, bytes.value, 'rename');
      if (!written.ok) return written;
      return this.remove(source, {}, context);
    }
    const from = this.#locate(source), to = this.#locate(destination);
    if (!from?.path || !to?.path) return err(denied('rename', source));
    if (from.folder !== to.folder) return err(new FileError('unknown', `EXDEV: cross-device link not permitted, rename '${source}' -> '${destination}'`, source));
    const existing = this.#kind(destination);
    if (existing === 'file') return err(notDirectory('rename', destination));
    if (existing === 'directory') {
      const entries = await this.listDir(destination, context);
      if (entries.ok && entries.value.length) return err(new FileError('unknown', `ENOTEMPTY: directory not empty, rename '${source}' -> '${destination}'`, source));
      this.#mounts.dirs.delete(destination);
    }
    try { if (from.folder.isFolder(from.path)) from.folder.move(from.path, to.path); }
    catch (caught) { return err(new FileError('permission_denied', caught instanceof Error ? caught.message : String(caught), source)); }
    for (const dir of [...this.#mounts.dirs]) if (dir === source || dir.startsWith(`${source}/`)) {
      this.#mounts.dirs.delete(dir);
      this.#mounts.dirs.add(destination + dir.slice(source.length));
    }
    this.#touch(destination);
    return ok(undefined);
  }
  async createDir(path: string, options: { recursive?: boolean } | undefined, context: Context): Promise<Result<void, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    const recursive = options?.recursive ?? true;
    const kind = this.#kind(absolute);
    if (kind === 'directory' && recursive) return ok(undefined);
    if (kind) return err(new FileError('unknown', `EEXIST: file already exists, mkdir '${absolute}'`, absolute));
    if (!this.#locate(absolute)) return err(denied('mkdir', absolute));
    const missing: string[] = [];
    for (let at = absolute; !this.#kind(at); at = dirnamePath(at)) missing.push(at);
    if (missing.length > 1 && !recursive) return err(notFound('mkdir', absolute));
    const blocking = dirnamePath(missing.at(-1)!);
    if (this.#kind(blocking) === 'file') return err(notDirectory('mkdir', absolute));
    for (const dir of missing) if (this.#locate(dir)?.path) this.#mounts.dirs.add(dir);
    return ok(undefined);
  }
  async remove(path: string, options: { recursive?: boolean; force?: boolean } | undefined, context: Context): Promise<Result<void, FileError>> {
    const absolute = this.#resolve(path);
    if (context.abortSignal?.aborted) return err(abortedFile(absolute));
    const kind = this.#kind(absolute);
    if (!kind) return options?.force ? ok(undefined) : err(notFound('rm', absolute));
    const at = this.#locate(absolute);
    if (!at?.path) return err(denied('rm', absolute));
    if (kind === 'directory' && !options?.recursive)
      return err(new FileError('is_directory', `Path is a directory: rm returned EISDIR (is a directory) ${absolute}`, absolute));
    try { if (at.folder.isFile(at.path) || at.folder.isFolder(at.path)) at.folder.remove(at.path); }
    catch (caught) { return err(new FileError('permission_denied', caught instanceof Error ? caught.message : String(caught), absolute)); }
    for (const dir of [...this.#mounts.dirs]) if (dir === absolute || dir.startsWith(`${absolute}/`)) this.#mounts.dirs.delete(dir);
    for (const file of [...this.#mounts.times.keys()]) if (file === absolute || file.startsWith(`${absolute}/`)) this.#mounts.times.delete(file);
    return ok(undefined);
  }
  async createTempDir(prefix: string | undefined, context: Context): Promise<Result<string, FileError>> {
    if (context.abortSignal?.aborted) return err(abortedFile());
    const dir = joinPath(TEMP, `${prefix ?? 'tmp-'}${randomName().slice(0, 8)}`);
    this.#mounts.dirs.add(dir);
    return ok(dir);
  }
  async createTempFile(options: { prefix?: string; suffix?: string } | undefined, context: Context): Promise<Result<string, FileError>> {
    const dir = await this.createTempDir('tmp-', context);
    if (!dir.ok) return dir;
    const file = joinPath(dir.value, `${options?.prefix ?? ''}${randomName()}${options?.suffix ?? ''}`);
    const written = this.#put(file, new Uint8Array(), 'open');
    return written.ok ? ok(file) : written;
  }

  // --- the shell --------------------------------------------------------------------------------------------------

  async exec(command: string | readonly string[], options: ShellExecOptions | undefined, context: Context):
      Promise<Result<ShellExecResult, ExecutionError>> {
    if (context.abortSignal?.aborted) return err(new ExecutionError('aborted', 'aborted'));
    const timeout = options?.timeout;
    if (timeout !== undefined && (!Number.isFinite(timeout) || timeout <= 0))
      return err(new ExecutionError('timeout', 'Invalid timeout: must be a finite number of seconds'));
    if (timeout !== undefined && timeout * 1000 > MAX_TIMEOUT_MS)
      return err(new ExecutionError('timeout', `Invalid timeout: maximum is ${MAX_TIMEOUT_MS / 1000} seconds`));
    const cwd = options?.cwd ? this.#resolve(options.cwd) : this.cwd;
    if ((cwd !== WORKSPACE && !cwd.startsWith(`${WORKSPACE}/`)) || this.#kind(cwd) !== 'directory')
      return err(new ExecutionError('spawn_error', `Working directory does not exist: ${cwd}\nCannot execute bash commands.`));
    let script: string, args: string[] | undefined;
    if (typeof command === 'string') script = command;
    else {
      const [program, ...rest] = command;
      if (program === undefined) return err(new ExecutionError('spawn_error', 'Empty argv: no program to run'));
      // An argv runs its program with the arguments as they are: the shell parses only the quoted program name.
      script = quote(program);
      args = rest;
    }

    const controller = new AbortController();
    let timedOut = false;
    const onAbort = () => controller.abort();
    context.abortSignal?.addEventListener('abort', onAbort, { once: true });
    const timer = timeout === undefined ? undefined : setTimeout(() => { timedOut = true; controller.abort(); }, timeout * 1000);
    this.#running.add(controller);
    let stdout = '', stderr = '', exitCode: number, changed: string[] = [];
    let failure: unknown;
    try {
      const result = await runFolderBash(this.folder, script, { cwd, network: this.#network, signal: controller.signal,
        ...(options?.env && Object.keys(options.env).length ? { env: options.env } : {}), ...(args ? { args } : {}) });
      ({ stdout, stderr, exitCode } = result);
      changed = result.changedPaths;
    } catch (caught) {
      // The folder shell refuses open-ended loops and recursion before running anything, as a syntax error.
      if (caught instanceof SyntaxError) { stderr = `bash: ${caught.message}\n`; exitCode = 2; }
      else { failure = caught; exitCode = 1; }
    } finally {
      if (timer) clearTimeout(timer);
      context.abortSignal?.removeEventListener('abort', onAbort);
      this.#running.delete(controller);
    }
    if (changed.length) this.#touch(...changed.map(path => joinPath(WORKSPACE, path)));

    let callbackError: ExecutionError | undefined;
    const emit = (text: string, stream: ShellOutputInfo['stream']) => {
      if (!text || !options?.onOutput || callbackError) return;
      try { options.onOutput(text, context, { stream }); }
      catch (caught) { callbackError = new ExecutionError('callback_error', caught instanceof Error ? caught.message : String(caught)); }
    };
    emit(stdout, 'stdout');
    emit(stderr, 'stderr');
    if (callbackError) return err(callbackError);

    // The complete output in a temporary file once it crosses either threshold, as a spawned command's would be.
    let spillPath: string | undefined;
    const output = stdout + stderr, spill = options?.spill;
    if (spill && output) {
      const bytes = new TextEncoder().encode(output);
      const lines = (output.match(/\n/g)?.length ?? 0) + (output.endsWith('\n') ? 0 : 1);
      if (bytes.length > spill.afterBytes || lines > spill.afterLines) {
        const file = await this.createTempFile({ prefix: 'pi-output-', suffix: '.log' }, context);
        if (!file.ok) return err(new ExecutionError('unknown', `Failed to preserve complete shell output: ${file.error.message}`));
        const written = this.#put(file.value, bytes, 'open');
        if (!written.ok) return err(new ExecutionError('unknown', `Failed to preserve complete shell output: ${written.error.message}`));
        spillPath = file.value;
      }
    }
    const interrupted = timedOut ? new ExecutionError('timeout', `timeout:${timeout}`) :
      context.abortSignal?.aborted ? new ExecutionError('aborted', 'aborted') : undefined;
    if (interrupted) {
      if (spillPath !== undefined) interrupted.spillPath = spillPath;
      return err(interrupted);
    }
    if (failure !== undefined) return err(new ExecutionError('unknown', failure instanceof Error ? failure.message : String(failure)));
    return ok({ exitCode, ...(spillPath === undefined ? {} : { spillPath }) });
  }

  async cleanup(): Promise<void> {
    for (const controller of this.#running) controller.abort();
    this.#running.clear();
  }
}
