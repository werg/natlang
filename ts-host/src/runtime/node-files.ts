/** Node filesystem access for the natlang loader and compiler. */
import { appendFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join, relative, resolve } from 'node:path';
import { callableMeta, callableTree, makeCallable, namedCallable, recordOf, rememberRecord, type NatlangCallable } from './callable.js';
import { findApplicationContext, loadCallableFolder, loadNamedFunction, type ItemRecord, type SourceFiles } from './loader.js';

/** Node source access; display paths are relative to `root`. */
export function nodeSourceFiles(root = process.cwd()): SourceFiles {
  const base = resolve(root);
  return { join, dirname, basename, extname,
    isFile: path => existsSync(path) && statSync(path).isFile(),
    isDirectory: path => existsSync(path) && statSync(path).isDirectory(),
    read: path => readFileSync(path, 'utf8'), list: readdirSync, readBytes: path => new Uint8Array(readFileSync(path)),
    relative: path => relative(base, resolve(path)).split('\\').join('/') };
}

/**
 * Load a named `.nl` function (with its companion folder) as a callable.
 *
 * `live`: the callable follows its sources. Each call first checks the files and folders the loader read (the
 * function, its owners, its companion folder, their types) and loads them again when one changed, so a long-running
 * process uses the newest definition from its next call on; a call that started keeps the definition it started with.
 * When the changed sources do not load, calls keep the last definition that loaded and the error is reported once.
 */
export function loadNatlang(path: string, root = dirname(resolve(path)), options: { live?: boolean } = {}): NatlangCallable {
  const absolute = resolve(path);
  const name = basename(absolute, '.nl');
  if (!options.live) return namedCallable(name, loadNamedFunction(absolute, nodeSourceFiles(root)));
  return liveCallable(name, () => {
    const watched = new Map<string, string>();
    const callable = namedCallable(name, loadNamedFunction(absolute, watchedSourceFiles(nodeSourceFiles(root), watched)));
    return { callable, watched };
  });
}

/** What a path looks like now, as far as the loader can tell: absent, a file's size and time, or a folder's names. */
function sourceState(path: string): string {
  if (!existsSync(path)) return 'absent';
  const stat = statSync(path);
  return stat.isDirectory() ? `dir:${readdirSync(path).sort().join('/')}` : `file:${stat.size}:${stat.mtimeMs}`;
}

/** `files`, remembering the state of every path the loader asked about. */
function watchedSourceFiles(files: SourceFiles, watched: Map<string, string>): SourceFiles {
  const watch = <T>(path: string, value: T): T => { if (!watched.has(path)) watched.set(path, sourceState(path)); return value; };
  return { ...files,
    isFile: path => watch(path, files.isFile(path)), isDirectory: path => watch(path, files.isDirectory(path)),
    read: path => watch(path, files.read(path)),
    ...(files.readBytes ? { readBytes: (path: string) => watch(path, files.readBytes!(path)) } : {}),
    list: path => watch(path, files.list(path)) };
}

function liveCallable(name: string, load: () => { callable: NatlangCallable; watched: Map<string, string> }): NatlangCallable {
  let current = load(), reported: string | undefined;
  const fresh = (): NatlangCallable => {
    if ([...current.watched].some(([path, state]) => sourceState(path) !== state)) {
      try { current = load(); reported = undefined; rememberRecord(live, recordOf(current.callable)!); syncChildren(); }
      catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        if (message !== reported) console.warn(`natlang: ${name} keeps its last definition; the changed sources do not load: ${message}`);
        reported = message;
      }
    }
    return current.callable;
  };
  const meta = callableMeta(current.callable)!;
  const live = makeCallable({ ...meta, invoke: (args, frame, at) => callableMeta(fresh())!.invoke(args, frame, at) });
  // What others read from the callable (its definition, its data captures) is the current definition's.
  for (const key of ['definition', 'captures'] as const) if (key in meta)
    Object.defineProperty(callableMeta(live)!, key, { get: () => callableMeta(current.callable)![key], enumerable: true });
  const children = new Set<string>();
  const syncChildren = () => {
    for (const child of Object.keys(current.callable)) if (!children.has(child)) {
      children.add(child);
      Object.defineProperty(live, child, { get: () => (fresh() as unknown as Record<string, unknown>)[child], enumerable: true });
    }
  };
  rememberRecord(live, recordOf(current.callable)!);
  syncChildren();
  return live;
}

/** Load a callable folder (for example `natlang.d/`) as a record of callables. */
export function loadCallables(dir: string, root = dirname(resolve(dir))): Record<string, unknown> {
  return callableTree(loadCallableFolder(resolve(dir), nodeSourceFiles(root)));
}

/** The record tree of the `natlang.d/` context that applies to `start`, if any. */
export function applicationContextRecords(start: string, root?: string): { dir: string; records: Record<string, ItemRecord> } | undefined {
  const files = nodeSourceFiles(root ?? start);
  const dir = findApplicationContext(resolve(start), files);
  return dir ? { dir, records: loadCallableFolder(dir, nodeSourceFiles(root ?? dirname(dir))) } : undefined;
}

/** A trace sink writing one JSONL file per natlang invocation into `dir`. */
export function fileTraceSink(dir: string): import('./runtime.js').TraceSink {
  const fileOf = (callId: string) => join(dir, `${callId.replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 160)}.jsonl`);
  // Events already appended per running call: each call's file grows as it runs and is rewritten whole when it ends.
  const written = new Map<string, number>();
  const sink = (trace: import('./runtime.js').InvocationTrace) => {
    mkdirSync(dir, { recursive: true });
    written.delete(trace.callId);
    writeFileSync(fileOf(trace.callId), trace.events.map(event => JSON.stringify(event)).join('\n') + '\n');
  };
  return Object.assign(sink, { stream(callId: string, events: readonly Record<string, unknown>[]) {
    const from = written.get(callId) ?? 0;
    if (events.length <= from) return;
    if (!from) mkdirSync(dir, { recursive: true });
    appendFileSync(fileOf(callId), events.slice(from).map(event => JSON.stringify(event)).join('\n') + '\n', { flag: from ? 'a' : 'w' });
    written.set(callId, events.length);
  } });
}
