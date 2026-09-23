/** Node filesystem access for the natlang loader and compiler. */
import { existsSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { basename, dirname, extname, join, relative, resolve } from 'node:path';
import { callableTree, namedCallable, type NatlangCallable } from './callable.js';
import { findApplicationContext, loadCallableFolder, loadNamedFunction, type ItemRecord, type SourceFiles } from './loader.js';

/** Node source access; display paths are relative to `root`. */
export function nodeSourceFiles(root = process.cwd()): SourceFiles {
  const base = resolve(root);
  return { join, dirname, basename, extname,
    isFile: path => existsSync(path) && statSync(path).isFile(),
    isDirectory: path => existsSync(path) && statSync(path).isDirectory(),
    read: path => readFileSync(path, 'utf8'), list: readdirSync,
    relative: path => relative(base, resolve(path)).split('\\').join('/') };
}

/** Load a named `.nl` function (with its companion folder) as a callable. */
export function loadNatlang(path: string, root = dirname(resolve(path))): NatlangCallable {
  const absolute = resolve(path);
  return namedCallable(basename(absolute, '.nl'), loadNamedFunction(absolute, nodeSourceFiles(root)));
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
export function fileTraceSink(dir: string): (trace: import('./runtime.js').InvocationTrace) => void {
  return trace => {
    mkdirSync(dir, { recursive: true });
    const name = trace.callId.replace(/[^A-Za-z0-9_.-]/g, '_').slice(0, 160);
    writeFileSync(join(dir, `${name}.jsonl`), trace.events.map(event => JSON.stringify(event)).join('\n') + '\n');
  };
}
