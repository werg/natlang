/** Node directories as natlang folders: lazy reads from disk, and atomic write-back of a folder's changes. */
import { closeSync, existsSync, lstatSync, mkdirSync, openSync, readFileSync, readdirSync, realpathSync, renameSync,
  rmSync, statSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { randomBytes } from 'node:crypto';
import { Folder, type ChangeSet, type FolderAccess, type FolderSource } from './scoped-fs.js';

/** Directory names never listed from disk. */
const IGNORED = new Set(['.git', 'node_modules']);

function inside(root: string, target: string): boolean {
  const fromRoot = relative(root, target);
  return fromRoot !== '..' && !fromRoot.startsWith(`..${sep}`) && !isAbsolute(fromRoot);
}

/** A directory on disk. The path list is taken once; file contents are read on first access and kept. */
class DiskSource implements FolderSource {
  private listed?: string[];
  private readonly cache = new Map<string, Uint8Array | undefined>();
  constructor(readonly root: string) {}
  paths(): Iterable<string> {
    if (this.listed) return this.listed;
    const found: string[] = [];
    const walk = (dir: string, prefix: string) => {
      for (const entry of readdirSync(dir, { withFileTypes: true })) {
        if (IGNORED.has(entry.name)) continue;
        const path = prefix ? `${prefix}/${entry.name}` : entry.name, absolute = join(dir, entry.name);
        let target = absolute;
        if (entry.isSymbolicLink()) {
          try { target = realpathSync(absolute); } catch { continue; }
          if (!inside(this.root, target)) continue;
        }
        const status = statSync(target);
        if (status.isDirectory()) { if (!entry.isSymbolicLink()) walk(absolute, path); }
        else if (status.isFile()) found.push(path);
      }
    };
    walk(this.root, '');
    return this.listed = found;
  }
  get(path: string): Uint8Array | undefined {
    if (this.cache.has(path)) return this.cache.get(path);
    let value: Uint8Array | undefined;
    const absolute = resolve(this.root, ...path.split('/'));
    try {
      const real = realpathSync(absolute);
      value = inside(this.root, real) && statSync(real).isFile() ? new Uint8Array(readFileSync(real)) : undefined;
    } catch { value = undefined; }
    this.cache.set(path, value);
    return value;
  }
}

/**
 * Open a directory as a folder. Reads are lazy; writes stay in the folder's overlay until
 * `saveFolder` writes them back.
 */
export function openFolder(root: string, access: FolderAccess = 'read'): Folder {
  const real = realpathSync(resolve(root));
  if (!statSync(real).isDirectory()) throw new RangeError(`not a directory: ${root}`);
  return new Folder(new DiskSource(real), access);
}

export type SavedChange = { path: string; kind: 'added' | 'modified' | 'deleted'; bytes: number };

/** Write a folder's changes (or a given change set) beneath `root`, each file by atomic replacement. */
export function saveFolder(root: string, changes: Folder | ChangeSet): SavedChange[] {
  const base = realpathSync(resolve(root));
  const set = changes instanceof Folder ? changes.diffSync() : changes;
  const planned = set.changes.map(change => {
    const target = resolve(base, ...change.path.split('/'));
    if (!inside(base, target)) throw new RangeError(`folder change escapes its root: ${change.path}`);
    if (existsSync(target) && !lstatSync(target).isFile()) throw new RangeError(`not a regular file: ${change.path}`);
    return { change, target };
  });
  return planned.map(({ change, target }) => {
    if (!change.after) { rmSync(target, { force: true }); return { path: change.path, kind: change.kind, bytes: 0 }; }
    mkdirSync(dirname(target), { recursive: true });
    const parent = realpathSync(dirname(target));
    if (!inside(base, parent)) throw new RangeError(`folder change escapes its root: ${change.path}`);
    const temporary = join(parent, `.natlang-write-${process.pid}-${randomBytes(8).toString('hex')}`);
    let descriptor: number | undefined;
    try {
      descriptor = openSync(temporary, 'wx');
      writeFileSync(descriptor, change.after);
      closeSync(descriptor); descriptor = undefined;
      renameSync(temporary, target);
    } catch (error) {
      if (descriptor !== undefined) closeSync(descriptor);
      try { unlinkSync(temporary); } catch { /* absent after rename or failed creation */ }
      throw error;
    }
    return { path: change.path, kind: change.kind, bytes: change.after.length };
  });
}
