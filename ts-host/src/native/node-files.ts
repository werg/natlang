import { closeSync, existsSync, mkdirSync, openSync, readFileSync, readdirSync, realpathSync, renameSync,
  statSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname, isAbsolute, join, relative, resolve, sep } from 'node:path';
import { randomBytes } from 'node:crypto';
import { LazyDict, type TreeEntry, type TreeProvider } from './host-tree.js';
import { validateFileWrites, type FileTreeLeaf, type FileWrite } from './file-tree.js';
export { FILE_TREE_LEAF_TYPE, FILE_WRITE_TYPE, type FileTreeLeaf, type FileWrite } from './file-tree.js';

class NodeFileProvider implements TreeProvider<FileTreeLeaf> {
  readonly root: string;
  constructor(root: string) { this.root = realpathSync(resolve(root)); }

  private target(path: readonly string[]): string {
    if (path.some(part => !part || part === '.' || part === '..' || part.includes('\0')))
      throw new RangeError(`file path escapes its root: ${path.join('/')}`);
    let absolute: string;
    try { absolute = realpathSync(resolve(this.root, ...path)); }
    catch { throw new RangeError(`no such file or directory: ${path.join('/')}`); }
    const fromRoot = relative(this.root, absolute);
    if (fromRoot === '..' || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot))
      throw new RangeError(`file path escapes its root: ${path.join('/')}`);
    return absolute;
  }

  list(path: readonly string[]): TreeEntry[] {
    const directory = this.target(path);
    if (!statSync(directory).isDirectory()) throw new RangeError(`not a directory: ${path.join('/')}`);
    return readdirSync(directory, { withFileTypes: true }).map(entry => {
      if (entry.isSymbolicLink()) {
        try { return { name: entry.name,
          kind: statSync(this.target([...path, entry.name])).isDirectory() ? 'branch' : 'leaf' }; }
        catch { return { name: entry.name, kind: 'leaf' }; }
      }
      return { name: entry.name, kind: entry.isDirectory() ? 'branch' : 'leaf' };
    });
  }

  read(path: readonly string[]): FileTreeLeaf {
    const file = this.target(path), status = statSync(file);
    if (!status.isFile()) throw new RangeError(`not a regular file: ${path.join('/')}`);
    const content = readFileSync(file);
    return content.includes(0) ? { kind: 'binary', bytes: content.length } :
      { kind: 'text', text: content.toString('utf8'), bytes: content.length };
  }
}

/** A filesystem-backed lazy Record<string, FileTreeLeaf>, suitable as an ordinary natlang input. */
export class NodeFileTree extends LazyDict<FileTreeLeaf> {
  constructor(root: string, label = 'project files') { super(new NodeFileProvider(root), label); }
}

export type FileWriteReceipt = { path: string; bytes: number; overwritten: boolean };

/** Commit a portable change plan beneath root, using same-directory atomic replacement. */
export function commitFileWrites(root: string, writes: unknown, options: { overwrite?: boolean } = {}): FileWriteReceipt[] {
  const base = realpathSync(resolve(root));
  if (!statSync(base).isDirectory()) throw new RangeError(`file-write root is not a directory: ${base}`);
  const checked = validateFileWrites(writes), overwrite = options.overwrite ?? true;
  const planned = checked.map((item: FileWrite) => {
    const target = resolve(base, ...item.path.split('/')), fromRoot = relative(base, target);
    if (fromRoot === '..' || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot))
      throw new RangeError(`file write escapes its root: ${item.path}`);
    if (existsSync(target) && !statSync(target).isFile())
      throw new RangeError(`file write target is not a regular file: ${item.path}`);
    if (existsSync(target) && !overwrite) throw new Error(`file already exists: ${item.path}`);
    return { item, target, overwritten: existsSync(target) };
  });
  return planned.map(({ item, target, overwritten }) => {
    mkdirSync(dirname(target), { recursive: true });
    const parent = realpathSync(dirname(target)), fromRoot = relative(base, parent);
    if (fromRoot === '..' || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot))
      throw new RangeError(`file write escapes its root: ${item.path}`);
    const temporary = join(parent, `.natlang-write-${process.pid}-${randomBytes(8).toString('hex')}`);
    let descriptor: number | undefined;
    try {
      descriptor = openSync(temporary, 'wx');
      writeFileSync(descriptor, item.text, 'utf8');
      closeSync(descriptor); descriptor = undefined;
      renameSync(temporary, target);
    } catch (error) {
      if (descriptor !== undefined) closeSync(descriptor);
      try { unlinkSync(temporary); } catch { /* absent after rename or failed creation */ }
      throw error;
    }
    return { path: item.path, bytes: Buffer.byteLength(item.text), overwritten };
  });
}
