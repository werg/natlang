import { closeSync, lstatSync, openSync, readSync, readdirSync, realpathSync, statSync } from 'node:fs';
import { isAbsolute, relative, resolve, sep } from 'node:path';
import { StringDecoder } from 'node:string_decoder';
import type { FileTreeEntry, FileTreeRead, ReadonlyFileTree } from './files.js';

const CHUNK = 64 * 1024;

/** Filesystem-backed tree. Directory metadata and file bytes are read only when requested. */
export class NodeFileTree implements ReadonlyFileTree {
  readonly root: string;
  constructor(root: string) { this.root = realpathSync(resolve(root)); }

  private target(path: string): { relative: string; absolute: string } {
    const normalized = path.replaceAll('\\', '/');
    const parts = normalized.split('/').filter(Boolean);
    if (isAbsolute(normalized) || parts.some(part => part === '.' || part === '..' || part.includes('\0')))
      throw new RangeError(`file path escapes its root: ${path}`);
    const requested = resolve(this.root, ...parts);
    let absolute: string;
    try { absolute = realpathSync(requested); }
    catch { throw new RangeError(`no such file or directory: ${parts.join('/')}`); }
    const fromRoot = relative(this.root, absolute);
    if (fromRoot === '..' || fromRoot.startsWith(`..${sep}`) || isAbsolute(fromRoot))
      throw new RangeError(`file path escapes its root: ${path}`);
    return { relative: parts.join('/'), absolute };
  }

  read(path: string, start?: number, end?: number): FileTreeRead {
    const target = this.target(path), status = statSync(target.absolute);
    if (status.isDirectory()) {
      const entries: FileTreeEntry[] = readdirSync(target.absolute, { withFileTypes: true }).map(entry => {
        const at = resolve(target.absolute, entry.name), link = lstatSync(at).isSymbolicLink();
        return { name: entry.name, kind: link ? 'symlink' : entry.isDirectory() ? 'directory' : 'file',
          ...(!link && entry.isFile() ? { bytes: statSync(at).size } : {}) };
      });
      return { kind: 'directory', path: target.relative,
        entries: entries.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0) };
    }
    if (!status.isFile()) throw new RangeError(`not a regular file: ${target.relative}`);
    return this.readFile(target.relative, target.absolute, status.size, start, end);
  }

  private readFile(path: string, absolute: string, bytes: number, start?: number, end?: number): FileTreeRead {
    const first = start ?? 1, last = end ?? (start === undefined ? first + 199 : first);
    if (!Number.isInteger(first) || !Number.isInteger(last) || first < 1 || last < first)
      throw new RangeError('file line range must satisfy 1 <= start <= end');
    const handle = openSync(absolute, 'r'), buffer = Buffer.allocUnsafe(CHUNK), decoder = new StringDecoder('utf8');
    let position = 0, pending = '', line = 1, truncated = false, binary = false, endedWithNewline = false;
    const selected: string[] = [];
    try {
      while (true) {
        const count = readSync(handle, buffer, 0, buffer.length, position);
        if (!count) break;
        if (buffer.subarray(0, count).includes(0)) { binary = true; break; }
        position += count; endedWithNewline = buffer[count - 1] === 10;
        pending += decoder.write(buffer.subarray(0, count));
        const parts = pending.split(/\r?\n/); pending = parts.pop() ?? '';
        for (let index = 0; index < parts.length; index++) {
          const item = parts[index]!;
          if (line >= first && line <= last) selected.push(item);
          line++;
          if (line > last) { truncated = position < bytes || index < parts.length - 1 || pending.length > 0 || endedWithNewline; break; }
        }
        if (line > last) break;
      }
      if (binary) return { kind: 'binary', path, bytes };
      if (line <= last) {
        pending += decoder.end();
        if (line >= first && line <= last && (pending.length || endedWithNewline)) selected.push(pending);
      }
      if (position < bytes) truncated = true;
      return { kind: 'text', path, text: selected.join('\n'), start: first,
        end: Math.max(first - 1, Math.min(last, line)), truncated, bytes };
    } finally { closeSync(handle); }
  }
}
