/** A read-only project tree exposed below `files/` in a natlang episode. */
export type FileTreeEntry = { name: string; kind: 'file' | 'directory' | 'symlink'; bytes?: number };
export type FileTreeRead =
  | { kind: 'directory'; path: string; entries: FileTreeEntry[] }
  | { kind: 'text'; path: string; text: string; start: number; end: number; truncated: boolean; bytes: number }
  | { kind: 'binary'; path: string; bytes: number };

export interface ReadonlyFileTree {
  read(path: string, start?: number, end?: number): FileTreeRead;
}

function clean(path: string): string {
  const parts = path.replaceAll('\\', '/').split('/').filter(Boolean);
  if (path.startsWith('/') || parts.some(part => part === '.' || part === '..' || part.includes('\0')))
    throw new RangeError(`file path escapes its root: ${path}`);
  return parts.join('/');
}

/** Portable provider for browser virtual files and already-memory-backed embeddings. */
export class MemoryFileTree implements ReadonlyFileTree {
  private readonly files: Record<string, string | Uint8Array>;
  constructor(files: Record<string, string | Uint8Array>) {
    this.files = Object.fromEntries(Object.entries(files).map(([path, value]) => [clean(path), value]));
  }

  read(path: string, start?: number, end?: number): FileTreeRead {
    const key = clean(path);
    if (Object.hasOwn(this.files, key)) {
      const value = this.files[key]!;
      if (value instanceof Uint8Array) return { kind: 'binary', path: key, bytes: value.byteLength };
      const lines = value.split(/\r?\n/), first = start ?? 1, last = end ?? (start === undefined ? Math.min(200, lines.length) : first);
      if (!Number.isInteger(first) || !Number.isInteger(last) || first < 1 || last < first)
        throw new RangeError('file line range must satisfy 1 <= start <= end');
      return { kind: 'text', path: key, text: lines.slice(first - 1, last).join('\n'),
        start: first, end: Math.min(last, lines.length), truncated: last < lines.length, bytes: new TextEncoder().encode(value).byteLength };
    }
    const prefix = key ? `${key}/` : '';
    const children = new Map<string, FileTreeEntry>();
    for (const [name, value] of Object.entries(this.files)) {
      if (!name.startsWith(prefix)) continue;
      const rest = name.slice(prefix.length), child = rest.split('/')[0]!;
      if (!child) continue;
      const nested = rest.includes('/');
      children.set(child, { name: child, kind: nested ? 'directory' : 'file',
        ...(!nested ? { bytes: typeof value === 'string' ? new TextEncoder().encode(value).byteLength : value.byteLength } : {}) });
    }
    if (!children.size && key) throw new RangeError(`no such file or directory: ${key}`);
    return { kind: 'directory', path: key, entries: [...children.values()].sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0) };
  }
}

export function formatFileTreeRead(result: FileTreeRead): string {
  if (result.kind === 'directory') return result.entries.length ? result.entries.map(entry =>
    `${entry.kind === 'directory' ? 'dir ' : entry.kind === 'symlink' ? 'link' : 'file'}  ${entry.name}${entry.bytes === undefined ? '' : `  ${entry.bytes} bytes`}`).join('\n') : '(empty directory)';
  if (result.kind === 'binary') return `${result.path}: binary file, ${result.bytes} bytes; content requires a host binary capability`;
  const suffix = result.truncated ? `\n… more content; read another line range from files/${result.path}` : '';
  return result.text + suffix;
}
