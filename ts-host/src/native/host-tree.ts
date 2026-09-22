export type TreeEntry = { name: string; kind: 'branch' | 'leaf' };
export interface TreeProvider<T = unknown> {
  list(path: readonly string[]): Iterable<TreeEntry>;
  read(path: readonly string[]): T;
}

type Cache = { provider: TreeProvider; branches: Map<string, TreeEntry[]>; leaves: Map<string, unknown> };
const key = (path: readonly string[]) => path.join('/');

/** Stable read-only view: each observed branch or leaf is fetched at most once. */
export class LazyDict<T = unknown> {
  readonly __natlangLazyDict = true;
  private readonly cache: Cache;
  constructor(provider: TreeProvider<T>, readonly label = 'host tree', readonly path: readonly string[] = [], cache?: Cache) {
    this.cache = cache ?? { provider, branches: new Map(), leaves: new Map() };
  }
  entries(): TreeEntry[] {
    const at = key(this.path), found = this.cache.branches.get(at);
    if (found) return found;
    const names = new Set<string>(), entries: TreeEntry[] = [];
    for (const entry of this.cache.provider.list(this.path)) {
      if (!entry.name || /[/\\]/.test(entry.name) || ['.', '..'].includes(entry.name) || entry.name.startsWith('$') ||
          !['branch', 'leaf'].includes(entry.kind)) throw new TypeError(`invalid host-tree entry ${JSON.stringify(entry)}`);
      if (names.has(entry.name)) throw new TypeError(`duplicate host-tree entry ${entry.name}`);
      names.add(entry.name); entries.push({ name: entry.name, kind: entry.kind });
    }
    entries.sort((a, b) => a.name < b.name ? -1 : a.name > b.name ? 1 : 0);
    this.cache.branches.set(at, entries); return entries;
  }
  child(name: string): LazyDict<T> | T {
    const entry = this.entries().find(item => item.name === name);
    if (!entry) throw new RangeError(`no such host-tree entry: ${name}`);
    const path = [...this.path, name], at = key(path);
    if (entry.kind === 'branch') return new LazyDict(this.cache.provider as TreeProvider<T>, this.label, path, this.cache);
    if (!this.cache.leaves.has(at)) this.cache.leaves.set(at, this.cache.provider.read(path));
    return this.cache.leaves.get(at) as T;
  }
}

export function isLazyDict(value: unknown): value is LazyDict {
  return !!value && typeof value === 'object' && (value as { __natlangLazyDict?: unknown }).__natlangLazyDict === true;
}

export class MemoryTreeProvider<T = unknown> implements TreeProvider<T> {
  private readonly leaves = new Map<string, T>();
  constructor(leaves: Record<string, T>) {
    for (const [raw, value] of Object.entries(leaves)) {
      const parts = raw.replaceAll('\\', '/').split('/').filter(Boolean);
      if (!parts.length || raw.startsWith('/') || parts.some(part => part === '.' || part === '..'))
        throw new TypeError(`invalid host-tree path: ${raw}`);
      this.leaves.set(parts.join('/'), value);
    }
  }
  list(path: readonly string[]): TreeEntry[] {
    const prefix = path.length ? `${key(path)}/` : '', children = new Map<string, 'branch' | 'leaf'>();
    for (const leaf of this.leaves.keys()) {
      if (!leaf.startsWith(prefix)) continue;
      const rest = leaf.slice(prefix.length), [name] = rest.split('/');
      if (!name) continue;
      const kind = rest.includes('/') ? 'branch' : 'leaf';
      if (children.get(name) !== 'branch') children.set(name, kind);
    }
    return [...children].map(([name, kind]) => ({ name, kind }));
  }
  read(path: readonly string[]): T {
    const at = key(path);
    if (!this.leaves.has(at)) throw new RangeError(`no such host-tree leaf: ${at}`);
    return this.leaves.get(at)!;
  }
}

export const lazyDict = <T>(leaves: Record<string, T>, label = 'lazy dictionary') =>
  new LazyDict(new MemoryTreeProvider(leaves), label);
