/** Standalone copy-on-write folder handles for scoped lambda filesystems. */

import { sha256 } from '@noble/hashes/sha2.js';

export type FolderAccess = 'read' | 'write' | 'overlay';
export type FileContents = string | Uint8Array;

const TOMBSTONE = Symbol('natlang-folder-tombstone');

function cleanPath(path: string, allowRoot = true): string {
  if (typeof path !== 'string') throw new TypeError('path must be a string');
  if (allowRoot && (path === '' || path === '.')) return '';
  if (!path || path.startsWith('/') || path.includes('\\') || path.includes('\0'))
    throw new RangeError(`invalid relative POSIX path: ${JSON.stringify(path)}`);
  const parts = path.split('/');
  if (parts.some(part => !part || part === '.' || part === '..'))
    throw new RangeError(`invalid relative POSIX path: ${JSON.stringify(path)}`);
  return parts.join('/');
}

function bytes(value: FileContents): Uint8Array {
  return typeof value === 'string' ? new TextEncoder().encode(value) : value.slice();
}

function text(value: Uint8Array): string { return new TextDecoder('utf-8', { fatal: true }).decode(value); }
function equalBytes(left: Uint8Array | undefined, right: Uint8Array | undefined): boolean {
  if (left === undefined || right === undefined) return left === right;
  return left.length === right.length && left.every((item, index) => item === right[index]);
}
function hexDigest(value: Uint8Array): string {
  return Array.from(sha256(value), byte => byte.toString(16).padStart(2, '0')).join('');
}
function under(path: string, parent: string): boolean {
  return !parent || path === parent || path.startsWith(`${parent}/`);
}
function pattern(pattern: string | undefined): ((path: string) => boolean) {
  if (pattern === undefined) return () => true;
  cleanPath(pattern.replaceAll('**', 'x'));
  const source = pattern.split('**').map(part => part.split('*').map(escapeRegExp).join('[^/]*')).join('.*');
  const expression = new RegExp(`^${source}$`);
  return path => expression.test(path);
}
function escapeRegExp(value: string): string { return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

export type EntryKind = 'file' | 'folder';
export type EntryStat = { path: string; kind: EntryKind; bytes: number; digest: string | null };
export type SearchMatch = { path: string; line: number; text: string };
export type ChangeKind = 'added' | 'modified' | 'deleted';
export type Change = { path: string; kind: ChangeKind; before?: Uint8Array; after?: Uint8Array };
export type ChangeSet = { changes: Change[]; moves: Array<[string, string]> };

function selectChanges(changes: ChangeSet, include?: string[], exclude?: string[]): ChangeSet {
  const includeMatches = (include ?? []).map(pattern), excludeMatches = (exclude ?? []).map(pattern);
  const selected = changes.changes.filter(change =>
    (!includeMatches.length || includeMatches.some(match => match(change.path))) &&
    !excludeMatches.some(match => match(change.path)));
  return { changes: selected,
    moves: changes.moves.filter(([from, to]) => selected.some(change => change.path === from || change.path === to)) };
}

export class FolderBusyError extends Error { constructor() { super('folder writer is busy'); } }
export class FolderConflictError extends Error { constructor(path: string) { super(`folder changed at ${path}`); } }

/** A per-root writer semaphore.  Read operations never acquire it. */
export class WriterLock {
  private busy = false;
  private waiters: Array<() => void> = [];

  async acquire(blocking = true): Promise<() => void> {
    if (!this.busy) { this.busy = true; return () => this.release(); }
    if (!blocking) throw new FolderBusyError();
    await new Promise<void>(resolve => this.waiters.push(resolve));
    this.busy = true;
    return () => this.release();
  }

  release(): void {
    if (!this.busy) throw new Error('folder writer is not held');
    const next = this.waiters.shift();
    if (next) next(); else this.busy = false;
  }
}

export class EntryHandle {
  constructor(readonly folder: Folder, readonly path: string) {}
  get name(): string { return this.path.split('/').at(-1) ?? ''; }
  get relativePath(): string { return this.path; }
  get parent(): FolderHandle | null {
    if (!this.path) return null;
    return new FolderHandle(this.folder, this.path.includes('/') ? this.path.slice(0, this.path.lastIndexOf('/')) : '');
  }
  async exists(): Promise<boolean> { return this.folder.exists(this.path); }
  async stat(): Promise<EntryStat> { return this.folder.stat(this.path); }
  async remove(): Promise<void> { this.folder.remove(this.path); }
  async moveTo(destination: FolderHandle | FileHandle | string): Promise<void> {
    const target = typeof destination === 'string' ? destination :
      destination instanceof FolderHandle ? this.folder.join(destination.path, this.name) : destination.path;
    this.folder.move(this.path, target);
  }
}

export class FileHandle extends EntryHandle {
  async readText(startLine?: number, endLine?: number): Promise<string> {
    return this.folder.readText(this.path, startLine, endLine);
  }
  async readBytes(): Promise<Uint8Array> { return this.folder.readBytes(this.path); }
  async readJson<T = unknown>(): Promise<T> { return JSON.parse(await this.readText()) as T; }
  async writeText(content: string): Promise<void> { this.folder.writeText(this.path, content); }
  async writeBytes(content: Uint8Array): Promise<void> { this.folder.writeBytes(this.path, content); }
  async writeJson(value: unknown): Promise<void> { await this.writeText(`${JSON.stringify(value, null, 2)}\n`); }
  async editText(find: string, replaceWith: string, fuzzy = false): Promise<Record<string, unknown>> {
    return this.folder.editText(this.path, find, replaceWith, fuzzy);
  }
}

export class FolderHandle extends EntryHandle {
  dir(path: string): FolderHandle { return new FolderHandle(this.folder, this.folder.join(this.path, path)); }
  file(path: string): FileHandle { return new FileHandle(this.folder, this.folder.join(this.path, path)); }
  entry(path: string): EntryHandle { const joined = this.folder.join(this.path, path); return this.folder.isFile(joined) ? new FileHandle(this.folder, joined) : new FolderHandle(this.folder, joined); }
  async entries(patternText?: string): Promise<EntryHandle[]> { return this.folder.list(this.path, patternText).map(item => this.folder.entry(item.path)); }
  async files(patternText?: string): Promise<FileHandle[]> { return this.folder.listFiles(this.path, patternText).map(item => new FileHandle(this.folder, item.path)); }
  async folders(patternText?: string): Promise<FolderHandle[]> { return this.folder.listFolders(this.path, patternText).map(item => new FolderHandle(this.folder, item.path)); }
  async *walk(patternText?: string): AsyncIterable<EntryHandle> { const entries = [...this.folder.listFiles(this.path, patternText), ...this.folder.listFolders(this.path, patternText)].sort((left, right) => left.path.localeCompare(right.path)); for (const entry of entries) yield this.folder.entry(entry.path); }
  async diff(): Promise<ChangeSet> { return this.folder.diff(this.path); }
  async beginTransaction(blocking = true): Promise<FolderTransaction> {
    return this.folder.beginTransaction(blocking, this.path);
  }
}

export class FolderTransaction {
  private closed = false;
  constructor(readonly parent: Folder, readonly folder: Folder, readonly prefix = '') {}
  get open(): boolean { return !this.closed; }
  private ensureOpen(): void { if (this.closed) throw new Error('folder transaction is closed'); }
  commitSync(include?: string[], exclude?: string[]): ChangeSet {
    this.ensureOpen();
    try {
      const selected = selectChanges(this.folder.diffSync(), include, exclude);
      const rooted = (path: string) => this.prefix ? this.parent.join(this.prefix, path) : path;
      this.parent.installLocked({ changes: selected.changes.map(change => ({ ...change, path: rooted(change.path) })),
        moves: selected.moves.map(([from, to]) => [rooted(from), rooted(to)]) });
      this.close(); return selected;
    }
    catch (error) { this.abort(); throw error; }
  }
  async commit(include?: string[], exclude?: string[]): Promise<ChangeSet> {
    return this.commitSync(include, exclude);
  }
  abort(): void { this.ensureOpen(); this.close(); }
  private close(): void { if (!this.closed) { this.closed = true; this.parent.writer().release(); } }
}

export class Folder {
  private readonly base: Map<string, Uint8Array>;
  private readonly overlay = new Map<string, Uint8Array | typeof TOMBSTONE>();
  private readonly moves: Array<[string, string]> = [];
  private readonly writerLock: WriterLock;

  constructor(files: Record<string, FileContents> = {}, readonly access: FolderAccess = 'write',
              options: { base?: Map<string, Uint8Array>; writer?: WriterLock } = {}) {
    if (!['read', 'write', 'overlay'].includes(access)) throw new RangeError('invalid folder access');
    this.base = options.base ?? new Map(Object.entries(files).map(([path, value]) => [cleanPath(path, false), bytes(value)]));
    this.writerLock = options.writer ?? new WriterLock();
  }

  static fromFiles(files: Record<string, FileContents>, access: FolderAccess = 'write'): Folder {
    return new Folder(files, access);
  }
  root(): FolderHandle { return new FolderHandle(this, ''); }
  dir(path = ''): FolderHandle { return new FolderHandle(this, cleanPath(path)); }
  file(path: string): FileHandle { return new FileHandle(this, cleanPath(path, false)); }
  entry(path: string): EntryHandle { const clean = cleanPath(path); return this.isFile(clean) ? new FileHandle(this, clean) : new FolderHandle(this, clean); }
  join(parent: string, child: string): string { const clean = cleanPath(child); return parent ? (clean ? cleanPath(`${parent}/${clean}`) : parent) : clean; }

  private effective(): Map<string, Uint8Array> {
    const result = new Map([...this.base].map(([path, value]) => [path, value.slice()]));
    for (const [path, value] of this.overlay) value === TOMBSTONE ? result.delete(path) : result.set(path, value.slice());
    return result;
  }
  private checkWrite(): void { if (this.access === 'read') throw new Error('folder is read-only'); }
  private read(path: string): Uint8Array {
    const value = this.effective().get(cleanPath(path, false));
    if (!value) throw new Error(`file not found: ${path}`);
    return value;
  }
  async exists(path: string): Promise<boolean> { const clean = cleanPath(path); const values = this.effective(); return values.has(clean) || [...values.keys()].some(item => under(item, clean)); }
  isFile(path: string): boolean { return this.effective().has(cleanPath(path)); }
  isFolder(path: string): boolean { const clean = cleanPath(path); return !clean || [...this.effective().keys()].some(item => item.startsWith(`${clean}/`)); }
  async stat(path: string): Promise<EntryStat> { return this.statSync(path); }
  private statSync(path: string): EntryStat {
    const clean = cleanPath(path), values = this.effective(), value = values.get(clean);
    if (value) return { path: clean, kind: 'file', bytes: value.length, digest: hexDigest(value) };
    if (!clean || [...values.keys()].some(item => item.startsWith(`${clean}/`))) return { path: clean, kind: 'folder', bytes: 0, digest: null };
    throw new Error(`entry not found: ${clean}`);
  }
  private paths(path = ''): string[] { const clean = cleanPath(path); return [...this.effective().keys()].filter(item => under(item, clean)).sort(); }

  list(path = '', patternText?: string): EntryStat[] {
    const clean = cleanPath(path), children = new Map<string, EntryKind>();
    for (const item of this.paths(clean)) {
      const rest = clean ? item.slice(clean.length + 1) : item, child = this.join(clean, rest.split('/')[0]!);
      children.set(child, rest.includes('/') || children.get(child) === 'folder' ? 'folder' : 'file');
    }
    const matches = pattern(patternText);
    return [...children].sort(([left], [right]) => left.localeCompare(right)).filter(([path]) => matches(path)).map(([path]) => this.statSync(path));
  }
  listFiles(path = '', patternText?: string): EntryStat[] { const matches = pattern(patternText); return this.paths(path).filter(path => this.isFile(path) && matches(path)).map(path => this.statSync(path)); }
  listFolders(path = '', patternText?: string): EntryStat[] {
    const clean = cleanPath(path), candidates = new Set<string>();
    for (const item of this.paths(clean)) { const parts = item.split('/'); for (let i = 1; i < parts.length; i++) candidates.add(parts.slice(0, i).join('/')); }
    const matches = pattern(patternText);
    return [...candidates].filter(item => item !== clean && under(item, clean) && matches(item)).sort().map(item => this.statSync(item));
  }
  async readBytes(path: string): Promise<Uint8Array> { return this.read(path).slice(); }
  async readText(path: string, startLine?: number, endLine?: number): Promise<string> {
    const value = text(this.read(path)); if (startLine === undefined && endLine === undefined) return value;
    const start = startLine ?? 1, end = endLine ?? start; if (start < 1 || end < start) throw new RangeError('invalid line range');
    return value.split(/(?<=\n)/).slice(start - 1, end).join('');
  }
  async readJson<T = unknown>(path: string): Promise<T> { return JSON.parse(await this.readText(path)) as T; }
  writeBytes(path: string, content: FileContents): void { this.checkWrite(); this.overlay.set(cleanPath(path, false), bytes(content)); }
  writeText(path: string, content: string): void { this.writeBytes(path, content); }
  async writeJson(path: string, value: unknown): Promise<void> { this.writeText(path, `${JSON.stringify(value, null, 2)}\n`); }
  remove(path: string): void { this.checkWrite(); const clean = cleanPath(path, false); if (!this.isFile(clean) && !this.isFolder(clean)) throw new Error(`entry not found: ${clean}`); for (const item of this.effective().keys()) if (item === clean || item.startsWith(`${clean}/`)) this.overlay.set(item, TOMBSTONE); }
  move(source: string, destination: string): void { this.checkWrite(); const from = cleanPath(source, false), to = cleanPath(destination, false); if (from === to || to.startsWith(`${from}/`)) throw new Error('cannot move an entry into itself'); const values = this.effective(), selected = [...values].filter(([path]) => path === from || path.startsWith(`${from}/`)); if (!selected.length) throw new Error(`entry not found: ${from}`); if (this.isFile(to) || this.isFolder(to)) throw new Error(`destination exists: ${to}`); for (const [path, value] of selected) { const suffix = path.slice(from.length).replace(/^\//, ''); this.overlay.set(`${to}${suffix ? `/${suffix}` : ''}`, value); this.overlay.set(path, TOMBSTONE); } this.moves.push([from, to]); }
  async editText(path: string, find: string, replaceWith: string, fuzzy = false): Promise<Record<string, unknown>> {
    let original = await this.readText(path), count = original.split(find).length - 1;
    if (count !== 1 && fuzzy) { const wanted = find.replace(/\s+/g, ' ').trim(); const candidates = [...original.matchAll(/[^\n]*(?:\n|$)/g)].filter(match => match[0].replace(/\s+/g, ' ').trim() === wanted); count = candidates.length; if (count === 1) { const match = candidates[0]!; const newline = match[0].endsWith('\n') ? '\n' : ''; original = original.slice(0, match.index!) + replaceWith + newline + original.slice(match.index! + match[0].length); this.writeText(path, original); return { path: cleanPath(path), changed: true, digest: hexDigest(new TextEncoder().encode(original)) }; } }
    if (count !== 1) throw new Error('edit requires exactly one matching span'); original = original.replace(find, replaceWith); this.writeText(path, original); return { path: cleanPath(path), changed: true, digest: hexDigest(new TextEncoder().encode(original)) };
  }
  async search(query: string, path = '', patternText?: string, regex = false): Promise<SearchMatch[]> {
    if (!query) throw new RangeError('search query must be nonempty'); const expression = regex ? new RegExp(query) : undefined, result: SearchMatch[] = [];
    for (const item of this.listFiles(path, patternText)) { let content: string; try { content = await this.readText(item.path); } catch { continue; } content.split(/\r?\n/).forEach((line, index) => { if (expression ? expression.test(line) : line.includes(query)) result.push({ path: item.path, line: index + 1, text: line }); }); }
    return result;
  }
  diffSync(path = ''): ChangeSet {
    const clean = cleanPath(path), current = this.effective(), paths = [...new Set([...this.base.keys(), ...current.keys()])].filter(item => under(item, clean)).sort(), changes: Change[] = [];
    for (const item of paths) { const before = this.base.get(item), after = current.get(item); if (equalBytes(before, after)) continue; changes.push({ path: item, kind: before === undefined ? 'added' : after === undefined ? 'deleted' : 'modified', ...(before === undefined ? {} : { before: before.slice() }), ...(after === undefined ? {} : { after: after.slice() }) }); }
    return { changes, moves: this.moves.map(move => [...move] as [string, string]) };
  }
  async diff(path = ''): Promise<ChangeSet> { return this.diffSync(path); }
  fork(access: FolderAccess = 'overlay'): Folder { return new Folder({}, access, { base: this.effective(), writer: this.writerLock }); }
  writer(): WriterLock { return this.writerLock; }
  async beginTransaction(blocking = true, path = ''): Promise<FolderTransaction> {
    this.checkWrite(); await this.writerLock.acquire(blocking);
    const prefix = cleanPath(path), source = this.effective(), files: Record<string, Uint8Array> = {};
    for (const [name, value] of source) if (under(name, prefix) && name !== prefix)
      files[prefix ? name.slice(prefix.length + 1) : name] = value;
    return new FolderTransaction(this, new Folder(files, 'overlay'), prefix);
  }
  installLocked(changes: ChangeSet, include?: string[], exclude?: string[]): ChangeSet {
    const selectedSet = selectChanges(changes, include, exclude), selected = selectedSet.changes;
    const current = this.effective(); for (const change of selected) if (!equalBytes(current.get(change.path), change.before)) throw new FolderConflictError(change.path);
    for (const change of selected) this.overlay.set(change.path, change.after === undefined ? TOMBSTONE : change.after.slice());
    return selectedSet;
  }
  async install(changes: ChangeSet, include?: string[], exclude?: string[], blocking = true): Promise<ChangeSet> {
    this.checkWrite();
    const release = await this.writerLock.acquire(blocking);
    try { return this.installLocked(changes, include, exclude); } finally { release(); }
  }
  async installFrom(child: Folder, include?: string[], exclude?: string[], blocking = true): Promise<ChangeSet> { return this.install(await child.diff(), include, exclude, blocking); }
}
