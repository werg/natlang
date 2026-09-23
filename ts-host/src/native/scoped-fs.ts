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

/**
 * A function that can run as a directory reducer on a folder exposes this method. `folder.apply(reducer, ...args)`
 * runs it with a private copy of the folder and installs its committed changes.
 */
export const APPLY_TO_FOLDER: unique symbol = Symbol.for('natlang.applyToFolder') as never;
type FolderReducer = { [APPLY_TO_FOLDER]?: (folder: FolderHandle, args: unknown[]) => Promise<unknown> };
function applyReducer(folder: FolderHandle, reducer: unknown, args: unknown[]): Promise<unknown> {
  const apply = (reducer as FolderReducer | undefined)?.[APPLY_TO_FOLDER];
  if (typeof apply !== 'function') throw new TypeError('folder.apply needs a natlang directory reducer');
  return apply.call(reducer, folder, args);
}

/** Replace exactly one span of `text`; `fuzzy` matches one whole line ignoring whitespace differences. */
export function editTextContent(text: string, find: string, replaceWith: string, fuzzy = false): string {
  let count = find ? text.split(find).length - 1 : 0;
  if (count !== 1 && fuzzy) {
    const wanted = find.replace(/\s+/g, ' ').trim();
    const candidates = [...text.matchAll(/[^\n]*(?:\n|$)/g)].filter(match => match[0] && match[0].replace(/\s+/g, ' ').trim() === wanted);
    count = candidates.length;
    if (count === 1) {
      const match = candidates[0]!, newline = match[0].endsWith('\n') ? '\n' : '';
      return text.slice(0, match.index!) + replaceWith + newline + text.slice(match.index! + match[0].length);
    }
  }
  if (count === 0) {
    if (fuzzy) throw new Error('edit found no matching span, even with fuzzy: true; inspect the current contents');
    throw new Error('edit found no matching span; retry with fuzzy: true for whitespace-tolerant matching or inspect the current contents');
  }
  if (count !== 1) throw new Error(`edit found ${count} matching spans; make find more specific so it matches exactly one span`);
  return text.replace(find, () => replaceWith);
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
  /** Run a directory reducer on this folder and install its committed changes; resolves to its typed result. */
  apply(reducer: unknown, ...args: unknown[]): Promise<unknown> { return applyReducer(this, reducer, args); }
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

/**
 * The unchanged contents beneath a folder: a list of file paths and a reader. Sources are read lazily,
 * so a folder can front a large directory on disk. Paths are clean relative POSIX paths.
 */
export interface FolderSource {
  paths(): Iterable<string>;
  get(path: string): Uint8Array | undefined;
}

class MapSource implements FolderSource {
  constructor(private readonly files: Map<string, Uint8Array>) {}
  paths(): Iterable<string> { return this.files.keys(); }
  get(path: string): Uint8Array | undefined { return this.files.get(path); }
}

/** A frozen view of a source with an overlay applied: forks and transactions start from one. */
class LayeredSource implements FolderSource {
  constructor(private readonly base: FolderSource, private readonly overlay: Map<string, Uint8Array | typeof TOMBSTONE>,
              private readonly prefix = '') {}
  *paths(): Iterable<string> {
    const seen = new Set<string>();
    for (const path of [...this.overlay.keys(), ...this.base.paths()]) {
      if (seen.has(path)) continue;
      seen.add(path);
      if (this.overlay.get(path) === TOMBSTONE || (this.prefix && !path.startsWith(`${this.prefix}/`))) continue;
      yield this.prefix ? path.slice(this.prefix.length + 1) : path;
    }
  }
  get(path: string): Uint8Array | undefined {
    const full = this.prefix ? `${this.prefix}/${path}` : path;
    const value = this.overlay.get(full);
    return value === TOMBSTONE ? undefined : value ?? this.base.get(full);
  }
}

export class Folder {
  private readonly source: FolderSource;
  private readonly overlay = new Map<string, Uint8Array | typeof TOMBSTONE>();
  private readonly moves: Array<[string, string]> = [];
  private readonly writerLock: WriterLock;

  constructor(files: Record<string, FileContents> | FolderSource = {}, readonly access: FolderAccess = 'write',
              options: { writer?: WriterLock } = {}) {
    if (!['read', 'write', 'overlay'].includes(access)) throw new RangeError('invalid folder access');
    this.source = isSource(files) ? files :
      new MapSource(new Map(Object.entries(files).map(([path, value]) => [cleanPath(path, false), bytes(value)])));
    this.writerLock = options.writer ?? new WriterLock();
  }

  static fromFiles(files: Record<string, FileContents>, access: FolderAccess = 'write'): Folder {
    return new Folder(files, access);
  }
  root(): FolderHandle { return new FolderHandle(this, ''); }
  dir(path = ''): FolderHandle { return new FolderHandle(this, cleanPath(path)); }
  file(path: string): FileHandle { return new FileHandle(this, cleanPath(path, false)); }
  /** Run a directory reducer on the whole folder and install its committed changes. */
  apply(reducer: unknown, ...args: unknown[]): Promise<unknown> { return applyReducer(this.root(), reducer, args); }
  entry(path: string): EntryHandle { const clean = cleanPath(path); return this.isFile(clean) ? new FileHandle(this, clean) : new FolderHandle(this, clean); }
  join(parent: string, child: string): string { const clean = cleanPath(child); return parent ? (clean ? cleanPath(`${parent}/${clean}`) : parent) : clean; }

  /** The current contents of one file, or undefined. */
  private current(path: string): Uint8Array | undefined {
    const value = this.overlay.get(path);
    return value === TOMBSTONE ? undefined : value ?? this.source.get(path);
  }
  /** Every current file path. */
  private allPaths(): string[] {
    const result = new Set<string>();
    for (const path of this.source.paths()) if (this.overlay.get(path) !== TOMBSTONE) result.add(path);
    for (const [path, value] of this.overlay) if (value !== TOMBSTONE) result.add(path);
    return [...result];
  }
  private checkWrite(): void { if (this.access === 'read') throw new Error('folder is read-only'); }
  private read(path: string): Uint8Array {
    const value = this.current(cleanPath(path, false));
    if (!value) throw new Error(`file not found: ${path}`);
    return value;
  }
  async exists(path: string): Promise<boolean> { const clean = cleanPath(path); return this.isFile(clean) || this.isFolder(clean); }
  isFile(path: string): boolean { const clean = cleanPath(path); return !!clean && this.current(clean) !== undefined; }
  isFolder(path: string): boolean { const clean = cleanPath(path); return !clean || this.allPaths().some(item => item.startsWith(`${clean}/`)); }
  async stat(path: string): Promise<EntryStat> { return this.statSync(path); }
  private statSync(path: string): EntryStat {
    const clean = cleanPath(path), value = clean ? this.current(clean) : undefined;
    if (value) return { path: clean, kind: 'file', bytes: value.length, digest: hexDigest(value) };
    if (this.isFolder(clean)) return { path: clean, kind: 'folder', bytes: 0, digest: null };
    throw new Error(`entry not found: ${clean}`);
  }
  private paths(path = ''): string[] { const clean = cleanPath(path); return this.allPaths().filter(item => under(item, clean)).sort(); }

  list(path = '', patternText?: string): EntryStat[] {
    const clean = cleanPath(path), children = new Map<string, EntryKind>();
    for (const item of this.paths(clean)) {
      const rest = clean ? item.slice(clean.length + 1) : item, child = this.join(clean, rest.split('/')[0]!);
      children.set(child, rest.includes('/') || children.get(child) === 'folder' ? 'folder' : 'file');
    }
    const matches = pattern(patternText);
    return [...children].sort(([left], [right]) => left.localeCompare(right)).filter(([path]) => matches(path)).map(([path]) => this.statSync(path));
  }
  listFiles(path = '', patternText?: string): EntryStat[] { const matches = pattern(patternText); return this.paths(path).filter(path => matches(path)).map(path => this.statSync(path)); }
  listFolders(path = '', patternText?: string): EntryStat[] {
    const clean = cleanPath(path), candidates = new Set<string>();
    for (const item of this.paths(clean)) { const parts = item.split('/'); for (let i = 1; i < parts.length; i++) candidates.add(parts.slice(0, i).join('/')); }
    const matches = pattern(patternText);
    return [...candidates].filter(item => item !== clean && under(item, clean) && matches(item)).sort().map(item => this.statSync(item));
  }
  async readBytes(path: string): Promise<Uint8Array> { return this.read(path).slice(); }
  readBytesSync(path: string): Uint8Array { return this.read(path).slice(); }
  async readText(path: string, startLine?: number, endLine?: number): Promise<string> {
    const value = text(this.read(path)); if (startLine === undefined && endLine === undefined) return value;
    const start = startLine ?? 1, end = endLine ?? start; if (start < 1 || end < start) throw new RangeError('invalid line range');
    return value.split(/(?<=\n)/).slice(start - 1, end).join('');
  }
  async readJson<T = unknown>(path: string): Promise<T> { return JSON.parse(await this.readText(path)) as T; }
  writeBytes(path: string, content: FileContents): void { this.checkWrite(); this.overlay.set(cleanPath(path, false), bytes(content)); }
  writeText(path: string, content: string): void { this.writeBytes(path, content); }
  async writeJson(path: string, value: unknown): Promise<void> { this.writeText(path, `${JSON.stringify(value, null, 2)}\n`); }
  remove(path: string): void {
    this.checkWrite(); const clean = cleanPath(path, false);
    if (!this.isFile(clean) && !this.isFolder(clean)) throw new Error(`entry not found: ${clean}`);
    for (const item of this.allPaths()) if (item === clean || item.startsWith(`${clean}/`)) this.overlay.set(item, TOMBSTONE);
  }
  move(source: string, destination: string): void {
    this.checkWrite(); const from = cleanPath(source, false), to = cleanPath(destination, false);
    if (from === to || to.startsWith(`${from}/`)) throw new Error('cannot move an entry into itself');
    const selected = this.allPaths().filter(path => path === from || path.startsWith(`${from}/`));
    if (!selected.length) throw new Error(`entry not found: ${from}`);
    if (this.isFile(to) || this.isFolder(to)) throw new Error(`destination exists: ${to}`);
    for (const path of selected) {
      const suffix = path.slice(from.length).replace(/^\//, '');
      this.overlay.set(`${to}${suffix ? `/${suffix}` : ''}`, this.current(path)!); this.overlay.set(path, TOMBSTONE);
    }
    this.moves.push([from, to]);
  }
  async editText(path: string, find: string, replaceWith: string, fuzzy = false): Promise<Record<string, unknown>> {
    const updated = editTextContent(await this.readText(path), find, replaceWith, fuzzy);
    this.writeText(path, updated);
    return { path: cleanPath(path), changed: true, digest: hexDigest(new TextEncoder().encode(updated)) };
  }
  async search(query: string, path = '', patternText?: string, regex = false): Promise<SearchMatch[]> {
    if (!query) throw new RangeError('search query must be nonempty'); const expression = regex ? new RegExp(query) : undefined, result: SearchMatch[] = [];
    for (const item of this.listFiles(path, patternText)) { let content: string; try { content = await this.readText(item.path); } catch { continue; } content.split(/\r?\n/).forEach((line, index) => { if (expression ? expression.test(line) : line.includes(query)) result.push({ path: item.path, line: index + 1, text: line }); }); }
    return result;
  }
  /** Changes relative to the source; only overlaid paths can differ. */
  diffSync(path = ''): ChangeSet {
    const clean = cleanPath(path), changes: Change[] = [];
    for (const item of [...this.overlay.keys()].filter(item => under(item, clean)).sort()) {
      const before = this.source.get(item), after = this.current(item);
      if (equalBytes(before, after)) continue;
      changes.push({ path: item, kind: before === undefined ? 'added' : after === undefined ? 'deleted' : 'modified',
        ...(before === undefined ? {} : { before: before.slice() }), ...(after === undefined ? {} : { after: after.slice() }) });
    }
    return { changes, moves: this.moves.map(move => [...move] as [string, string]) };
  }
  async diff(path = ''): Promise<ChangeSet> { return this.diffSync(path); }
  /** A copy-on-write fork of the current contents, sharing this folder's writer lock. */
  fork(access: FolderAccess = 'overlay'): Folder { return new Folder(this.snapshot(), access, { writer: this.writerLock }); }
  private snapshot(prefix = ''): FolderSource { return new LayeredSource(this.source, new Map(this.overlay), prefix); }
  writer(): WriterLock { return this.writerLock; }
  async beginTransaction(blocking = true, path = ''): Promise<FolderTransaction> {
    this.checkWrite(); await this.writerLock.acquire(blocking);
    const prefix = cleanPath(path);
    return new FolderTransaction(this, new Folder(this.snapshot(prefix), 'overlay'), prefix);
  }
  installLocked(changes: ChangeSet, include?: string[], exclude?: string[]): ChangeSet {
    const selectedSet = selectChanges(changes, include, exclude), selected = selectedSet.changes;
    for (const change of selected) if (!equalBytes(this.current(change.path), change.before)) throw new FolderConflictError(change.path);
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

function isSource(value: unknown): value is FolderSource {
  return !!value && typeof (value as FolderSource).paths === 'function' && typeof (value as FolderSource).get === 'function';
}
