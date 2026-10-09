/** Standalone copy-on-write folder handles for scoped lambda filesystems. */

import { currentFrame } from '../runtime/context.js';
import { FolderIteration } from '../runtime/folder-iteration.js';
import { sha256 } from '@noble/hashes/sha2.js';
import { folderFromData, type FolderDataLayout } from './data-layout.js';

export type FolderAccess = 'read' | 'write' | 'overlay';
export type FileContents = string | Uint8Array;
type WriteFence = { path: string | null; displayPrefix: string };

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
/** What list_files shows the model: one `path  N bytes` line per file. */
export function fileListingText(entries: EntryStat[]): string {
  return entries.length ? entries.map(entry => `${entry.path}  ${entry.bytes} bytes`).join('\n') : '(no files)';
}
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
export class FolderScopeError extends Error {
  constructor(path: string) {
    super(`write to ${JSON.stringify(path)} is outside the supplied FileHandle scope; pass that output FileHandle to the child or return the value to the parent to write`);
    this.name = 'FolderScopeError';
  }
}

/** A subtree writer lock. Disjoint directories may run and commit together. */
export class WriterLock {
  private readonly held = new Set<string>();
  private waiters: Array<() => void> = [];

  async acquire(blocking = true, path = ''): Promise<() => void> {
    const conflicts = () => [...this.held].some(item => under(item, path) || under(path, item));
    while (conflicts()) {
      if (!blocking) throw new FolderBusyError();
      await new Promise<void>(resolve => this.waiters.push(resolve));
    }
    this.held.add(path);
    return () => this.release(path);
  }

  release(path = ''): void {
    if (!this.held.delete(path)) throw new Error('folder writer is not held');
    const waiters = this.waiters.splice(0);
    for (const wake of waiters) wake();
  }
}

/**
 * A function that can run as a directory reducer on a folder exposes this method. `folder.apply(reducer, ...args)`
 * runs it with a private copy of the folder and installs its committed changes.
 */
export const APPLY_TO_FOLDER: unique symbol = Symbol.for('natlang.applyToFolder') as never;
export type FolderReducer = { [APPLY_TO_FOLDER]?: (folder: FolderHandle, args: unknown[]) => Promise<unknown> };
function applyReducer(folder: FolderHandle, reducer: unknown, args: unknown[]): Promise<unknown> {
  const apply = (reducer as FolderReducer | undefined)?.[APPLY_TO_FOLDER];
  if (typeof apply === 'function') return apply.call(reducer, folder, args);
  // A natural-language function that is not a directory reducer takes no folder; it is not applied as a plain function.
  if (typeof reducer === 'function' && !Object.hasOwn(reducer, Symbol.for('natlang.callable'))) return applyFunction(folder, reducer as (folder: FolderHandle, ...args: unknown[]) => unknown, args);
  throw new TypeError('folder.apply needs a directory reducer: a natlang `kind: directory-reducer` function or a host/TypeScript function (folder, ...args)');
}
/**
 * A plain (host or TypeScript) function as a directory reducer. It has the contract of a natlang one: it runs on a
 * private transaction copy of the folder and receives that copy as its first argument; its changes are validated and
 * installed when it returns, and discarded when it throws.
 */
async function applyFunction(folder: FolderHandle, reducer: (folder: FolderHandle, ...args: unknown[]) => unknown, args: unknown[]): Promise<unknown> {
  const transaction = await folder.beginTransaction(true);
  try {
    const value = await reducer(transaction.folder.root(), ...args);
    transaction.validateSync();
    transaction.commitSync();
    return value;
  } finally { if (transaction.open) transaction.abort(); }
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

/** The handle kind a path needs: asking for a file handle on a folder (or the reverse) is a mistake worth naming. */
function checkedKind(folder: Folder, path: string, requested: string, want: 'file' | 'folder'): void {
  if (want === 'file' && (requested.endsWith('/') || (path && folder.isFolder(path) && !folder.isFile(path))))
    throw new Error(`${requested} is a folder; use dir(${JSON.stringify(requested.replace(/\/+$/, ''))})`);
  if (want === 'folder' && folder.isFile(path)) throw new Error(`${requested} is a file; use file(${JSON.stringify(requested)})`);
}

export class EntryHandle {
  // The backing folder is not enumerable, so inspecting a handle shows its path, not the store behind it.
  declare readonly folder: Folder;
  constructor(folder: Folder, readonly path: string) { Object.defineProperty(this, 'folder', { value: folder, enumerable: false }); }
  get name(): string { return this.path.split('/').at(-1) ?? ''; }
  get relativePath(): string { return this.path; }
  get parent(): FolderHandle | null {
    if (!this.path) return null;
    return new FolderHandle(this.folder, this.path.includes('/') ? this.path.slice(0, this.path.lastIndexOf('/')) : '');
  }
  async exists(): Promise<boolean> { return this.folder.exists(this.path); }
  async stat(): Promise<EntryStat> { return this.folder.stat(this.path); }
  async remove(): Promise<void> { this.folder.remove(this.path); }
  /** Move like `mv`: into a folder handle, into a path ending in "/" or naming an existing folder, else to that path. */
  async moveTo(destination: FolderHandle | FileHandle | string): Promise<void> {
    const into = (folderPath: string) => this.folder.join(folderPath, this.name);
    const target = typeof destination !== 'string' ?
      (destination instanceof FolderHandle ? into(destination.path) : destination.path) :
      destination.endsWith('/') || (this.folder.isFolder(destination) && !this.folder.isFile(destination)) ?
        into(destination.replace(/\/+$/, '')) : destination;
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
  dir(path: string): FolderHandle {
    const joined = this.folder.join(this.path, path.replace(/\/+$/, ''));
    checkedKind(this.folder, joined, path, 'folder');
    return new FolderHandle(this.folder, joined);
  }
  file(path: string): FileHandle {
    const joined = this.folder.join(this.path, path.replace(/\/+$/, ''));
    checkedKind(this.folder, joined, path, 'file');
    return new FileHandle(this.folder, joined);
  }
  entry(path: string): EntryHandle { const joined = this.folder.join(this.path, path); return this.folder.isFile(joined) ? new FileHandle(this.folder, joined) : new FolderHandle(this.folder, joined); }
  /** Handle globs are relative to this subtree, as file() and dir() paths are. */
  private scopedPattern(patternText?: string): string | undefined {
    if (patternText === undefined || !this.path || patternText.startsWith(`${this.path}/`)) return patternText;
    return `${this.path}/${patternText}`;
  }
  async entries(patternText?: string): Promise<EntryHandle[]> { return this.folder.list(this.path, this.scopedPattern(patternText)).map(item => this.folder.entry(item.path)); }
  async files(patternText?: string): Promise<FileHandle[]> { return this.folder.listFiles(this.path, this.scopedPattern(patternText)).map(item => new FileHandle(this.folder, item.path)); }
  async folders(patternText?: string): Promise<FolderHandle[]> { return this.folder.listFolders(this.path, this.scopedPattern(patternText)).map(item => new FolderHandle(this.folder, item.path)); }
  async *walk(patternText?: string): AsyncIterable<EntryHandle> { const glob = this.scopedPattern(patternText); const entries = [...this.folder.listFiles(this.path, glob), ...this.folder.listFolders(this.path, glob)].sort((left, right) => left.path.localeCompare(right.path)); for (const entry of entries) yield this.folder.entry(entry.path); }
  snapshot(): FolderSnapshot { return this.folder.snapshot(this.path); }
  at(sourceId:string):FolderSnapshot{return this.folder.at(sourceId,this.path);}
  propose<A extends unknown[], R>(reducer: (folder: any, ...args: A) => Promise<R>, ...args: A): Promise<FolderProposal<R>>;
  propose<R = unknown>(reducer: unknown, ...args: unknown[]): Promise<FolderProposal<R>>;
  propose<R = unknown>(reducer: unknown, ...args: unknown[]): Promise<FolderProposal<R>> { return this.folder.proposeAt<R>(this.path, reducer, args); }
  accept<R>(proposal: FolderProposal<R>): Promise<FolderSnapshot> { return this.folder.acceptAt(this.path, proposal); }
  select(snapshot: FolderSnapshot): Promise<void> { return this.folder.selectAt(this.path, snapshot); }
  iterateOn<S>(reducer: unknown, initial: S, ...args: unknown[]): FolderIteration<S> { return new FolderIteration(this, reducer, initial, args); }
  async diff(): Promise<ChangeSet> { return this.folder.diff(this.path); }
  async beginTransaction(blocking = true): Promise<FolderTransaction> {
    return this.folder.beginTransaction(blocking, this.path);
  }
  /** Run a directory reducer on this folder and install its committed changes; resolves to its typed result. */
  apply<A extends unknown[], R>(reducer: (folder: any, ...args: A) => Promise<R>, ...args: A): Promise<R>;
  apply(reducer: unknown, ...args: unknown[]): Promise<unknown>;
  apply(reducer: unknown, ...args: unknown[]): Promise<unknown> { return applyReducer(this, reducer, args); }
}

export class FolderTransaction {
  private closed = false;
  constructor(readonly parent: Folder, readonly folder: Folder, readonly prefix = '',
    private readonly release: () => void = () => {}, readonly scopedFile?: string) {}
  get open(): boolean { return !this.closed; }
  private ensureOpen(): void { if (this.closed) throw new Error('folder transaction is closed'); }
  private rootedChanges(include?: string[], exclude?: string[]): ChangeSet {
    const selected = selectChanges(this.folder.diffSync(), include, exclude);
    if (this.scopedFile && (selected.changes.some(change => change.path !== this.scopedFile) || selected.moves.length)) {
      const path = selected.changes.find(change => change.path !== this.scopedFile)?.path ?? selected.moves[0]?.[0] ?? '';
      throw new FolderScopeError(this.prefix ? this.parent.join(this.prefix, path) : path);
    }
    const rooted = (path: string) => this.prefix ? this.parent.join(this.prefix, path) : path;
    return { changes: selected.changes.map(change => ({ ...change, path: rooted(change.path) })),
      moves: selected.moves.map(([from, to]) => [rooted(from), rooted(to)] as [string, string]) };
  }
  validateSync(include?: string[], exclude?: string[]): void {
    this.ensureOpen(); this.parent.validateInstall(this.rootedChanges(include, exclude));
  }
  commitSync(include?: string[], exclude?: string[]): ChangeSet {
    this.ensureOpen();
    try {
      const selected = selectChanges(this.folder.diffSync(), include, exclude);
      this.parent.installLocked(this.rootedChanges(include, exclude));
      this.close(); return selected;
    }
    catch (error) { this.abort(); throw error; }
  }
  async commit(include?: string[], exclude?: string[]): Promise<ChangeSet> {
    return this.commitSync(include, exclude);
  }
  abort(): void { this.ensureOpen(); this.close(); }
  private close(): void { if (!this.closed) { this.closed = true; this.release(); } }
}

/**
 * The unchanged contents beneath a folder: a list of file paths and a reader. Sources are read lazily,
 * so a folder can front a large directory on disk. Paths are clean relative POSIX paths.
 */
export interface FolderSource {
  paths(): Iterable<string>;
  get(path: string): Uint8Array | undefined;
  size?(path: string): number | undefined;
}

class MapSource implements FolderSource {
  constructor(private readonly files: Map<string, Uint8Array>) {}
  paths(): Iterable<string> { return this.files.keys(); }
  get(path: string): Uint8Array | undefined { return this.files.get(path); }
  size(path: string): number | undefined { return this.files.get(path)?.length; }
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
  size(path: string): number | undefined {
    const full = this.prefix ? `${this.prefix}/${path}` : path;
    const value = this.overlay.get(full);
    return value === TOMBSTONE ? undefined : value?.length ?? this.base.size?.(full) ?? this.base.get(full)?.length;
  }
}

export class Folder {
  private readonly source: FolderSource;
  private readonly overlay = new Map<string, Uint8Array | typeof TOMBSTONE>();
  private readonly moves: Array<[string, string]> = [];
  private readonly writerLock: WriterLock;
  private revisionNumber = 0;
  private readonly lineage: object;
  private readonly scope: string;
  private readonly computed: Map<string, (folder: Folder) => Uint8Array>;
  private readonly writeFence?: WriteFence;

  constructor(files: Record<string, FileContents> | FolderSource = {}, readonly access: FolderAccess = 'write',
              options: { writer?: WriterLock; computed?: Map<string, (folder: Folder) => Uint8Array>; lineage?: object; scope?: string; writeFence?: WriteFence } = {}) {
    if (!['read', 'write', 'overlay'].includes(access)) throw new RangeError('invalid folder access');
    this.source = isSource(files) ? files :
      new MapSource(new Map(Object.entries(files).map(([path, value]) => [cleanPath(path, false), bytes(value)])));
    this.writerLock = options.writer ?? new WriterLock();
    this.computed = new Map(options.computed);
    this.lineage = options.lineage ?? {};
    this.scope = options.scope ?? '';
    this.writeFence = options.writeFence;
    for (const field of ['source', 'overlay', 'moves', 'writerLock', 'revisionNumber', 'computed', 'lineage', 'scope', 'writeFence']) Object.defineProperty(this, field, { enumerable: false });
  }

  static fromFiles(files: Record<string, FileContents>, access: FolderAccess = 'write'): Folder {
    return new Folder(files, access);
  }
  static fromData(records: Record<string, unknown>[], layout: FolderDataLayout,
    access: FolderAccess = 'write'): Folder { return folderFromData(records, layout, access); }
  root(): FolderHandle { return new FolderHandle(this, ''); }
  registerComputedFile(path: string, render: (folder: Folder) => Uint8Array): void {
    const clean = cleanPath(path, false);
    if (this.isFile(clean) || this.isFolder(clean)) throw new Error(`computed file collides with entry: ${clean}`);
    this.computed.set(clean, render);
  }
  filePaths(): string[] { return this.allPaths(); }
  dir(path = ''): FolderHandle {
    const clean = cleanPath(path.replace(/\/+$/, ''));
    checkedKind(this, clean, path, 'folder');
    return new FolderHandle(this, clean);
  }
  file(path: string): FileHandle {
    const clean = cleanPath(path.replace(/\/+$/, ''), false);
    checkedKind(this, clean, path, 'file');
    return new FileHandle(this, clean);
  }
  /** Run a directory reducer on the whole folder and install its committed changes. */
  apply<A extends unknown[], R>(reducer: (folder: any, ...args: A) => Promise<R>, ...args: A): Promise<R>;
  apply(reducer: unknown, ...args: unknown[]): Promise<unknown>;
  apply(reducer: unknown, ...args: unknown[]): Promise<unknown> { return applyReducer(this.root(), reducer, args); }
  propose<A extends unknown[], R>(reducer: (folder: any, ...args: A) => Promise<R>, ...args: A): Promise<FolderProposal<R>>;
  propose<R = unknown>(reducer: unknown, ...args: unknown[]): Promise<FolderProposal<R>>;
  propose<R = unknown>(reducer: unknown, ...args: unknown[]): Promise<FolderProposal<R>> { return this.proposeAt<R>('', reducer, args); }
  accept<R>(proposal: FolderProposal<R>): Promise<FolderSnapshot> { return this.acceptAt('', proposal); }
  select(snapshot: FolderSnapshot): Promise<void> { return this.selectAt('', snapshot); }
  iterateOn<S>(reducer: unknown, initial: S, ...args: unknown[]): FolderIteration<S> { return new FolderIteration(this.root(), reducer, initial, args); }
  /** Retrieve an already materialized immutable revision of this authorized folder. */
  at(sourceId:string,prefix=''):FolderSnapshot {const found=revisions.get(this.lineage)?.get(this.join(this.scope,cleanPath(prefix))+'\0'+sourceId);if(!found)throw new FolderConflictError('unknown source revision in this folder');return found;}
  /** Materialize immutable bytes: disk-backed sources cannot change this checkpoint afterwards. */
  snapshot(prefix = ''): FolderSnapshot {
    const clean = cleanPath(prefix), files: Record<string, Uint8Array> = {};
    for (const path of this.paths(clean)) files[clean ? path.slice(clean.length + 1) : path] = this.readBytesSync(path);
    return new FolderSnapshot(files, this.lineage, this.join(this.scope, clean));
  }
  async proposeAt<R>(path: string, reducer: unknown, args: unknown[]): Promise<FolderProposal<R>> {
    this.checkWrite();
    currentFrame()?.task.runtime.options.onFolderProposal?.();
    const revision = this.revision(), base = this.snapshot(path), draft = base.branch();
    const value = await draft.apply(reducer, ...args) as R;
    const proposal = new FolderProposal(draft.snapshot(), value, revision, draft.diffSync());
    proposals.set(proposal, { parent: this, path, revision, base: base.digest });
    return proposal;
  }
  async acceptAt<R>(path: string, proposal: FolderProposal<R>): Promise<FolderSnapshot> {
    this.checkWrite();
    const authority = proposals.get(proposal);
    if (!authority || authority.parent !== this || authority.path !== path) throw new FolderConflictError('foreign proposal');
    const release = await this.writerLock.acquire(true, path);
    try {
      if (this.revision() !== authority.revision || this.snapshot(path).digest !== authority.base) throw new FolderConflictError('stale proposal');
      this.replaceSnapshot(path, proposal.folder);
      proposals.delete(proposal);
      return this.snapshot(path);
    } finally { release(); }
  }
  async selectAt(path: string, snapshot: FolderSnapshot): Promise<void> {
    this.checkWrite();
    const authority = snapshots.get(snapshot);
    if (!authority || authority.lineage !== this.lineage || authority.scope !== this.join(this.scope, path)) throw new FolderConflictError('foreign snapshot');
    const revision = this.revision(), base = this.snapshot(path).digest;
    const release = await this.writerLock.acquire(true, path);
    try {
      if (this.revision() !== revision || this.snapshot(path).digest !== base) throw new FolderConflictError('stale selection');
      this.replaceSnapshot(path, snapshot);
    } finally { release(); }
  }
  private replaceSnapshot(path: string, snapshot: FolderSnapshot): void {
    const before = this.snapshot(path), paths = new Set([...before.filePaths(), ...snapshot.filePaths()]);
    const changes: Change[] = [];
    for (const relative of paths) {
      const old = before.isFile(relative) ? before.readBytesSync(relative) : undefined;
      const next = snapshot.isFile(relative) ? snapshot.readBytesSync(relative) : undefined;
      if (!equalBytes(old, next)) changes.push({ path: this.join(path, relative), kind: old === undefined ? 'added' : next === undefined ? 'deleted' : 'modified', before: old, after: next });
    }
    for (const change of changes) this.checkWritablePath(change.path);
    this.installLocked({ changes, moves: [] });
  }
  entry(path: string): EntryHandle { const clean = cleanPath(path); return this.isFile(clean) ? new FileHandle(this, clean) : new FolderHandle(this, clean); }
  join(parent: string, child: string): string { const clean = cleanPath(child); return parent ? (clean ? cleanPath(`${parent}/${clean}`) : parent) : clean; }

  /** The current contents of one file, or undefined. */
  private current(path: string): Uint8Array | undefined {
    const computed = this.computed.get(path);
    if (computed) return computed(this);
    const value = this.overlay.get(path);
    return value === TOMBSTONE ? undefined : value ?? this.source.get(path);
  }
  /** Every current file path. */
  private allPaths(): string[] {
    const result = new Set<string>();
    for (const path of this.source.paths()) if (this.overlay.get(path) !== TOMBSTONE) result.add(path);
    for (const [path, value] of this.overlay) if (value !== TOMBSTONE) result.add(path);
    for (const path of this.computed.keys()) result.add(path);
    return [...result];
  }
  private checkWrite(): void { if (this.access === 'read') throw new Error('folder is read-only'); }
  private checkWritablePath(path: string): void {
    this.checkWrite();
    if (this.writeFence && this.writeFence.path !== path) throw new FolderScopeError(this.writeFence.displayPrefix ? `${this.writeFence.displayPrefix}/${path}` : path);
    if (this.computed.has(path)) throw new Error(`computed file is read-only: ${path}`);
  }
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
    const clean = cleanPath(path), value = clean ? this.overlay.get(clean) : undefined;
    const size = this.computed.has(clean) ? this.computed.get(clean)!(this).length :
      value === TOMBSTONE ? undefined : value?.length ?? this.source.size?.(clean) ?? (clean ? this.source.get(clean)?.length : undefined);
    if (size !== undefined) return { path: clean, kind: 'file', bytes: size, digest: null };
    if (this.isFolder(clean)) return { path: clean, kind: 'folder', bytes: 0, digest: null };
    throw new Error(`entry not found: ${clean}`);
  }
  revision(): number { return this.revisionNumber; }
  kind(path: string): 'file' | 'dir' | null { return this.isFile(path) ? 'file' : this.isFolder(path) ? 'dir' : null; }
  childNames(path = ''): string[] {
    const prefix = path ? `${cleanPath(path)}/` : '';
    return this.list(path).map(entry => entry.path.slice(prefix.length));
  }
  size(path: string): number { return this.statSync(path).bytes; }
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
  writeBytes(path: string, content: FileContents): void {
    const clean = cleanPath(path, false); this.checkWritablePath(clean);
    this.overlay.set(clean, bytes(content)); this.revisionNumber++;
  }
  writeText(path: string, content: string): void { this.writeBytes(path, content); }
  async writeJson(path: string, value: unknown): Promise<void> { this.writeText(path, `${JSON.stringify(value, null, 2)}\n`); }
  remove(path: string): void {
    const clean = cleanPath(path, false); this.checkWritablePath(clean);
    if ([...this.computed.keys()].some(item => under(item, clean))) throw new Error(`computed file is read-only: ${clean}`);
    if (!this.isFile(clean) && !this.isFolder(clean)) throw new Error(`entry not found: ${clean}`);
    for (const item of this.allPaths()) if (item === clean || item.startsWith(`${clean}/`)) this.overlay.set(item, TOMBSTONE);
    this.revisionNumber++;
  }
  move(source: string, destination: string): void {
    const from = cleanPath(source, false), to = cleanPath(destination, false);
    this.checkWritablePath(from); this.checkWritablePath(to);
    if ([...this.computed.keys()].some(item => under(item, from) || under(item, to)))
      throw new Error(`computed file is read-only: ${from}`);
    if (from === to || to.startsWith(`${from}/`)) throw new Error('cannot move an entry into itself');
    const selected = this.allPaths().filter(path => path === from || path.startsWith(`${from}/`));
    if (!selected.length) throw new Error(`entry not found: ${from}`);
    if (this.isFile(to) || this.isFolder(to)) throw new Error(`destination exists: ${to}`);
    for (const path of selected) {
      const suffix = path.slice(from.length).replace(/^\//, '');
      this.overlay.set(`${to}${suffix ? `/${suffix}` : ''}`, this.current(path)!); this.overlay.set(path, TOMBSTONE);
    }
    this.moves.push([from, to]);
    this.revisionNumber++;
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
  fork(access: FolderAccess = 'overlay'): Folder { return new Folder(this.snapshotSource(), access, { writer: this.writerLock, computed: this.computed, lineage: this.lineage, scope: this.scope, writeFence: this.writeFence }); }
  private snapshotSource(prefix = ''): FolderSource { return new LayeredSource(this.source, new Map(this.overlay), prefix); }
  writer(): WriterLock { return this.writerLock; }
  async beginTransaction(blocking = true, path = ''): Promise<FolderTransaction> {
    const prefix = cleanPath(path);
    const release = this.access === 'read' ? () => {} : await this.writerLock.acquire(blocking, prefix);
    const computed = new Map([...this.computed].filter(([item]) => under(item, prefix)).map(([item, render]) =>
      [prefix ? item.slice(prefix.length + 1) : item, render] as [string, (folder: Folder) => Uint8Array]));
    const writeFence = this.writeFence ? {
      path: this.writeFence.path === null ? null : this.writeFence.path === prefix ? null :
        prefix && this.writeFence.path.startsWith(`${prefix}/`) ? this.writeFence.path.slice(prefix.length + 1) :
          !prefix && this.writeFence.path ? this.writeFence.path : null,
      displayPrefix: this.writeFence.displayPrefix,
    } : undefined;
    return new FolderTransaction(this, new Folder(this.snapshotSource(prefix), this.access === 'read' ? 'read' : 'overlay', { computed, lineage: this.lineage, scope: this.join(this.scope, prefix), writeFence }), prefix, release);
  }
  /** A one-file capability for a child call. Reads and writes are limited to this file; pass another FileHandle for an output. */
  async beginFileTransaction(path: string, blocking = true): Promise<FolderTransaction> {
    const clean = cleanPath(path, false);
    const name = clean.split('/').at(-1)!, parent = clean.includes('/') ? clean.slice(0, clean.lastIndexOf('/')) : '';
    const release = this.access === 'read' ? () => {} : await this.writerLock.acquire(blocking, clean);
    try {
      const content = this.readBytesSync(clean);
      return new FolderTransaction(this, new Folder({ [name]: content }, this.access === 'read' ? 'read' : 'overlay', {
        writeFence: { path: name, displayPrefix: parent },
      }), parent, release, name);
    } catch (error) { release(); throw error; }
  }
  installLocked(changes: ChangeSet, include?: string[], exclude?: string[]): ChangeSet {
    const selectedSet = selectChanges(changes, include, exclude), selected = selectedSet.changes;
    if (selected.length) this.checkWrite();
    for (const change of selected) this.checkWritablePath(change.path);
    this.validateInstall(selectedSet);
    for (const change of selected) this.overlay.set(change.path, change.after === undefined ? TOMBSTONE : change.after.slice());
    if (selected.length) this.revisionNumber++;
    return selectedSet;
  }
  validateInstall(changes: ChangeSet): void {
    for (const change of changes.changes)
      if (!equalBytes(this.current(change.path), change.before)) throw new FolderConflictError(change.path);
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

/** Proposals are capabilities, not serializable patches. Only their originating folder may accept them. */
/** Live proposal capability: preserve identity across interpreter eval boundaries. */
export class FolderProposal<R = unknown> {
  constructor(readonly folder: FolderSnapshot, readonly value: R, readonly baseRevision: number, readonly diff: ChangeSet) { Object.freeze(this); }
}
const proposals = new WeakMap<object, { parent: Folder; path: string; revision: number; base: string }>();
const revisions=new WeakMap<object,Map<string,FolderSnapshot>>();
const snapshots = new WeakMap<object, { lineage: object; scope: string }>();
/** Immutable, content-addressed source. Branching preserves target ownership, never shares a writer lock. */
export class FolderSnapshot extends Folder {
  readonly digest: string;
  constructor(files: Record<string, FileContents>, lineage: object = {}, scope = '') {
    super(files, 'read', { lineage, scope });
    snapshots.set(this, { lineage, scope });
    this.digest = hexDigest(new TextEncoder().encode(JSON.stringify(this.filePaths().sort().map(path => [path, hexDigest(this.readBytesSync(path))]))));
    let known=revisions.get(lineage);if(!known){known=new Map();revisions.set(lineage,known);}known.set(scope+'\0'+this.digest,this);
    Object.freeze(this);
  }
  branch(): Folder {
    const authority = snapshots.get(this)!;
    return new Folder(Object.fromEntries(this.filePaths().map(path => [path, this.readBytesSync(path)])), 'overlay', authority);
  }
  override registerComputedFile(): never { throw new Error('snapshot is immutable'); }
  override fork(access: FolderAccess = 'overlay'): Folder { return access === 'read' ? this : this.branch(); }
}
