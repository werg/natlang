/** Standalone copy-on-write folder handles for scoped lambda filesystems. */
import { currentFrame } from '../runtime/context.js';
import { FolderIteration } from '../runtime/folder-iteration.js';
import { sha256 } from '@noble/hashes/sha2.js';
import { folderFromData } from './data-layout.js';
const TOMBSTONE = Symbol('natlang-folder-tombstone');
function cleanPath(path, allowRoot = true) {
    if (typeof path !== 'string')
        throw new TypeError('path must be a string');
    if (allowRoot && (path === '' || path === '.'))
        return '';
    if (!path || path.startsWith('/') || path.includes('\\') || path.includes('\0'))
        throw new RangeError(`invalid relative POSIX path: ${JSON.stringify(path)}`);
    const parts = path.split('/');
    if (parts.some(part => !part || part === '.' || part === '..'))
        throw new RangeError(`invalid relative POSIX path: ${JSON.stringify(path)}`);
    return parts.join('/');
}
function bytes(value) {
    return typeof value === 'string' ? new TextEncoder().encode(value) : value.slice();
}
function text(value) { return new TextDecoder('utf-8', { fatal: true }).decode(value); }
function equalBytes(left, right) {
    if (left === undefined || right === undefined)
        return left === right;
    return left.length === right.length && left.every((item, index) => item === right[index]);
}
function hexDigest(value) {
    return Array.from(sha256(value), byte => byte.toString(16).padStart(2, '0')).join('');
}
function under(path, parent) {
    return !parent || path === parent || path.startsWith(`${parent}/`);
}
function pattern(pattern) {
    if (pattern === undefined)
        return () => true;
    cleanPath(pattern.replaceAll('**', 'x'));
    // A globstar directory prefix includes zero directories: **/*.md must also
    // find root-level Markdown files in a one-file FileHandle transaction.
    let source = '';
    for (let index = 0; index < pattern.length; index++) {
        if (pattern[index] !== '*') {
            source += escapeRegExp(pattern.charAt(index));
            continue;
        }
        if (pattern[index + 1] !== '*') {
            source += '[^/]*';
            continue;
        }
        index++;
        if (pattern[index + 1] === '/' && (index === 1 || pattern[index - 2] === '/')) {
            source += '(?:[^/]+/)*';
            index++;
        }
        else
            source += '.*';
    }
    const expression = new RegExp(`^${source}$`);
    return path => expression.test(path);
}
function escapeRegExp(value) { return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }
/** What list_files shows the model: one `path  N bytes` line per file. */
export function fileListingText(entries) {
    return entries.length ? entries.map(entry => `${entry.path}  ${entry.bytes} bytes`).join('\n') : '(no files)';
}
function selectChanges(changes, include, exclude) {
    const includeMatches = (include ?? []).map(pattern), excludeMatches = (exclude ?? []).map(pattern);
    const selected = changes.changes.filter(change => (!includeMatches.length || includeMatches.some(match => match(change.path))) &&
        !excludeMatches.some(match => match(change.path)));
    return { changes: selected,
        moves: changes.moves.filter(([from, to]) => selected.some(change => change.path === from || change.path === to)) };
}
export class FolderBusyError extends Error {
    constructor() { super('folder writer is busy'); }
}
export class FolderConflictError extends Error {
    constructor(path) { super(`folder changed at ${path}`); }
}
export class FolderScopeError extends Error {
    constructor(path) {
        super(`write to ${JSON.stringify(path)} is outside the supplied FileHandle scope; pass that output FileHandle to the child or return the value to the parent to write`);
        this.name = 'FolderScopeError';
    }
}
/** A subtree writer lock. Disjoint directories may run and commit together. */
export class WriterLock {
    held = new Set();
    waiters = [];
    async acquire(blocking = true, path = '') {
        const conflicts = () => [...this.held].some(item => under(item, path) || under(path, item));
        while (conflicts()) {
            if (!blocking)
                throw new FolderBusyError();
            await new Promise(resolve => this.waiters.push(resolve));
        }
        this.held.add(path);
        return () => this.release(path);
    }
    release(path = '') {
        if (!this.held.delete(path))
            throw new Error('folder writer is not held');
        const waiters = this.waiters.splice(0);
        for (const wake of waiters)
            wake();
    }
}
/**
 * A function that can run as a directory reducer on a folder exposes this method. `folder.apply(reducer, ...args)`
 * runs it with a private copy of the folder and installs its committed changes.
 */
export const APPLY_TO_FOLDER = Symbol.for('natlang.applyToFolder');
function applyReducer(folder, reducer, args) {
    const apply = reducer?.[APPLY_TO_FOLDER];
    if (typeof apply === 'function')
        return apply.call(reducer, folder, args);
    // A natural-language function that is not a directory reducer takes no folder; it is not applied as a plain function.
    if (typeof reducer === 'function' && !Object.hasOwn(reducer, Symbol.for('natlang.callable')))
        return applyFunction(folder, reducer, args);
    throw new TypeError('folder.apply needs a directory reducer: a natlang `kind: directory-reducer` function or a host/TypeScript function (folder, ...args)');
}
/**
 * A plain (host or TypeScript) function as a directory reducer. It has the contract of a natlang one: it runs on a
 * private transaction copy of the folder and receives that copy as its first argument; its changes are validated and
 * installed when it returns, and discarded when it throws.
 */
async function applyFunction(folder, reducer, args) {
    const transaction = await folder.beginTransaction(true);
    try {
        const value = await reducer(transaction.folder.root(), ...args);
        transaction.validateSync();
        transaction.commitSync();
        return value;
    }
    finally {
        if (transaction.open)
            transaction.abort();
    }
}
/** Replace exactly one span of `text`; `fuzzy` matches one whole line ignoring whitespace differences. */
export function editTextContent(text, find, replaceWith, fuzzy = false) {
    let count = find ? text.split(find).length - 1 : 0;
    if (count !== 1 && fuzzy) {
        const wanted = find.replace(/\s+/g, ' ').trim();
        const candidates = [...text.matchAll(/[^\n]*(?:\n|$)/g)].filter(match => match[0] && match[0].replace(/\s+/g, ' ').trim() === wanted);
        count = candidates.length;
        if (count === 1) {
            const match = candidates[0], newline = match[0].endsWith('\n') ? '\n' : '';
            return text.slice(0, match.index) + replaceWith + newline + text.slice(match.index + match[0].length);
        }
    }
    if (count === 0) {
        if (fuzzy)
            throw new Error('edit found no matching span, even with fuzzy: true; inspect the current contents');
        throw new Error('edit found no matching span; retry with fuzzy: true for whitespace-tolerant matching or inspect the current contents');
    }
    if (count !== 1)
        throw new Error(`edit found ${count} matching spans; make find more specific so it matches exactly one span`);
    return text.replace(find, () => replaceWith);
}
/** The handle kind a path needs: asking for a file handle on a folder (or the reverse) is a mistake worth naming. */
function checkedKind(folder, path, requested, want) {
    if (want === 'file' && (requested.endsWith('/') || (path && folder.isFolder(path) && !folder.isFile(path))))
        throw new Error(`${requested} is a folder; use dir(${JSON.stringify(requested.replace(/\/+$/, ''))})`);
    if (want === 'folder' && folder.isFile(path))
        throw new Error(`${requested} is a file; use file(${JSON.stringify(requested)})`);
}
export class EntryHandle {
    path;
    constructor(folder, path) {
        this.path = path;
        Object.defineProperty(this, 'folder', { value: folder, enumerable: false });
    }
    get name() { return this.path.split('/').at(-1) ?? ''; }
    get relativePath() { return this.path; }
    get parent() {
        if (!this.path)
            return null;
        return new FolderHandle(this.folder, this.path.includes('/') ? this.path.slice(0, this.path.lastIndexOf('/')) : '');
    }
    async exists() { return this.folder.exists(this.path); }
    async stat() { return this.folder.stat(this.path); }
    async remove() { this.folder.remove(this.path); }
    /** Move like `mv`: into a folder handle, into a path ending in "/" or naming an existing folder, else to that path. */
    async moveTo(destination) {
        const into = (folderPath) => this.folder.join(folderPath, this.name);
        const target = typeof destination !== 'string' ?
            (destination instanceof FolderHandle ? into(destination.path) : destination.path) :
            destination.endsWith('/') || (this.folder.isFolder(destination) && !this.folder.isFile(destination)) ?
                into(destination.replace(/\/+$/, '')) : destination;
        this.folder.move(this.path, target);
    }
}
export class FileHandle extends EntryHandle {
    async readText(startLine, endLine) {
        return this.folder.readText(this.path, startLine, endLine);
    }
    async readBytes() { return this.folder.readBytes(this.path); }
    async readJson() { return JSON.parse(await this.readText()); }
    async writeText(content) { this.folder.writeText(this.path, content); }
    async writeBytes(content) { this.folder.writeBytes(this.path, content); }
    async writeJson(value) { await this.writeText(`${JSON.stringify(value, null, 2)}\n`); }
    async editText(find, replaceWith, fuzzy = false) {
        return this.folder.editText(this.path, find, replaceWith, fuzzy);
    }
}
export class FolderHandle extends EntryHandle {
    dir(path) {
        const joined = this.folder.join(this.path, path.replace(/\/+$/, ''));
        checkedKind(this.folder, joined, path, 'folder');
        return new FolderHandle(this.folder, joined);
    }
    file(path) {
        const joined = this.folder.join(this.path, path.replace(/\/+$/, ''));
        checkedKind(this.folder, joined, path, 'file');
        return new FileHandle(this.folder, joined);
    }
    entry(path) { const joined = this.folder.join(this.path, path); return this.folder.isFile(joined) ? new FileHandle(this.folder, joined) : new FolderHandle(this.folder, joined); }
    /** Handle globs are relative to this subtree, as file() and dir() paths are. */
    scopedPattern(patternText) {
        if (patternText === undefined || !this.path || patternText.startsWith(`${this.path}/`))
            return patternText;
        return `${this.path}/${patternText}`;
    }
    async entries(patternText) { return this.folder.list(this.path, this.scopedPattern(patternText)).map(item => this.folder.entry(item.path)); }
    async files(patternText) { return this.folder.listFiles(this.path, this.scopedPattern(patternText)).map(item => new FileHandle(this.folder, item.path)); }
    async folders(patternText) { return this.folder.listFolders(this.path, this.scopedPattern(patternText)).map(item => new FolderHandle(this.folder, item.path)); }
    async *walk(patternText) { const glob = this.scopedPattern(patternText); const entries = [...this.folder.listFiles(this.path, glob), ...this.folder.listFolders(this.path, glob)].sort((left, right) => left.path.localeCompare(right.path)); for (const entry of entries)
        yield this.folder.entry(entry.path); }
    snapshot() { return this.folder.snapshot(this.path); }
    at(sourceId) { return this.folder.at(sourceId, this.path); }
    propose(reducer, ...args) { return this.folder.proposeAt(this.path, reducer, args); }
    accept(proposal) { return this.folder.acceptAt(this.path, proposal); }
    select(snapshot) { return this.folder.selectAt(this.path, snapshot); }
    iterateOn(reducer, initial, ...args) { return new FolderIteration(this, reducer, initial, args); }
    async diff() { return this.folder.diff(this.path); }
    async beginTransaction(blocking = true) {
        return this.folder.beginTransaction(blocking, this.path);
    }
    apply(reducer, ...args) { return applyReducer(this, reducer, args); }
}
export class FolderTransaction {
    parent;
    folder;
    prefix;
    release;
    scopedFile;
    closed = false;
    constructor(parent, folder, prefix = '', release = () => { }, scopedFile) {
        this.parent = parent;
        this.folder = folder;
        this.prefix = prefix;
        this.release = release;
        this.scopedFile = scopedFile;
    }
    get open() { return !this.closed; }
    ensureOpen() { if (this.closed)
        throw new Error('folder transaction is closed'); }
    rootedChanges(include, exclude) {
        const selected = selectChanges(this.folder.diffSync(), include, exclude);
        if (this.scopedFile && (selected.changes.some(change => change.path !== this.scopedFile) || selected.moves.length)) {
            const path = selected.changes.find(change => change.path !== this.scopedFile)?.path ?? selected.moves[0]?.[0] ?? '';
            throw new FolderScopeError(this.prefix ? this.parent.join(this.prefix, path) : path);
        }
        const rooted = (path) => this.prefix ? this.parent.join(this.prefix, path) : path;
        return { changes: selected.changes.map(change => ({ ...change, path: rooted(change.path) })),
            moves: selected.moves.map(([from, to]) => [rooted(from), rooted(to)]) };
    }
    validateSync(include, exclude) {
        this.ensureOpen();
        this.parent.validateInstall(this.rootedChanges(include, exclude));
    }
    commitSync(include, exclude) {
        this.ensureOpen();
        try {
            const selected = selectChanges(this.folder.diffSync(), include, exclude);
            this.parent.installLocked(this.rootedChanges(include, exclude));
            this.close();
            return selected;
        }
        catch (error) {
            this.abort();
            throw error;
        }
    }
    async commit(include, exclude) {
        return this.commitSync(include, exclude);
    }
    abort() { this.ensureOpen(); this.close(); }
    close() { if (!this.closed) {
        this.closed = true;
        this.release();
    } }
}
class MapSource {
    files;
    constructor(files) {
        this.files = files;
    }
    paths() { return this.files.keys(); }
    get(path) { return this.files.get(path); }
    size(path) { return this.files.get(path)?.length; }
}
/** A frozen view of a source with an overlay applied: forks and transactions start from one. */
class LayeredSource {
    base;
    overlay;
    prefix;
    constructor(base, overlay, prefix = '') {
        this.base = base;
        this.overlay = overlay;
        this.prefix = prefix;
    }
    *paths() {
        const seen = new Set();
        for (const path of [...this.overlay.keys(), ...this.base.paths()]) {
            if (seen.has(path))
                continue;
            seen.add(path);
            if (this.overlay.get(path) === TOMBSTONE || (this.prefix && !path.startsWith(`${this.prefix}/`)))
                continue;
            yield this.prefix ? path.slice(this.prefix.length + 1) : path;
        }
    }
    get(path) {
        const full = this.prefix ? `${this.prefix}/${path}` : path;
        const value = this.overlay.get(full);
        return value === TOMBSTONE ? undefined : value ?? this.base.get(full);
    }
    size(path) {
        const full = this.prefix ? `${this.prefix}/${path}` : path;
        const value = this.overlay.get(full);
        return value === TOMBSTONE ? undefined : value?.length ?? this.base.size?.(full) ?? this.base.get(full)?.length;
    }
}
export class Folder {
    access;
    source;
    overlay = new Map();
    moves = [];
    writerLock;
    revisionNumber = 0;
    lineage;
    scope;
    computed;
    writeFence;
    constructor(files = {}, access = 'write', options = {}) {
        this.access = access;
        if (!['read', 'write', 'overlay'].includes(access))
            throw new RangeError('invalid folder access');
        this.source = isSource(files) ? files :
            new MapSource(new Map(Object.entries(files).map(([path, value]) => [cleanPath(path, false), bytes(value)])));
        this.writerLock = options.writer ?? new WriterLock();
        this.computed = new Map(options.computed);
        this.lineage = options.lineage ?? {};
        this.scope = options.scope ?? '';
        this.writeFence = options.writeFence;
        for (const field of ['source', 'overlay', 'moves', 'writerLock', 'revisionNumber', 'computed', 'lineage', 'scope', 'writeFence'])
            Object.defineProperty(this, field, { enumerable: false });
    }
    static fromFiles(files, access = 'write') {
        return new Folder(files, access);
    }
    static fromData(records, layout, access = 'write') { return folderFromData(records, layout, access); }
    root() { return new FolderHandle(this, ''); }
    registerComputedFile(path, render) {
        const clean = cleanPath(path, false);
        if (this.isFile(clean) || this.isFolder(clean))
            throw new Error(`computed file collides with entry: ${clean}`);
        this.computed.set(clean, render);
    }
    filePaths() { return this.allPaths(); }
    dir(path = '') {
        const clean = cleanPath(path.replace(/\/+$/, ''));
        checkedKind(this, clean, path, 'folder');
        return new FolderHandle(this, clean);
    }
    file(path) {
        const clean = cleanPath(path.replace(/\/+$/, ''), false);
        checkedKind(this, clean, path, 'file');
        return new FileHandle(this, clean);
    }
    apply(reducer, ...args) { return applyReducer(this.root(), reducer, args); }
    propose(reducer, ...args) { return this.proposeAt('', reducer, args); }
    accept(proposal) { return this.acceptAt('', proposal); }
    select(snapshot) { return this.selectAt('', snapshot); }
    iterateOn(reducer, initial, ...args) { return new FolderIteration(this.root(), reducer, initial, args); }
    /** Retrieve an already materialized immutable revision of this authorized folder. */
    at(sourceId, prefix = '') { const found = revisions.get(this.lineage)?.get(this.join(this.scope, cleanPath(prefix)) + '\0' + sourceId); if (!found)
        throw new FolderConflictError('unknown source revision in this folder'); return found; }
    /** Materialize immutable bytes: disk-backed sources cannot change this checkpoint afterwards. */
    snapshot(prefix = '') {
        const clean = cleanPath(prefix), files = {};
        for (const path of this.paths(clean))
            files[clean ? path.slice(clean.length + 1) : path] = this.readBytesSync(path);
        return new FolderSnapshot(files, this.lineage, this.join(this.scope, clean));
    }
    async proposeAt(path, reducer, args) {
        this.checkWrite();
        currentFrame()?.task.runtime.options.onFolderProposal?.();
        const revision = this.revision(), base = this.snapshot(path), draft = base.branch();
        const value = await draft.apply(reducer, ...args);
        const proposal = new FolderProposal(draft.snapshot(), value, revision, draft.diffSync());
        proposals.set(proposal, { parent: this, path, revision, base: base.digest });
        return proposal;
    }
    async acceptAt(path, proposal) {
        this.checkWrite();
        const authority = proposals.get(proposal);
        if (!authority || authority.parent !== this || authority.path !== path)
            throw new FolderConflictError('foreign proposal');
        const release = await this.writerLock.acquire(true, path);
        try {
            if (this.revision() !== authority.revision || this.snapshot(path).digest !== authority.base)
                throw new FolderConflictError('stale proposal');
            this.replaceSnapshot(path, proposal.folder);
            proposals.delete(proposal);
            return this.snapshot(path);
        }
        finally {
            release();
        }
    }
    async selectAt(path, snapshot) {
        this.checkWrite();
        const authority = snapshots.get(snapshot);
        if (!authority || authority.lineage !== this.lineage || authority.scope !== this.join(this.scope, path))
            throw new FolderConflictError('foreign snapshot');
        const revision = this.revision(), base = this.snapshot(path).digest;
        const release = await this.writerLock.acquire(true, path);
        try {
            if (this.revision() !== revision || this.snapshot(path).digest !== base)
                throw new FolderConflictError('stale selection');
            this.replaceSnapshot(path, snapshot);
        }
        finally {
            release();
        }
    }
    replaceSnapshot(path, snapshot) {
        const before = this.snapshot(path), paths = new Set([...before.filePaths(), ...snapshot.filePaths()]);
        const changes = [];
        for (const relative of paths) {
            const old = before.isFile(relative) ? before.readBytesSync(relative) : undefined;
            const next = snapshot.isFile(relative) ? snapshot.readBytesSync(relative) : undefined;
            if (!equalBytes(old, next))
                changes.push({ path: this.join(path, relative), kind: old === undefined ? 'added' : next === undefined ? 'deleted' : 'modified', before: old, after: next });
        }
        for (const change of changes)
            this.checkWritablePath(change.path);
        this.installLocked({ changes, moves: [] });
    }
    entry(path) { const clean = cleanPath(path); return this.isFile(clean) ? new FileHandle(this, clean) : new FolderHandle(this, clean); }
    join(parent, child) { const clean = cleanPath(child); return parent ? (clean ? cleanPath(`${parent}/${clean}`) : parent) : clean; }
    /** The current contents of one file, or undefined. */
    current(path) {
        const computed = this.computed.get(path);
        if (computed)
            return computed(this);
        const value = this.overlay.get(path);
        return value === TOMBSTONE ? undefined : value ?? this.source.get(path);
    }
    /** Every current file path. */
    allPaths() {
        const result = new Set();
        for (const path of this.source.paths())
            if (this.overlay.get(path) !== TOMBSTONE)
                result.add(path);
        for (const [path, value] of this.overlay)
            if (value !== TOMBSTONE)
                result.add(path);
        for (const path of this.computed.keys())
            result.add(path);
        return [...result];
    }
    checkWrite() { if (this.access === 'read')
        throw new Error('folder is read-only'); }
    checkWritablePath(path) {
        this.checkWrite();
        if (this.writeFence && this.writeFence.path !== path)
            throw new FolderScopeError(this.writeFence.displayPrefix ? `${this.writeFence.displayPrefix}/${path}` : path);
        if (this.computed.has(path))
            throw new Error(`computed file is read-only: ${path}`);
    }
    read(path) {
        const value = this.current(cleanPath(path, false));
        if (!value)
            throw new Error(`file not found: ${path}`);
        return value;
    }
    async exists(path) { const clean = cleanPath(path); return this.isFile(clean) || this.isFolder(clean); }
    isFile(path) { const clean = cleanPath(path); return !!clean && this.current(clean) !== undefined; }
    isFolder(path) { const clean = cleanPath(path); return !clean || this.allPaths().some(item => item.startsWith(`${clean}/`)); }
    async stat(path) { return this.statSync(path); }
    statSync(path) {
        const clean = cleanPath(path), value = clean ? this.overlay.get(clean) : undefined;
        const size = this.computed.has(clean) ? this.computed.get(clean)(this).length :
            value === TOMBSTONE ? undefined : value?.length ?? this.source.size?.(clean) ?? (clean ? this.source.get(clean)?.length : undefined);
        if (size !== undefined)
            return { path: clean, kind: 'file', bytes: size, digest: null };
        if (this.isFolder(clean))
            return { path: clean, kind: 'folder', bytes: 0, digest: null };
        throw new Error(`entry not found: ${clean}`);
    }
    revision() { return this.revisionNumber; }
    kind(path) { return this.isFile(path) ? 'file' : this.isFolder(path) ? 'dir' : null; }
    childNames(path = '') {
        const prefix = path ? `${cleanPath(path)}/` : '';
        return this.list(path).map(entry => entry.path.slice(prefix.length));
    }
    size(path) { return this.statSync(path).bytes; }
    paths(path = '') { const clean = cleanPath(path); return this.allPaths().filter(item => under(item, clean)).sort(); }
    list(path = '', patternText) {
        const clean = cleanPath(path), children = new Map();
        for (const item of this.paths(clean)) {
            const rest = clean ? item.slice(clean.length + 1) : item, child = this.join(clean, rest.split('/')[0]);
            children.set(child, rest.includes('/') || children.get(child) === 'folder' ? 'folder' : 'file');
        }
        const matches = pattern(patternText);
        return [...children].sort(([left], [right]) => left.localeCompare(right)).filter(([path]) => matches(path)).map(([path]) => this.statSync(path));
    }
    listFiles(path = '', patternText) { const matches = pattern(patternText); return this.paths(path).filter(path => matches(path)).map(path => this.statSync(path)); }
    listFolders(path = '', patternText) {
        const clean = cleanPath(path), candidates = new Set();
        for (const item of this.paths(clean)) {
            const parts = item.split('/');
            for (let i = 1; i < parts.length; i++)
                candidates.add(parts.slice(0, i).join('/'));
        }
        const matches = pattern(patternText);
        return [...candidates].filter(item => item !== clean && under(item, clean) && matches(item)).sort().map(item => this.statSync(item));
    }
    async readBytes(path) { return this.read(path).slice(); }
    readBytesSync(path) { return this.read(path).slice(); }
    async readText(path, startLine, endLine) {
        const value = text(this.read(path));
        if (startLine === undefined && endLine === undefined)
            return value;
        const start = startLine ?? 1, end = endLine ?? start;
        if (start < 1 || end < start)
            throw new RangeError('invalid line range');
        return value.split(/(?<=\n)/).slice(start - 1, end).join('');
    }
    async readJson(path) { return JSON.parse(await this.readText(path)); }
    writeBytes(path, content) {
        const clean = cleanPath(path, false);
        this.checkWritablePath(clean);
        this.overlay.set(clean, bytes(content));
        this.revisionNumber++;
    }
    writeText(path, content) { this.writeBytes(path, content); }
    async writeJson(path, value) { this.writeText(path, `${JSON.stringify(value, null, 2)}\n`); }
    remove(path) {
        const clean = cleanPath(path, false);
        this.checkWritablePath(clean);
        if ([...this.computed.keys()].some(item => under(item, clean)))
            throw new Error(`computed file is read-only: ${clean}`);
        if (!this.isFile(clean) && !this.isFolder(clean))
            throw new Error(`entry not found: ${clean}`);
        for (const item of this.allPaths())
            if (item === clean || item.startsWith(`${clean}/`))
                this.overlay.set(item, TOMBSTONE);
        this.revisionNumber++;
    }
    move(source, destination) {
        const from = cleanPath(source, false), to = cleanPath(destination, false);
        this.checkWritablePath(from);
        this.checkWritablePath(to);
        if ([...this.computed.keys()].some(item => under(item, from) || under(item, to)))
            throw new Error(`computed file is read-only: ${from}`);
        if (from === to || to.startsWith(`${from}/`))
            throw new Error('cannot move an entry into itself');
        const selected = this.allPaths().filter(path => path === from || path.startsWith(`${from}/`));
        if (!selected.length)
            throw new Error(`entry not found: ${from}`);
        if (this.isFile(to) || this.isFolder(to))
            throw new Error(`destination exists: ${to}`);
        for (const path of selected) {
            const suffix = path.slice(from.length).replace(/^\//, '');
            this.overlay.set(`${to}${suffix ? `/${suffix}` : ''}`, this.current(path));
            this.overlay.set(path, TOMBSTONE);
        }
        this.moves.push([from, to]);
        this.revisionNumber++;
    }
    async editText(path, find, replaceWith, fuzzy = false) {
        const updated = editTextContent(await this.readText(path), find, replaceWith, fuzzy);
        this.writeText(path, updated);
        return { path: cleanPath(path), changed: true, digest: hexDigest(new TextEncoder().encode(updated)) };
    }
    async search(query, path = '', patternText, regex = false) {
        if (!query)
            throw new RangeError('search query must be nonempty');
        const expression = regex ? new RegExp(query) : undefined, result = [];
        for (const item of this.listFiles(path, patternText)) {
            let content;
            try {
                content = await this.readText(item.path);
            }
            catch {
                continue;
            }
            content.split(/\r?\n/).forEach((line, index) => { if (expression ? expression.test(line) : line.includes(query))
                result.push({ path: item.path, line: index + 1, text: line }); });
        }
        return result;
    }
    /** Changes relative to the source; only overlaid paths can differ. */
    diffSync(path = '') {
        const clean = cleanPath(path), changes = [];
        for (const item of [...this.overlay.keys()].filter(item => under(item, clean)).sort()) {
            const before = this.source.get(item), after = this.current(item);
            if (equalBytes(before, after))
                continue;
            changes.push({ path: item, kind: before === undefined ? 'added' : after === undefined ? 'deleted' : 'modified',
                ...(before === undefined ? {} : { before: before.slice() }), ...(after === undefined ? {} : { after: after.slice() }) });
        }
        return { changes, moves: this.moves.map(move => [...move]) };
    }
    async diff(path = '') { return this.diffSync(path); }
    /** A copy-on-write fork of the current contents, sharing this folder's writer lock. */
    fork(access = 'overlay') { return new Folder(this.snapshotSource(), access, { writer: this.writerLock, computed: this.computed, lineage: this.lineage, scope: this.scope, writeFence: this.writeFence }); }
    snapshotSource(prefix = '') { return new LayeredSource(this.source, new Map(this.overlay), prefix); }
    writer() { return this.writerLock; }
    async beginTransaction(blocking = true, path = '') {
        const prefix = cleanPath(path);
        const release = this.access === 'read' ? () => { } : await this.writerLock.acquire(blocking, prefix);
        const computed = new Map([...this.computed].filter(([item]) => under(item, prefix)).map(([item, render]) => [prefix ? item.slice(prefix.length + 1) : item, render]));
        const writeFence = this.writeFence ? {
            path: this.writeFence.path === null ? null : this.writeFence.path === prefix ? null :
                prefix && this.writeFence.path.startsWith(`${prefix}/`) ? this.writeFence.path.slice(prefix.length + 1) :
                    !prefix && this.writeFence.path ? this.writeFence.path : null,
            displayPrefix: this.writeFence.displayPrefix,
        } : undefined;
        return new FolderTransaction(this, new Folder(this.snapshotSource(prefix), this.access === 'read' ? 'read' : 'overlay', { computed, lineage: this.lineage, scope: this.join(this.scope, prefix), writeFence }), prefix, release);
    }
    /** A one-file capability for a child call. Reads and writes are limited to this file; pass another FileHandle for an output. */
    async beginFileTransaction(path, blocking = true) {
        const clean = cleanPath(path, false);
        const name = clean.split('/').at(-1), parent = clean.includes('/') ? clean.slice(0, clean.lastIndexOf('/')) : '';
        const release = this.access === 'read' ? () => { } : await this.writerLock.acquire(blocking, clean);
        try {
            const content = this.readBytesSync(clean);
            return new FolderTransaction(this, new Folder({ [name]: content }, this.access === 'read' ? 'read' : 'overlay', {
                writeFence: { path: name, displayPrefix: parent },
            }), parent, release, name);
        }
        catch (error) {
            release();
            throw error;
        }
    }
    installLocked(changes, include, exclude) {
        const selectedSet = selectChanges(changes, include, exclude), selected = selectedSet.changes;
        if (selected.length)
            this.checkWrite();
        for (const change of selected)
            this.checkWritablePath(change.path);
        this.validateInstall(selectedSet);
        for (const change of selected)
            this.overlay.set(change.path, change.after === undefined ? TOMBSTONE : change.after.slice());
        if (selected.length)
            this.revisionNumber++;
        return selectedSet;
    }
    validateInstall(changes) {
        for (const change of changes.changes)
            if (!equalBytes(this.current(change.path), change.before))
                throw new FolderConflictError(change.path);
    }
    async install(changes, include, exclude, blocking = true) {
        this.checkWrite();
        const release = await this.writerLock.acquire(blocking);
        try {
            return this.installLocked(changes, include, exclude);
        }
        finally {
            release();
        }
    }
    async installFrom(child, include, exclude, blocking = true) { return this.install(await child.diff(), include, exclude, blocking); }
}
function isSource(value) {
    return !!value && typeof value.paths === 'function' && typeof value.get === 'function';
}
/** Proposals are capabilities, not serializable patches. Only their originating folder may accept them. */
/** Live proposal capability: preserve identity across interpreter eval boundaries. */
export class FolderProposal {
    folder;
    value;
    baseRevision;
    diff;
    constructor(folder, value, baseRevision, diff) {
        this.folder = folder;
        this.value = value;
        this.baseRevision = baseRevision;
        this.diff = diff;
        Object.freeze(this);
    }
}
const proposals = new WeakMap();
const revisions = new WeakMap();
const snapshots = new WeakMap();
/** Immutable, content-addressed source. Branching preserves target ownership, never shares a writer lock. */
export class FolderSnapshot extends Folder {
    digest;
    constructor(files, lineage = {}, scope = '') {
        super(files, 'read', { lineage, scope });
        snapshots.set(this, { lineage, scope });
        this.digest = hexDigest(new TextEncoder().encode(JSON.stringify(this.filePaths().sort().map(path => [path, hexDigest(this.readBytesSync(path))]))));
        let known = revisions.get(lineage);
        if (!known) {
            known = new Map();
            revisions.set(lineage, known);
        }
        known.set(scope + '\0' + this.digest, this);
        Object.freeze(this);
    }
    branch() {
        const authority = snapshots.get(this);
        return new Folder(Object.fromEntries(this.filePaths().map(path => [path, this.readBytesSync(path)])), 'overlay', authority);
    }
    registerComputedFile() { throw new Error('snapshot is immutable'); }
    fork(access = 'overlay') { return access === 'read' ? this : this.branch(); }
}
