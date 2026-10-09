/**
 * Repository migration in natural language. migrate.nl understands the request, surveys the sites that mention the old
 * thing, classifies each usage, plans, writes and checks one patch per site, applies them, runs the checks, and
 * repairs from the failures in a bounded loop with a natural-language stopping judgment. This file is the outside
 * world those stages act on, as the `repository` service: candidate revisions in memory, search, exact line ranges
 * and counts, applying exact patches, materializing a candidate in a temporary directory to run the declared checks,
 * and the report of changed files. The original checkout is never written. Manifest and check commands are trusted.
 */
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';
import type { NatlangRuntime } from '@natlang/node';
import migrateFlow from './migrate.nl';
import type { Check, ChangedFile, Migration, MigrationReport, Patch, RepoSnapshot, SearchHit, SearchResult, Validation } from './types.js';

export type * from './types.js';
export type CheckCommand = { id: string, argv: string[], timeoutMs?: number };

/** The pluggable policy points, and the implementation each runs when the host does not choose. */
export type PolicyPoint = 'exact' | 'settled';
export type Implementation = 'crisp' | 'natural-language';
export const DEFAULT_POLICY: Record<PolicyPoint, Implementation> = { exact: 'natural-language', settled: 'natural-language' };

/** What the natural-language stages see of the `repository` service. Types are those of types.ts. */
export const repositoryDeclaration = `/** The repository under migration: candidate revisions held in memory, the original checkout untouched. */
/** Which implementation runs a pluggable policy point: 'crisp' or 'natural-language'. */
export function implementation(point: 'exact' | 'settled'): 'crisp' | 'natural-language';
/** Search every manifest file of a revision (default: the base) for an exact string. Hits carry path, 1-based line, offset and an excerpt. */
export function search(query: string, revision?: string): SearchResult;
/** The text of lines from..to (1-based, inclusive) of a file at a revision, exactly as stored. */
export function lines(path: string, from: number, to: number, revision?: string): string;
/** The whole text of a file at a revision. */
export function read(path: string, revision?: string): string;
/** How many times text occurs in a file at a revision (a patch's old text must occur once). */
export function count(path: string, text: string, revision?: string): number;
/** The files of a revision with digests and line counts. */
export function snapshot(revision?: string): RepoSnapshot;
/** Apply patches in order to a revision and return the new revision. Rejects with the path and reason when a path is not in the manifest, old is empty or equals new, or old occurs zero or several times. */
export function apply(base: string, patches: Patch[]): RepoSnapshot;
/** Materialize a revision in a temporary directory, run the declared checks there and remove the directory. */
export function validate(revision: string): Promise<Validation>;
/** The exact report for a validated revision: its status, the files changed against the base with digests, and the checks. */
export function report(revision: string, validation: Validation): MigrationReport;`;

const hash = (value: string) => createHash('sha256').update(value).digest('hex');
const safe = (path: unknown) => typeof path === 'string' && path.length > 0 && !path.startsWith('/') &&
  !path.split(/[\\/]/).some(part => part === '..' || part === '');

function runCheck(check: CheckCommand, directory: string): Promise<Check> {
  return new Promise(resolveCheck => {
    const env = { ...process.env };
    // A nested Node test runner otherwise reports success while skipping files.
    delete env.NODE_TEST_CONTEXT;
    const child = spawn(check.argv[0]!, check.argv.slice(1), { cwd: directory, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let tail = '', bytes = 0, failed = false, timedOut = false;
    const capture = (chunk: Buffer) => { bytes += chunk.length; tail = (tail + chunk.toString('utf8')).slice(-4000); };
    child.stdout.on('data', capture); child.stderr.on('data', capture);
    child.on('error', error => { failed = true; capture(Buffer.from(error.message)); });
    const timeout = check.timeoutMs && check.timeoutMs > 0 ? setTimeout(() => { timedOut = true; child.kill('SIGKILL'); }, check.timeoutMs) : null;
    child.on('close', code => {
      if (timeout) clearTimeout(timeout);
      resolveCheck({ id: check.id, status: !failed && !timedOut && code === 0 ? 'passed' : 'failed', output: tail,
        output_bytes: bytes, truncated: bytes > 4000, ...(timedOut ? { detail: 'timeout' } : {}) });
    });
  });
}

export class RepositoryMigration {
  base = '';
  private readonly policy: Record<PolicyPoint, Implementation>;
  private readonly root: string;
  private readonly files: string[];
  private readonly checks: CheckCommand[];
  private readonly revisions = new Map<string, Record<string, string>>();
  private readonly events: Record<string, unknown>[] = [];

  constructor(root: string, { files, checks = [], policy = {} }: { files: string[], checks?: CheckCommand[], policy?: Partial<Record<PolicyPoint, Implementation>> }) {
    this.policy = { ...DEFAULT_POLICY, ...policy };
    this.root = resolve(root); this.files = [...files]; this.checks = checks;
    if (!this.files.length || this.files.some(path => !safe(path)) || new Set(this.files).size !== this.files.length)
      throw new Error('invalid file manifest');
    if (checks.some(check => !Array.isArray(check.argv) || !check.argv.length || check.argv.some(arg => typeof arg !== 'string') ||
        check.timeoutMs !== undefined && (!Number.isSafeInteger(check.timeoutMs) || check.timeoutMs < 1))) throw new Error('invalid check');
  }

  async open(): Promise<RepoSnapshot> {
    const contents: Record<string, string> = {};
    for (const path of this.files) {
      const absolute = resolve(this.root, path);
      if (!absolute.startsWith(`${this.root}${sep}`)) throw new Error('path escape');
      contents[path] = await readFile(absolute, 'utf8');
    }
    const id = this.identity(contents);
    this.revisions.set(id, contents);
    this.base = id;
    return this.snapshot(id);
  }

  private identity(contents: Record<string, string>): string {
    return hash(JSON.stringify(Object.entries(contents).sort(([a], [b]) => a.localeCompare(b))));
  }

  private contents(revision: string): Record<string, string> {
    const contents = this.revisions.get(revision);
    if (!contents) throw new Error('unknown revision');
    return contents;
  }

  implementation(point: PolicyPoint): Implementation {
    if (!Object.hasOwn(this.policy, point)) throw new Error(`unknown policy point: ${point}`);
    return this.policy[point];
  }

  /** Lines from..to (1-based, inclusive) exactly as stored. */
  lines(path: string, from: number, to: number, revision = this.base): string {
    if (!Number.isSafeInteger(from) || !Number.isSafeInteger(to) || from < 1 || to < from) throw new Error(`invalid line range: ${from}..${to}`);
    return this.read(path, revision).split('\n').slice(from - 1, to).join('\n');
  }

  /** How many times text occurs in a file at a revision. */
  count(path: string, text: string, revision = this.base): number {
    if (typeof text !== 'string' || !text) throw new Error('empty text');
    const contents = this.read(path, revision);
    let found = 0;
    for (let at = contents.indexOf(text); at >= 0; at = contents.indexOf(text, at + text.length)) found++;
    return found;
  }

  snapshot(revision = this.base): RepoSnapshot {
    return { revision, files: Object.entries(this.contents(revision)).map(([path, text]) => ({ path, sha256: hash(text), lines: text.split('\n').length })) };
  }

  search(query: string, revision = this.base): SearchResult {
    if (typeof query !== 'string' || !query) throw new Error('empty query');
    const hits: SearchHit[] = [];
    for (const [path, text] of Object.entries(this.contents(revision)))
      for (let at = text.indexOf(query); at >= 0; at = text.indexOf(query, at + query.length))
        hits.push({ path, offset: at, line: text.slice(0, at).split('\n').length,
          excerpt: text.slice(Math.max(0, at - 50), Math.min(text.length, at + query.length + 50)) });
    this.events.push({ operation: 'repository.search', query, revision, hits: hits.length });
    return { revision, hits };
  }

  read(path: string, revision = this.base): string {
    const contents = this.revisions.get(revision);
    if (!contents || !Object.hasOwn(contents, path)) throw new Error('unknown file or revision');
    return contents[path]!;
  }

  apply(base: string, patches: Patch[]): RepoSnapshot {
    const contents = this.revisions.get(base);
    if (!contents || !Array.isArray(patches) || !patches.length) throw new Error('unknown base or empty patches');
    const updated = { ...contents };
    for (const patch of patches) {
      if (!Object.hasOwn(updated, patch.path)) throw new Error(`invalid patch: ${patch.path} is not a file of the manifest`);
      if (typeof patch.old !== 'string' || !patch.old || typeof patch.new !== 'string' || patch.old === patch.new)
        throw new Error(`invalid patch: ${patch.path} needs a non-empty old text and a new text that differs`);
      const text = updated[patch.path]!, at = text.indexOf(patch.old);
      if (at < 0) throw new Error(`patch context missing: ${patch.path} does not contain the old text ${JSON.stringify(patch.old.slice(0, 80))}`);
      if (text.indexOf(patch.old, at + patch.old.length) >= 0)
        throw new Error(`patch context ambiguous: ${patch.path} contains the old text ${JSON.stringify(patch.old.slice(0, 80))} more than once; extend it with neighboring text`);
      updated[patch.path] = text.slice(0, at) + patch.new + text.slice(at + patch.old.length);
    }
    const revision = this.identity(updated);
    this.revisions.set(revision, updated);
    this.events.push({ operation: 'repository.apply', base, revision,
      patches: patches.map(row => ({ path: row.path, old_sha256: hash(row.old), new_sha256: hash(row.new) })) });
    return this.snapshot(revision);
  }

  async validate(revision: string): Promise<Validation> {
    const contents = this.contents(revision);
    const directory = await mkdtemp(join(tmpdir(), 'natlang-migration-'));
    try {
      for (const [path, text] of Object.entries(contents)) {
        const absolute = join(directory, path);
        await mkdir(dirname(absolute), { recursive: true });
        await writeFile(absolute, text);
      }
      const results: Check[] = [];
      for (const check of this.checks) results.push(await runCheck(check, directory));
      const status = results.every(row => row.status === 'passed') ? 'passed' : 'failed';
      this.events.push({ operation: 'repository.validate', revision, status, checks: results.map(row => ({ id: row.id, status: row.status })) });
      return { revision, status, checks: results };
    } finally { await rm(directory, { recursive: true, force: true }); }
  }

  report(revision: string, validation: Validation): MigrationReport {
    const base = this.contents(this.base), candidate = this.revisions.get(revision);
    if (!candidate || validation.revision !== revision) throw new Error('revision mismatch');
    const changed = this.files.filter(path => candidate[path] !== base[path])
      .map(path => ({ path, before_sha256: hash(base[path]!), after_sha256: hash(candidate[path]!) }));
    return { status: validation.status === 'passed' ? 'reviewable' : 'checks-failed', base: this.base, revision, changed, checks: validation.checks };
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** The options that give natural-language stages the repository as the `repository` service. */
export function migrationServices(repository: RepositoryMigration) {
  return { services: { repository }, serviceDeclarations: { repository: repositoryDeclaration } };
}

/**
 * Migrate the repository as the request says with the natural-language stages: at most `attempts` repair rounds after
 * the first candidate. seeds are extra search strings. The original checkout is untouched.
 */
export async function migrate(runtime: NatlangRuntime, repository: RepositoryMigration, request: string,
  { attempts = 3, seeds = [] as string[], checks = [] as string[] } = {}): Promise<Migration> {
  const snapshot = repository.snapshot();
  return runtime.run(() => migrateFlow(request, snapshot, checks, seeds, attempts), migrationServices(repository));
}
