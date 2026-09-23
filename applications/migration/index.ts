/**
 * Repository migration workbench. Candidate revisions live in memory; each is materialized only in
 * a temporary directory for the declared checks, so the original checkout is never written. Natlang
 * proposes exact patches from search evidence and repairs a failing candidate from its check output,
 * inside a bounded `iterateOn` loop. Manifest and check commands are trusted configuration.
 */
import { createHash } from 'node:crypto';
import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';
import { IterationLimitError, iterateOn } from '@natlang/node';
import propose from './propose.nl';
import type { Check, Patch, RepoSnapshot, SearchHit, SearchResult, Validation } from './types.js';

export type * from './types.js';
export type CheckCommand = { id: string, argv: string[], timeoutMs?: number };
export type ChangedFile = { path: string, before_sha256: string, after_sha256: string };
export type MigrationReport = { status: 'reviewable' | 'checks-failed', base: string, revision: string, changed: ChangedFile[], checks: Check[] };

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
  private readonly root: string;
  private readonly files: string[];
  private readonly checks: CheckCommand[];
  private readonly revisions = new Map<string, Record<string, string>>();
  private readonly events: Record<string, unknown>[] = [];

  constructor(root: string, { files, checks = [] }: { files: string[], checks?: CheckCommand[] }) {
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
      if (!Object.hasOwn(updated, patch.path) || typeof patch.old !== 'string' || !patch.old || typeof patch.new !== 'string' ||
          patch.old === patch.new) throw new Error('invalid patch');
      const text = updated[patch.path]!, at = text.indexOf(patch.old);
      if (at < 0 || text.indexOf(patch.old, at + patch.old.length) >= 0) throw new Error(`patch context missing or ambiguous: ${patch.path}`);
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

/**
 * Propose a candidate for the request, validate it, and repair it from check output until the checks
 * pass or `attempts` candidates have been tried. The original checkout is untouched.
 */
export async function migrate(repository: RepositoryMigration, request: string, query: string, { attempts = 3 } = {}): Promise<MigrationReport> {
  type Candidate = { snapshot: RepoSnapshot, validation?: Validation };
  const next = async ({ snapshot, validation }: Candidate): Promise<Candidate> => {
    const patches = await propose(request, snapshot, repository.search(query, snapshot.revision), validation);
    const candidate = repository.apply(snapshot.revision, patches);
    return { snapshot: candidate, validation: await repository.validate(candidate.revision) };
  };
  const final = await iterateOn(next, { snapshot: repository.snapshot() } as Candidate)
    .withLimit({ maxSteps: attempts }).until(candidate => candidate.validation?.status === 'passed')
    .catch(error => { if (error instanceof IterationLimitError) return error.lastState as Candidate; throw error; });
  return repository.report(final.snapshot.revision, final.validation!);
}
