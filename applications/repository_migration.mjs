/** Isolated candidate revisions for semantic repository migrations. */
import { createHash } from 'node:crypto';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, join, resolve, sep } from 'node:path';

const exec = promisify(execFile);
const hash = value => createHash('sha256').update(value).digest('hex');
const safe = path => typeof path === 'string' && path.length > 0 &&
  !path.startsWith('/') && !path.split(/[\\/]/).some(part => part === '..' || part === '');

export class RepositoryMigration {
  constructor(root, { files, checks = [] }) {
    this.root = resolve(root); this.files = [...files]; this.checks = checks;
    if (!this.files.length || this.files.some(path => !safe(path)) ||
        new Set(this.files).size !== this.files.length) throw new Error('invalid file manifest');
    if (checks.some(check => !Array.isArray(check.argv) || !check.argv.length ||
        check.argv.some(arg => typeof arg !== 'string'))) throw new Error('invalid check');
    this.revisions = new Map(); this.events = [];
  }

  async open() {
    const contents = {};
    for (const path of this.files) {
      const absolute = resolve(this.root, path);
      if (!absolute.startsWith(`${this.root}${sep}`)) throw new Error('path escape');
      contents[path] = await readFile(absolute, 'utf8');
    }
    const id = this.#identity(contents);
    this.revisions.set(id, contents);
    this.base = id;
    return this.snapshot(id);
  }

  #identity(contents) {
    return hash(JSON.stringify(Object.entries(contents).sort(([a], [b]) => a.localeCompare(b))));
  }

  snapshot(revision = this.base) {
    const contents = this.revisions.get(revision);
    if (!contents) throw new Error('unknown revision');
    return { revision, files: Object.entries(contents).map(([path, text]) => ({
      path, sha256: hash(text), lines: text.split('\n').length })) };
  }

  search(query, revision = this.base) {
    if (typeof query !== 'string' || !query) throw new Error('empty query');
    const contents = this.revisions.get(revision);
    if (!contents) throw new Error('unknown revision');
    const hits = [];
    for (const [path, text] of Object.entries(contents)) {
      let cursor = 0;
      while (true) {
        const at = text.indexOf(query, cursor);
        if (at < 0) break;
        hits.push({ path, offset: at, line: text.slice(0, at).split('\n').length,
          excerpt: text.slice(Math.max(0, at - 50), Math.min(text.length, at + query.length + 50)) });
        cursor = at + query.length;
      }
    }
    this.events.push({ operation: 'repository.search', query, revision, hits: hits.length });
    return { revision, hits };
  }

  read(path, revision = this.base) {
    const contents = this.revisions.get(revision);
    if (!contents || !Object.hasOwn(contents, path)) throw new Error('unknown file or revision');
    return contents[path];
  }

  apply(base, patches) {
    const contents = this.revisions.get(base);
    if (!contents || !Array.isArray(patches) || !patches.length)
      throw new Error('unknown base or empty patches');
    const updated = { ...contents };
    for (const patch of patches) {
      if (!Object.hasOwn(updated, patch.path) || typeof patch.old !== 'string' ||
          !patch.old || typeof patch.new !== 'string' || patch.old === patch.new)
        throw new Error('invalid patch');
      const text = updated[patch.path], at = text.indexOf(patch.old);
      if (at < 0 || text.indexOf(patch.old, at + patch.old.length) >= 0)
        throw new Error(`patch context missing or ambiguous: ${patch.path}`);
      updated[patch.path] = text.slice(0, at) + patch.new + text.slice(at + patch.old.length);
    }
    const revision = this.#identity(updated);
    this.revisions.set(revision, updated);
    this.events.push({ operation: 'repository.apply', base, revision,
      patches: patches.map(row => ({ path: row.path, old_sha256: hash(row.old),
        new_sha256: hash(row.new) })) });
    return this.snapshot(revision);
  }

  async validate(revision) {
    const contents = this.revisions.get(revision);
    if (!contents) throw new Error('unknown revision');
    const directory = await mkdtemp(join(tmpdir(), 'natlang-migration-'));
    try {
      for (const [path, text] of Object.entries(contents)) {
        const absolute = join(directory, path);
        await mkdir(dirname(absolute), { recursive: true });
        await writeFile(absolute, text);
      }
      const results = [];
      for (const check of this.checks) {
        try {
          const env = { ...process.env };
          // A nested Node test runner otherwise reports success while skipping files.
          delete env.NODE_TEST_CONTEXT;
          const result = await exec(check.argv[0], check.argv.slice(1), {
            cwd: directory, timeout: check.timeoutMs ?? 0,
            maxBuffer: 1024 * 1024, env });
          results.push({ id: check.id, status: 'passed',
            output: (result.stdout + result.stderr).slice(-4000) });
        } catch (error) {
          results.push({ id: check.id, status: 'failed',
            output: String((error.stdout ?? '') + (error.stderr ?? '') || error.message).slice(-4000) });
        }
      }
      const status = results.every(row => row.status === 'passed') ? 'passed' : 'failed';
      this.events.push({ operation: 'repository.validate', revision, status,
        checks: results.map(row => ({ id: row.id, status: row.status })) });
      return { revision, status, checks: results };
    } finally { await rm(directory, { recursive: true, force: true }); }
  }

  report(revision, validation) {
    const base = this.revisions.get(this.base), candidate = this.revisions.get(revision);
    if (!candidate || validation.revision !== revision) throw new Error('revision mismatch');
    const changed = this.files.filter(path => candidate[path] !== base[path]).map(path => ({
      path, before_sha256: hash(base[path]), after_sha256: hash(candidate[path]) }));
    return { status: validation.status === 'passed' ? 'reviewable' : 'checks-failed',
      base: this.base, revision, changed, checks: validation.checks };
  }

  drainEvents() { return this.events.splice(0); }
}
