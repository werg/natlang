// Adopting a call's code fix (WS1 L3): a successful call that repaired its program with edit_code left its revisions in
// its call record (`code_revisions`). They are a proposal, not a change: this builds the proposed source, runs
// `natlang check` and `natlang test` on it, and installs it only when both pass and nothing that judges the program
// changed. The install is the source-adoption transaction (adoption.ts): stale checkouts are refused, an adoption record
// is written first, and `natlang improve rollback RECORD` restores the exact bytes.
import { cpSync, existsSync, mkdtempSync, readFileSync, readdirSync, rmSync, statSync, symlinkSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, join, relative, resolve, sep } from 'node:path';
import { tmpdir } from 'node:os';
import YAML from 'yaml';
import { Folder } from '../native/scoped-fs.js';
import { hexDigest } from '../native/hash.js';
import { isProtectedSource, protectedSources } from '../runtime/loader.js';
import type { CallRecord, CodeRevisionRecord } from '../calls/types.js';
import { adoptSource, type AdoptionRecord, type SourceManifest } from './adoption.js';

/** What `natlang check` and `natlang test` said about the proposed source (materialized at `directory`). */
export type AdoptionVerdict = { check: { ok: boolean; diagnostics: string[] };
  test: { ok: boolean; passed?: number; failed?: number; skipped?: string; detail?: string } };
export type AdoptCallOptions = {
  /** The program checkout to install into; the call's recorded `program_root` by default. */
  into?: string;
  /** Runs `natlang check` and `natlang test` on a materialized copy of the proposal. */
  verify: (directory: string) => Promise<AdoptionVerdict>;
  /** Where the adoption record is written; `.natlang/adoptions/<call>.json` in the checkout by default. */
  recordPath?: string;
  /** Check and test, but install nothing. */
  dryRun?: boolean;
};
export type AdoptCallResult = { status: 'accepted' | 'rejected' | 'checked'; call: string; directory: string; changed: string[];
  reasons: string[]; verdict?: AdoptionVerdict; record?: AdoptionRecord; recordPath?: string };

type StoreLike = { call(id: string): CallRecord | undefined; value(ref: CodeRevisionRecord['after']): unknown };

const SKIPPED = new Set(['node_modules', '.git', '.natlang', 'dist', '.venv']);

/** Every `.nl` file's `protect:` list and every `types.ts`, as the sources that judge the program (relative paths). */
function judgedSources(directory: string, files: Record<string, string>): { protected: string[]; lists: Record<string, string> } {
  const found: string[] = [], lists: Record<string, string> = {};
  const visit = (path: string) => {
    for (const entry of readdirSync(path)) {
      if (SKIPPED.has(entry) || entry.startsWith('.')) continue;
      const full = join(path, entry), rel = relative(directory, full).split(sep).join('/');
      if (statSync(full).isDirectory()) { visit(full); continue; }
      if (!entry.endsWith('.nl')) continue;
      const text = files[rel] ?? readFileSync(full, 'utf8');
      const front = /^---\r?\n([\s\S]*?)\r?\n---/.exec(text)?.[1];
      let protect: unknown;
      try { protect = front ? (YAML.parse(front) as Record<string, unknown> | null)?.protect : undefined; } catch { protect = undefined; }
      const entries = Array.isArray(protect) ? protect.filter((item): item is string => typeof item === 'string') : [];
      lists[rel] = JSON.stringify(entries);
      found.push(...protectedSources(rel, entries));
    }
  };
  visit(directory);
  return { protected: found, lists };
}

/** Materialize the checkout with the proposed files in place, for check and test (node_modules is linked, not copied). */
function materialize(directory: string, files: Record<string, string>): string {
  const target = mkdtempSync(join(tmpdir(), 'natlang-adopt-call-'));
  for (const entry of readdirSync(directory)) {
    if (entry === '.git' || entry === '.natlang') continue;
    if (entry === 'node_modules') { symlinkSync(join(directory, entry), join(target, entry), 'dir'); continue; }
    cpSync(join(directory, entry), join(target, entry), { recursive: true, filter: source => !source.split(sep).includes('node_modules') });
  }
  for (const [path, text] of Object.entries(files)) { mkdirSync(dirname(join(target, path)), { recursive: true }); writeFileSync(join(target, path), text); }
  return target;
}

/**
 * Check, test and (when both pass) install the code a call changed. A rejection leaves the checkout untouched and
 * says which gate failed; nothing is installed on a dry run.
 */
export async function adoptCall(store: StoreLike, callId: string, options: AdoptCallOptions): Promise<AdoptCallResult> {
  const record = store.call(callId);
  if (!record) throw new Error(`no recorded call ${callId}; natlang traces list shows recent calls`);
  const into = options.into ?? record.program_root;
  if (!into) throw new Error(`call ${callId} did not record its program directory; pass --into DIR`);
  const directory = resolve(into);
  const base = { call: callId, directory, changed: [] as string[] };
  if (record.outcome !== 'done')
    return { ...base, status: 'rejected', reasons: [`the call ended ${record.outcome}; only a call that succeeded proposes its code changes`] };
  const revisions = record.code_revisions ?? [];
  if (!revisions.length) return { ...base, status: 'rejected', reasons: ['the call changed no code; there is nothing to adopt'] };

  // The last revision of each source is the proposal; the first says which text the call started from.
  const proposed: Record<string, string> = {}, before: Record<string, string> = {}, reasons: string[] = [];
  for (const revision of revisions) {
    const path = revision.source;
    const target = resolve(directory, path), rel = relative(directory, target);
    if (!rel || rel.startsWith('..') || rel.split(sep).join('/') !== path) {
      reasons.push(`${path} is outside ${directory}; pass --into the directory the call loaded its program from`); continue;
    }
    if (!(path in before)) {
      if (!existsSync(target)) { reasons.push(`${path} does not exist in ${directory}; pass --into the program the call ran`); continue; }
      const current = readFileSync(target, 'utf8');
      if (hexDigest(current) !== revision.effectiveHash) {
        reasons.push(`${path} changed since the call edited it; rerun the call on the current source to propose the fix again`); continue;
      }
      before[path] = current;
    }
    if (!revision.after.complete) { reasons.push(`the recorded text of ${path} was too large to keep; adopt it by hand from natlang traces show ${callId} --events`); continue; }
    const text = store.value(revision.after);
    if (typeof text !== 'string' || hexDigest(text) !== revision.patchedHash) { reasons.push(`the recorded text of ${path} does not match its revision hash`); continue; }
    proposed[path] = text;
  }
  const changed = Object.keys(proposed).filter(path => proposed[path] !== before[path]).sort();
  if (reasons.length) return { ...base, changed, status: 'rejected', reasons };
  if (!changed.length) return { ...base, status: 'rejected', reasons: ['the edits leave every file as it is; there is nothing to adopt'] };

  // Nothing that judges the program may change: types.ts, items a protect: list names, or a protect: list itself.
  const judged = judgedSources(directory, {}), after = judgedSources(directory, proposed);
  for (const path of changed) {
    if (/(^|\/)types\.ts$/.test(path)) reasons.push(`${path} declares the types results are checked against, so a call cannot change it`);
    else if (isProtectedSource(path, judged.protected) || isProtectedSource(path, after.protected))
      reasons.push(`${path} decides whether calls succeed (protect:), so a call cannot change it`);
    else if (judged.lists[path] !== after.lists[path]) reasons.push(`${path}: an edit cannot change its protect: list`);
  }
  if (reasons.length) return { ...base, changed, status: 'rejected', reasons };

  const files = Object.fromEntries(changed.map(path => [path, proposed[path]!]));
  const baseFiles = Object.fromEntries(changed.map(path => [path, before[path]!]));
  const manifest: SourceManifest = { schema: 'natlang.program-source/1', source: Folder.fromFiles(files).snapshot().digest,
    base: Folder.fromFiles(baseFiles).snapshot().digest, files, baseFiles, contract: { adoptedFrom: callId } };
  let verdict: AdoptionVerdict | undefined;
  const verify = async () => {
    const copy = materialize(directory, files);
    try { verdict = await options.verify(copy); } finally { rmSync(copy, { recursive: true, force: true }); }
    const diagnostics = [...(verdict.check.ok ? [] : ['natlang check failed', ...verdict.check.diagnostics]),
      ...(verdict.test.ok ? [] : [`natlang test failed${verdict.test.detail ? `: ${verdict.test.detail}` : ''}`])];
    return { source: manifest.source, valid: verdict.check.ok && verdict.test.ok, diagnostics, contract: manifest.contract as never };
  };
  if (options.dryRun) {
    const checked = await verify();
    return { ...base, changed, status: checked.valid ? 'checked' : 'rejected', reasons: checked.diagnostics, verdict };
  }
  const recordPath = resolve(options.recordPath ?? join(directory, '.natlang', 'adoptions', `${callId.replace(/[^\w.-]+/g, '_')}.json`));
  try {
    const adopted = await adoptSource(directory, manifest, verify, recordPath);
    return { ...base, changed, status: 'accepted', reasons: [], verdict, record: adopted, recordPath };
  } catch (error) {
    const message = (error as Error).message;
    if (!message.startsWith('adoption build failed')) throw error;
    return { ...base, changed, status: 'rejected', reasons: message.replace(/^adoption build failed: /, '').split('\n'), verdict };
  }
}
